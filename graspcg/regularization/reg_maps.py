# graspcg/regularization/reg_maps.py
from __future__ import annotations
from typing import Protocol, Tuple, Dict, Optional, Sequence, List
import torch
from ..core.roles import Roles

class RegMap(Protocol):
    """
    Mapping from solver variable (x_base) to regularizer input u, with adjoint pullback.

    forward(ctx, x_ext, interior) -> u_ext
      - x_ext: halo-extended read tensor from workspace (shape like ws.x shard, maybe with halo).
      - return u_ext, same batch/halo semantics; can change inner-axis lengths.
    pullback(ctx, g_u_interior, out_g, interior) -> None
      - Accumulate gradient wrt base variable into out_g[interior].
    Optional helpers:
      - roles_for_u(roles): if mapping changes semantic axis categories; default is identity.
      - halo_extra(roles, axes): if the map needs extra halo; default is none.
    """
    def forward(self, ctx, x_ext: torch.Tensor, interior: Tuple[slice, ...]) -> torch.Tensor: ...
    def pullback(self, ctx, g_u_interior: torch.Tensor, out_g: torch.Tensor,
                 interior: Tuple[slice, ...]) -> None: ...
    def roles_for_u(self, roles: Roles) -> Roles: return roles
    def halo_extra(self, roles: Roles, axes: Tuple[int, ...]) -> Dict[int, int]: return {}

# -------------------- Identity --------------------

class IdentityMap:
    def forward(self, ctx, x_ext: torch.Tensor, interior: Tuple[slice, ...]) -> torch.Tensor:
        return x_ext
    def pullback(self, ctx, g_u_interior: torch.Tensor, out_g: torch.Tensor,
                 interior: Tuple[slice, ...]) -> None:
        out_g[interior].add_(g_u_interior)

# -------------------- Scale (uses ctx.ws.scale if present) --------------------

class ScaleMap:
    """
    Uses per-shard (1/s) from workspace if available:
        inv_s = ws.scale.inv_for_shard(shard, anchor=...)
    Extends (1/s) across halo along the sharded batch axis and applies chain rule.
    """
    def forward(self, ctx, x_ext: torch.Tensor, interior: Tuple[slice, ...]) -> torch.Tensor:
        ws = getattr(ctx, "ws", None)
        sh_i = getattr(ctx, "shard_index", None)
        if ws is None or sh_i is None or not hasattr(ws, "scale"):
            return x_ext
        sh = ws.shard_for_index(sh_i)
        inv_s = ws.scale.inv_for_shard(sh, anchor=x_ext)
        B_ext = int(x_ext.shape[0])
        start = int(interior[0].start or 0)
        stop  = int(interior[0].stop  or B_ext)
        left  = start
        right = B_ext - stop
        if left > 0 or right > 0:
            inv_ext = torch.cat([
                inv_s[0:1].expand(left,  *inv_s.shape[1:]),
                inv_s,
                inv_s[-1:].expand(right, *inv_s.shape[1:]),
            ], dim=0)
        else:
            inv_ext = inv_s
        return x_ext * inv_ext

    def pullback(self, ctx, g_u_interior: torch.Tensor, out_g: torch.Tensor,
                 interior: Tuple[slice, ...]) -> None:
        ws = getattr(ctx, "ws", None)
        sh_i = getattr(ctx, "shard_index", None)
        if ws is None or sh_i is None or not hasattr(ws, "scale"):
            out_g[interior].add_(g_u_interior)
            return
        sh = ws.shard_for_index(sh_i)
        inv_s = ws.scale.inv_for_shard(sh, anchor=g_u_interior)
        out_g[interior].add_(g_u_interior * inv_s)

# -------------------- TemporalBasis: u = U @ v along an inner axis --------------------

class TemporalBasisMap:
    """
    Let v have an inner axis of length R (coeffs). Produce frames u with length T via U (T x R).
    inner_axis: index among *inner* dims (exclude batch at dim 0). For example, if v has
      shape (B, R, H, W), inner_axis=0 maps that R axis. If (B, H, W, R), inner_axis=2.
    """
    def __init__(self, U: torch.Tensor, inner_axis: int):
        assert U.ndim == 2, "U must be (T,R)"
        self.U_cpu = U.detach().cpu()
        self.inner_axis = int(inner_axis)
        self._cache: Dict[torch.device, torch.Tensor] = {}

    def _U(self, dev: torch.device, dtype: torch.dtype) -> torch.Tensor:
        U = self._cache.get(dev)
        if U is None or U.dtype != dtype:
            U = self.U_cpu.to(device=dev, dtype=dtype, non_blocking=True)
            self._cache[dev] = U
        return U

    def forward(self, ctx, x_ext: torch.Tensor, interior: Tuple[slice, ...]) -> torch.Tensor:
        dev = x_ext.device
        dtype = x_ext.dtype
        ax_abs = 1 + self.inner_axis  # skip batch dim
        v_perm = x_ext.movedim(ax_abs, -1)   # (..., R)
        U = self._U(dev, dtype.real)         # (T, R)
        u_perm = torch.tensordot(v_perm, U.t(), dims=([-1], [-1]))  # (..., T)
        u_ext  = u_perm.movedim(-1, ax_abs)
        return u_ext

    def pullback(self, ctx, g_u_interior: torch.Tensor, out_g: torch.Tensor,
                 interior: Tuple[slice, ...]) -> None:
        dev = g_u_interior.device
        ax_abs = 1 + self.inner_axis
        gu_perm = g_u_interior.movedim(ax_abs, -1)               # (..., T)
        U = self._U(dev, g_u_interior.real.dtype).t().conj()     # (R, T)
        gv_perm = torch.tensordot(gu_perm, U.t(), dims=([-1], [-1]))  # (..., R)
        gv = gv_perm.movedim(-1, ax_abs)
        out_g[interior].add_(gv)

# -------------------- Compose maps --------------------

class ComposeMap:
    """Compose multiple maps: forward left->right, pullback right->left (adjoints)."""
    def __init__(self, ops: Sequence[RegMap]):
        self.ops: List[RegMap] = list(ops)
    def forward(self, ctx, x_ext: torch.Tensor, interior: Tuple[slice, ...]) -> torch.Tensor:
        u = x_ext
        for op in self.ops:
            u = op.forward(ctx, u, interior)
        return u
    def pullback(self, ctx, g_u_interior: torch.Tensor, out_g: torch.Tensor,
                 interior: Tuple[slice, ...]) -> None:
        grad = g_u_interior
        # cascade adjoints using a temp buffer shaped like out_g[interior]
        tmp = torch.zeros_like(out_g[interior])
        for idx, op in enumerate(reversed(self.ops)):
            tmp.zero_()
            op.pullback(ctx, grad, tmp, interior)
            grad = tmp
        out_g[interior].add_(grad)
