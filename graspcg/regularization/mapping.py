# graspcg/regularization/mapping.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Tuple, Dict, Optional, Sequence, List
import torch

from .base import Regularizer, RegContext
from ..core.roles import Roles
from ..ops.ndops import fwd_diff, tv_iso_energy, tv_aniso_energy  # for quantile sampling

# ------------------------ Unified operator API ------------------------

class Op(Protocol):
    """
    Functional operator that can be composed:
      u_ext = forward_apply(z_ext, ctx, interior)
      grad_z_int = adjoint_apply(grad_u_int, ctx, interior)

    Notes:
      • 'z_ext' / 'u_ext' live on the shard device and may include halo along the sharded batch axis (dim 0).
      • 'interior' is the write slice (drop halo) in dim 0.
      • adjoint_apply MUST NOT write into ctx.g or any ws buffers unless it's a ParamOp
        that targets parameters (in that case, it returns a ZERO tensor for the chain).
    """
    def forward_apply(self, z_ext: torch.Tensor, ctx: RegContext, interior: Tuple[slice, ...]) -> torch.Tensor: ...
    def adjoint_apply(self, grad_u_int: torch.Tensor, ctx: RegContext, interior: Tuple[slice, ...]) -> torch.Tensor: ...
    def halo_extra(self, roles: Roles) -> Dict[int, int]: return {}
    def roles_for_output(self, roles: Roles) -> Roles: return roles

# ------------------------ Concrete ops ------------------------

class IdentityOp:
    def forward_apply(self, z_ext: torch.Tensor, ctx: RegContext, interior: Tuple[slice, ...]) -> torch.Tensor:
        return z_ext
    def adjoint_apply(self, grad_u_int: torch.Tensor, ctx: RegContext, interior: Tuple[slice, ...]) -> torch.Tensor:
        return grad_u_int
    def halo_extra(self, roles: Roles) -> Dict[int, int]: return {}
    def roles_for_output(self, roles: Roles) -> Roles:
        # Identity mapping doesn't change axis semantics
        return roles
class ScaleOp:
    """
    Multiply by (1/s) from ws.scale.inv_for_shard. Acts on FIELD (x).
    """
    def forward_apply(self, z_ext: torch.Tensor, ctx: RegContext, interior: Tuple[slice, ...]) -> torch.Tensor:
        ws = getattr(ctx, "ws", None)
        i  = getattr(ctx, "shard_index", None)
        if ws is None or i is None or not hasattr(ws, "scale"):
            return z_ext
        sh = ws.shard_for_index(i)
        inv_s = ws.scale.inv_for_shard(sh, anchor=z_ext)  # (B_loc,1,...,1)
        B_ext = int(z_ext.shape[0])
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
        return z_ext * inv_ext

    def adjoint_apply(self, grad_u_int: torch.Tensor, ctx: RegContext, interior: Tuple[slice, ...]) -> torch.Tensor:
        ws = getattr(ctx, "ws", None)
        i  = getattr(ctx, "shard_index", None)
        if ws is None or i is None or not hasattr(ws, "scale"):
            return grad_u_int
        sh = ws.shard_for_index(i)
        inv_s = ws.scale.inv_for_shard(sh, anchor=grad_u_int)
        return grad_u_int * inv_s

    def halo_extra(self, roles: Roles) -> Dict[int, int]: return {}
    def roles_for_output(self, roles: Roles) -> Roles:
        # Pointwise scaling doesn't change axis semantics
        return roles
@dataclass
class TemporalBasisOp:
    """
    Parameter-space op: u = U @ V, where V is read from workspace, and adjoint writes to g_V.

    U : (B_total, K) or (T, K) aligned with the sharded batch axis.
    param_key: ws buffer name for V (K, C, H, W, ...)
    grad_key : ws buffer name for g_V
    """
    U: torch.Tensor
    param_key: str = "V"
    grad_key: str  = "g_V"

    def _rows(self, row0: int, row1: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.U[row0:row1].to(device=device, dtype=dtype, non_blocking=True)

    def forward_apply(self, z_ext: torch.Tensor, ctx: RegContext, interior: Tuple[slice, ...]) -> torch.Tensor:
        # ignore z_ext; build from params
        assert ctx.ws is not None and ctx.shard_index is not None, "TemporalBasisOp requires ws/shard_index"
        ws, i = ctx.ws, ctx.shard_index
        x_i = ws.get("x", i)
        B_loc = int(x_i.shape[0])
        sh = ws.shard_for_index(i)
        B0 = int(sh.b_start)

        # halo inferred from interior
        B_ext = int(z_ext.shape[0])
        start = int(interior[0].start or 0)
        stop  = int(interior[0].stop  or B_ext)
        left  = start
        right = B_ext - stop

        row0, row1 = B0 - left, B0 + B_loc + right

        V = ws.get(self.param_key, i)                 # (K, C, H, W, ...) complex or real
        dev = V.device
        U_ext = self._rows(row0, row1, dev, V.dtype)  # (B_ext, K) -- match V dtype (complex-safe)
        N = int(V.numel() // V.shape[0])             # flatten non-K dims
        V2 = V.reshape(V.shape[0], N)                # (K, N)
        u2 = U_ext @ V2                               # (B_ext, N)
        return u2.reshape((U_ext.shape[0],) + tuple(V.shape[1:]))

    def adjoint_apply(self, grad_u_int: torch.Tensor, ctx: RegContext, interior: Tuple[slice, ...]) -> torch.Tensor:
        # Write ∂E/∂V into workspace and return zeros to previous stage
        assert ctx.ws is not None and ctx.shard_index is not None
        ws, i = ctx.ws, ctx.shard_index
        sh = ws.shard_for_index(i)
        B_loc = int(ws.get("x", i).shape[0])
        dev = grad_u_int.device
        V = ws.get(self.param_key, i)                 # dtype target for adjoint (complex-safe)
        U_int = self.U[int(sh.b_start): int(sh.b_start) + B_loc].to(device=dev, dtype=V.dtype, non_blocking=True)
        gu2 = grad_u_int.reshape(B_loc, -1)          # (B_loc, N)
        gV2 = U_int.conj().transpose(0, 1) @ gu2     # (K, N)
        gV  = gV2.reshape(V.shape)
        ws.get(self.grad_key, i).add_(gV)

        # return zeros (no gradient to a previous z)
        return torch.zeros_like(ctx.g[interior])

    def halo_extra(self, roles: Roles) -> Dict[int, int]: return {}
    def roles_for_output(self, roles: Roles) -> Roles:
        # u = U @ V changes only the batch/time extent, but axis *semantics* stay the same
        # for the TV regularizer. Keep roles unchanged.
        return roles
# ------------------------ Composition ------------------------

class ComposeOp:
    """Compose multiple ops: forward left->right, adjoint right->left."""
    def __init__(self, ops: Sequence[Op]):
        self.ops: List[Op] = list(ops)
    def forward_apply(self, z_ext: torch.Tensor, ctx: RegContext, interior: Tuple[slice, ...]) -> torch.Tensor:
        u = z_ext
        for op in self.ops:
            u = op.forward_apply(u, ctx, interior)
        return u
    def adjoint_apply(self, grad_u_int: torch.Tensor, ctx: RegContext, interior: Tuple[slice, ...]) -> torch.Tensor:
        grad = grad_u_int
        for op in reversed(self.ops):
            grad = op.adjoint_apply(grad, ctx, interior)
        return grad
    def halo_extra(self, roles: Roles) -> Dict[int, int]:
        merged: Dict[int, int] = {}
        for op in self.ops:
            for ax, k in op.halo_extra(roles).items():
                merged[ax] = max(merged.get(ax, 0), int(k))
        return merged
    def roles_for_output(self, roles: Roles) -> Roles:
        out = roles
        for op in self.ops:
            out = op.roles_for_output(out)
        return out

# ------------------------ LinOp adapter ------------------------

class LinOpAdapter:
    """
    Adapt a simple linear operator with methods:
        y = fwd(x)
        x = adj(y)
    to the unified Op API. It ignores ws/interior and has no halo requests.
    """
    def __init__(self, linop) -> None:
        self.linop = linop
    def forward_apply(self, z_ext: torch.Tensor, ctx: RegContext, interior: Tuple[slice, ...]) -> torch.Tensor:
        return self.linop.fwd(z_ext)
    def adjoint_apply(self, grad_u_int: torch.Tensor, ctx: RegContext, interior: Tuple[slice, ...]) -> torch.Tensor:
        return self.linop.adj(grad_u_int)
    def halo_extra(self, roles: Roles) -> Dict[int, int]: return {}
    def roles_for_output(self, roles: Roles) -> Roles: return roles

# ------------------------ Mapped regularizer ------------------------

@dataclass
class MappedRegularizer(Regularizer):
    """
    Wrap a regularizer 'inner' to act on u = Op(x, ws, ...).
    The Op can be a field transform (returning grad to x) or a parameter op
    (writing grads to workspace and returning zeros to the chain).
    """
    name: str
    inner: Regularizer
    op: Op

    @property
    def params(self):
        return getattr(self.inner, "params", None)

    @params.setter
    def params(self, v):
        setattr(self.inner, "params", v)

    def halo(self, roles: Roles):
        h1 = {}
        try:
            h1 = self.inner.halo(roles) or {}
        except Exception:
            pass
        h2 = {}
        try:
            h2 = self.op.halo_extra(roles) or {}
        except Exception:
            pass
        out: Dict[int, int] = dict(h1)
        for ax, k in h2.items():
            out[ax] = max(out.get(ax, 0), int(k))
        return out

    def add_diag(self, ctx: RegContext) -> None:
        # Only safe for identity mapping; otherwise no‑op.
        try:
            if isinstance(self.op, IdentityOp):
                return self.inner.add_diag(ctx)
        except Exception:
            pass
        return None

    def energy_grad_fixed_axes(self, ctx: RegContext, axes: Tuple[int, ...]) -> torch.Tensor:
        """
        Compile‑friendly path: manager resolves axes once and calls this.
        Returns a 0‑D REAL tensor (shard energy) and accumulates grad into ctx.g.
        """
        interior = ctx.write_interior_slice or (slice(None),) * ctx.x.ndim
        # If the mapping changes axis semantics, reflect that here (fallback to identity):
        roles_u = self.op.roles_for_output(ctx.roles_image) if hasattr(self.op, "roles_for_output") else ctx.roles_image

        # Forward through op chain
        u_ext = self.op.forward_apply(ctx.x, ctx, interior)
        g_u   = torch.zeros_like(u_ext, device=u_ext.device, dtype=ctx.dtype_c)

        sub = RegContext(
            x=u_ext, g=g_u, diag=None,
            roles_image=roles_u, device=u_ext.device,
            dtype_c=ctx.dtype_c, dtype_r=ctx.dtype_r,
            axes_resolver=ctx.axes_resolver,  # manager-provided resolver ok if roles unchanged
            arena=ctx.arena, write_interior_slice=interior,
            ws=ctx.ws, shard_index=ctx.shard_index, halo_map=ctx.halo_map,
        )

        # Call inner (prefer fixed-axes if available)
        if hasattr(self.inner, "energy_grad_fixed_axes"):
            E = self.inner.energy_grad_fixed_axes(sub, axes)
        else:
            E = self.inner.energy_grad(sub)

        # Adjoint through op chain, write back to ctx.g (field) or to parameter grads
        g_u_int = g_u[interior] if interior is not None else g_u
        g_prev  = self.op.adjoint_apply(g_u_int, ctx, interior)
        if g_prev is not None:
            ctx.g[interior].add_(g_prev)

        return E

    def energy_grad(self, ctx: RegContext) -> torch.Tensor:
        """
        Convenience path when caller didn’t pre-resolve axes.
        We resolve axes using roles AFTER mapping (roles_for_output).
        """
        # Decide axes using roles_for_output to be robust to role changes (fallback to identity)
        roles_u = self.op.roles_for_output(ctx.roles_image) if hasattr(self.op, "roles_for_output") else ctx.roles_image
        try:
            spec = getattr(self.inner, "params", None)
            spec = getattr(spec, "axes", "spatial")
        except Exception:
            spec = "spatial"

        # Prefer a resolver if the manager injected one; otherwise use roles_u directly
        if ctx.axes_resolver is not None:
            try:
                axes = tuple(int(a) for a in ctx.axes_resolver(spec))
            except Exception:
                axes = roles_u.resolve_axes(spec)
        else:
            axes = roles_u.resolve_axes(spec)

        return self.energy_grad_fixed_axes(ctx, axes)

    # Continuation: sample TV magnitude on the mapped field (scalar-only path)
    def quantile_sample_shard(self, ctx: RegContext, axes: Tuple[int, ...], K_shard: int, q: float) -> Optional[torch.Tensor]:
        """
        Return a small 1‑D tensor of TV magnitudes sampled from the mapped field u,
        using the provided axes (already resolved by the manager).
        """
        interior = ctx.write_interior_slice or (slice(None),) * ctx.x.ndim
        u_ext = self.op.forward_apply(ctx.x, ctx, interior)
        grads = [fwd_diff(u_ext, ax) for ax in axes]

        eps  = torch.as_tensor(getattr(getattr(self.inner, "params", None), "eps", 0.0),
                               device=u_ext.device, dtype=ctx.dtype_r)
        isot = bool(getattr(getattr(self.inner, "params", None), "isotropic", True))

        e_den_full = tv_iso_energy(grads, eps) if isot else tv_aniso_energy(grads, eps)
        e_den = e_den_full[interior] if interior is not None else e_den_full
        flat = e_den.reshape(-1)
        nvox = int(flat.numel())
        if nvox == 0:
            return None

        K = max(1, int(K_shard))
        stride = max(1, nvox // K)
        sample = flat[::stride]
        if int(sample.numel()) > K:
            sample = sample[:K]
        return sample