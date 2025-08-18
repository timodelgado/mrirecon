# graspcg/regularization/mapped.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Protocol
import torch
from .base import Regularizer, RegContext
from ..ops.ndops import fwd_diff, tv_iso_energy, tv_aniso_energy  # for quantile samples

class FieldMap(Protocol):
    """Map from current shard (ctx) -> u_ext (with halo) and pullback of grad(u)."""
    def forward_ext(self, ctx: RegContext, interior: Tuple[slice, ...]) -> torch.Tensor: ...
    def pullback_interior(self, ctx: RegContext, gu_int: torch.Tensor, interior: Tuple[slice, ...]) -> None: ...

@dataclass
class MappedRegularizer(Regularizer):
    """Wrap an inner regularizer to act on u = Φ(params) instead of x."""
    name: str
    inner: Regularizer
    fmap: FieldMap

    @property
    def params(self): return self.inner.params
    @params.setter
    def params(self, v): setattr(self.inner, "params", v)

    def energy_grad(self, ctx: RegContext) -> torch.Tensor:
        interior = ctx.write_interior_slice or (slice(None),) * ctx.x.ndim
        u_ext = self.fmap.forward_ext(ctx, interior)
        gu = torch.zeros_like(u_ext, device=u_ext.device, dtype=ctx.dtype_c)
        inner_ctx = RegContext(
            x=u_ext, g=gu, diag=None,
            roles_image=ctx.roles_image, device=u_ext.device,
            dtype_c=ctx.dtype_c, dtype_r=ctx.dtype_r,
            axes_resolver=ctx.axes_resolver, arena=ctx.arena,
            write_interior_slice=interior,
            ws=ctx.ws, shard_index=ctx.shard_index, halo_map=ctx.halo_map,
        )
        E = self.inner.energy_grad(inner_ctx)
        gu_int = gu[interior] if interior is not None else gu
        self.fmap.pullback_interior(ctx, gu_int, interior)
        return E

    # Optional: scalar sampling used by RegManager for continuation (quantiles)
    def quantile_sample_shard(self, ctx: RegContext, axes: Tuple[int, ...],
                              K_shard: int, q: float) -> Optional[torch.Tensor]:
        interior = ctx.write_interior_slice or (slice(None),) * ctx.x.ndim
        u_ext = self.fmap.forward_ext(ctx, interior)
        # energy density on mapped field
        grads = [fwd_diff(u_ext, ax) for ax in axes]
        eps = torch.as_tensor(getattr(self.inner.params, "eps", 0.0), device=u_ext.device, dtype=ctx.dtype_r)
        isot = bool(getattr(self.inner.params, "isotropic", True))
        e_den_full = tv_iso_energy(grads, eps) if isot else tv_aniso_energy(grads, eps)
        e_den = e_den_full[interior] if interior is not None else e_den_full
        flat = e_den.reshape(-1)
        nvox = int(flat.numel())
        if nvox == 0:
            return None
        stride = max(1, nvox // max(1, K_shard))
        sample = flat[::stride]
        if int(sample.numel()) > K_shard:
            sample = sample[:K_shard]
        return sample

    # Pass-throughs
    def add_diag(self, ctx: RegContext) -> None:
        try: return self.inner.add_diag(ctx)
        except Exception: return None
    def continuation_update(self, stats) -> bool:
        try: return bool(self.inner.continuation_update(stats))
        except Exception: return False
    def scaling_policy(self, ctx: RegContext): return None
    def prox_inplace(self, ctx: RegContext, step: float) -> None: pass
    def majorizer_diag(self, ctx: RegContext):
        try: return self.inner.majorizer_diag(ctx)
        except Exception: return None
    def halo(self, roles):
        try: return self.inner.halo(roles)
        except Exception: return {}

# Example FieldMap: temporal linear mixing with optional per-frame 1/s
@dataclass
class LinearTimeMix(FieldMap):
    U: torch.Tensor            # (B, K) on CPU or device
    param_key: str = "V"       # ws buffer with coeffs: (K, C, H, W, ...)
    grad_key:  str = "g_V"     # ws buffer to accumulate ∂E/∂V
    apply_inv_s: bool = False  # if True: u_ext = (U_ext @ V) * (1/s_ext)

    def _U_rows(self, start: int, stop: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.U[start:stop].to(device=device, dtype=dtype, non_blocking=True)

    def forward_ext(self, ctx: RegContext, interior: Tuple[slice, ...]) -> torch.Tensor:
        assert ctx.ws is not None and ctx.shard_index is not None, "MappedRegularizer requires ws/shard_index in RegContext"
        ws, i = ctx.ws, ctx.shard_index

        # local sizes and shard start
        x_i = ws.get("x", i)
        B_loc = int(x_i.shape[0])
        sh = ws.shard_for_index(i)
        B0 = int(sh.b_start)

        # halo (temporal) from ctx.halo_map
        t_halo = int((ctx.halo_map or {}).get(0, 0))
        left  = int(interior[0].start or 0)
        # right halo equals t_halo if there is a next shard, else 0
        num_shards = getattr(ws, "num_shards", getattr(ws.plan, "num_shards", None))
        has_next = (num_shards is not None and (i + 1) < int(num_shards))
        right = t_halo if has_next else 0

        row0 = B0 - left
        row1 = B0 + B_loc + right

        V = ws.get(self.param_key, i)                         # (K, C, H, W, ...)
        dev = V.device
        U_ext = self._U_rows(row0, row1, dev, V.real.dtype)   # (B_ext, K)

        # (B_ext, K) @ (K, N) -> (B_ext, N)
        N = int(V.numel() // V.shape[0])
        V2 = V.reshape(V.shape[0], N)
        u2 = U_ext @ V2
        u_ext = u2.reshape((U_ext.shape[0],) + tuple(V.shape[1:]))

        if self.apply_inv_s and getattr(ws, "scale", None) is not None:
            inv_s = ws.scale.inv_for_shard(sh, anchor=V)  # (B_loc,1,...,1)
            if left > 0 or right > 0:
                inv_ext = torch.cat([
                    inv_s[0:1].expand(left,  *inv_s.shape[1:]),
                    inv_s,
                    inv_s[-1:].expand(right, *inv_s.shape[1:]),
                ], dim=0)
            else:
                inv_ext = inv_s
            u_ext = u_ext * inv_ext
        return u_ext

    def pullback_interior(self, ctx: RegContext, gu_int: torch.Tensor, interior: Tuple[slice, ...]) -> None:
        assert ctx.ws is not None and ctx.shard_index is not None
        ws, i = ctx.ws, ctx.shard_index
        sh = ws.shard_for_index(i)
        B_loc = int(ws.get("x", i).shape[0])
        dev = gu_int.device

        # If scaling was applied in forward: chain rule w.r.t. z = U V
        if self.apply_inv_s and getattr(ws, "scale", None) is not None:
            inv_s = ws.scale.inv_for_shard(sh, anchor=gu_int)
            gu_int = gu_int * inv_s

        # ∂E/∂V = U_int^H @ ∂E/∂z  (use the interior B_loc rows)
        U_int = self.U[int(sh.b_start): int(sh.b_start) + B_loc].to(device=dev, dtype=gu_int.real.dtype, non_blocking=True)
        gu2 = gu_int.reshape(B_loc, -1)
        gV2 = U_int.conj().transpose(0, 1) @ gu2
        gV  = gV2.reshape(ws.get(self.param_key, i).shape)
        ws.get(self.grad_key, i).add_(gV)
