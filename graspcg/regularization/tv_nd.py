# graspcg/regularization/tv_nd.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple, List, Union
import torch

from .base import Regularizer, RegParams, RegContext, AxesSpec
from ..core.roles import Roles
from ..ops.ndops import (  # compile‑friendly TV stencils
    fwd_diff, tv_aniso_div, tv_aniso_flux, tv_aniso_energy,
    tv_iso_div,  tv_iso_flux,  tv_iso_energy,
)



# --------------------------- params ---------------------------

@dataclass(frozen=True)
class TVParams(RegParams):
    """
    N‑D Total Variation parameters.
    • weight: λ
    • eps   : Huber knee
    • axes  : tokens or explicit indices
    • isotropic: ℓ2 coupling across axes if True; else ℓ1 across axes
    • axis_weights: optional per‑axis scalars (same length/order as 'axes')
    """
    weight: float = 0.0
    eps: float = 1e-3
    axes: AxesSpec = "spatial"
    isotropic: bool = True
    axis_weights: Optional[Sequence[float]] = None

class TVND(Regularizer):
    """
    N‑D TV (scale‑free). If you need TV((U V)/s), wrap TVND with a mapped regularizer.

    Contract:
      • energy_grad[_fixed_axes] -> 0‑D REAL on device, accumulates into ctx.g
      • honors ctx.write_interior_slice (halo)
      • supports per‑axis weights via params.axis_weights
    """
    Params = TVParams

    def __init__(self, name: str, params: TVParams):
        self.name = name
        self.params = params

    def energy_grad(self, ctx: RegContext) -> torch.Tensor:
        axes = self._resolve_axes(ctx)
        return self.energy_grad_fixed_axes(ctx, axes)

    def energy_grad_fixed_axes(self, ctx: RegContext, axes: Tuple[int, ...]) -> torch.Tensor:
        dev, dr = ctx.device, ctx.dtype_r
        if len(axes) == 0 or self.params.weight == 0.0:
            return torch.zeros((), device=dev, dtype=dr)

        # per-axis weights (anisotropic dimensions)
        if self.params.axis_weights is not None:
            if len(self.params.axis_weights) != len(axes):
                raise ValueError(f"axis_weights must match len(axes).")
            w_axes = [torch.tensor(float(w), device=dev, dtype=dr) for w in self.params.axis_weights]
        else:
            w_axes = [torch.tensor(1.0, device=dev, dtype=dr) for _ in axes]

        x_ext = ctx.x
        interior = ctx.write_interior_slice or (slice(None),) * x_ext.ndim

        grads = [wi * fwd_diff(x_ext, ax) for wi, ax in zip(w_axes, axes)]
        w   = torch.tensor(self.params.weight, device=dev, dtype=dr)
        eps = torch.tensor(self.params.eps,    device=dev, dtype=dr)

        if self.params.isotropic:
            e_den_full = tv_iso_energy(grads, eps)
            flux = tv_iso_flux(grads, eps)
            div_p_full = tv_iso_div(flux, axes)
        else:
            e_den_full = tv_aniso_energy(grads, eps)
            flux = tv_aniso_flux(grads, eps)
            div_p_full = tv_aniso_div(flux, axes)

        e_den = e_den_full[interior] if interior is not None else e_den_full
        div_p = div_p_full[interior] if interior is not None else div_p_full

        ctx.g.add_(div_p.neg_().mul_(w))
        E = (e_den * w).sum()
        return E if E.dtype == dr else E.to(dr)

    def add_diag(self, ctx: RegContext) -> None:
        if ctx.diag is None or self.params.weight == 0.0:
            return False

        axes = self._resolve_axes(ctx)
        if not axes or self.params.weight == 0.0 or ctx.diag is None:
            return False

        interior = ctx.write_interior_slice or (slice(None),) * ctx.x.ndim
        Dint     = ctx.diag[interior]
        lam      = torch.as_tensor(self.params.weight, device=Dint.device, dtype=Dint.dtype)
        eps      = torch.as_tensor(self.params.eps,    device=Dint.device, dtype=Dint.dtype)

        # per-axis parameter weights
        if self.params.axis_weights is not None:
            w2 = [float(w)**2 for w in self.params.axis_weights]
        else:
            w2 = [1.0 for _ in axes]

        # optional temporal scaling (B_loc,1,1,...)
        inv_s2 = None
        ws, i  = getattr(ctx, "ws", None), getattr(ctx, "shard_index", None)
        if ws is not None and i is not None and hasattr(ws, "scale"):
            sh = ws.shard_for_index(i)
            inv_s2 = ws.scale.inv_for_shard(sh, anchor=Dint)
            inv_s2 = inv_s2 * inv_s2  # (B_loc,1,1,...)

        for ax, wa2 in zip(axes, w2):
            # per-axis grad (on x_ext to respect halo)
            gi = fwd_diff(ctx.x, ax)

            # anisotropic weight (REAL field), reuse gi storage lifetime
            wi = torch.rsqrt((gi.real*gi.real + gi.imag*gi.imag) + eps*eps)

            scale_field = lam * wa2 * wi
            if inv_s2 is not None:
                scale_field = scale_field * inv_s2  # broadcasts

            n = int(Dint.shape[ax])
            s = [slice(None)] * Dint.ndim

            if n >= 1:
                s[ax] = 0
                Dint[tuple(s)].add_(scale_field[tuple(s)])
            if n >= 2:
                s[ax] = n - 1
                Dint[tuple(s)].add_(scale_field[tuple(s)])
            if n >= 3:
                s[ax] = slice(1, n-1)
                Dint[tuple(s)].add_(scale_field[tuple(s)] * 2.0)

        return True

    def majorizer_diag(self, ctx: RegContext) -> Optional[torch.Tensor]:
        """
        Return a 0‑D REAL tensor 'k' such that a diagonal preconditioner
        on the *mapped* field can be formed by pushing k back through the op.
        This matches the constant add_diag used for identity mapping.
        """
        axes = self._resolve_axes(ctx)
        if not axes or self.params.weight == 0.0:
            return torch.zeros((), device=ctx.device, dtype=ctx.dtype_r)

        if self.params.axis_weights is not None:
            if len(self.params.axis_weights) != len(axes):
                # If mismatched, be conservative and skip
                return torch.zeros((), device=ctx.device, dtype=ctx.dtype_r)
            sum_w2 = sum(float(w)**2 for w in self.params.axis_weights)
        else:
            sum_w2 = float(len(axes))

        k = torch.as_tensor(2.0 * self.params.weight * sum_w2,
                            device=ctx.device, dtype=ctx.dtype_r)
        return k
    def majorizer_profile(self, ctx: RegContext) -> Optional[List[Tuple[int, torch.Tensor]]]:
        """
        Return [(0, v_t)] where v_t is the 1‑D Laplacian degree along the batch/time axis,
        scaled by 2 * λ * (w_t)^2. If time axis is not included, return None.
        """
        axes = self._resolve_axes(ctx)
        if self.params.weight == 0.0 or 0 not in axes:
            return None

        # axis weight for axis 0 (default 1.0)
        if self.params.axis_weights is not None:
            if len(self.params.axis_weights) != len(axes):
                return None
            w_map = {ax: float(w) for ax, w in zip(axes, self.params.axis_weights)}
            w0 = w_map.get(0, 1.0)
        else:
            w0 = 1.0

        interior = ctx.write_interior_slice or (slice(None),) * ctx.x.ndim
        B_int = int(ctx.x[interior].shape[0])

        v = torch.ones(B_int, device=ctx.device, dtype=ctx.dtype_r)
        if B_int > 2:
            v[1:-1] = 2.0

        scale = torch.as_tensor(2.0 * self.params.weight * (w0 ** 2),
                                device=ctx.device, dtype=ctx.dtype_r)
        v.mul_(scale)
        return [(0, v)]    
    def continuation_update(self, stats: Mapping[str, float]) -> bool:
        return False

    def scaling_policy(self, ctx: RegContext):
        return None

    def prox_inplace(self, ctx: RegContext, step: float) -> None:
        pass

    def halo(self, roles: Roles) -> Mapping[int, int]:
        # request 1‑sample halo along temporal (absolute axis 0) if included
        h = {}
        axes = roles.resolve_axes(self.params.axes)
        if any(ax == 0 for ax in axes):
            h[0] = 1
        return h

    def _resolve_axes(self, ctx: RegContext) -> Tuple[int, ...]:
        if getattr(ctx, "axes_resolver", None) is not None:
            return tuple(int(a) for a in ctx.axes_resolver(self.params.axes))
        return ctx.roles_image.resolve_axes(self.params.axes)


