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

# --------------------------- helpers ---------------------------

def _resolve_axes_local(spec: AxesSpec, roles: Roles) -> Tuple[int, ...]:
    u, l, n = int(roles.unlike), int(roles.like), int(roles.nufft)
    total = u + l + n
    def rng(s, c): return list(range(s, s+c))
    tok = {
        "temporal": rng(0, u), "time": rng(0, u), "unlike": rng(0, u),
        "like": rng(u, l),
        "spatial": rng(u+l, n), "nufft": rng(u+l, n), "image": rng(u+l, n),
    }
    out: List[int] = []
    if isinstance(spec, str):
        out.extend(tok.get(spec.lower(), []))
    elif isinstance(spec, (list, tuple)):
        for s in spec:
            if isinstance(s, str):
                out.extend(tok.get(s.lower(), []))
            elif isinstance(s, int):
                ax = s if s >= 0 else (total + s)
                if ax < 0 or ax >= total:
                    raise ValueError(f"Axis index {s} out of range for dims={total}")
                out.append(ax)
            else:
                raise TypeError(f"Bad axis element: {type(s)}")
    else:
        raise TypeError(f"Bad axes spec type: {type(spec)}")
    # dedup preserve order
    seen, uniq = set(), []
    for a in out:
        if a not in seen:
            seen.add(a); uniq.append(a)
    return tuple(uniq)

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
            return
        axes = self._resolve_axes(ctx)
        if not axes:
            return
        if self.params.axis_weights is not None:
            if len(self.params.axis_weights) != len(axes):
                return
            sum_w2 = sum(float(w)**2 for w in self.params.axis_weights)
        else:
            sum_w2 = float(len(axes))
        k = torch.as_tensor(2.0 * self.params.weight * sum_w2, device=ctx.device, dtype=ctx.dtype_r)
        ctx.diag.add_(k)

    def continuation_update(self, stats: Mapping[str, float]) -> bool:
        return False

    def scaling_policy(self, ctx: RegContext):
        return None

    def prox_inplace(self, ctx: RegContext, step: float) -> None:
        pass

    def majorizer_diag(self, ctx: RegContext) -> Optional[torch.Tensor]:
        return None

    def halo(self, roles: Roles) -> Mapping[int, int]:
        # request 1‑sample halo along temporal (absolute axis 0) if included
        h = {}
        axes = self._resolve_axes_local(self.params.axes, roles)
        if any(ax == 0 for ax in axes):
            h[0] = 1
        return h

    def _resolve_axes(self, ctx: RegContext) -> Tuple[int, ...]:
        if getattr(ctx, "axes_resolver", None) is not None:
            return tuple(int(a) for a in ctx.axes_resolver(self.params.axes))
        return self._resolve_axes_local(self.params.axes, ctx.roles_image)

    @staticmethod
    def _resolve_axes_local(spec: AxesSpec, roles: Roles) -> Tuple[int, ...]:
        return _resolve_axes_local(spec, roles)
