from __future__ import annotations

from typing import Sequence

import torch


# ---------------------------
# Internal helpers (real math)
# ---------------------------

def _abs2(x: torch.Tensor) -> torch.Tensor:
    """Return |x|^2 as a REAL tensor, regardless of complex/real input."""
    if torch.is_complex(x):
        return x.real.mul(x.real).add(x.imag.mul(x.imag))
    return x.mul(x)


def _real0_like(x: torch.Tensor) -> torch.Tensor:
    """Real zeros with same shape/device as x."""
    return torch.zeros_like(x.real, dtype=x.real.dtype)


# ---------------------------------------
# Finite differences (TV-friendly stencils)
# ---------------------------------------

def fwd_diff(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Forward difference with same shape:
        D_f x = roll(x, -1, dim) - x;    last slice along `dim` is set to 0.

    This pairs with `bwd_diff` below via the standard TV identity:
        <D_f x, p> = - <x, div p>,   where div p = bwd_diff(p, dim)
    """
    xp1 = torch.roll(x, shifts=-1, dims=dim)
    out = xp1 - x
    # Zero the last slice along `dim`
    idx = [slice(None)] * out.ndim
    idx[dim] = -1
    out[tuple(idx)] = 0
    return out


def bwd_diff(y: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Backward difference used as the **divergence component** (negative adjoint of fwd_diff):
        div p = p - roll(p, +1, dim);   first slice along `dim` is set to p[0]

    With this pairing:
        <fwd_diff(x), p> = - <x, bwd_diff(p)>   (discrete Green's identity).
    """
    ym1 = torch.roll(y, shifts=+1, dims=dim)
    out = y - ym1
    # At the "upper" boundary, set the first slice to y[0] (consistent divergence BC)
    idx = [slice(None)] * out.ndim
    idx[dim] = 0
    out[tuple(idx)] = y[tuple(idx)]
    return out


# ---------------------------------------
# Isotropic TV (ℓ2 over axis stack)
# ---------------------------------------

def tv_iso_energy(grads: Sequence[torch.Tensor], eps: float) -> torch.Tensor:
    """
    Per-pixel isotropic TV energy density:
        ϕ_iso = sqrt( sum_i |g_i|^2 + eps^2 )   (REAL tensor)
    The caller typically reduces (sum) it outside this function.
    """
    assert len(grads) >= 1, "Need at least one gradient tensor"
    # Accumulate |g_i|^2 in REAL dtype
    s = _real0_like(grads[0])
    for gi in grads:
        s = s + _abs2(gi)
    return torch.sqrt(s + (eps * eps))


def tv_iso_flux(grads: Sequence[torch.Tensor], eps: float) -> Sequence[torch.Tensor]:
    """
    Normalized isotropic TV flux per axis:
        p_i = g_i / sqrt( sum_j |g_j|^2 + eps^2 )
    Returns tensors with same shapes as grads[i].
    """
    denom = tv_iso_energy(grads, eps)  # real, >= eps
    denom = denom.clamp_min(1e-12)     # numerical safety
    return [gi / denom for gi in grads]


def tv_iso_div(flux: Sequence[torch.Tensor], axes: Sequence[int]) -> torch.Tensor:
    """
    Divergence of isotropic flux:
        div p = sum_i bwd_diff( p_i, axes[i] )
    Returns a tensor with the same shape as each p_i.
    """
    assert len(flux) == len(axes), "flux and axes must have same length"
    out = torch.zeros_like(flux[0])
    for pi, ax in zip(flux, axes):
        out = out + bwd_diff(pi, ax)
    return out


# ---------------------------------------
# Anisotropic TV (ℓ1 across axes)
# ---------------------------------------

def tv_aniso_energy(grads: Sequence[torch.Tensor], eps: float) -> torch.Tensor:
    """
    Per-pixel anisotropic TV energy density (sum of per-axis smooth ℓ1):
        ϕ_aniso = sum_i sqrt( |g_i|^2 + eps^2 )   (REAL tensor)
    """
    assert len(grads) >= 1, "Need at least one gradient tensor"
    e = _real0_like(grads[0])
    for gi in grads:
        e = e + torch.sqrt(_abs2(gi) + (eps * eps))
    return e


def tv_aniso_flux(grads: Sequence[torch.Tensor], eps: float) -> Sequence[torch.Tensor]:
    """
    Per-axis anisotropic flux:
        p_i = g_i / sqrt( |g_i|^2 + eps^2 )
    """
    out = []
    for gi in grads:
        denom = torch.sqrt(_abs2(gi) + (eps * eps)).clamp_min(1e-12)
        out.append(gi / denom)
    return out


def tv_aniso_div(flux: Sequence[torch.Tensor], axes: Sequence[int]) -> torch.Tensor:
    """
    Divergence of anisotropic flux:
        div p = sum_i bwd_diff( p_i, axes[i] )
    """
    return tv_iso_div(flux, axes)