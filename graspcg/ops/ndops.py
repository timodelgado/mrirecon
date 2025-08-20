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

def tv_iso_energy(grads, eps: torch.Tensor):
    """
    Isotropic TV energy density with baseline subtraction:
        e(x) = sqrt( sum_i |g_i|^2 + eps^2 ) - eps
    grads: list[Tensor] (real or complex)
    returns: Tensor of same shape as x
    """
    if not grads:
        raise ValueError("tv_iso_energy: empty grads")
    # sum of squared magnitudes (handles complex)
    ss = None
    for g in grads:
        g2 = (g.real * g.real + g.imag * g.imag) if g.is_complex() else (g * g)
        ss = g2 if ss is None else (ss + g2)
    den = torch.sqrt(ss + eps * eps)
    return den - eps


def tv_aniso_energy(grads, eps: torch.Tensor):
    """
    Anisotropic TV energy density with baseline subtraction:
        e(x) = sum_i ( sqrt(|g_i|^2 + eps^2) - eps )
    grads: list[Tensor] (real or complex)
    returns: Tensor of same shape as x
    """
    if not grads:
        raise ValueError("tv_aniso_energy: empty grads")
    out = None
    for g in grads:
        g2 = (g.real * g.real + g.imag * g.imag) if g.is_complex() else (g * g)
        den = torch.sqrt(g2 + eps * eps) - eps
        out = den if out is None else (out + den)
    return out


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


def tv_iso_flux(grads: Sequence[torch.Tensor], eps: torch.Tensor) -> Sequence[torch.Tensor]:
    """
    Per-axis isotropic flux:
        p_i = g_i / sqrt( sum_j |g_j|^2 + eps^2 )
    Shared denominator across axes for isotropic TV.
    """
    if not grads:
        raise ValueError("tv_iso_flux: empty grads")
    # Sum of squared magnitudes (handles complex)
    ss = None
    for gi in grads:
        g2 = (gi.real * gi.real + gi.imag * gi.imag) if gi.is_complex() else (gi * gi)
        ss = g2 if ss is None else (ss + g2)
    denom = torch.sqrt(ss + (eps * eps)).clamp_min(1e-12)  # real, broadcast to each axis tensor
    return [gi / denom for gi in grads]


def tv_iso_div(flux: Sequence[torch.Tensor], axes: Sequence[int]) -> torch.Tensor:
    """
    Divergence of isotropic (or anisotropic) flux:
        div p = sum_i bwd_diff( p_i, axes[i] )
    """
    if not flux:
        raise ValueError("tv_iso_div: empty flux")
    out = None
    for pi, ax in zip(flux, axes):
        term = bwd_diff(pi, ax)
        out = term if out is None else (out + term)
    return out