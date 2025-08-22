from __future__ import annotations

from typing import Optional, Sequence, Tuple, Callable
import math
import torch


# -------------------------------
# Core helpers (strict, no fallbacks)
# -------------------------------

def _ensure_complex_dtype(dtype: torch.dtype) -> None:
    if dtype not in (torch.complex64, torch.complex128):
        raise ValueError("Expected a complex dtype (complex64 or complex128).")


def _center_index(spatial: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(s) // 2 for s in spatial)


def sum_sens2(maps: torch.Tensor) -> torch.Tensor:
    """
    Σ_c |S_c|^2 as (1, X, Y[, Z]) float32.

    maps: (C, X, Y[, Z]) complex
    """
    if maps.ndim not in (3, 4):
        raise ValueError("maps must be (C,X,Y) or (C,X,Y,Z).")
    # compute on maps' device; return float32 (frontend may move/cast)
    s2 = (maps.real * maps.real + maps.imag * maps.imag).sum(dim=0, keepdim=True)
    return s2.to(torch.float32)


# -------------------------------
# Per‑frame profile α_b
# -------------------------------

@torch.no_grad()
def alpha_profile(traj_BndK: torch.Tensor,
                  dcf_BK: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Per‑frame proxy α_b:
        α_b = sum_k dcf[b,k]   if DCF is provided,
        α_b = K                otherwise.

    traj_BndK: (B, nd, K) float32  (radians)
    dcf_BK   : (B, K) float32 or None

    Returns: (B,) float32 (on traj device)
    """
    if traj_BndK.ndim != 3:
        raise ValueError("traj_BndK must be (B, nd, K).")
    B, _, K = (int(traj_BndK.shape[0]), int(traj_BndK.shape[1]), int(traj_BndK.shape[2]))
    dev = traj_BndK.device
    if dcf_BK is None:
        return torch.full((B,), float(K), device=dev, dtype=torch.float32)
    if dcf_BK.ndim != 2 or int(dcf_BK.shape[0]) not in (1, B) or int(dcf_BK.shape[1]) != K:
        raise ValueError("dcf_BK must be (B,K) or (1,K) matching traj_BndK.")
    if int(dcf_BK.shape[0]) == 1:
        dcf_BK = dcf_BK.expand(B, -1)
    return dcf_BK.sum(dim=1).to(torch.float32)


# -------------------------------
# Scalar α via PSF median (robust)
# -------------------------------

@torch.no_grad()
def alpha_scalar_via_psf(
    AH: Callable[[torch.Tensor], torch.Tensor],
    maps: torch.Tensor,
    traj_BndK: torch.Tensor,
    K: int,
    dtype_c: torch.dtype,
    device: torch.device,
    trim: Tuple[float, float] = (0.10, 0.90),
) -> torch.Tensor:
    """
    Robust scalar α from trimmed median of AH(ones)/Σ|S|^2.

    AH:     function taking (B, C, K) -> (B, 1, X, Y[,Z]) complex (adjoint)
    maps:   (C, X, Y[,Z]) complex
    traj_BndK: (B, nd, K) float32, to match B
    K:      int, samples per frame
    dtype_c: complex dtype for ones
    device : torch.device for compute
    trim   : central quantile to keep (lo, hi)

    Returns: 0-D float32 tensor on CPU
    """
    _ensure_complex_dtype(dtype_c)
    if maps.ndim not in (3, 4):
        raise ValueError("maps must be (C,X,Y) or (C,X,Y,Z).")
    if traj_BndK.ndim != 3:
        raise ValueError("traj_BndK must be (B, nd, K).")
    B = int(traj_BndK.shape[0])
    C = int(maps.shape[0])

    if K <= 0:
        raise ValueError("K must be positive.")

    ones = torch.ones((B, C, K), dtype=dtype_c, device=device)
    psf  = AH(ones)                # (B,1,X,Y[,Z])
    psf  = psf.sum(dim=1)          # (B,X,Y[,Z]) – already coil‑combined
    s2   = sum_sens2(maps).to(device)  # (1,X,Y[,Z])
    num  = psf.real
    den  = s2.clamp_min(1e-8)
    ratio = (num / den).reshape(-1)

    n = int(ratio.numel())
    if n == 0:
        alpha = torch.tensor(1.0, dtype=torch.float32)
    else:
        lo = int(max(0, math.floor(trim[0] * n)))
        hi = int(min(n, math.ceil(trim[1] * n)))
        r_sorted, _ = torch.sort(ratio)
        alpha = r_sorted[lo:hi].median()

    return alpha.detach().to('cpu', dtype=torch.float32).reshape(())


# -------------------------------
# Scalar α via delta (precise)
# -------------------------------

@torch.no_grad()
def alpha_scalar_via_delta(
    A: Callable[[torch.Tensor], torch.Tensor],
    AH: Callable[[torch.Tensor], torch.Tensor],
    maps: torch.Tensor,
    traj_BndK: torch.Tensor,
    dtype_c: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Precise scalar α by applying AᴴA to a unit impulse at the image center, then
    normalizing by Σ|S|²(center). Uses a symmetric delta over all coils.

    A:       (B, C, X, Y[,Z]) -> (B, C, K) complex
    AH:      (B, C, K) -> (B, 1, X, Y[,Z]) complex
    maps:    (C, X, Y[,Z]) complex
    traj_BndK: (B, nd, K) float32  (B must match backend plan)
    dtype_c: complex dtype
    device : torch.device

    Returns: 0-D float32 tensor on CPU
    """
    _ensure_complex_dtype(dtype_c)
    if maps.ndim not in (3, 4):
        raise ValueError("maps must be (C,X,Y) or (C,X,Y,Z).")
    if traj_BndK.ndim != 3:
        raise ValueError("traj_BndK must be (B, nd, K).")

    B = int(traj_BndK.shape[0])
    C = int(maps.shape[0])
    spatial = tuple(int(s) for s in maps.shape[1:])
    idx = _center_index(spatial)

    # symmetric delta across coils at center, only in the first frame
    x0 = torch.zeros((B, C) + spatial, dtype=dtype_c, device=device)
    for c in range(C):
        x0[0, c][idx] = 1.0 + 0.0j

    y = A(x0)          # (B, C, K)
    z = AH(y)          # (B, 1, X, Y[,Z])

    s2_center = (maps.abs() ** 2).sum(dim=0)[idx].clamp_min(1e-20)
    alpha = z[0, 0][idx].real / s2_center

    return torch.as_tensor(float(alpha), dtype=torch.float32, device='cpu').reshape(())
