import torch
from scipy.signal import medfilt, savgol_filter
import cupy as cp
import mrinufft
import numpy as np
from scipy.signal import savgol_filter, medfilt


@torch.no_grad()
def get_ktraj2D_cufi(Ndth, Ndr, device = "cuda"):
    # k-space trajectory calculation based on number of radial angles and 
    # Golden angle in radians
    ga = cp.deg2rad(180 / ((1 + cp.sqrt(cp.float32(5))) / 2))

    ga = cp.deg2rad(180 / ((1 + cp.sqrt(cp.float32(5))) / 2))

    # Allocate kx, ky on GPU (float32)
    kx = cp.zeros((Ndr, Ndth), dtype=cp.float32)
    ky = cp.zeros((Ndr, Ndth), dtype=cp.float32)

    # First spoke (kx[:, 0] = linspace(...))
    kx[:, 0] = cp.linspace(-Ndr / 2, Ndr / 2 - 1, Ndr, dtype=cp.float32)

    # Rotate subsequent spokes
    for i in range(1, Ndth):
        c = cp.cos(ga)
        s = cp.sin(ga)
        ky[:, i] = c * ky[:, i - 1] - s * kx[:, i - 1]
        kx[:, i] = s * ky[:, i - 1] + c * kx[:, i - 1]

    # Transpose dimensions: (Ndth, Ndr)
    ky = ky.T  
    kx = kx.T

    # Stack along axis=0 => shape: (2, Ndth, Ndr)
    # Then scale by pi/(Ndr/2) - cufi uses -pi to pi for coordinates
    ktrajcp = (cp.pi * cp.stack((kx, ky), axis=0) / (Ndr / 2)).astype(cp.float32)
    ktraj = torch.tensor(ktrajcp.get(), dtype=torch.float32, device=device).contiguous()


    del kx, ky, ga, c, s, ktrajcp
    torch.cuda.empty_cache()
    cp._default_memory_pool.free_all_blocks()

    return ktraj

@torch.no_grad()
def get_dcomp_2d(ktraj,dtype = torch.complex64):
    """
    ktraj can have shape:
      (2, #samples) 
       or 
      (2, #spokes, #readout) 
    We'll flatten internally for Voronoi.

    Returns:
      dcomp : shape [#samples] if input is 2D
              shape [#spokes, #readout] if input is 3D
    """
    if ktraj.ndim == 3:
        # (2, Ndth, Ndr)
        Ndth = ktraj.shape[1]
        Ndr  = ktraj.shape[2]
        # Flatten -> shape (Ndth*Ndr, 2)
        ktraj_cp = cp.asarray(ktraj.permute(1,2,0).contiguous().view(-1,2))  
    else:
        # (2, #samples)
        Ndth = None
        Ndr  = None
        # shape (#samples, 2)
        ktraj_cp = cp.asarray(ktraj.permute(1,0).contiguous())

    # Compute Voronoi
    dcomp_np = mrinufft.density.geometry_based.voronoi(ktraj_cp.get())
    dcomp_torch = torch.as_tensor(dcomp_np, dtype=dtype, device = ktraj.device)

    # Normalize to mean=1
    dcomp_torch.div_(dcomp_torch.mean())

    # If 3D input, reshape result
    if ktraj.ndim == 3:
        dcomp_torch = dcomp_torch.view(Ndth, Ndr)
    del dcomp_np, ktraj_cp

    torch.cuda.empty_cache()
    cp._default_memory_pool.free_all_blocks()
    return dcomp_torch

@torch.no_grad()
def frame_scaling(organized_k_space):
    # Calculate scaling factor 
    s_t = cp.array(organized_k_space.view(organized_k_space.shape[0],-1).abs().pow(2).mean(1).sqrt())
    s_t /= s_t.mean()                               # normalise
    s_cpu = cp.asnumpy(s_t)
    s_cpu = medfilt(s_cpu, 15)
    s_cpu = savgol_filter(s_cpu, window_length=121, polyorder=3)
    s_t = cp.asarray(s_cpu, dtype=s_t.dtype)
    #s_t = cp.clip(s_t, a_min=None, a_max=None)
    s_t /= s_t.mean()                               # renormalise
    s_t = torch.as_tensor(s_t, dtype=torch.float32,
                        device=organized_k_space.device).view(-1,1,1,1).detach().requires_grad_(False)

    del s_cpu
    return s_t



def _safe_smooth_1d(y: np.ndarray,
                    preferred_win: int = 121,
                    poly: int = 3) -> np.ndarray:
    """Robust 1‑D smoothing that never throws on tiny inputs."""
    y = np.asarray(y, dtype=np.float32)
    n = y.size
    if n == 0:
        return y
    # median prefilter if long enough
    if n >= 7:
        y = medfilt(y, kernel_size=5)
    # choose a valid odd window ≤ n and > poly
    win = min(preferred_win, n if n % 2 == 1 else n - 1)
    if win <= poly or win < 5:
        # fallback: 5‑point (or smaller) moving average at edges
        k = min(5, n)
        if k <= 1:
            return y
        c = np.convolve(y, np.ones(k, dtype=y.dtype) / k, mode="same")
        return c
    try:
        return savgol_filter(y, window_length=win, polyorder=min(poly, win - 1),
                             mode="interp")
    except Exception:
        # absolute safety net
        return y