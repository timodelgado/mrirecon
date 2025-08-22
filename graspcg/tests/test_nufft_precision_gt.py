# -----------------------------------------------------------------------------
# test_nufft_precision_gt.py
# Numerical precision vs NUDFT ground truth (double-precision CPU).
# -----------------------------------------------------------------------------
import math
import pytest
import torch
import os

from ..nufft.api import NUFFT, NUFFTConfig
from ..nufft.layout import AxisSpec

def _have_cufi():
    try:
        import cufinufft  # noqa: F401
        return True
    except Exception:
        return False

def _have_torchkbnufft():
    try:
        import torchkbnufft  # noqa: F401
        return True
    except Exception:
        return False

def _axis(ndim: int) -> AxisSpec:
    if ndim == 2:
        return AxisSpec(
            image_labels=('B','L','coil','X','Y'),
            kspace_labels=('B','L','coil','K'),
            image_fft_labels=('X','Y'),
            coil_label='coil',
        )
    else:
        return AxisSpec(
            image_labels=('B','L','coil','X','Y','Z'),
            kspace_labels=('B','L','coil','K'),
            image_fft_labels=('X','Y','Z'),
            coil_label='coil',
        )

def _complex_rand(shape, dtype, device):
    r = torch.randn(shape, dtype=torch.float64 if dtype==torch.complex128 else torch.float32, device=device)
    i = torch.randn_like(r)
    return (r + 1j*i).to(dtype)

def _coords_centered(spatial, device):
    # centered indices in [-S/2, ..., S/2-1]
    axes = [torch.arange(S, device=device, dtype=torch.float64) - (S // 2) for S in spatial]
    mesh = torch.meshgrid(*axes, indexing='ij')
    coords = torch.stack([m.reshape(-1) for m in mesh], dim=1)  # (R, nd)
    return coords  # float64

def _nudft_A(maps, x, traj_b, dcf_b=None):
    """
    Ground truth forward (double CPU):
      x: (L, 1, *S)  complex128
      maps: (C, *S)  complex128
      traj_b: (nd, K) radians
      dcf_b:  (K,)    float64 or None (balanced → multiply sqrt after)
    returns: (L, C, K) complex128
    """
    L = x.shape[0]
    C = maps.shape[0]
    spatial = x.shape[2:]
    R = int(torch.prod(torch.tensor(spatial)))
    coords = _coords_centered(spatial, x.device)            # (R,nd) float64
    omega = traj_b.transpose(0,1).to(torch.float64)         # (K,nd)

    # E: (K,R) complex128  with exp(-i * ω·r)
    phase = omega @ coords.T                                 # (K,R)
    E = torch.exp(-1j * phase)

    # Flatten images and apply maps per coil
    X_LR = x.reshape(L, R)                                   # (L,R)
    M_CR = maps.reshape(C, R)                                # (C,R)

    Y_LCK = torch.empty((L, C, E.shape[0]), dtype=torch.complex128, device=x.device)
    for c in range(C):
        # broadcast over L: (L,R) * (1,R) -> (L,R)
        XR = X_LR * M_CR[c].unsqueeze(0)
        # (L,R) @ (R,K) -> (L,K)
        Y_LK = XR @ E.T
        Y_LCK[:, c, :] = Y_LK

    if dcf_b is not None:
        Y_LCK *= torch.sqrt(dcf_b.clamp_min(0)).to(Y_LCK.dtype).unsqueeze(0).unsqueeze(0)
    return Y_LCK

def _nudft_AH(maps, y, traj_b, dcf_b=None):
    """
    Ground truth adjoint (double CPU):
      y: (L, C, K)  complex128
      maps: (C,*S)  complex128
      traj_b: (nd,K) radians
      dcf_b: (K,)    float64 or None  (balanced → multiply sqrt before)
    returns: (L, 1, *S) complex128 (coil-combined)
    """
    L, C, K = y.shape
    spatial = maps.shape[1:]
    R = int(torch.prod(torch.tensor(spatial)))
    coords = _coords_centered(spatial, y.device)            # (R,nd)
    omega = traj_b.transpose(0,1).to(torch.float64)         # (K,nd)

    # conj(E): (R,K) complex128 with exp(+i * ω·r)
    phase = omega @ coords.T                                # (K,R)
    E_conj = torch.exp(1j * phase).T                        # (R,K)

    X_LR = torch.zeros((L, R), dtype=torch.complex128, device=y.device)
    for c in range(C):
        Y_LK = y[:, c, :]
        if dcf_b is not None:
            Y_LK = Y_LK * torch.sqrt(dcf_b.clamp_min(0)).to(Y_LK.dtype).unsqueeze(0)
        # (L,K) @ (K,R) -> (L,R)
        Xc_LR = Y_LK @ E_conj
        # coil combine with conj(maps[c])
        M_R = maps[c].reshape(R).conj()
        X_LR += Xc_LR * M_R.unsqueeze(0)

    return X_LR.reshape(L, 1, *spatial)

@pytest.mark.parametrize("backend", ["torchkb","cufi"])
def test_precision_vs_nudft_2d_small(backend):
    # Backend availability
    if backend == "cufi" and not _have_cufi():
        pytest.skip("CUFINUFFT not available")
    if backend == "torchkb" and not _have_torchkbnufft():
        pytest.skip("torchkbnufft not available")

    # Small 2D problem to keep NUDFT tractable
    dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    # Ground truth computed on CPU/double for determinism
    gt_dev = torch.device("cpu")

    B, L, C = 2, 2, 3
    H, W = 32, 32
    K = 1024
    dtype = torch.complex64
    axis = _axis(2)

    # Data
    maps = _complex_rand((C, H, W), torch.complex128, gt_dev) / math.sqrt(C)
    traj = (torch.rand((B, 2, K), device=gt_dev) * (2*math.pi) - math.pi)  # radians
    dcf  = torch.rand((B, K), device=gt_dev) + 0.1

    # Operator (on dev)
    cfg = NUFFTConfig(ndim=2, backend=backend, traj_units='rad', dcf_mode='balanced')
    op  = NUFFT(maps=maps.to(dev, dtype=dtype),
                traj=traj.to(dev),
                dcf=dcf.to(dev),
                axis=axis, config=cfg, dtype=dtype, device=dev)

    x = _complex_rand((B, L, 1, H, W), torch.complex128, gt_dev)  # GT space (double, CPU)
    y = torch.empty((B, L, C, K), dtype=dtype, device=dev)
    xhat = torch.empty_like(x.to(dev, dtype=dtype))

    # Backend results
    op.A(x.to(dev, dtype=dtype), out=y)
    op.AH(y, out=xhat)

    # Ground truth (double CPU)
    y_gt = torch.empty((B, L, C, K), dtype=torch.complex128, device=gt_dev)
    x_gt = torch.empty_like(x)

    for b in range(B):
        y_gt[b] = _nudft_A(maps, x[b], traj[b], dcf[b])
        x_gt[b] = _nudft_AH(maps, y_gt[b], traj[b], dcf[b])

    # Compare (move backend results to CPU/double)
    y_b = y.detach().to(gt_dev, dtype=torch.complex128)
    x_b = xhat.detach().to(gt_dev, dtype=torch.complex128)

    # Tolerances: KB/PTNUFFT and CUFI should be ~1e-3–1e-2 rel; TorchKb typically ≤1e-3
    tol_A  = float(os.environ.get("NUFFT_TOL_A",  "2.0e-2"))
    tol_AH = float(os.environ.get("NUFFT_TOL_AH", "2.0e-2"))

    relA  = (y_b - y_gt).norm()  / (y_gt.norm()  + 1e-12)
    relAH = (x_b - x_gt).norm() / (x_gt.norm() + 1e-12)

    print(f"[{backend}] relA={relA:.3e}  relAH={relAH:.3e}  (tols {tol_A}, {tol_AH})")
    assert relA  <= tol_A
    assert relAH <= tol_AH
