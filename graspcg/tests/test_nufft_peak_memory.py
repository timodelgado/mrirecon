# -----------------------------------------------------------------------------
# test_nufft_peak_memory.py
# Compare peak CUDA memory across single-device backends (A and AH).
# Prints metrics and asserts basic invariants; skips unavailable backends.
# -----------------------------------------------------------------------------
import os
import math
import pytest
import torch

# Public API
from ..nufft.api import NUFFT, NUFFTConfig
from ..nufft.layout import AxisSpec

def _have_cufi():
    try:
        import cufinufft  # noqa: F401
        return torch.cuda.is_available()
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
            image_labels=('B','L','X','Y','Z','coil'),
            kspace_labels=('B','L','coil','K'),
            image_fft_labels=('X','Y','Z'),
            coil_label='coil',
        )

def _complex_rand(shape, dtype, device):
    r = torch.randn(shape, dtype=torch.float32 if dtype==torch.complex64 else torch.float64, device=device)
    i = torch.randn_like(r)
    return (r + 1j*i).to(dtype)

@pytest.mark.parametrize("backend", ["torchkb","cufi"])
def test_peak_memory_single_device(backend):
    # Skip backends not available on this host
    if backend == "cufi" and not _have_cufi():
        pytest.skip("CUFINUFFT/CUDA not available")
    if backend == "torchkb" and not _have_torchkbnufft():
        pytest.skip("torchkbnufft not available")

    # Small but nontrivial 2D case to keep runtime low
    dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.complex64
    B, L, C = 2, 3, 4
    H, W = 192, 192
    K = 64000 if dev.type == "cuda" else 8000  # keep CPU runtime reasonable
    axis = _axis(2)

    # Synthesize inputs (balanced policy expects sqrt(dcf) preconditioning externally)
    maps = _complex_rand((C, H, W), dtype, dev) / math.sqrt(C)
    traj = (torch.rand((B, 2, K), device=dev) * (2*math.pi) - math.pi)  # radians
    dcf  = torch.rand((B, K), device=dev) + 0.1

    cfg = NUFFTConfig(ndim=2, backend=backend, traj_units='rad', dcf_mode='balanced')
    op  = NUFFT(maps=maps, traj=traj, dcf=dcf, axis=axis, config=cfg, dtype=dtype, device=dev)

    x = _complex_rand((B, L, 1, H, W), dtype, dev)
    y = torch.empty((B, L, C, K), dtype=dtype, device=dev)
    x2 = torch.empty_like(x)

    # Warmup to de-jit / cache plans
    for _ in range(2):
        op.A(x, out=y)
    for _ in range(2):
        op.AH(y, out=x2)

    # Measure peak memory with OUT buffers (best practice)
    def _peak(fn, *args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated()
        return 0

    peak_A_with_out  = _peak(op.A, x, out=y)
    peak_AH_with_out = _peak(op.AH, y, out=x2)

    # For comparison, also let the op allocate its result (OUT=None)
    peak_A_no_out  = _peak(op.A, x, out=None)
    peak_AH_no_out = _peak(op.AH, y, out=None)

    # Print human-friendly metrics (visible under -s)
    def _fmt(n):
        for u in ['B','KB','MB','GB','TB']:
            if n < 1024: return f"{n:.1f} {u}"
            n /= 1024
        return f"{n:.1f} PB"

    print(f"[{backend}] peak A with out:  {_fmt(peak_A_with_out)} ; without: {_fmt(peak_A_no_out)}")
    print(f"[{backend}] peak AH with out: {_fmt(peak_AH_with_out)} ; without: {_fmt(peak_AH_no_out)}")

    # Basic invariants
    #  - correctness: shapes preserved
    assert y.shape == (B, L, C, K)
    assert x2.shape == (B, L, 1, H, W)
    #  - memory with OUT should not exceed memory without OUT (allow small jitter)
    if torch.cuda.is_available():
        assert peak_A_with_out  <= peak_A_no_out  + 8*1024*1024
        assert peak_AH_with_out <= peak_AH_no_out + 8*1024*1024
