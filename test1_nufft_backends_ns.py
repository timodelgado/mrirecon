# =============================================================================
# NUFFT API tests (new front-end + non-canonical layouts)
# =============================================================================
from __future__ import annotations
import math
import os
from typing import Optional, Tuple

import pytest
import torch

# New front-end + layout
from graspcg.nufft.api import NUFFT, NUFFTConfig
from graspcg.nufft.layout import AxisSpec

# -----------------------------------------------------------------------------
# Backend availability
# -----------------------------------------------------------------------------
try:
    import torchkbnufft  # noqa: F401
    HAVE_TKB = True
except Exception:
    HAVE_TKB = False

try:
    import cufinufft  # noqa: F401
    HAVE_CUFI = True
except Exception:
    HAVE_CUFI = False

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _rand_cplx(shape: Tuple[int, ...], device, scale=0.1, seed: int = 1234) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    r = torch.randn(*shape, generator=g, device=device)
    i = torch.randn(*shape, generator=g, device=device)
    return scale * (r + 1j * i).to(torch.complex64)

def _make_maps(C: int, spatial: Tuple[int, ...], device) -> torch.Tensor:
    # Mildly varying coil sensitivities with a sensible sum-of-squares magnitude
    maps = _rand_cplx((C,)+spatial, device, scale=0.7, seed=777)
    # avoid pathological zeros
    return maps + 0.05 * (1.0 + 0j)

def _traj_norm(B: int, K: int, nd: int, device) -> torch.Tensor:
    # Normalized cycles/pixel in [-0.5, 0.5]; adapters scale to radians internally
    g = torch.Generator(device=device); g.manual_seed(2025 + B + K + nd)
    t = torch.empty((B, nd, K), device=device)
    t.uniform_(-0.5, 0.5, generator=g)
    return t.to(torch.float32)

def _dcf_uniform(B: int, K: int, device) -> torch.Tensor:
    return torch.ones((B, K), device=device, dtype=torch.float32)

def _devices():
    devs = [torch.device("cpu")]
    if torch.cuda.is_available():
        devs.append(torch.device("cuda"))
    return devs

# -----------------------------------------------------------------------------
# Parameterizations
# -----------------------------------------------------------------------------

BACKENDS = []
if HAVE_TKB:
    BACKENDS.append("torchkb")
if HAVE_CUFI and torch.cuda.is_available():
    BACKENDS.append("cufi")

@pytest.mark.skipif(len(BACKENDS) == 0, reason="No enabled NUFFT backends found (torchkbnufft/cufinufft).")
class TestNUFFT:

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("device", _devices())
    def test_shapes_and_adjointness_noncanonical(self, backend, device):
        # CUFI requires CUDA
        if backend == "cufi" and device.type != "cuda":
            pytest.skip("CuFiNUFFT requires CUDA device.")

        # Problem sizes
        B, C, P = 3, 2, 4
        spatial = (28, 26)      # (Fx, Fy) labels in AxisSpec below
        nd, K = 2, 2048

        maps = _make_maps(C, spatial, device)
        traj = _traj_norm(B, K, nd, device)
        dcf  = _dcf_uniform(B, K, device)

        # Non-canonical axis order: image ('P','Fx','Fy','B'), k-space ('coil','B','K','P')
        axis = AxisSpec(
            image_labels=('P','Fx','Fy','B'),
            kspace_labels=('coil','B','K','P'),
            image_fft_labels=('Fx','Fy'),
            kspace_fft_labels=('K',),
            coil_label='coil',
        )

        cfg = NUFFTConfig(
            ndim=2, backend=backend,
            apply_dcf_in_fwd=False, apply_dcf_in_adj=True
        )

        op = NUFFT(maps, traj, dcf, axis=axis, config=cfg, dtype=torch.complex64, device=device)

        # x: object images (coil-combined by default): (B,P,X,Y) but as per axis order -> ('P','Fx','Fy','B')
        x = _rand_cplx((P, spatial[0], spatial[1], B), device, scale=0.05, seed=11)

        # y: phantom k-space to test adjointness, shaped ('coil','B','K','P') per AxisSpec
        y = _rand_cplx((C, B, K, P), device, scale=0.05, seed=12)

        # Forward & adjoint
        Ax  = op.A(x)            # → ('coil','B','K','P')
        AHy = op.AH(y)           # → ('P','Fx','Fy','B'), coil-combined

        # Shape checks (non-canonical order must be preserved)
        assert Ax.shape == (C, B, K, P)
        assert AHy.shape == (P, spatial[0], spatial[1], B)

        # Adjointness: <Ax, y> == <x, AHy>
        lhs = (Ax.conj() * y).sum().real
        rhs = (x.conj() * AHy).sum().real
        rel = (lhs - rhs).abs() / rhs.abs().clamp_min(1e-12)

        # Tight for TorchKb, looser for CUFI
        tol = 5e-3 if backend == "torchkb" else 2e-2
        assert float(rel) < tol, f"adjointness rel={float(rel):.3e} exceeds tol={tol:.1e} for {backend}/{device}"

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("device", _devices())
    def test_profile_and_alpha_scalar(self, backend, device):
        if backend == "cufi" and device.type != "cuda":
            pytest.skip("CuFiNUFFT requires CUDA device.")

        B, C = 4, 3
        spatial = (24, 22)
        nd, K = 2, 1536

        maps = _make_maps(C, spatial, device)
        traj = _traj_norm(B, K, nd, device)
        dcf  = _dcf_uniform(B, K, device)

        axis = AxisSpec(
            image_labels=('B','Fx','Fy'),     # no like dims
            kspace_labels=('B','coil','K'),   # no like dims
            image_fft_labels=('Fx','Fy'),
            kspace_fft_labels=('K',),
            coil_label='coil',
        )
        cfg = NUFFTConfig(ndim=2, backend=backend, apply_dcf_in_adj=True, apply_dcf_in_fwd=False)
        op = NUFFT(maps, traj, dcf, axis=axis, config=cfg, dtype=torch.complex64, device=device)

        # Profile must be constant across frames when DCF is uniform: sum dcf = K
        if hasattr(op, "_backend") and hasattr(op._backend, "diag_AHA_profile"):
            prof = op._backend.diag_AHA_profile()
            assert prof.shape == (B,)
            assert torch.allclose(prof, torch.full((B,), float(K), device=device)), "profile != K"
        # Calibrated scalar should be finite and positive
        if hasattr(op, "_backend") and hasattr(op._backend, "diag_AHA_scalar"):
            alpha = float(op._backend.diag_AHA_scalar())
            assert math.isfinite(alpha) and alpha > 0.0

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("device", _devices())
    def test_out_semantics_and_vectorization_over_like(self, backend, device):
        if backend == "cufi" and device.type != "cuda":
            pytest.skip("CuFiNUFFT requires CUDA device.")

        # Add a like dim (partitions) and ensure vectorization works + out= honored
        B, C, P = 2, 2, 3
        spatial = (20, 18)
        nd, K = 2, 1024

        maps = _make_maps(C, spatial, device)
        traj = _traj_norm(B, K, nd, device)
        dcf  = _dcf_uniform(B, K, device)

        axis = AxisSpec(
            image_labels=('B','P','Fx','Fy'),
            kspace_labels=('B','coil','P','K'),
            image_fft_labels=('Fx','Fy'),
            kspace_fft_labels=('K',),
            coil_label='coil',
        )
        cfg = NUFFTConfig(ndim=2, backend=backend, apply_dcf_in_adj=True, apply_dcf_in_fwd=False)
        op = NUFFT(maps, traj, dcf, axis=axis, config=cfg, dtype=torch.complex64, device=device)

        # Allocate out buffers with correct user order
        y_out = torch.empty((B, C, P, K), dtype=torch.complex64, device=device)
        x_out = torch.empty((B, P, spatial[0], spatial[1]), dtype=torch.complex64, device=device)

        x = _rand_cplx((B, P, spatial[0], spatial[1]), device, scale=0.03, seed=101)
        y = _rand_cplx((B, C, P, K), device, scale=0.03, seed=102)

        # Forward
        y1 = op.A(x)
        y2 = op.A(x, out=y_out)
        assert y2.data_ptr() == y_out.data_ptr()
        assert torch.allclose(y1, y2, atol=2e-3, rtol=2e-2)

        # Adjoint
        x1 = op.AH(y)
        x2 = op.AH(y, out=x_out)
        assert x2.data_ptr() == x_out.data_ptr()
        assert torch.allclose(x1, x2, atol=2e-3, rtol=2e-2)

    @pytest.mark.skipif(not (HAVE_TKB and HAVE_CUFI and torch.cuda.is_available()),
                        reason="Needs both TorchKb and CUFI on CUDA for cross-backend check.")
    def test_backend_similarity_cuda(self):
        device = torch.device("cuda")
        B, C, P = 2, 2, 3
        spatial = (24, 20)
        nd, K = 2, 2048

        maps = _make_maps(C, spatial, device)
        traj = _traj_norm(B, K, nd, device)
        dcf  = _dcf_uniform(B, K, device)

        axis = AxisSpec(
            image_labels=('B','P','Fx','Fy'),
            kspace_labels=('B','coil','P','K'),
            image_fft_labels=('Fx','Fy'),
            kspace_fft_labels=('K',),
            coil_label='coil',
        )

        torchkb = NUFFT(maps, traj, dcf, axis=axis,
                        config=NUFFTConfig(ndim=2, backend='torchkb',
                                           apply_dcf_in_fwd=False, apply_dcf_in_adj=True),
                        dtype=torch.complex64, device=device)

        cufi   = NUFFT(maps, traj, dcf, axis=axis,
                        config=NUFFTConfig(ndim=2, backend='cufi',
                                           apply_dcf_in_fwd=False, apply_dcf_in_adj=True),
                        dtype=torch.complex64, device=device)

        x = _rand_cplx((B, P, spatial[0], spatial[1]), device, scale=0.05, seed=21)

        y_kb = torchkb.A(x)
        y_cf = cufi.A(x)
        # Kernels differ; allow moderate tolerance
        assert torch.allclose(y_kb, y_cf, atol=2e-3, rtol=2e-2)

        xk = torchkb.AH(y_kb)
        xc = cufi.AH(y_kb)  # feed TorchKb's y into CUFI adjoint
        assert torch.allclose(xk, xc, atol=2e-3, rtol=2e-2)
