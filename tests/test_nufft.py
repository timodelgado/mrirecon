#%%
# tests/test_nufft.py
# ---------------------------------------------------------------------
# NUFFT integration tests:
#   • Uses a real-like image (skimage.camera if available; else smooth phantom)
#   • Simulated 2D radial k-space trajectory + density compensation
#   • Adjointness check: <A x, y> ≈ <x, A^H y>
#   • Round-trip fidelity: A^H(A img) ≈ α·img
#   • Sliding-window organizer returns a VIEW (no giant copies)
#   • out= buffer semantics & no input mutation
#   • Runs on "torchkb" backend; tries "cufi" if CUDA+cufinufft present
# ---------------------------------------------------------------------

import os
import pytest
import torch
import math

import graspcg.nufft.organize_k_space_slidingwindow as orgmod
from graspcg.nufft.organize_k_space_slidingwindow import organize_k_space_slidingwindow
from graspcg.nufft.nufftoperator_cufi import get_cufi_opSENSE2 as CufiOp
from graspcg.nufft.nufftoperator_torchkb import SenseNUFFT_TORCHKB as TorchKbOp
from graspcg.nufft.overlap_scheduler import AH_overlap, A_overlap  # A_overlap used in similar patterns
from graspcg.nufft.utils import get_ktraj2D_cufi, get_dcomp_2d

# -------------------------- helpers --------------------------

def _seed():
    torch.manual_seed(7)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(7)


def _device_for_backend(backend: str) -> torch.device:
    if backend == "cufi":
        pytest.importorskip("cufinufft")
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for CUFINUFFT backend")
        return torch.device("cuda:0")
    # torchkb can run CPU or GPU
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _try_real_image(Ndz: int, Nx: int, Ny: int, device: torch.device) -> torch.Tensor:
    """Return (Ndz, Nx, Ny) complex64 image: skimage.camera if available, else smooth phantom."""
    img = None
    try:
        from skimage import data
        from skimage.transform import resize
        base = data.camera().astype("float32")  # (512,512)
        base = resize(base, (Nx, Ny), anti_aliasing=True, preserve_range=True).astype("float32")
        base = torch.from_numpy(base)
    except Exception:
        # Simple smooth "phantom": sum of Gaussians + ring
        x = torch.linspace(-1.0, 1.0, Nx)
        y = torch.linspace(-1.0, 1.0, Ny)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        R = torch.sqrt(X**2 + Y**2)
        base = (
            torch.exp(-((X + 0.25) ** 2 + (Y + 0.1) ** 2) / (2 * 0.15**2))
            + 0.7 * torch.exp(-((X - 0.3) ** 2 + (Y - 0.2) ** 2) / (2 * 0.2**2))
            + 0.3 * torch.exp(-((R - 0.45) ** 2) / (2 * 0.03**2))
        ).float()
    base -= base.min()
    base /= (base.max() + 1e-8)
    img2d = base.to(device)  # (Nx,Ny)
    img3d = img2d.unsqueeze(0).repeat(Ndz, 1, 1)  # (Ndz,Nx,Ny)
    return img3d.to(torch.complex64)


def _coil_sensitivities(Nch: int, Ndz: int, Nx: int, Ny: int, device: torch.device) -> torch.Tensor:
    """
    Return coil maps (Nch, Ndz, Nx, Ny) complex64 with smooth magnitudes and light phase,
    normalized to sum-of-squares ≈ 1 per-pixel.
    """
    x = torch.linspace(-1.0, 1.0, Nx, device=device)
    y = torch.linspace(-1.0, 1.0, Ny, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    csm = torch.empty((Nch, Ndz, Nx, Ny), dtype=torch.complex64, device=device)
    sigma = 0.5
    for c in range(Nch):
        ang = 2.0 * math.pi * c / Nch
        cx, cy = 0.35 * math.cos(ang), 0.35 * math.sin(ang)
        mag = torch.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma**2))
        # very mild linear phase
        ph = 2.0 * math.pi * (0.03 * (X * math.cos(ang) + Y * math.sin(ang)))
        phase = torch.complex(torch.cos(ph), torch.sin(ph))
        for z in range(Ndz):
            csm[c, z] = (mag * phase).to(torch.complex64)
    sos = torch.sqrt(torch.sum(torch.abs(csm) ** 2, dim=0, keepdim=True)) + 1e-8
    return (csm / sos).contiguous()


def _build_op(backend: str, csm: torch.Tensor, Ndth: int, Ndr: int) -> object:
    """
    Return an operator with A(image,ktraj,dcomp,out=) and AH(k,ktraj,dcomp,out=).
    Balanced √dcomp convention by default.
    """
    Nch, Ndz, Nx, Ny = map(int, csm.shape)
    if backend == "cufi":
        op_params = {
            "Ndth": Ndth, "Ndr": Ndr,
            "Ndx": Nx, "Ndy": Ny,
            "Ndz": Ndz, "Nch": Nch,
            "device_id": csm.device.index,
        }
        return CufiOp(op_params, csm, dcomp_mode="sqrt")
    else:
        return TorchKbOp(Ndx=Nx, Ndy=Ny, Ndth=Ndth, Ndr=Ndr,
                         Ndz=Ndz, Nch=Nch, coil_sensitivity_maps=csm,
                         dcomp_mode="sqrt")


def _vdot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Conjugate dot product: <a,b> = sum(conj(a) * b)"""
    a = a.reshape(-1)
    b = b.reshape(-1)
    return torch.vdot(a, b)


# -------------------------- parameters --------------------------

@pytest.mark.parametrize("backend", ["torchkb", "cufi"])
def test_nufft_adjointness_and_out_semantics(backend):
    _seed()
    device = _device_for_backend(backend)

    # modest sizes for quick test
    Nx = Ny = 96
    Ndz = 2
    Nch = 4
    Ndth = 96
    Ndr = 128

    # data
    img = _try_real_image(Ndz, Nx, Ny, device)         # (Ndz,Nx,Ny) complex64
    csm = _coil_sensitivities(Nch, Ndz, Nx, Ny, device)
    ktraj = get_ktraj2D_cufi(Ndth, Ndr).to(device)      # (2,Ndth,Ndr) float32-ish
    dcomp = get_dcomp_2d(ktraj, dtype=torch.float32).to(device)  # (Ndth,Ndr) float32

    op = _build_op(backend, csm, Ndth, Ndr)

    # random test vectors
    x = torch.randn_like(img) + 1j * torch.randn_like(img)
    y = torch.randn((Nch, Ndz, Ndth, Ndr), dtype=torch.complex64, device=device)
    x_immut = x.clone()
    y_immut = y.clone()

    # A and AH with out= buffers
    out_k = torch.empty((Nch, Ndz, Ndth, Ndr), dtype=torch.complex64, device=device)
    Ax = op.A(x, ktraj, dcomp, out=out_k)
    assert Ax.data_ptr() == out_k.data_ptr(), "A() must honor out="
    out_x = torch.empty((Ndz, Nx, Ny), dtype=torch.complex64, device=device)
    AHy = op.AH(y, ktraj, dcomp, out=out_x)
    assert AHy.data_ptr() == out_x.data_ptr(), "AH() must honor out="

    # no input mutation
    assert torch.allclose(x, x_immut), "Operator mutated image input"
    assert torch.allclose(y, y_immut), "Operator mutated k-space input"

    # adjointness: <A x, y> ≈ <x, A^H y>
    lhs = _vdot(Ax, y)
    rhs = _vdot(x, AHy)
    rel = (lhs - rhs).abs() / (torch.maximum(lhs.abs(), rhs.abs()) + 1e-12)
    # Allow slightly looser tol for GPU/cuFINUFFT kernel variability
    tol = 5e-3 if backend == "cufi" else 2e-3
    assert rel.item() < tol, f"Adjointness failed for {backend}: rel={rel.item():.3e}, lhs={lhs}, rhs={rhs}"


@pytest.mark.parametrize("backend", ["torchkb", "cufi"])
def test_nufft_roundtrip_fidelity(backend):
    _seed()
    device = _device_for_backend(backend)

    Nx = Ny = 128
    Ndz = 1
    Nch = 4
    Ndth = 128
    Ndr = 192

    img = _try_real_image(Ndz, Nx, Ny, device)         # (Ndz,Nx,Ny)
    csm = _coil_sensitivities(Nch, Ndz, Nx, Ny, device)
    ktraj = get_ktraj2D_cufi(Ndth, Ndr).to(device)
    dcomp = get_dcomp_2d(ktraj, dtype=torch.float32).to(device)

    op = _build_op(backend, csm, Ndth, Ndr)

    k = op.A(img, ktraj, dcomp)                        # (Nch,Ndz,Ndth,Ndr)
    xhat = op.AH(k, ktraj, dcomp)                      # (Ndz,Nx,Ny)

    # best complex scalar α to align
    num = _vdot(xhat, img)
    den = _vdot(img, img)
    alpha = (num / (den + 1e-12)).detach()
    rel_err = (xhat - alpha * img).abs().pow(2).sum().sqrt() / (img.abs().pow(2).sum().sqrt() + 1e-12)

    # A^H A is a (weighted) PSF; with these spokes we expect decent fidelity
    # keep threshold modest to be robust across backends
    assert rel_err.item() < 0.25, f"Round-trip too lossy for {backend}: rel={rel_err.item():.3f}"


@pytest.mark.parametrize("backend", ["torchkb", "cufi"])
def test_sliding_window_view_and_frames(backend):
    """
    Build a full acquisition (Ndth_total spokes), organize into sliding windows,
    verify that org_k is a VIEW (when policy != 'preweight_cache'), then
    reconstruct frame-wise images and compare the center frame with ground truth.
    """
    _seed()
    device = _device_for_backend(backend)

    # image / coils
    Nx = Ny = 96
    Ndz = 1
    Nch = 4
    Ndth_total = 128
    Ndr = 128
    temp_window = 64
    temp_slide = 32
    Nframes = (Ndth_total - temp_window) // temp_slide + 1

    img = _try_real_image(Ndz, Nx, Ny, device)
    csm = _coil_sensitivities(Nch, Ndz, Nx, Ny, device)

    # full kspace simulation (Ndth_total)
    full_ktraj = get_ktraj2D_cufi(Ndth_total, Ndr).to(device)
    full_dcomp = get_dcomp_2d(full_ktraj, dtype=torch.float32).to(device)

    # build a "full" op matching Ndth_total
    if backend == "cufi":
        op_full = CufiOp(
            {"Ndth": Ndth_total, "Ndr": Ndr, "Ndx": Nx, "Ndy": Ny, "Ndz": Ndz,
             "Nch": Nch, "device_id": device.index},
            csm, dcomp_mode="sqrt"
        )
    else:
        op_full = TorchKbOp(Ndx=Nx, Ndy=Ny, Ndth=Ndth_total, Ndr=Ndr,
                            Ndz=Ndz, Nch=Nch, coil_sensitivity_maps=csm,
                            dcomp_mode="sqrt")

    k_space_full = op_full.A(img, full_ktraj, full_dcomp)  # (Nch,Ndz,Ndth_total,Ndr)

    # organize with view optimization; let operator apply √dcomp
    org_k, frame_op = organize_k_space_slidingwindow(
        k_space_full, csm, temp_window, temp_slide,
        backend=backend, dcomp_policy="op_sqrt", devices=[device.index] if device.type == "cuda" else None
    )

    # org_k must be a VIEW that shares storage with k_space_full
    assert org_k.untyped_storage().data_ptr() == k_space_full.untyped_storage().data_ptr(), \
        "organizer did not return a view (storage differs)"

    # reconstruct frames
    frames = frame_op.AH(org_k)  # (Nframes, Ndz, Nx, Ny)
    assert frames.shape == (Nframes, Ndz, Nx, Ny)

    # center frame should resemble ground truth up to a complex scale
    imc = frames[Nframes // 2, 0]       # (Nx,Ny)
    gt = img[0]
    alpha = _vdot(imc, gt) / (_vdot(gt, gt) + 1e-12)
    rel = (imc - alpha * gt).abs().pow(2).sum().sqrt() / (gt.abs().pow(2).sum().sqrt() + 1e-12)

    # Because each frame sees only temp_window spokes, relax compared to full round-trip
    assert rel.item() < 0.40, f"Frame reconstruction too far from ground truth (rel={rel.item():.3f})"

    # out= buffer on frame operator
    out_frames = torch.empty_like(frames)
    ret = frame_op.AH(org_k, out=out_frames)
    assert ret.data_ptr() == out_frames.data_ptr(), "Frame AH() must honor out="


# -------------------------- marks & skips --------------------------

def pytest_collection_modifyitems(config, items):
    """
    Reorder/mark tests so torchkb runs first; CUFI tests skip cleanly on CPU
    and when cufinufft is not present.
    """
    for item in items:
        backend = item.callspec.params["backend"] if hasattr(item, "callspec") else None
        if backend == "cufi":
            # Will skip at runtime if CUDA or cufinufft missing
            item.add_marker(pytest.mark.cufi)
        elif backend == "torchkb":
            item.add_marker(pytest.mark.torchkb)
            
            
import pytest
import torch

# 1) Chunking equivalence (torchkbnufft): all-at-once vs small chunks
@pytest.mark.parametrize("backend", ["torchkb"])
def test_torchkb_chunking_equivalence(backend):
    _seed()
    device = _device_for_backend(backend)
    Nx = Ny = 64
    Ndz = 1
    Nch = 3
    Ndth_total = 96
    Ndr = 96
    temp_window = 48
    temp_slide = 24

    img = _try_real_image(Ndz, Nx, Ny, device)
    csm = _coil_sensitivities(Nch, Ndz, Nx, Ny, device)

    full_ktraj = get_ktraj2D_cufi(Ndth_total, Ndr).to(device)
    full_dcomp = get_dcomp_2d(full_ktraj, dtype=torch.float32).to(device)

    op_full = TorchKbOp(Ndx=Nx, Ndy=Ny, Ndth=Ndth_total, Ndr=Ndr,
                        Ndz=Ndz, Nch=Nch, coil_sensitivity_maps=csm,
                        dcomp_mode="sqrt")
    k_space_full = op_full.A(img, full_ktraj, full_dcomp)

    org_k, frame_op = organize_k_space_slidingwindow(
        k_space_full, csm, temp_window, temp_slide,
        backend=backend, dcomp_policy="op_sqrt"
    )

    ref = frame_op.AH(org_k, frames_per_batch=None)   # all frames at once
    chk = frame_op.AH(org_k, frames_per_batch=2)      # small chunks

    err = (ref - chk).abs().max() / (ref.abs().max() + 1e-12)
    assert err.item() < 1e-6, f"Chunked AH differs from all-at-once: rel={err.item():.3e}"


# 2) CUFI per-frame plan cache reuse: same frame returns the same bound object
@pytest.mark.parametrize("backend", ["cufi"])
def test_cufi_frame_plan_cache_reuse(backend):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for CUFI plan cache test")

    _seed()
    device = _device_for_backend(backend)

    Nx = Ny = 64
    Ndz = 1
    Nch = 3
    Ndth_total = 96
    Ndr = 96
    temp_window = 48
    temp_slide = 24

    img = _try_real_image(Ndz, Nx, Ny, device)
    csm = _coil_sensitivities(Nch, Ndz, Nx, Ny, device)

    full_ktraj = get_ktraj2D_cufi(Ndth_total, Ndr).to(device)
    full_dcomp = get_dcomp_2d(full_ktraj, dtype=torch.float32).to(device)

    op_full = CufiOp(
        {"Ndth": Ndth_total, "Ndr": Ndr, "Ndx": Nx, "Ndy": Ny, "Ndz": Ndz,
         "Nch": Nch, "device_id": device.index},
        csm, dcomp_mode="sqrt"
    )
    k_space_full = op_full.A(img, full_ktraj, full_dcomp)

    org_k, frame_op = organize_k_space_slidingwindow(
        k_space_full, csm, temp_window, temp_slide,
        backend=backend, dcomp_policy="op_sqrt"
    )

    assert hasattr(frame_op, "_cufi_factory") and frame_op._cufi_factory is not None, "Factory not initialized"
    # Acquire the same frame twice; the factory should return the exact same bound object
    b0 = frame_op._cufi_factory.get(0, ktraj=frame_op.org_trj[0], dcomp=frame_op.org_cmp[0])
    b0_again = frame_op._cufi_factory.get(0, ktraj=frame_op.org_trj[0], dcomp=frame_op.org_cmp[0])
    assert b0 is b0_again, "Per-frame CUFI plan was not reused"


# 3) torch.compile vs eager on torchkbnufft path
@pytest.mark.parametrize("backend", ["torchkb"])
def test_torchkb_compile_matches_eager(backend):
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available in this environment")

    _seed()
    device = _device_for_backend(backend)

    Nx = Ny = 64
    Ndz = 1
    Nch = 2
    Ndth_total = 64
    Ndr = 64
    temp_window = 32
    temp_slide = 16

    img = _try_real_image(Ndz, Nx, Ny, device)
    csm = _coil_sensitivities(Nch, Ndz, Nx, Ny, device)

    full_ktraj = get_ktraj2D_cufi(Ndth_total, Ndr).to(device)
    full_dcomp = get_dcomp_2d(full_ktraj, dtype=torch.float32).to(device)

    op_full = TorchKbOp(Ndx=Nx, Ndy=Ny, Ndth=Ndth_total, Ndr=Ndr,
                        Ndz=Ndz, Nch=Nch, coil_sensitivity_maps=csm,
                        dcomp_mode="sqrt")
    k_space_full = op_full.A(img, full_ktraj, full_dcomp)

    org_k, frame_op_eager = organize_k_space_slidingwindow(
        k_space_full, csm, temp_window, temp_slide,
        backend=backend, dcomp_policy="op_sqrt"
    )
    # clone a compiled sibling
    _, frame_op_comp = organize_k_space_slidingwindow(
        k_space_full, csm, temp_window, temp_slide,
        backend=backend, dcomp_policy="op_sqrt"
    )
    enabled = frame_op_comp.enable_compile("static")
    if not enabled:
        pytest.skip("compile() failed to enable; skipping equivalence check")

    ref = frame_op_eager.AH(org_k)
    got = frame_op_comp.AH(org_k)

    diff = (ref - got).abs().max() / (ref.abs().max() + 1e-12)
    assert diff.item() < 1e-6, f"compiled vs eager mismatch: rel={diff.item():.3e}"
    
    
    
# --- New tests: organizer units + cache key ----------------------------------

@pytest.mark.parametrize("backend", ["torchkb", "cufi"])
def test_units_and_cache_key(backend):
    _seed()
    device = _device_for_backend(backend)

    Nx = Ny = 64
    Ndz = 1
    Nch = 2
    Ndth_total = 96
    Ndr = 96
    temp_window = 48
    temp_slide = 24

    img = _try_real_image(Ndz, Nx, Ny, device)
    csm = _coil_sensitivities(Nch, Ndz, Nx, Ny, device)

    full_ktraj = get_ktraj2D_cufi(Ndth_total, Ndr).to(device)
    full_dcomp = get_dcomp_2d(full_ktraj, dtype=torch.float32).to(device)

    # Build "full" operator and simulate full k-space
    if backend == "cufi":
        op_full = CufiOp(
            {"Ndth": Ndth_total, "Ndr": Ndr, "Ndx": Nx, "Ndy": Ny, "Ndz": Ndz,
             "Nch": Nch, "device_id": device.index},
            csm, dcomp_mode="sqrt"
        )
    else:
        op_full = TorchKbOp(Ndx=Nx, Ndy=Ny, Ndth=Ndth_total, Ndr=Ndr,
                            Ndz=Ndz, Nch=Nch, coil_sensitivity_maps=csm,
                            dcomp_mode="sqrt")

    k_space_full = op_full.A(img, full_ktraj, full_dcomp)  # (Nch,Ndz,Ndth_total,Ndr)

    org_k, frame_op = organize_k_space_slidingwindow(
        k_space_full, csm, temp_window, temp_slide,
        backend=backend, dcomp_policy="op_sqrt",
        devices=[device.index] if device.type == "cuda" else None
    )

    # 1) trajectory units check
    max_abs = frame_op.org_trj.abs().max().item()
    if backend == "torchkb":
        assert max_abs <= 0.51, f"torchkbnufft should use cycles; max |ktraj|={max_abs:.3f}"
    else:
        assert max_abs > 1.0, f"cufinufft should use radians; max |ktraj|={max_abs:.3f}"

    # 2) dcomp cache filename encodes units
    base_dir = os.path.dirname(orgmod.__file__)
    dcmp_dir = os.path.join(base_dir, "dcomp")
    units_tag = "cycles" if backend == "torchkb" else "rad"
    expected = os.path.join(dcmp_dir, f"dcomp_{Ndth_total}_{Ndr}_w{temp_window}_s{temp_slide}_{units_tag}.pt")
    assert os.path.exists(expected), f"Expected dcomp cache not found: {expected}"


# --- New tests: torchkbnufft chunking equivalence ----------------------------

@pytest.mark.parametrize("backend", ["torchkb"])
def test_torchkb_chunking_equivalence(backend):
    _seed()
    device = _device_for_backend(backend)

    Nx = Ny = 64
    Ndz = 1
    Nch = 3
    Ndth_total = 96
    Ndr = 96
    temp_window = 48
    temp_slide = 24

    img = _try_real_image(Ndz, Nx, Ny, device)
    csm = _coil_sensitivities(Nch, Ndz, Nx, Ny, device)

    full_ktraj = get_ktraj2D_cufi(Ndth_total, Ndr).to(device)
    full_dcomp = get_dcomp_2d(full_ktraj, dtype=torch.float32).to(device)

    op_full = TorchKbOp(Ndx=Nx, Ndy=Ny, Ndth=Ndth_total, Ndr=Ndr,
                        Ndz=Ndz, Nch=Nch, coil_sensitivity_maps=csm,
                        dcomp_mode="sqrt")
    k_space_full = op_full.A(img, full_ktraj, full_dcomp)

    org_k, frame_op = organize_k_space_slidingwindow(
        k_space_full, csm, temp_window, temp_slide,
        backend=backend, dcomp_policy="op_sqrt"
    )

    ref = frame_op.AH(org_k, frames_per_batch=None)  # all frames at once
    chk = frame_op.AH(org_k, frames_per_batch=2)     # small chunks

    err = (ref - chk).abs().max() / (ref.abs().max() + 1e-12)
    assert err.item() < 1e-6, f"Chunked AH differs from all-at-once: rel={err.item():.3e}"


# --- New tests: overlap scheduler vs naive (CUDA only) -----------------------
@pytest.mark.parametrize("backend", ["torchkb", "cufi"])
def test_overlap_scheduler_matches_naive(backend):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for overlap scheduler tests")

    _seed()
    device = _device_for_backend(backend)

    Nx = Ny = 64
    Ndz = 1
    Nch = 3
    Ndth_total = 96
    Ndr = 96
    temp_window = 48
    temp_slide = 24

    img = _try_real_image(Ndz, Nx, Ny, device)
    csm = _coil_sensitivities(Nch, Ndz, Nx, Ny, device)

    full_ktraj = get_ktraj2D_cufi(Ndth_total, Ndr).to(device)
    full_dcomp = get_dcomp_2d(full_ktraj, dtype=torch.float32).to(device)

    if backend == "cufi":
        op_full = CufiOp(
            {"Ndth": Ndth_total, "Ndr": Ndr, "Ndx": Nx, "Ndy": Ny, "Ndz": Ndz,
             "Nch": Nch, "device_id": device.index},
            csm, dcomp_mode="sqrt"
        )
    else:
        op_full = TorchKbOp(Ndx=Nx, Ndy=Ny, Ndth=Ndth_total, Ndr=Ndr,
                            Ndz=Ndz, Nch=Nch, coil_sensitivity_maps=csm,
                            dcomp_mode="sqrt")

    k_space_full = op_full.A(img, full_ktraj, full_dcomp)

    org_k, frame_op = organize_k_space_slidingwindow(
        k_space_full, csm, temp_window, temp_slide,
        backend=backend, dcomp_policy="op_sqrt"
    )

    # Naive reference on the same device as org_k
    ref = frame_op.AH(org_k)

    # Gather overlap results to the same device as ref
    ol1 = AH_overlap(frame_op, org_k, devices=[device.index],
                     prestage_constants=False, gather="device", out_device=ref.device)
    ol2 = AH_overlap(frame_op, org_k, devices=[device.index],
                     prestage_constants=True, gather="device", out_device=ref.device)

    def _rel(a, b):
        # ensure both are on same device
        a = a.to(b.device)
        num = (a - b).abs().max()
        den = b.abs().max() + 1e-12
        return (num / den).item()

    r1 = _rel(ol1, ref)
    r2 = _rel(ol2, ref)
    rtol = 5e-6 if backend == "torchkb" else 1e-6
    assert r1 < rtol, f"Overlap scheduler (no prestage) deviates: {r1:.3e}"
    assert r2 < rtol, f"Overlap scheduler (prestage) deviates: {r2:.3e}"


# --- New tests: CUFI per-frame plan cache reuse ------------------------------

@pytest.mark.parametrize("backend", ["cufi"])
def test_cufi_frame_plan_cache_reuse(backend):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for CUFI plan cache test")

    _seed()
    device = _device_for_backend(backend)

    Nx = Ny = 64
    Ndz = 1
    Nch = 3
    Ndth_total = 96
    Ndr = 96
    temp_window = 48
    temp_slide = 24

    img = _try_real_image(Ndz, Nx, Ny, device)
    csm = _coil_sensitivities(Nch, Ndz, Nx, Ny, device)

    full_ktraj = get_ktraj2D_cufi(Ndth_total, Ndr).to(device)
    full_dcomp = get_dcomp_2d(full_ktraj, dtype=torch.float32).to(device)

    op_full = CufiOp(
        {"Ndth": Ndth_total, "Ndr": Ndr, "Ndx": Nx, "Ndy": Ny, "Ndz": Ndz,
         "Nch": Nch, "device_id": device.index},
        csm, dcomp_mode="sqrt"
    )
    k_space_full = op_full.A(img, full_ktraj, full_dcomp)

    org_k, frame_op = organize_k_space_slidingwindow(
        k_space_full, csm, temp_window, temp_slide,
        backend=backend, dcomp_policy="op_sqrt"
    )

    assert hasattr(frame_op, "_cufi_factory") and frame_op._cufi_factory is not None, "Factory not initialized"
    # same frame twice -> same bound object
    b0 = frame_op._cufi_factory.get(0, ktraj=frame_op.org_trj[0], dcomp=frame_op.org_cmp[0])
    b0_again = frame_op._cufi_factory.get(0, ktraj=frame_op.org_trj[0], dcomp=frame_op.org_cmp[0])
    assert b0 is b0_again, "Per-frame CUFI plan was not reused"


# --- New tests: torch.compile vs eager (torchkbnufft) ------------------------

@pytest.mark.parametrize("backend", ["torchkb"])
def test_torchkb_compile_matches_eager(backend):
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available in this environment")

    _seed()
    device = _device_for_backend(backend)

    Nx = Ny = 64
    Ndz = 1
    Nch = 2
    Ndth_total = 64
    Ndr = 64
    temp_window = 32
    temp_slide = 16

    img = _try_real_image(Ndz, Nx, Ny, device)
    csm = _coil_sensitivities(Nch, Ndz, Nx, Ny, device)

    full_ktraj = get_ktraj2D_cufi(Ndth_total, Ndr).to(device)
    full_dcomp = get_dcomp_2d(full_ktraj, dtype=torch.float32).to(device)

    op_full = TorchKbOp(Ndx=Nx, Ndy=Ny, Ndth=Ndth_total, Ndr=Ndr,
                        Ndz=Ndz, Nch=Nch, coil_sensitivity_maps=csm,
                        dcomp_mode="sqrt")
    k_space_full = op_full.A(img, full_ktraj, full_dcomp)

    org_k, frame_op_eager = organize_k_space_slidingwindow(
        k_space_full, csm, temp_window, temp_slide,
        backend=backend, dcomp_policy="op_sqrt"
    )
    # sibling op we can compile
    _, frame_op_comp = organize_k_space_slidingwindow(
        k_space_full, csm, temp_window, temp_slide,
        backend=backend, dcomp_policy="op_sqrt"
    )
    enabled = frame_op_comp.enable_compile("static")
    if not enabled:
        pytest.skip("compile() failed to enable; skipping equivalence check")

    ref = frame_op_eager.AH(org_k)
    got = frame_op_comp.AH(org_k)

    diff = (ref - got).abs().max() / (ref.abs().max() + 1e-12)
    assert diff.item() < 1e-6, f"compiled vs eager mismatch: rel={diff.item():.3e}"