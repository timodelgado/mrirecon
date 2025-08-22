# -*- coding: utf-8 -*-
import math
from typing import Tuple

import pytest
import torch

from graspcg.workspace.unified_arena import DeviceArena
from graspcg.workspace.workspace import Workspace, BufSpec
from graspcg.solvers.cg import CGSolver, CGConfig
from graspcg.regularization.tv_nd import TVND, TVParams
from graspcg.regularization.manager import RegManager
from graspcg.numerics.line_search import search as ls_search
from graspcg.numerics.directions import DirFR
from graspcg.core.roles import Roles

# -----------------------------
# Helpers
# -----------------------------
def _has_cuda(min_devices: int = 1) -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() >= min_devices

def _rand_cplx(shape, device, dtype=torch.complex64, scale=1.0, seed=123):
    g = torch.Generator(device="cpu").manual_seed(seed)
    re = torch.randn(*shape, generator=g) * scale
    im = torch.randn(*shape, generator=g) * scale
    return (re + 1j * im).to(device=device, dtype=dtype)

def _scatter_full_x_into_ws(ws: Workspace, x_full: torch.Tensor) -> None:
    for sh, i in ws.iter_shards():
        xs = ws.get("x", i)
        xs.copy_(x_full[int(sh.b_start):int(sh.b_stop)].to(xs.device))

def _gather_ws_tensor(ws: Workspace, name: str, dtype=torch.complex64, device="cpu") -> torch.Tensor:
    parts = []
    for sh, i in ws.iter_shards():
        parts.append(ws.get(name, i).to(device))
    return torch.cat(parts, dim=0) if parts else torch.empty(0, dtype=dtype, device=device)


# A minimal Cartesian FFT operator satisfying the NUFFTLike contract used by Workspace/CGSolver.
# A / AH are simple FFTs on the last two dims; image_shape equals y.shape.
class CartesianFFT:
    # Explicit roles (B is the unlike axis; C is like; H,W are nufft)
    roles_image  = Roles(unlike=1, like=1, nufft=2)
    roles_kspace = Roles(unlike=1, like=1, nufft=2)
    def image_shape(self, y: torch.Tensor) -> Tuple[int, ...]:
        return tuple(int(s) for s in y.shape)

    def A(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        y = torch.fft.fftn(x, dim=(-2, -1))
        if out is None:
            return y
        out.copy_(y)
        return out

    def AH(self, y: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.fft.ifftn(y, dim=(-2, -1))
        if out is None:
            return x
        out.copy_(x)
        return out


# -----------------------------
# DeviceArena tests
# -----------------------------

def test_arena_compute_device_resolution_cpu():
    arena = DeviceArena(compute="cpu")
    assert arena.compute_device().type == "cpu"

@pytest.mark.skipif(not _has_cuda(1), reason="CUDA not available")
def test_arena_compute_device_resolution_cuda():
    arena = DeviceArena(compute="cuda")
    assert arena.compute_device().type == "cuda"

@pytest.mark.skipif(not _has_cuda(1), reason="CUDA not available")
def test_arena_streams_and_request_anchor():
    arena = DeviceArena(compute="cuda")
    primary = arena.compute_device()
    # stream_for should create and memoize a stream per device
    s0 = arena.stream_for(primary)
    s1 = arena.stream_for(primary)
    assert s0 is s1

    # request() with an anchor must place on the anchor's device
    anchor = torch.empty(0, device=primary, dtype=torch.float32)
    buf = arena.request(1024, torch.complex64, anchor=anchor)
    assert buf.device == primary and buf.numel() >= 1024
    arena.release(buf)


# -----------------------------
# Workspace planning & buffers
# -----------------------------

def _manifest_for_basic():
    # minimal manifest exercising image/kspace/spatial + per_shard/global
    CPLX, REAL = torch.complex64, torch.float32
    return [
        BufSpec("x",    "image",  "per_shard", "image",   CPLX, init="zeros"),
        BufSpec("g",    "image",  "per_shard", "image",   CPLX, init="zeros"),
        BufSpec("dx",   "image",  "per_shard", "image",   CPLX, init="zeros"),
        BufSpec("diag", "image",  "per_shard", "spatial", REAL, init="ones"),
        BufSpec("r_sh", "kspace", "per_shard", "kspace",  CPLX, init="zeros", lifetime="ls"),
    ]


def test_workspace_shards_cover_batch_cpu():
    B, C, H, W = 6, 1, 8, 8
    y = _rand_cplx((B, C, H, W), device="cpu", scale=0.01)
    op = CartesianFFT()
    arena = DeviceArena(compute="cpu")
    ws = Workspace(y, op, arena=arena, buf_specs=_manifest_for_basic(),
                   kspace_mode="sharded", benchmark=False)

    # shards cover [0, B) contiguously without overlap
    spans = sorted((int(sh.b_start), int(sh.b_stop)) for sh in (s for s, _ in ws.iter_shards()))
    assert spans[0][0] == 0 and spans[-1][1] == B
    # disjoint & contiguous
    for (a0, a1), (b0, b1) in zip(spans[:-1], spans[1:]):
        assert a1 == b0 and a0 < a1 and b0 < b1

    # buffer shapes: x per_shard should match shard.shape; diag is per-spatial
    for sh, i in ws.iter_shards():
        x = ws.get("x", i)
        d = ws.get("diag", i)
        r = ws.get("r_sh", i)
        assert tuple(x.shape) == (sh.span, C, H, W)
        assert tuple(d.shape) == (C, H, W)
        assert tuple(r.shape) == (sh.span, C, H, W)


@pytest.mark.skipif(not _has_cuda(2), reason="need at least 2 CUDA devices for multi-GPU shard test")
def test_workspace_shards_multi_gpu_distribution():
    B, C, H, W = 8, 1, 8, 8
    y = _rand_cplx((B, C, H, W), device="cuda:0", scale=0.01)
    op = CartesianFFT()
    # compute on cuda:0, helpers left to default (other GPUs)
    arena = DeviceArena(compute="cuda:0")
    ws = Workspace(y, op, arena=arena, buf_specs=_manifest_for_basic(),
                   kspace_mode="sharded", benchmark=False)

    devices_used = sorted({sh.device.index for sh, _ in ws.iter_shards() if sh.device.type == "cuda"})
    assert len(devices_used) >= 2, f"Expected shards across >=2 GPUs, got {devices_used}"


# -----------------------------
# Integration with CG solver
# -----------------------------

@pytest.mark.parametrize("use_cuda", [False, pytest.param(True, marks=pytest.mark.skipif(not _has_cuda(1), reason="CUDA not available"))])
def test_cg_integration_smoke(use_cuda: bool):
    dev = torch.device("cuda:0") if use_cuda else torch.device("cpu")
    B, C, H, W = 4, 1, 16, 16
    y = _rand_cplx((B, C, H, W), device=dev, scale=0.1, seed=777)

    op = CartesianFFT()
    regs = [TVND(name="tv", params=TVParams(weight=1e-3, eps=1e-3, axes="spatial", isotropic=True))]
    regm = RegManager(regs)

    cfg = CGConfig(
        devices="cuda" if use_cuda else "cpu",
        direction="fr",
        ls_name="armijo",
        max_iter=3,
        record_history=True,
        verbose=False,
    )
    solver = CGSolver(y=y, nufft_op=op, regm=regm, cfg=cfg)
    solver.run(max_iter=3)

    hist = solver.history()
    assert len(hist) >= 1
    for rec in hist:
        # ensure finite floats and required keys present
        assert all(k in rec for k in ("iter", "f", "gdot", "step", "xnorm", "gnorm", "stepnorm"))
        assert all(math.isfinite(float(v)) for v in rec.values())


# -----------------------------
# Compile‑friendliness characterization (0‑D device tensors)
# -----------------------------

@pytest.mark.parametrize("use_cuda", [False, pytest.param(True, marks=pytest.mark.skipif(not _has_cuda(1), reason="CUDA not available"))])
def test_line_search_and_direction_return_0d_tensors(use_cuda: bool):
    dev = torch.device("cuda:0") if use_cuda else torch.device("cpu")
    B, C, H, W = 3, 1, 8, 8
    y = _rand_cplx((B, C, H, W), device=dev, scale=0.05, seed=313)
    op = CartesianFFT()
    regs = [TVND(name="tv", params=TVParams(weight=0.0, eps=1e-3))]  # zero weight to keep it simple here
    regm = RegManager(regs)
    cfg = CGConfig(devices="cuda" if use_cuda else "cpu", direction="fr", ls_name="armijo", max_iter=1, record_history=False)
    solver = CGSolver(y=y, nufft_op=op, regm=regm, cfg=cfg)

    # Prime g and a descent direction the same way the solver does
    solver.obj.begin_linesearch(solver.ws)
    try:
        t0 = torch.zeros((), device=dev, dtype=y.real.dtype)
        f0, g0d = solver.obj.f_g_tensor(solver.ws, t0)
        for sh, i in solver.ws.iter_shards():
            g, d, D = solver.ws.bind(i, "g", "dx", "diag")
            d.copy_(g).div_(D).neg_()
        # DirFR.init_state may cache a rho; then do one update
        dirfr = DirFR(solver.ws)
        dirfr.init_state()
    finally:
        solver.obj.end_linesearch(solver.ws)

    ok, t, f_t, gdot_t = ls_search(solver)
    assert all(isinstance(z, torch.Tensor) and z.dim() == 0 for z in (ok, t, f_t, gdot_t))
    beta = dirfr.update_inplace(solver.ws)
    assert isinstance(beta, torch.Tensor) and beta.dim() == 0


# -----------------------------
# TV correctness: shards ≡ single device
# -----------------------------

@pytest.mark.skipif(not _has_cuda(1), reason="TV cross-shard equivalence requires CUDA to compare sharded vs single-device")
def test_tv_energy_and_grad_equivalence_sharded_vs_single_device():
    B, C, H, W = 6, 1, 24, 24
    # Use a fixed CPU master tensor for reproducibility; scatter to workspaces
    x_full_cpu = _rand_cplx((B, C, H, W), device="cpu", scale=0.2, seed=202405)

    # Common reg manager
    tv = TVND(name="tv", params=TVParams(weight=1e-2, eps=1e-3, axes="spatial", isotropic=True))
    regm = RegManager([tv])

    # Build a k-space "observation" y that is irrelevant (we only use ws for TV buffers)
    y_cpu = torch.zeros_like(x_full_cpu)
    op = CartesianFFT()

    # ----- single-device workspace (CUDA primary only; still one shard by design) -----
    arena_single = DeviceArena(compute="cuda:0", helpers=[])
    ws_single = Workspace(y_cpu.to("cuda:0"), op, arena=arena_single,
                          buf_specs=_manifest_for_basic(), kspace_mode="sharded", benchmark=False)
    _scatter_full_x_into_ws(ws_single, x_full_cpu)
    # Zero grads before eval
    for sh, i in ws_single.iter_shards():
        ws_single.get("g", i).zero_()

    E_single = regm.energy_and_grad(ws_single)
    g_single = _gather_ws_tensor(ws_single, "g", device="cpu")

    # ----- sharded workspace (primary + helpers) -----
    arena_sharded = DeviceArena(compute="cuda")  # default helpers = all other GPUs
    ws_sharded = Workspace(y_cpu.to(arena_sharded.compute_device()), op, arena=arena_sharded,
                           buf_specs=_manifest_for_basic(), kspace_mode="sharded", benchmark=False)
    _scatter_full_x_into_ws(ws_sharded, x_full_cpu)
    for sh, i in ws_sharded.iter_shards():
        ws_sharded.get("g", i).zero_()

    E_sharded = regm.energy_and_grad(ws_sharded)
    g_sharded = _gather_ws_tensor(ws_sharded, "g", device="cpu")

    # ----- compare -----
    # Energies equal within tight tolerance (FFT/TV are deterministic given same x)
    assert torch.allclose(E_single.to("cpu"), E_sharded.to("cpu"), rtol=1e-5, atol=1e-6)
    # Gradients equal element-wise on CPU
    assert torch.allclose(g_single, g_sharded, rtol=1e-5, atol=1e-6)