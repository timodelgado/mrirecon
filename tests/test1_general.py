# tests/test_general.py
# Run with:  pytest -q
import math
import types
import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers / stubs used by multiple tests
# ---------------------------------------------------------------------------

class FakeNUFFT:
    """
    Minimal linear operator with adjoint = identity.
    Supports out=..., accumulate=... paths.
    """
    def __init__(self, scale_emp: float = 1.0, device=None, dtype=torch.complex64):
        self.scale_emp = float(scale_emp)
        self.device = device or torch.device("cpu")
        self.dtype = dtype

    @torch.no_grad()
    def A(self, x, out=None, accumulate=False):
        if out is None:
            return x.clone()
        if accumulate:
            out.add_(x)
        else:
            out.copy_(x)

    @torch.no_grad()
    def AH(self, y, out=None):
        if out is None:
            return y.clone()
        out.copy_(y)

class DummyScale:
    @torch.no_grad()
    def inv_s2_for_shard(self, sh, anchor=None):
        # Shape: (B_loc, 1, 1, ...) with ones
        shape = [sh.x.shape[0]] + [1] * (sh.x.ndim - 1)
        return torch.ones(shape, dtype=torch.float32, device=sh.x.device)

    @torch.no_grad()
    def inv_s_for_shard(self, sh, anchor=None):
        return torch.ones_like(self.inv_s2_for_shard(sh, anchor)).sqrt()

class Shard:
    def __init__(self, shape, device=torch.device("cpu"), dtype=torch.complex64,
                 b_slice=None):
        self.x   = torch.zeros(shape, dtype=dtype, device=device)
        self.g   = torch.zeros_like(self.x)
        self.dx  = torch.zeros_like(self.x)
        # Real diag, same spatial shape as x.real (full per-voxel)
        self.diag = torch.ones_like(self.x.real, dtype=torch.float32)
        # For APIs that slice k-space by frames:
        B = shape[0]
        self.b_slice = b_slice or slice(0, B)

class FakeWorkspace:
    """
    Enough of a workspace to test objective, directions, preconditioner, etc.
    """
    def __init__(self, y, nufft_op, arena, shards):
        self.y = y
        self.nufft_op = nufft_op
        self.arena = arena
        self._shards = [(s, i) for i, s in enumerate(shards)]
        self.scale = DummyScale()

    def iter_shards(self):
        yield from self._shards

# ---------------------------------------------------------------------------
# Imports from the codebase (skip gracefully if a module is missing)
# ---------------------------------------------------------------------------
Arena = pytest.importorskip("graspcg.workspace.unified_arena").UnifiedArena
DevicePool = pytest.importorskip("graspcg.workspace.device_pool").DevicePool
DeviceCfg = pytest.importorskip("graspcg.workspace.device_cfg").DeviceCfg
ops = pytest.importorskip("graspcg.utils.operations")
precond = pytest.importorskip("graspcg.regularization.preconditioner")
objective_mod = pytest.importorskip("graspcg.ops.objective")
init_scaling = pytest.importorskip("graspcg.ops.init_scaling")
line_search = pytest.importorskip("graspcg.numerics.line_search")
directions = pytest.importorskip("graspcg.numerics.directions")

# Optional modules (skip if not present or still under construction)
tvt_mod = pytest.importorskip("graspcg.regularization.tvt")
tvspatial_mod = pytest.importorskip("graspcg.regularization.tv_spatial")

# ---------------------------------------------------------------------------
# Basic device facts
# ---------------------------------------------------------------------------
CUDA = torch.cuda.is_available()
CPU  = torch.device("cpu")
CPLX = torch.complex64
REAL = torch.float32

# ---------------------------------------------------------------------------
# UnifiedArena & DevicePool / DeviceCfg
# ---------------------------------------------------------------------------
def test_arena_request_release_basic():
    arena = Arena()
    a = torch.empty(8, dtype=CPLX, device=CPU)
    buf = arena.request(100, CPLX, anchor=a)
    assert buf.numel() >= 100
    assert buf.device == a.device and buf.dtype == CPLX
    arena.release(buf)  # should not throw

def test_arena_reuse_and_accumulate_path():
    # Ensure we can allocate, release, and reallocate; and use accumulate path
    arena = Arena()
    y = torch.zeros(16, dtype=CPLX)
    nuf = FakeNUFFT()
    x1 = arena.request(16, CPLX, anchor=y).view_as(y).zero_()
    x2 = torch.ones_like(y)
    nuf.A(x2, out=x1, accumulate=True)  # x1 += x2
    assert torch.allclose(x1, x2)
    arena.release(x1)

def test_device_pool_claim_order_cpu_fallback():
    pool = DevicePool(gpu_ids=[])  # No GPUs enumerated -> only CPU target
    chosen = pool.claim(bytes_needed=10)
    assert chosen.type == "cpu"

def test_device_cfg_streams_and_helpers():
    cfg = DeviceCfg(compute="cpu")  # force CPU
    with pytest.raises(ValueError):
        cfg.stream_for("cpu")
    assert isinstance(cfg.cuda_devices(), list)  # may be empty

# ---------------------------------------------------------------------------
# ops.utilities
# ---------------------------------------------------------------------------
def test_dot_chunked_matches_dense_with_and_without_diag():
    a = torch.randn(3, 4, dtype=CPLX)
    b = torch.randn(3, 4, dtype=CPLX)
    diag = torch.rand(3, 4, dtype=REAL).add_(1e-2)

    dense = torch.real((a.conj() * b).sum()).item()
    chunked = ops.dot_chunked(a, b)
    assert math.isclose(chunked, dense, rel_tol=1e-5, abs_tol=1e-7)

    dense_pc = torch.real((a.conj() * (b / diag)).sum()).item()
    chunked_pc = ops.dot_chunked(a, b, diag=diag)
    assert math.isclose(chunked_pc, dense_pc, rel_tol=1e-5, abs_tol=1e-7)

def test_suggest_tile_nonzero():
    arena = Arena()
    dev = CPU
    tile = ops.suggest_tile(target_elems=64, arena=arena, dtype=CPLX, dev=dev)
    assert isinstance(tile, int) and tile >= 1

# ---------------------------------------------------------------------------
# Preconditioner
# ---------------------------------------------------------------------------
def test_build_precond_diag_calls_reg_add_diag_shard():
    # Prepare a workspace with two shards
    B, S1, S2 = 2, 3, 4
    shards = [Shard((B, S1, S2), device=CPU), Shard((B, S1, S2), device=CPU)]
    y = torch.zeros((2*B, S1, S2), dtype=CPLX)  # matches frame slicing logic
    arena = Arena()
    ws = FakeWorkspace(y, FakeNUFFT(scale_emp=1.0), arena, shards)

    # Dummy reg that adds +2 to diag (per element)
    class Reg:
        @torch.no_grad()
        def add_diag_shard(self, ws, sh, diag):
            diag.add_(2.0)

    class RegManagerObj:
        def __init__(self): self.regs = {"dummy": Reg()}
        @torch.no_grad()
        def add_diag_shard(self, ws, sh, diag): self.regs["dummy"].add_diag_shard(ws, sh, diag)

    regm = RegManagerObj()

    # Build diag = nufft_norm (1.0) + 2.0 = 3.0
    precond.build_precond_diag(ws, regm, mode="full", use_nufft_norm=True)
    for sh, _ in ws.iter_shards():
        assert torch.allclose(sh.diag, torch.full_like(sh.diag, 3.0))

# ---------------------------------------------------------------------------
# Init scaling
# ---------------------------------------------------------------------------
def test_initial_backproj_and_scaling_identity_operator():
    B, S1, S2 = 3, 2, 2
    # Two shards concatenated along B axis in k-space
    y = torch.randn((2*B, S1, S2), dtype=CPLX)
    shards = [Shard((B, S1, S2), device=CPU), Shard((B, S1, S2), device=CPU)]
    ws = FakeWorkspace(y, FakeNUFFT(scale_emp=1.0), Arena(), shards)

    out = init_scaling.initial_backproj_and_scaling(ws, regm=types.SimpleNamespace(estimate_from_pilot=None),
                                                    xfactor=4.0, stats_cfg=None, verbose=False)
    # x := AH y = y shard-wise; with identity A/AH, E_data == E_est ⇒ scale = sqrt(4) = 2
    for sh, _ in ws.iter_shards():
        assert torch.allclose(sh.x, 2.0 * y[0:B] if _ == 0 else 2.0 * y[B:2*B])

    assert "E_data" in out and isinstance(out["E_data"], float)

# ---------------------------------------------------------------------------
# Objective caches & f_g (with null regularizer)
# ---------------------------------------------------------------------------
def test_objective_begin_end_and_f_g_no_reg():
    B, S1, S2 = 2, 2, 2
    y = torch.randn((B, S1, S2), dtype=CPLX)
    shards = [Shard((B, S1, S2), device=CPU)]
    ws = FakeWorkspace(y, FakeNUFFT(scale_emp=1.0), Arena(), shards)

    class NullRegManager:
        @torch.no_grad()
        def energy_and_grad(self, ws): return 0.0

    obj = objective_mod.Objective(ws.nufft_op, y, regm=NullRegManager())
    # dx=0 ⇒ Adx=0; Ax is zero initially; r = -y
    obj.begin_linesearch(ws)
    f, gdot = obj.f_g(ws, t=0.0)
    # f_data should be 0.5 * ||y||^2; gdot should be 0 with d=0
    target = 0.5 * (y.real.square().sum() + y.imag.square().sum()).item()
    assert math.isclose(f, target, rel_tol=1e-5, abs_tol=1e-7)
    assert math.isclose(gdot, 0.0, abs_tol=1e-12)
    obj.end_linesearch(ws)

# ---------------------------------------------------------------------------
# Directions (PR+, DY, FR) — update rule sanity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dir_name", ["prplus", "dy", "fr"])
def test_directions_first_update_is_precond_steepest_descent(dir_name):
    # Two shards; initial d = 0 ⇒ first update must be d = -M^{-1} g
    B, S1 = 3, 2
    shards = [Shard((B, S1), device=CPU), Shard((B, S1), device=CPU)]
    for sh, lam in zip(shards, [2.0, 0.5]):
        sh.g.normal_()
        sh.dx.zero_()
        sh.diag.fill_(lam)
    ws = FakeWorkspace(y=torch.zeros(2*B, S1, dtype=CPLX),
                       nufft_op=FakeNUFFT(), arena=Arena(), shards=shards)

    dir_obj = directions.build_direction(dir_name, ws)
    # Seed state
    if hasattr(dir_obj, "init_state"):
        dir_obj.init_state()
    # Update in place
    dir_obj.update_inplace(ws)
    # Should now be preconditioned negative gradient
    for sh in [s for s, _ in ws.iter_shards()]:
        assert torch.allclose(sh.dx, -sh.g / sh.diag)

# ---------------------------------------------------------------------------
# Line search (Armijo and Wolfe) on a synthetic 1-D quadratic along d
# ---------------------------------------------------------------------------
class QuadObjective:
    """
    f(t) = f0 + g0d * t + 0.5 * a * t^2
    g(t)^T d = g0d + a t
    """
    def __init__(self, a=1.0, f0=1.0, g0d=-1.0):
        self.a = float(a); self.f0 = float(f0); self.g0d = float(g0d)

    @torch.no_grad()
    def f_g(self, ws, t: float):
        f_t = self.f0 + self.g0d * t + 0.5 * self.a * t * t
        gdot = self.g0d + self.a * t
        return float(f_t), float(gdot)

class MiniSolver:
    def __init__(self, ls_name="wolfe", c1=1e-4, c2=0.9, it=20, zoom=True):
        self.ls_name = ls_name
        self.c1 = c1
        self.c2 = c2 if ls_name != "armijo" else 0.0
        self.ls_max_iter = it
        self.ls_zoom = zoom
        self.ws = object()  # unused
        self.obj = QuadObjective(a=2.0, f0=1.0, g0d=-1.0)

@pytest.mark.parametrize("ls_name", ["armijo", "wolfe"])
def test_line_search_on_quadratic(ls_name):
    solver = MiniSolver(ls_name=ls_name)
    ok, t, f_new, gdot = line_search.search(solver, f0=1.0, g0d=-1.0)
    assert ok
    # For a convex quadratic with g0d<0, acceptable t should be >0 and finite
    assert 0.0 < t < 1e6
    # Strong Wolfe or Armijo ensure decrease
    assert f_new < 1.0

# ---------------------------------------------------------------------------
# Temporal and Spatial TV (energy/grad/diag) sanity
# ---------------------------------------------------------------------------
def test_temporal_tv_energy_grad_and_diag_sanity():
    # Skip if module is incomplete
    TemporalTV = tvt_mod.TemporalTV

    B, S1, S2 = 4, 2, 2
    shards = [Shard((B, S1, S2), device=CPU)]
    # Build a simple ramp along time so TV is positive and predictable
    ramp = torch.arange(B, dtype=REAL).view(B, 1, 1).expand(B, S1, S2).to(CPLX)
    shards[0].x.copy_(ramp)
    y = torch.zeros((B, S1, S2), dtype=CPLX)
    ws = FakeWorkspace(y, FakeNUFFT(), Arena(), shards)

    tv = TemporalTV(weight=1.0, eps=1e-3, tile=None, apply_scale=True)
    # Energy should be > 0
    E = tv.energy_and_grad(ws)
    assert E > 0.0
    # Grad must be finite, and adding diag should increase diag
    before = shards[0].diag.clone()
    tv.add_diag_shard(ws, shards[0], shards[0].diag)
    assert torch.all(shards[0].diag >= before)

def test_spatial_tv_diag_and_energy_nonnegative():
    SpatialTV = tvspatial_mod.SpatialTV
    B, S1, S2, S3 = 2, 3, 3, 3
    shards = [Shard((B, S1, S2, S3), device=CPU)]
    y = torch.zeros((B, S1, S2, S3), dtype=CPLX)
    ws = FakeWorkspace(y, FakeNUFFT(), Arena(), shards)
    # minimal config blob imitation
    ws.regs = {"tv_s": {"weight": 0.5, "eps": 1e-3, "voxel_size": (1.0, 1.0, 1.0)}}

    tvs = SpatialTV.from_ws(ws)
    E = tvs.energy_and_grad(ws)  # should not crash; energy >= 0 by construction
    assert E >= 0.0
    before = shards[0].diag.clone()
    tvs.add_diag_shard(ws, shards[0], shards[0].diag)
    assert torch.all(shards[0].diag >= before)

# ---------------------------------------------------------------------------
# Smoke tests for modules that may still be evolving
# ---------------------------------------------------------------------------
def test_modules_importable_and_have_expected_symbols():
    # These asserts ensure the modules expose the tested functions/classes.
    assert hasattr(ops, "dot_chunked")
    assert hasattr(precond, "build_precond_diag")
    assert hasattr(objective_mod, "Objective")
    assert hasattr(init_scaling, "initial_backproj_and_scaling")
    assert hasattr(line_search, "search")
    assert hasattr(directions, "build_direction")
    assert hasattr(tvt_mod, "TemporalTV")
    assert hasattr(tvspatial_mod, "SpatialTV")

# ---------------------------------------------------------------------------
# (Optional) CUDA-only sanity checks
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CUDA, reason="CUDA not available")
def test_arena_request_on_cuda_device():
    arena = Arena()
    dev = torch.device("cuda", torch.cuda.current_device())
    buf = arena.request(32, CPLX, device=dev)
    assert buf.device.type == "cuda"