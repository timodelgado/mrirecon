# graspcg/tests/test_workspace_stack.py
from __future__ import annotations
import math
import types
import sys
import importlib

import pytest
import torch

from graspcg.workspace.workspace import Workspace, BufSpec, Roles
from graspcg.workspace.unified_arena import DeviceArena
from graspcg.ops.objective import Objective
from graspcg.ops.init_scaling import initial_backproj_and_scaling
from graspcg.solvers.cg import CGSolver
from graspcg.numerics.directions import DirPRPlus, DirDY
from graspcg.numerics import line_search as ls_mod


# ────────────────────────────────────────────────────────────────────────
# Helpers / Fixtures
# ────────────────────────────────────────────────────────────────────────

class FakeNUFFTIdentity:
    """
    NUFFT stub where A = I, AH = I. K-space and image shapes are identical.
    Roles enforce (unlike=1, like=0, nufft=2) i.e., (B, H, W).
    """
    def __init__(self):
        self.roles_image  = Roles(unlike=1, like=0, nufft=2)
        self.roles_kspace = Roles(unlike=1, like=0, nufft=2)
        self.scale_emp = 1.0
        self.calls_A = 0
        self.calls_AH = 0

    def image_shape(self, y: torch.Tensor):
        return (int(y.shape[0]), *y.shape[1:])

    @torch.no_grad()
    def A(self, x: torch.Tensor, out: torch.Tensor | None = None, accumulate: bool = False):
        self.calls_A += 1
        if out is None:
            return x.clone()
        if accumulate:
            out.add_(x)
        else:
            out.copy_(x)

    @torch.no_grad()
    def AH(self, y: torch.Tensor, out: torch.Tensor | None = None, accumulate: bool = False):
        self.calls_AH += 1
        if out is None:
            return y.clone()
        if accumulate:
            out.add_(y)
        else:
            out.copy_(y)


class FakeNUFFTScalar(FakeNUFFTIdentity):
    """
    NUFFT stub where A = α I and AH = α I (self-adjoint scalar).
    Useful for scaling tests.
    """
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = float(alpha)

    @torch.no_grad()
    def A(self, x: torch.Tensor, out: torch.Tensor | None = None, accumulate: bool = False):
        self.calls_A += 1
        if out is None:
            return x.mul(self.alpha)
        if accumulate:
            out.add_(x, alpha=self.alpha)
        else:
            out.copy_(x * self.alpha)

    @torch.no_grad()
    def AH(self, y: torch.Tensor, out: torch.Tensor | None = None, accumulate: bool = False):
        self.calls_AH += 1
        if out is None:
            return y.mul(self.alpha)
        if accumulate:
            out.add_(y, alpha=self.alpha)
        else:
            out.copy_(y * self.alpha)


class NullRegManager:
    """Minimal reg manager: contributes no energy, no gradient."""
    def __init__(self):
        self.regs = {}

    @torch.no_grad()
    def energy_and_grad(self, ws) -> float:
        return 0.0


def make_specs(cplx: torch.dtype, real: torch.dtype, with_kspace_caches: bool = True):
    specs = [
        BufSpec("x",      "image",  "per_shard", "image",   cplx, init="zeros"),
        BufSpec("g",      "image",  "per_shard", "image",   cplx, init="zeros"),
        BufSpec("dx",     "image",  "per_shard", "image",   cplx, init="zeros"),
        BufSpec("diag",   "image",  "per_shard", "spatial", real, init="ones"),
    ]
    if with_kspace_caches:
        specs += [
            BufSpec("Ax_sh","kspace","per_shard","kspace", cplx, init="zeros", lifetime="ls"),
            BufSpec("Ad_sh","kspace","per_shard","kspace", cplx, init="zeros", lifetime="ls"),
            BufSpec("r_sh", "kspace","per_shard","kspace", cplx, init="zeros", lifetime="ls"),
        ]
    return specs


# ────────────────────────────────────────────────────────────────────────
# Workspace & roles
# ────────────────────────────────────────────────────────────────────────

def test_workspace_roles_validation_pass_cpu():
    torch.manual_seed(1)
    B, H, W = 4, 3, 2
    y = torch.randn((B, H, W), dtype=torch.complex64, device="cpu")
    nufft = FakeNUFFTIdentity()
    arena = DeviceArena(compute="cpu")
    ws = Workspace(y, nufft, arena, buf_specs=make_specs(y.dtype, torch.float32, with_kspace_caches=True))

    assert ws.plan.image_shape == (B, H, W)
    assert ws.plan.kspace_shape == (B, H, W)
    assert ws.plan.roles_image.unlike == 1 and ws.plan.roles_image.nufft == 2
    assert ws.plan.roles_kspace.unlike == 1 and ws.plan.roles_kspace.nufft == 2

    shards = list(ws.iter_shards())
    assert len(shards) == 1
    (sh, i) = shards[0]
    assert sh.span == B
    # accessors work
    x = ws.get("x", i)
    assert x.shape == (B, H, W)
    g, dx, D = ws.bind(i, "g", "dx", "diag")
    assert g.shape == x.shape and dx.shape == x.shape and D.shape == (H, W)


def test_workspace_roles_validation_fail_mismatch_dims():
    torch.manual_seed(0)
    # y has 2 dims after B (H only) but roles expect 2 (H,W) → ValueError
    B, H = 2, 3
    y = torch.randn((B, H), dtype=torch.complex64)
    nufft = FakeNUFFTIdentity()
    arena = DeviceArena(compute="cpu")
    with pytest.raises(ValueError):
        Workspace(y, nufft, arena, buf_specs=make_specs(y.dtype, torch.float32))


def test_workspace_get_bind_concat_and_errors():
    torch.manual_seed(2)
    B, H, W = 3, 2, 2
    y = torch.randn((B, H, W), dtype=torch.complex64)
    ws = Workspace(y, FakeNUFFTIdentity(), DeviceArena(compute="cpu"),
                   buf_specs=make_specs(y.dtype, torch.float32))

    # fill x per shard (single shard on CPU)
    for sh, i in ws.iter_shards():
        x = ws.get("x", i)
        x.copy_(torch.arange(x.numel(), dtype=torch.float32, device=x.device).to(x.dtype).view_as(x))

    x_all = ws.concat("x")
    assert x_all.shape == (B, H, W)

    # error cases
    with pytest.raises(ValueError):
        ws.get("x")  # per_shard requires index
    # create a tiny global to test the other error branch
    specs2 = make_specs(y.dtype, torch.float32)
    specs2.append(BufSpec("scalar_g", "scalar", "global", "scalar", torch.float32, init=None))
    ws2 = Workspace(y, FakeNUFFTIdentity(), DeviceArena(compute="cpu"), buf_specs=specs2)
    with pytest.raises(ValueError):
        ws2.get("scalar_g", 0)  # global must not pass shard_idx


# ────────────────────────────────────────────────────────────────────────
# Objective: sharded caches vs global fallback
# ────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("with_caches", [True, False])
def test_objective_f_g_consistency_sharded_vs_global(with_caches: bool):
    torch.manual_seed(3)
    B, H, W = 4, 2, 2
    y = torch.randn((B, H, W), dtype=torch.complex64)
    nufft = FakeNUFFTIdentity()
    specs = make_specs(y.dtype, torch.float32, with_kspace_caches=with_caches)

    ws = Workspace(y, nufft, DeviceArena(compute="cpu"), buf_specs=specs)
    obj = Objective(nufft, y, NullRegManager())

    # seed x=0, dx=random
    for sh, i in ws.iter_shards():
        ws.get("x", i).zero_()
        ws.get("dx", i).copy_(torch.randn((B, H, W), dtype=y.dtype))  # overspec assign OK; we slice below

    obj.begin_linesearch(ws)
    f0, g0d = obj.f_g(ws, t=0.25)
    assert math.isfinite(f0) and math.isfinite(g0d)

    # g should equal AH(Ax + tAdx - y); with x=0 and A=I, AH=I -> g = t*dx - y
    g_cat = ws.concat("g")
    dx_cat = ws.concat("dx")
    expect = dx_cat * 0.25 - y
    assert torch.allclose(g_cat, expect, atol=1e-6, rtol=1e-6)
    obj.end_linesearch(ws)


# ────────────────────────────────────────────────────────────────────────
# init_scaling
# ────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("alpha", [1.0, 2.0, 0.5])
def test_initial_backproj_and_scaling_scalar_nufft(alpha: float):
    torch.manual_seed(4)
    B, H, W = 5, 2, 2
    y = torch.randn((B, H, W), dtype=torch.complex64)
    nufft = FakeNUFFTScalar(alpha)
    ws = Workspace(y, nufft, DeviceArena(compute="cpu"),
                   buf_specs=make_specs(y.dtype, torch.float32))

    out = initial_backproj_and_scaling(ws, NullRegManager(), xfactor=1.0, verbose=False)
    scale = out["scale"]

    # x <- AH(y) = α y ; A(x) = α (α y) = α^2 y ; E_est = ||α^2 y||^2 = α^4 ||y||^2
    # scale = sqrt( ||y||^2 / (α^4 ||y||^2) ) = 1/α^2
    expect_scale = 1.0 / (alpha * alpha)
    assert math.isclose(scale, expect_scale, rel_tol=1e-4, abs_tol=1e-4)

    x_cat = ws.concat("x")
    # final x = (AH y) * scale = α y * (1/α^2) = y / α
    assert torch.allclose(x_cat, y / alpha, atol=1e-5, rtol=1e-5)

    # continuation hook exists and acts multiplicatively
    t = torch.ones_like(x_cat)
    ws.scale.divide_inplace(t)
    assert torch.allclose(t, torch.ones_like(t) * (1.0 / scale), atol=1e-6)


# ────────────────────────────────────────────────────────────────────────
# CGSolver one step (steepest descent seed)
# ────────────────────────────────────────────────────────────────────────

def test_cgsolver_run_one_initial_direction_is_steepest_descent():
    torch.manual_seed(5)
    B, H, W = 3, 2, 2
    y = torch.randn((B, H, W), dtype=torch.complex64)
    solver = CGSolver(y, FakeNUFFTIdentity(), NullRegManager(), devices=["cpu"])

    # run a single setup iteration (no line search accept)
    solver.run_one()
    ws = solver.ws

    # With A=I, x=0 → r = -y ; g = AH r = -y ; dx = -g/diag = y (diag = 1)
    g = ws.concat("g")
    dx = ws.concat("dx")
    assert torch.allclose(g, -y, atol=1e-6)
    assert torch.allclose(dx,  y, atol=1e-6)


# ────────────────────────────────────────────────────────────────────────
# Directions (PR+, DY): first update from zero step
# ────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("DirCls", [DirPRPlus, DirDY])
def test_directions_first_update_is_precond_steepest_descent(DirCls):
    torch.manual_seed(6)
    B, H, W = 4, 2, 2
    y = torch.randn((B, H, W), dtype=torch.complex64)
    nufft = FakeNUFFTIdentity()

    # Build a manifest that includes g_prev ONLY for these direction tests
    specs = make_specs(y.dtype, torch.float32)
    specs.append(BufSpec("g_prev", "image", "per_shard", "image", y.dtype, init="zeros"))

    ws = Workspace(y, nufft, DeviceArena(compute="cpu"), buf_specs=specs)

    # Build data gradient at x=0, t=0: g = -y
    obj = Objective(nufft, y, NullRegManager())
    obj.begin_linesearch(ws)
    f0, g0d = obj.f_g(ws, t=0.0)
    assert torch.allclose(ws.concat("g"), -y, atol=1e-6)

    # zero initial direction; diag = 1
    for sh, i in ws.iter_shards():
        ws.get("dx", i).zero_()

    D = DirCls(ws)
    D.init_state()
    beta = D.update_inplace(ws)

    # Since g_prev=g at first step ⇒ β=0 ; dx = -g/diag
    assert beta >= 0.0
    assert torch.allclose(ws.concat("dx"), -ws.concat("g").real / 1.0 + 0j, atol=1e-6)
    obj.end_linesearch(ws)


# ────────────────────────────────────────────────────────────────────────
# Line search (Armijo / Strong Wolfe) on simple quadratic
# ────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ls_name", ["armijo", "wolfe"])
def test_line_search_quadratic_minimizes_at_t_one(ls_name: str):
    torch.manual_seed(7)
    B, H, W = 3, 2, 2
    y = torch.randn((B, H, W), dtype=torch.complex64)
    nufft = FakeNUFFTIdentity()
    ws = Workspace(y, nufft, DeviceArena(compute="cpu"),
                   buf_specs=make_specs(y.dtype, torch.float32))
    obj = Objective(nufft, y, NullRegManager())

    # x=0; choose direction dx=y so the minimizer of 0.5||t*dx - y||^2 is t*=1
    for sh, i in ws.iter_shards():
        ws.get("x", i).zero_()
        ws.get("dx", i).copy_(y)

    obj.begin_linesearch(ws)
    f0, g0d = obj.f_g(ws, t=0.0)  # computes g
    # set direction to steepest descent: dx = -g = y
    for sh, i in ws.iter_shards():
        ws.get("dx", i).copy_(-ws.get("g", i))

    class Dummy:
        pass
    solver = Dummy()
    solver.ws = ws
    solver.obj = obj
    solver.ls_name = ls_name
    solver.c1 = 1e-4
    solver.c2 = 0.9
    solver.ls_max_iter = 20
    solver.ls_zoom = True

    ok, t, f_t, gdot_t = ls_mod.search(solver, f0, g0d)
    assert ok
    assert abs(t - 1.0) <= 1e-6
    obj.end_linesearch(ws)


# ────────────────────────────────────────────────────────────────────────
# Continuation (stub stats helper, no preconditioner rebuild)
# ────────────────────────────────────────────────────────────────────────

def test_continuation_updates_regm_with_stubbed_helper():
    torch.manual_seed(8)
    B, H, W = 4, 2, 2
    y = torch.randn((B, H, W), dtype=torch.complex64)

    # Ensure the registry module exists before importing continuation
    modname = "graspcg.regularization.reg_registry"
    if modname not in sys.modules:
        stub = types.ModuleType(modname)
        stub.STATS_HELPERS = {}
        sys.modules[modname] = stub
    else:
        # refresh helpers for a clean start
        sys.modules[modname].STATS_HELPERS = {}

    # Import (or reload) continuation now that the stub is in place
    from graspcg.numerics import continuation as cont
    importlib.reload(cont)

    # Install a simple helper under key "dummy" that returns percentile & mean
    def helper(ws, xs, *, percentile: float, eps_floor: float):
        mags = xs.abs().float().reshape(xs.shape[0], -1)  # (B, P)
        # per-batch percentile, then mean across frames
        q = torch.quantile(mags, q=torch.tensor(percentile, dtype=torch.float32), dim=1)
        eps = float(torch.clamp(q.mean(), min=eps_floor))
        sigma = float(mags.mean())
        return eps, sigma

    cont.STATS_HELPERS["dummy"] = helper

    # Build workspace and a dx pilot (init_scaling also attaches ws.scale)
    nufft = FakeNUFFTIdentity()
    ws = Workspace(y, nufft, DeviceArena(compute="cpu"),
                   buf_specs=make_specs(y.dtype, torch.float32))
    # Seed x with AH(y) to get a nontrivial pilot; also prepares ws.scale
    initial_backproj_and_scaling(ws, NullRegManager(), xfactor=1.0, verbose=False)

    # Set up a minimal reg manager with one entry
    class RM(NullRegManager):
        def __init__(self):
            super().__init__()
            self.regs = {"dummy": {"apply_scale_to_data": True, "alpha": 0.0}}  # no EMA for test

    regm = RM()
    cfg = cont.ContinuationConfig(every=1, update_diag=False)  # avoid precond rebuild path for now
    cman = cont.ContinuationManager(cfg, regm)

    # Perform update at k_iter=1 (multiple of 'every')
    changed = cman.maybe_update(ws, k_iter=1)
    assert changed
    # Values were written
    assert "weight" in regm.regs["dummy"] and "eps" in regm.regs["dummy"]
    assert regm.regs["dummy"]["weight"] > 0.0
    assert regm.regs["dummy"]["eps"] >= cfg.eps_floor


# ────────────────────────────────────────────────────────────────────────
# DeviceArena basics (CPU-only; GPU branch optional)
# ────────────────────────────────────────────────────────────────────────

def test_device_arena_request_release_cpu_anchor_and_device():
    arena = DeviceArena(compute="cpu")
    a = torch.zeros((2, 3), dtype=torch.float32)  # anchor on CPU

    # anchor takes precedence
    t1 = arena.request(100, torch.float32, anchor=a)
    assert t1.device.type == "cpu"
    assert t1.numel() >= 100

    # explicit device
    t2 = arena.request(50, torch.float32, device="cpu")
    assert t2.device.type == "cpu"

    arena.release(t1)
    arena.release(t2)
    # after release, a similar request should reuse (numel may be >= requested)
    t3 = arena.request(80, torch.float32, device="cpu")
    assert t3.device.type == "cpu"
    assert t3.numel() >= 80


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_arena_stream_and_request_on_cuda():
    arena = DeviceArena(compute="cuda")
    devs = arena.cuda_devices()
    assert len(devs) >= 1
    s = arena.stream_for(devs[0])
    assert isinstance(s, torch.cuda.Stream)

    t = arena.request(128, torch.float32, device=devs[0])
    assert t.device.type == "cuda"
    arena.release(t)