import math
import pytest
import torch

from graspcg.solvers.cg import CGSolver, CGConfig
from graspcg.regularization.manager import RegManager
from graspcg.regularization.tv_nd import TVND, TVParams


# -------------------------
# Helpers / Fixtures
# -------------------------

def device_list():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))
        devices.append(torch.device("cuda:1"))
    return devices


class CartesianFFT:
    """
    Simple Cartesian FFT operator with optional `out` argument.
    A = FFT (unitary, 'ortho'); AH = IFFT (inverse).
    Works cross-device and cross-dtype; if `out=None`, returns a new tensor.
    """
    def __init__(self, dims=None, norm="ortho"):
        self.dims = dims
        self.norm = norm
        # Declare axis semantics for Workspace:
        # tensors are shaped (B, C, H, W) -> unlike=1 (B), like=1 (C), nufft=2 (H,W)
        self.roles_image = (1, 1, 2)
        self.roles_kspace = (1, 1, 2)

    def _dims(self, x):
        if self.dims is not None:
            return self.dims
        # default: act on the last two spatial dims
        return tuple(range(-2, 0))

    def A(self, x, out=None):
        import torch
        k = torch.fft.fftn(x, dim=self._dims(x), norm=self.norm)
        if out is None:
            return k
        if k.device != out.device or k.dtype != out.dtype:
            k = k.to(device=out.device, dtype=out.dtype, non_blocking=True)
        out.copy_(k)
        return out

    def AH(self, k, out=None):
        import torch
        x = torch.fft.ifftn(k, dim=self._dims(k), norm=self.norm)
        if out is None:
            return x
        if x.device != out.device or x.dtype != out.dtype:
            x = x.to(device=out.device, dtype=out.dtype, non_blocking=True)
        out.copy_(x)
        return out

    # --- Discovery helpers for Workspace ----------------------------
    def roles(self):
        # Return in a generic form (dict/tuples) that Workspace.infer_plan understands.
        return {"image": self.roles_image, "kspace": self.roles_kspace}

    def image_shape(self, y):
        # Cartesian FFT keeps (B,C,H,W) shape between domains.
        return tuple(y.shape)


def _rand_cplx(shape, device, dtype=torch.complex64, scale=1.0, seed=123):
    if isinstance(device, torch.device) and device.type == "cuda":
        g = torch.Generator(device=device)
    else:
        g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    re = torch.randn(*shape, generator=g, device=device, dtype=torch.float32)
    im = torch.randn(*shape, generator=g, device=device, dtype=torch.float32)
    return (re + 1j * im).to(dtype) * scale


def _build_solver(device, *, direction="fr", ls_name="armijo", with_tv=False):
    # Small problem
    B, C, H, W = 6, 1, 8, 8
    y = torch.zeros((B, C, H, W), device=device, dtype=torch.complex64)

    # Reg manager (optionally TV)
    regs = []
    if with_tv:
        tv = TVND("tv_sp", TVParams(weight=0.3, eps=1e-3, axes="spatial", isotropic=True))
        regs = [tv]
    regm = RegManager(regs, compile_kernels=False)

    cfg = CGConfig(
        devices=device, direction=direction, ls_name=ls_name,
        max_iter=4, ls_max_iter=20, ls_zoom=True, verbose=False,
        record_history=False,
    )
    solver = CGSolver(y=y, nufft_op=CartesianFFT(), regm=regm, cfg=cfg)
    return solver


# -------------------------
# Objective tests
# -------------------------

@pytest.mark.parametrize("device", device_list())
def test_objective_fg_identity_quadratic(device):
    """
    For A=I, y=0, x random and d=-x:
      r(t) = (1-t) x; f(t)=0.5||(1-t)x||^2; g(t)=r(t); g^T d = -(1-t)||x||^2
    """
    solver = _build_solver(device, direction="fr", ls_name="armijo", with_tv=False)
    ws, obj = solver.ws, solver.obj

    # Initialize x and set direction d = -x
    for sh, i in ws.iter_shards():
        x = _rand_cplx(ws.get("x", i).shape, device, scale=0.1, seed=42)
        ws.get("x", i).copy_(x)
        ws.get("dx", i).copy_(-x)

    # f(0) and g(0)^T d
    obj.begin_linesearch(ws)
    f0, g0d = obj.f_g_tensor(ws, torch.zeros((), device=device, dtype=ws.get("x", 0).real.dtype))
    obj.end_linesearch(ws)

    # f0 > 0 and g0d < 0
    assert torch.isfinite(f0) and torch.isfinite(g0d)
    assert f0.item() > 0
    assert g0d.item() < -1e-12

    # f(1) ≈ 0, g(1)^T d ≈ 0
    obj.begin_linesearch(ws)
    f1, g1d = obj.f_g_tensor(ws, torch.ones_like(f0))
    obj.end_linesearch(ws)
    assert f1.abs().item() <= 1e-6 * (1.0 + f0.abs().item())
    assert g1d.abs().item() <= 1e-6 * (1.0 + g0d.abs().item())


@pytest.mark.parametrize("device", device_list())
def test_objective_with_tv_increases_energy_for_nonconstant(device):
    """
    TV should add strictly positive energy for non-constant x (baseline-subtracted TV -> 0 for constants).
    """
    # Build two solvers: with and without TV
    s0 = _build_solver(device, with_tv=False)
    s1 = _build_solver(device, with_tv=True)

    # Non-constant x; dx=0 to evaluate at current x
    for ws in (s0.ws, s1.ws):
        for sh, i in ws.iter_shards():
            x = torch.zeros_like(ws.get("x", i))
            x[..., : x.shape[-1] // 2] = 1.0  # step edge -> non-constant
            ws.get("x", i).copy_(x)
            ws.get("dx", i).zero_()

    # Evaluate
    for s in (s0, s1):
        s.obj.begin_linesearch(s.ws)
    f0, _ = s0.obj.f_g_tensor(s0.ws, torch.zeros((), device=device, dtype=s0.y.real.dtype))
    f1, _ = s1.obj.f_g_tensor(s1.ws, torch.zeros((), device=device, dtype=s1.y.real.dtype))
    for s in (s0, s1):
        s.obj.end_linesearch(s.ws)

    assert f1.item() > f0.item() + 1e-6


# -------------------------
# Line-search tests
# -------------------------

@pytest.mark.parametrize("device", device_list())
def test_line_search_armijo_accepts_full_step(device):
    """
    With A=I, y=0, d=-x, Armijo should accept t=1 immediately.
    """
    solver = _build_solver(device, ls_name="armijo", with_tv=False)

    # Seed x, d = -x
    for sh, i in solver.ws.iter_shards():
        x = _rand_cplx(solver.ws.get("x", i).shape, device, scale=0.1, seed=123)
        solver.ws.get("x", i).copy_(x)
        solver.ws.get("dx", i).copy_(-x)

    from graspcg.numerics.line_search import search as line_search
    ok, t, f_t, gdot_t = line_search(solver)

    assert bool(ok.item())
    assert torch.allclose(t, torch.ones_like(t), atol=1e-6)
    # f(1) ~ 0, g·d(1) ~ 0
    assert f_t.abs().item() <= 1e-6
    assert gdot_t.abs().item() <= 1e-6


@pytest.mark.parametrize("device", device_list())
def test_line_search_wolfe_zoom_path(device):
    """
    Force a zoom by using a very large c1 so that Armijo at t=1 fails.
    For A=I, y=0, d=-x, the Wolfe zoom should land near t=0.5.
    """
    solver = _build_solver(device, ls_name="wolfe", with_tv=False)
    # make Armijo harder: c1 > 0.5 guarantees violation at t=1 for quadratic
    solver.c1 = 0.75
    solver.c2 = 0.9
    solver.ls_zoom = True

    for sh, i in solver.ws.iter_shards():
        x = _rand_cplx(solver.ws.get("x", i).shape, device, scale=0.1, seed=321)
        solver.ws.get("x", i).copy_(x)
        solver.ws.get("dx", i).copy_(-x)

    from graspcg.numerics.line_search import search as line_search
    ok, t, f_t, gdot_t = line_search(solver)
    assert bool(ok.item())
    # For quadratic f(t)=0.5(1-t)^2||x||^2, zoom mid should be near 0.5 (Armijo equality)
    assert torch.allclose(t, 0.5 * torch.ones_like(t), atol=0.05)
    # Strong Wolfe curvature at that t: |g·d| <= -c2*g0d
    # Re-evaluate g0d at t=0
    solver.obj.begin_linesearch(solver.ws)
    f0, g0d = solver.obj.f_g_tensor(solver.ws, torch.zeros_like(t))
    solver.obj.end_linesearch(solver.ws)
    assert g0d.item() < 0
    assert gdot_t.abs().item() <= (-solver.c2 * g0d).item() + 1e-6


@pytest.mark.parametrize("device", device_list())
def test_line_search_rejects_non_descent(device):
    """If g·d ≥ 0, line search should reject the step and return t=0."""
    solver = _build_solver(device, ls_name="armijo", with_tv=False)

    for sh, i in solver.ws.iter_shards():
        x = _rand_cplx(solver.ws.get("x", i).shape, device, scale=0.1, seed=4321)
        solver.ws.get("x", i).copy_(x)
        solver.ws.get("dx", i).copy_(x)

    from graspcg.numerics.line_search import search as line_search
    ok, t, f_t, gdot_t = line_search(solver)

    assert not bool(ok.item())
    assert torch.allclose(t, torch.zeros_like(t), atol=1e-6)

    solver.obj.begin_linesearch(solver.ws)
    f0, g0d = solver.obj.f_g_tensor(solver.ws, torch.zeros_like(t))
    solver.obj.end_linesearch(solver.ws)
    assert torch.allclose(f_t, f0)
    assert torch.allclose(gdot_t, g0d)


# -------------------------
# Directions tests
# -------------------------

@pytest.mark.parametrize("device", device_list())
@pytest.mark.parametrize("dir_name", ["fr", "prplus", "dy"])
def test_directions_update_produces_descent(device, dir_name):
    """
    After recomputing the gradient at a new point, the updated direction should be a descent direction:
      g^T d_new < 0  (allowing small numerical slack)
    """
    solver = _build_solver(device, direction=dir_name, with_tv=False)
    ws, obj = solver.ws, solver.obj
    dtype_r = solver.y.real.dtype

    # Initialize x and d = -g/D at the first point
    for sh, i in ws.iter_shards():
        x = _rand_cplx(ws.get("x", i).shape, device, scale=0.2, seed=777)
        ws.get("x", i).copy_(x)
        ws.get("dx", i).zero_()

    obj.begin_linesearch(ws)
    _f0, _g0d = obj.f_g_tensor(ws, torch.zeros((), device=device, dtype=dtype_r))  # fills g
    for sh, i in ws.iter_shards():
        g, d, D = ws.bind(i, "g", "dx", "diag")
        d.copy_(g).div_(D).neg_()
    # Seed direction state
    try:
        solver.dir.init_state()
    except Exception:
        pass
    obj.end_linesearch(ws)

    # Take a small synthetic step to change gradient (not using line search here)
    step = torch.tensor(0.1, device=device, dtype=dtype_r)
    for sh, i in ws.iter_shards():
        x, d = ws.bind(i, "x", "dx")
        x.add_(d * step.to(d.device, dtype=d.real.dtype))

    # Recompute g at new x
    obj.begin_linesearch(ws)
    _f1, _g1d = obj.f_g_tensor(ws, torch.zeros((), device=device, dtype=dtype_r))
    obj.end_linesearch(ws)

    # Update direction in-place
    beta = solver.dir.update_inplace(ws)
    assert torch.isfinite(beta).all()
    if dir_name == "prplus":
        # PR+ clamps to >= 0
        assert beta.item() >= -1e-12

    # Descent test: g^T d_new < 0
    # (Use a 0-D real dot across shards)
    gdot = None
    for sh, i in ws.iter_shards():
        g, d = ws.bind(i, "g", "dx")
        v = (g.conj() * d).real.sum()
        gdot = v if gdot is None else (gdot + v.to(gdot.device))
    assert gdot.item() < 1e-8  # allow small slack


# -------------------------
# End-to-end CG smoke (all directions, both line searches)
# -------------------------

@pytest.mark.parametrize("device", device_list())
@pytest.mark.parametrize("dir_name", ["fr", "prplus", "dy"])
@pytest.mark.parametrize("ls_name", ["armijo", "wolfe"])
def test_cg_solver_smoke_all_variants(device, dir_name, ls_name):
    """
    Run a few iterations across all directions and both line-searches to ensure integration stability.
    Checks the objective value decreases (roughly) and no NaNs are produced.
    """
    solver = _build_solver(device, direction=dir_name, ls_name=ls_name, with_tv=True)

    # Seed x and dx
    for sh, i in solver.ws.iter_shards():
        x = _rand_cplx(solver.ws.get("x", i).shape, device, scale=0.1, seed=888 + i)
        solver.ws.get("x", i).copy_(x)
        solver.ws.get("dx", i).zero_()

    # Prime gradient and steepest descent as in run()
    solver.obj.begin_linesearch(solver.ws)
    _f0, _g0d = solver.obj.f_g_tensor(solver.ws, torch.zeros((), device=device, dtype=solver.y.real.dtype))
    for sh, i in solver.ws.iter_shards():
        g, d, D = solver.ws.bind(i, "g", "dx", "diag")
        d.copy_(g).div_(D).neg_()
    try:
        solver.dir.init_state()
    except Exception:
        pass
    solver.obj.end_linesearch(solver.ws)

    # Run a couple of iterations
    solver.run(max_iter=3)

    hist = solver.history()
    # Expect at least one step and finite stats
    assert len(hist) >= 1
    for rec in hist:
        for k, v in rec.items():
            assert math.isfinite(float(v))
    # Final step size or objective should be non-increasing-ish
    if len(hist) >= 2:
        assert hist[-1]["f"] <= hist[0]["f"] + 1e-4