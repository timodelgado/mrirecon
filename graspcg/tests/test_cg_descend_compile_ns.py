# File: graspcg/tests/test_cg_descend_compile_ns.py
import math
import pytest
import torch

from graspcg.core.roles import Roles
from graspcg.regularization.tv_nd import TVND, TVParams
from graspcg.regularization.manager import RegManager
from graspcg.solvers.cg import CGSolver, CGConfig


# ----------------------------
# Helpers
# ----------------------------
def _has_cuda(n: int = 1) -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() >= n

def _rand_cplx(shape, device, dtype=torch.complex64, scale=1.0, seed=123):
    """Deterministic complex tensor using a CPU generator (avoids CUDA generator quirks)."""
    cpu_g = torch.Generator(device="cpu").manual_seed(seed)
    rdt = torch.float32 if dtype == torch.complex64 else torch.float64
    re = torch.randn(*shape, generator=cpu_g, device="cpu", dtype=rdt).to(device)
    im = torch.randn(*shape, generator=cpu_g, device="cpu", dtype=rdt).to(device)
    z = (re + 1j * im).to(dtype)
    if scale != 1.0:
        z = z * float(scale)
    return z


# ----------------------------
# Cartesian FFT operator (unitary) with explicit roles
# ----------------------------
class CartesianFFT:
    """
    Minimal NUFFT-like operator for tests.
    Unitary A = FFT, AH = IFFT (ortho). Keeps the same shape B,C,H,W.
    Provides explicit roles so Workspace doesn't infer/normalize axes.
    """
    roles_image  = Roles(unlike=1, like=1, nufft=2)
    roles_kspace = Roles(unlike=1, like=1, nufft=2)

    def image_shape(self, y: torch.Tensor):
        # Same shape in image/k-space under unitary FFT
        return tuple(int(s) for s in y.shape)

    def A(self, x: torch.Tensor, out: torch.Tensor | None = None):
        y = torch.fft.fftn(x, dim=(-2, -1), norm="ortho")
        if out is None:
            return y
        out.copy_(y)
        return out

    def AH(self, y: torch.Tensor, out: torch.Tensor | None = None):
        x = torch.fft.ifftn(y, dim=(-2, -1), norm="ortho")
        if out is None:
            return x
        out.copy_(x)
        return out


# ----------------------------
# Fixtures
# ----------------------------
def device_list():
    devs = [torch.device("cpu")]
    if _has_cuda(1):
        devs.append(torch.device("cuda:0"))
        devs.append(torch.device("cuda:1"))
    return devs


# ==========================================================
# 1) CG descent / line-search acceptance
# ==========================================================
@pytest.mark.parametrize("device", device_list())
@pytest.mark.parametrize("direction", ["fr", "prplus", "dy"])
def test_cg_armijo_is_monotone_and_accepted(device, direction):
    """
    Under Armijo backtracking, accepted steps should satisfy f_{k+1} <= f_k + c1 * alpha * gdot_k
    and in well-conditioned quadratic-ish cases should be monotone non-increasing.
    """
    B, C, H, W = 4, 1, 16, 16
    op = CartesianFFT()
    # Ground truth x*, build y = A x*
    x_star = _rand_cplx((B, C, H, W), device=device, scale=0.1, seed=123)
    y = op.A(x_star)

    # Mild TV to keep it smooth-ish (Huber eps)
    regm = RegManager([TVND("tv", TVParams(weight=1e-3, eps=1e-3, axes="spatial", isotropic=True))])
    cfg = CGConfig(
        devices=("cuda" if device.type == "cuda" else "cpu"),
        direction=direction,
        ls_name="armijo",
        max_iter=6,
        record_history=True,
        verbose=False,
    )
    solver = CGSolver(y=y, nufft_op=op, regm=regm, cfg=cfg)

    # Initialize x and dx
    for sh, i in solver.ws.iter_shards():
        solver.ws.get("x", i).copy_(_rand_cplx(solver.ws.get("x", i).shape, device, scale=0.05, seed=888+i))
        solver.ws.get("dx", i).zero_()

    # Run a few iterations
    solver.run(max_iter=4)
    hist = solver.history()
    assert len(hist) >= 1, "Expected at least one accepted step with Armijo LS"

    # Check Armijo inequality at each recorded step
    # (history stores 'f', 'gdot', 'step' as scalars)
    c1 = getattr(solver, "c1", 1e-4)
    for k in range(len(hist) - 1):
        f_k    = hist[k]["f"]
        gdot_k = hist[k]["gdot"]
        a_k    = hist[k]["step"]
        f_kp1  = hist[k+1]["f"]
        assert f_kp1 <= f_k + c1 * a_k * gdot_k + 1e-8, "Armijo condition violated"

    # Monotonic non-increasing (soft tolerance)
    fvals = [h["f"] for h in hist]
    for k in range(len(fvals) - 1):
        assert fvals[k+1] <= fvals[k] + 1e-10, "Expected monotone decrease under Armijo"


@pytest.mark.parametrize("device", device_list())
@pytest.mark.parametrize("direction", ["fr", "prplus", "dy"])
def test_cg_wolfe_satisfies_strong_conditions(device, direction):
    """
    For strong Wolfe, test the two Wolfe conditions at accepted steps using history:
      • Armijo: f_{k+1} <= f_k + c1 * α_k * g'(0)
      • Curvature (strong): |g'(α_k)| <= c2 * |g'(0)|
    Note: Strong Wolfe does not guarantee monotonicity, so we don't test that here.
    """
    B, C, H, W = 4, 1, 16, 16
    op = CartesianFFT()
    x_star = _rand_cplx((B, C, H, W), device=device, scale=0.1, seed=321)
    y = op.A(x_star)

    regm = RegManager([TVND("tv", TVParams(weight=5e-4, eps=1e-3, axes="spatial", isotropic=True))])
    cfg = CGConfig(
        devices=("cuda" if device.type == "cuda" else "cpu"),
        direction=direction,
        ls_name="wolfe",
        max_iter=6,
        record_history=True,
        verbose=False,
    )
    solver = CGSolver(y=y, nufft_op=op, regm=regm, cfg=cfg)

    for sh, i in solver.ws.iter_shards():
        solver.ws.get("x", i).copy_(_rand_cplx(solver.ws.get("x", i).shape, device, scale=0.05, seed=999+i))
        solver.ws.get("dx", i).zero_()

    solver.run(max_iter=4)
    hist = solver.history()
    assert len(hist) >= 1, "Expected at least one accepted step with Wolfe LS"

    c1 = getattr(solver, "c1", 1e-4)
    c2 = getattr(solver, "c2", 0.9)
    # history 'gdot' is derivative at t=0 for that iter; to check |g'(α_k)| we need the next record
    for k in range(len(hist) - 1):
        f_k, f_kp1 = hist[k]["f"], hist[k+1]["f"]
        g0, ak = hist[k]["gdot"], hist[k]["step"]
        # Prefer the exact slope at the accepted step if available.
        g_a = hist[k].get("gdot_at_step", None)
        if g_a is None:
            # Fallback for older history semantics (less exact).
            g_a = hist[k+1].get("gdot", None)
        # small extra slack for cross-device reductions
        assert f_kp1 <= f_k + c1 * ak * g0 + 5e-8, "Strong Wolfe: Armijo part violated"
        # Curvature (use absolute strong condition if we have g'(α))
        if g_a is not None:
            assert abs(g_a) <= c2 * abs(g0) + 1e-8, "Strong Wolfe: curvature part violated"


# ==========================================================
# 2) Directional derivative matches finite-difference slope
# ==========================================================
@pytest.mark.parametrize("device", device_list())
def test_directional_derivative_matches_finite_difference(device):
    B, C, H, W = 3, 1, 12, 12
    op = CartesianFFT()
    x_star = _rand_cplx((B, C, H, W), device=device, scale=0.1, seed=2024)
    y = op.A(x_star)

    regm = RegManager([TVND("tv", TVParams(weight=1e-3, eps=1e-3, axes="spatial", isotropic=True))])

    cfg = CGConfig(
        devices=("cuda" if device.type == "cuda" else "cpu"),
        direction="fr",
        ls_name="armijo",
        max_iter=1,
        record_history=False,
        verbose=False,
    )
    solver = CGSolver(y=y, nufft_op=op, regm=regm, cfg=cfg)

    # Seed x and a *fixed* direction d
    for sh, i in solver.ws.iter_shards():
        solver.ws.get("x", i).copy_(_rand_cplx(solver.ws.get("x", i).shape, device, scale=0.05, seed=555+i))
        d = _rand_cplx(solver.ws.get("dx", i).shape, device, scale=0.05, seed=777+i)
        solver.ws.get("dx", i).copy_(d)

    # Evaluate phi(0) and phi'(0) with the objective (keeps x fixed, varies t along dx)
    solver.obj.begin_linesearch(solver.ws)
    try:
        t0 = torch.zeros((), device=device, dtype=solver.y.real.dtype)
        f0, g0d = solver.obj.f_g_tensor(solver.ws, t0)
        # symmetric finite-difference around t0
        h = torch.tensor(1e-4, device=device, dtype=solver.y.real.dtype)
        fp, _ = solver.obj.f_g_tensor(solver.ws, t0 + h)
        fm, _ = solver.obj.f_g_tensor(solver.ws, t0 - h)
    finally:
        solver.obj.end_linesearch(solver.ws)

    def fd_slope(obj, ws, t0, h):
        fp, _ = obj.f_g_tensor(ws, t0 + h)
        fm, _ = obj.f_g_tensor(ws, t0 - h)
        return float(((fp - fm) / (2*h)).item())

    h  = torch.tensor(1e-4, device=device, dtype=solver.y.real.dtype)
    g1 = fd_slope(solver.obj, solver.ws, t0, h)
    g2 = fd_slope(solver.obj, solver.ws, t0, 2*h)
    g_fd = (4*g1 - g2) / 3.0  # Richardson
    assert math.isfinite(g_fd)
    # Require close directional derivative
    assert abs(g_fd - g0d) <= max(1e-6, 1e-2 * max(1.0, abs(g0d))), \
        f"Directional derivative mismatch: g'(0)={g0d:.6e} vs FD={g_fd:.6e}"


# ==========================================================
# 3) Objective: compiled vs eager equivalence
# ==========================================================
@pytest.mark.parametrize("device", device_list())
def test_objective_compiled_matches_eager(device):
    """
    Build two solvers: one with eager RegManager, one with compile_kernels=True.
    With identical (x, dx), f(t) and g'(t) must match bitwise or to tight tolerance.
    """
    B, C, H, W = 3, 1, 10, 10
    op = CartesianFFT()

    # Common y from a fixed x*
    x_star = _rand_cplx((B, C, H, W), device=device, scale=0.1, seed=42)
    y = op.A(x_star)

    tv = TVND("tv", TVParams(weight=1e-3, eps=1e-3, axes="spatial", isotropic=True))
    regm_eager   = RegManager([tv], compile_kernels=False)
    regm_compiled= RegManager([tv], compile_kernels=True)

    cfg = CGConfig(
        devices=("cuda" if device.type == "cuda" else "cpu"),
        direction="fr",
        ls_name="armijo",
        max_iter=1,
        record_history=False,
        verbose=False,
    )

    s_eager    = CGSolver(y=y, nufft_op=op, regm=regm_eager,    cfg=cfg)
    s_compiled = CGSolver(y=y, nufft_op=op, regm=regm_compiled, cfg=cfg)

    # Make (x, dx) identical across both workspaces
    for (sh0, i0), (sh1, i1) in zip(s_eager.ws.iter_shards(), s_compiled.ws.iter_shards()):
        x0  = _rand_cplx(s_eager.ws.get("x", i0).shape, device, scale=0.05, seed=1000+i0)
        dx0 = _rand_cplx(s_eager.ws.get("dx", i0).shape, device, scale=0.05, seed=2000+i0)
        s_eager.ws.get("x", i0).copy_(x0);     s_compiled.ws.get("x", i1).copy_(x0)
        s_eager.ws.get("dx", i0).copy_(dx0);   s_compiled.ws.get("dx", i1).copy_(dx0)

    # Evaluate at the same t
    t = torch.zeros((), device=device, dtype=y.real.dtype)
    s_eager.obj.begin_linesearch(s_eager.ws)
    f_e, g_e = s_eager.obj.f_g_tensor(s_eager.ws, t)
    s_eager.obj.end_linesearch(s_eager.ws)

    s_compiled.obj.begin_linesearch(s_compiled.ws)
    f_c, g_c = s_compiled.obj.f_g_tensor(s_compiled.ws, t)
    s_compiled.obj.end_linesearch(s_compiled.ws)

    # Tight tolerances (identical math paths)
    assert abs(f_e - f_c) <= 1e-7 * max(1.0, abs(f_e)), f"Energy mismatch eager={f_e} compiled={f_c}"
    assert abs(g_e - g_c) <= 1e-7 * max(1.0, abs(g_e)), f"g·d mismatch eager={g_e} compiled={g_c}"