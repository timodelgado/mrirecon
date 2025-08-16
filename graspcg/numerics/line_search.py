# graspcg/numerics/line_search.py
from __future__ import annotations
from typing import Tuple
import torch

# This file provides a line-search that:
#   • accepts only `solver`
#   • internally runs begin_linesearch / end_linesearch
#   • evaluates the objective via obj.f_g_tensor(ws, t) to keep heavy kernels compilable
#   • returns (ok: bool, t: 0-D real tensor, f_t: 0-D real tensor, gdot_t: 0-D real tensor)

# ---------------------------------------------------------------------------

def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype is torch.complex64:  return torch.float32
    if dtype is torch.complex128: return torch.float64
    return dtype

@torch.no_grad()
def search(solver) -> Tuple[bool, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Line-search driver (Armijo or Strong-Wolfe).

    Returns:
        ok      : Python bool (accepted step?)
        t       : 0-D REAL torch tensor (on obj.y.device)
        f_t     : 0-D REAL objective at accepted step
        gdot_t  : 0-D REAL directional derivative at accepted step
    """
    ws, obj = solver.ws, solver.obj

    # Defaults with graceful fallbacks
    algo        = getattr(solver, "ls_name", "armijo").lower()
    itmax       = int(getattr(solver, "ls_max_iter", 20))
    c1          = float(getattr(solver, "c1", 1e-4))
    c2          = float(getattr(solver, "c2", 0.9))
    do_zoom     = bool(getattr(solver, "ls_zoom", True))
    t_init      = float(getattr(solver, "t_init", getattr(solver, "ls_t0", 1.0)))
    shrink      = float(getattr(solver, "armijo_shrink", 0.5))
    t_growth    = float(getattr(solver, "wolfe_growth", 2.0))
    t_max_cap   = float(getattr(solver, "t_max_cap", 1e6))

    # Prepare caches for (A x) and (A d) etc. in a compile-friendly way
    obj.begin_linesearch(ws)
    try:
        dtype_r = _real_dtype(obj.y.dtype)
        dev_y   = obj.y.device

        # Seed: f(0), g(0)^T d and a tensorized step t0
        t0 = torch.zeros((), device=dev_y, dtype=dtype_r)
        f0, g0d = obj.f_g_tensor(ws, t0)  # fills ws.g at current x
        # If not a descent direction, reject quickly
        if float(g0d.item()) >= 0.0:
            return False, t0, f0, g0d

        if algo.startswith("armijo"):
            ok, t, f_t, g_t = _armijo_impl(
                solver, f0=f0, g0d=g0d,
                t_init=t_init, shrink=shrink, itmax=itmax, dtype_r=dtype_r, dev=dev_y
            )
        else:
            ok, t, f_t, g_t = _wolfe_impl(
                solver, f0=f0, g0d=g0d,
                t_init=t_init, c1=c1, c2=c2, itmax=itmax, do_zoom=do_zoom,
                growth=t_growth, t_cap=t_max_cap, dtype_r=dtype_r, dev=dev_y
            )

        return ok, t, f_t, g_t
    finally:
        # Always release caches (global path) even on early returns
        obj.end_linesearch(ws)

# ---------------------------------------------------------------------------
# Armijo backtracking
# ---------------------------------------------------------------------------

@torch.no_grad()
def _armijo_impl(solver,
                 *,
                 f0: torch.Tensor,
                 g0d: torch.Tensor,
                 t_init: float,
                 shrink: float,
                 itmax: int,
                 dtype_r: torch.dtype,
                 dev: torch.device):
    ws, obj = solver.ws, solver.obj

    t_val = max(1e-12, float(t_init))
    f_best, g_best = f0, g0d
    t_best = torch.tensor(t_val, device=dev, dtype=dtype_r)

    for _ in range(int(itmax)):
        t = torch.tensor(t_val, device=dev, dtype=dtype_r)
        f_t, g_t = obj.f_g_tensor(ws, t)

        # Armijo: f(t) <= f(0) + c1 * t * g(0)^T d
        rhs = f0 + float(getattr(solver, "c1", 1e-4)) * t * g0d
        if bool((f_t <= rhs).item()):
            return True, t, f_t, g_t

        # keep last seen values in case we need to report them
        f_best, g_best, t_best = f_t, g_t, t
        t_val *= float(shrink)

    return False, t_best, f_best, g_best

# ---------------------------------------------------------------------------
# Strong Wolfe (with optional zoom)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _wolfe_impl(solver,
                *,
                f0: torch.Tensor,
                g0d: torch.Tensor,
                t_init: float,
                c1: float,
                c2: float,
                itmax: int,
                do_zoom: bool,
                growth: float,
                t_cap: float,
                dtype_r: torch.dtype,
                dev: torch.device):
    ws, obj = solver.ws, solver.obj

    t_prev = 0.0
    f_prev = f0
    g_prev = g0d
    t_val  = max(1e-12, float(t_init))

    for i in range(int(itmax)):
        t = torch.tensor(t_val, device=dev, dtype=dtype_r)
        f_t, g_t = obj.f_g_tensor(ws, t)

        # Armijo violation or non-improvement triggers zoom/bracket handling
        armijo_rhs = f0 + c1 * t * g0d
        if bool((f_t > armijo_rhs).item()) or (i > 0 and bool((f_t >= f_prev).item())):
            if do_zoom:
                ok, tz, fz, gz = _zoom_impl(
                    solver, f0=f0, g0d=g0d,
                    t_lo=t_prev, f_lo=f_prev,
                    t_hi=t_val,  f_hi=f_t,
                    c1=c1, c2=c2, itmax=itmax,
                    dtype_r=dtype_r, dev=dev
                )
                return ok, tz, fz, gz
            # Fallback: Armijo backtracking
            return _armijo_impl(solver, f0=f0, g0d=g0d, t_init=min(t_val, 1.0),
                                shrink=float(getattr(solver, "armijo_shrink", 0.5)),
                                itmax=itmax, dtype_r=dtype_r, dev=dev)

        # Curvature condition |g(t)^T d| <= -c2 * g(0)^T d
        if bool((g_t.abs() <= (-c2) * g0d).item()):
            return True, t, f_t, g_t

        # If slope positive, zoom between (t, t_prev)
        if bool((g_t >= 0).item()):
            if do_zoom:
                ok, tz, fz, gz = _zoom_impl(
                    solver, f0=f0, g0d=g0d,
                    t_lo=t_val, f_lo=f_t,
                    t_hi=t_prev, f_hi=f_prev,
                    c1=c1, c2=c2, itmax=itmax,
                    dtype_r=dtype_r, dev=dev
                )
                return ok, tz, fz, gz
            # If no zoom, accept (often good enough)
            return True, t, f_t, g_t

        # Otherwise, increase step and continue
        t_prev, f_prev, g_prev = t_val, f_t, g_t
        t_val = min(t_val * float(growth), float(t_cap))

    # Exhausted iterations: return the last improving bracket point
    t_last = torch.tensor(t_prev, device=dev, dtype=dtype_r)
    return False, t_last, f_prev, g_prev

@torch.no_grad()
def _zoom_impl(solver,
               *,
               f0: torch.Tensor,
               g0d: torch.Tensor,
               t_lo: float,
               f_lo: torch.Tensor,
               t_hi: float,
               f_hi: torch.Tensor,
               c1: float,
               c2: float,
               itmax: int,
               dtype_r: torch.dtype,
               dev: torch.device):
    """
    Simple bisection zoom:
      - maintains Armijo bracket
      - tries to satisfy strong curvature
    """
    ws, obj = solver.ws, solver.obj

    tL, fL = float(t_lo), f_lo
    tH, fH = float(t_hi), f_hi
    t_best, f_best, g_best = torch.tensor(tL, device=dev, dtype=dtype_r), fL, g0d

    for _ in range(int(itmax)):
        tm = 0.5 * (tL + tH)
        t = torch.tensor(tm, device=dev, dtype=dtype_r)
        f_m, g_m = obj.f_g_tensor(ws, t)

        armijo_rhs = f0 + c1 * t * g0d
        if bool((f_m > armijo_rhs).item()) or bool((f_m >= fL).item()):
            tH, fH = tm, f_m
        else:
            # Armijo ok; check curvature
            if bool((g_m.abs() <= (-c2) * g0d).item()):
                return True, t, f_m, g_m
            if tm == tH or tm == tL:
                # Degenerate interval; accept best Armijo point
                return True, t, f_m, g_m
            if bool((g_m * (tH - tL) >= 0).item()):
                tH, fH = tL, fL
            tL, fL = tm, f_m
            t_best, f_best, g_best = t, f_m, g_m

    # Return best Armijo point we had in the zoom
    return True, t_best, f_best, g_best
