# graspcg/numerics/line_search.py
from __future__ import annotations
import torch
from typing import Tuple

@torch.no_grad()
def search(solver, f0: float, g0d: float) -> Tuple[bool, float, float, float]:
    """
    Driver that dispatches to Armijo or strong Wolfe based on solver.ls_name.
    Returns (ok, t, f(t), g(t)^T d).
    """
    if solver.ls_name == "armijo":
        return _armijo(solver, f0, g0d)
    return _wolfe(solver, f0, g0d)

# ──────────────────────────────────────────────────────────────────────
# Armijo backtracking
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _armijo(solver, f0: float, g0d: float):
    ws, obj = solver.ws, solver.obj
    c1, itmax = solver.c1, solver.ls_max_iter

    t = 1.0
    f_new, gdot = f0, g0d
    for _ in range(itmax):
        f_new, gdot = obj.f_g(ws, t)
        if f_new <= f0 + c1 * t * g0d:
            return True, float(t), float(f_new), float(gdot)
        t *= 0.5
    return False, float(t), float(f_new), float(gdot)

# ──────────────────────────────────────────────────────────────────────
# Strong Wolfe with zoom
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _wolfe(solver, f0: float, g0d: float):
    ws, obj = solver.ws, solver.obj
    c1, c2, itmax, do_zoom = solver.c1, solver.c2, solver.ls_max_iter, solver.ls_zoom

    t_prev = 0.0
    f_prev = f0
    g_prev = g0d
    t = 1.0

    for i in range(itmax):
        f_t, gdot_t = obj.f_g(ws, t)

        # Armijo violation or non‑improving -> zoom
        if (f_t > f0 + c1 * t * g0d) or (i > 0 and f_t >= f_prev):
            if do_zoom:
                return _zoom(solver, t_low=t_prev, f_low=f_prev, t_high=t,
                             f0=f0, g0d=g0d)
            # fallback: backtrack like Armijo
            return _armijo(solver, f0, g0d)

        # Curvature condition satisfied?
        if abs(gdot_t) <= -c2 * g0d:
            return True, float(t), float(f_t), float(gdot_t)

        # If slope positive -> zoom between t and t_prev
        if gdot_t >= 0.0:
            if do_zoom:
                return _zoom(solver, t_low=t, f_low=f_t, t_high=t_prev,
                             f0=f0, g0d=g0d)
            return True, float(t), float(f_t), float(gdot_t)

        # otherwise increase step and continue
        t_prev, f_prev, g_prev = t, f_t, gdot_t
        t = min(2.0 * t, 1e6)

    return False, float(t_prev), float(f_prev), float(g_prev)

@torch.no_grad()
def _zoom(solver, *, t_low: float, f_low: float,
          t_high: float, f0: float, g0d: float):
    """
    Simple bisection zoom that enforces Armijo and tries to satisfy curvature.
    """
    ws, obj = solver.ws, solver.obj
    c1, c2, itmax = solver.c1, solver.c2, solver.ls_max_iter

    for _ in range(itmax):
        t_mid = 0.5 * (t_low + t_high)
        f_mid, gdot_mid = obj.f_g(ws, t_mid)

        if (f_mid > f0 + c1 * t_mid * g0d) or (f_mid >= f_low):
            t_high = t_mid
        else:
            # Armijo ok; check curvature
            if abs(gdot_mid) <= -c2 * g0d:
                return True, float(t_mid), float(f_mid), float(gdot_mid)
            if gdot_mid * (t_high - t_low) >= 0:
                t_high = t_low
            t_low, f_low = t_mid, f_mid

    # give best Armijo point we had
    return True, float(t_low), float(f_low), float(g0d)
