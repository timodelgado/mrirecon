# graspcg/numerics/line_search.py
from __future__ import annotations
from typing import Tuple
import torch

@torch.no_grad()
def search(solver) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Line-search driver.
    Returns (ok, t, f_t, gdot_t) as 0‑D REAL tensors on the solver's primary device.
    Dispatches on solver.ls_name: "armijo" (default) or "wolfe".
    """
    if getattr(solver, "ls_name", "armijo") == "wolfe":
        return _wolfe(solver)
    return _armijo(solver)

# ──────────────────────────────────────────────────────────────────────
# Armijo backtracking (tensor-native)
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _armijo(solver):
    ws, obj = solver.ws, solver.obj
    c1 = torch.tensor(getattr(solver, "c1", 1e-4), device=ws.primary_device(), dtype=ws.dtype_r)

    obj.begin_linesearch(ws)
    try:
        # t, f0, g0d are 0‑D REAL tensors
        t = torch.ones((), device=ws.primary_device(), dtype=ws.dtype_r)
        f0, g0d = obj.f_g_tensor(ws, torch.zeros_like(t))   # fills g at x

        # if descent is not valid, bail out with t=0 (no step)
        if (g0d >= 0).logical_or(torch.isnan(g0d)).logical_or(torch.isinf(g0d)):
            return torch.zeros((), device=t.device, dtype=t.dtype), t*0, f0, g0d

        itmax = int(getattr(solver, "ls_max_iter", 20))
        f_t, gdot_t = f0, g0d
        for _ in range(itmax):
            f_t, gdot_t = obj.f_g_tensor(ws, t)
            # Armijo: f(t) ≤ f(0) + c1 * t * g(0)^T d
            if f_t <= f0 + c1 * t * g0d:
                return torch.ones_like(t), t, f_t, gdot_t
            t.mul_(0.5)

        return torch.zeros_like(t), t, f_t, gdot_t
    finally:
        obj.end_linesearch(ws)

# ──────────────────────────────────────────────────────────────────────
# Strong Wolfe (simple zoom). Still tensor-native for f/gdot.
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _wolfe(solver):
    ws, obj = solver.ws, solver.obj
    dev = ws.primary_device()
    dtype = ws.dtype_r

    c1 = torch.tensor(getattr(solver, "c1", 1e-4), device=dev, dtype=dtype)
    c2 = torch.tensor(getattr(solver, "c2", 0.9),  device=dev, dtype=dtype)
    itmax = int(getattr(solver, "ls_max_iter", 20))
    do_zoom = bool(getattr(solver, "ls_zoom", True))

    obj.begin_linesearch(ws)
    try:
        t = torch.ones((), device=dev, dtype=dtype)
        f0, g0d = obj.f_g_tensor(ws, torch.zeros_like(t))  # fills g at x

        if (g0d >= 0).logical_or(torch.isnan(g0d)).logical_or(torch.isinf(g0d)):
            return torch.zeros((), device=dev, dtype=dtype), t*0, f0, g0d

        t_prev = torch.zeros_like(t)
        f_prev = f0
        g_prev = g0d

        for i in range(itmax):
            f_t, gdot_t = obj.f_g_tensor(ws, t)

            # Armijo violation or non‑improving -> zoom or fallback
            if (f_t > f0 + c1 * t * g0d) or (i > 0 and f_t >= f_prev):
                if do_zoom:
                    return _zoom(solver, t_low=t_prev, f_low=f_prev, t_high=t, f0=f0, g0d=g0d)
                # fallback to Armijo
                return _armijo(solver)

            # Curvature condition |g(t)^T d| ≤ −c2 g(0)^T d
            if gdot_t.abs() <= (-c2 * g0d):
                return torch.ones_like(t), t, f_t, gdot_t

            # Positive slope -> zoom between t and t_prev
            if gdot_t >= 0:
                if do_zoom:
                    return _zoom(solver, t_low=t, f_low=f_t, t_high=t_prev, f0=f0, g0d=g0d)
                return torch.ones_like(t), t, f_t, gdot_t

            # Otherwise increase step
            t_prev, f_prev, g_prev = t, f_t, gdot_t
            t = torch.minimum(2.0 * t, torch.tensor(1e6, device=dev, dtype=dtype))

        return torch.zeros_like(t), t_prev, f_prev, g_prev
    finally:
        obj.end_linesearch(ws)

@torch.no_grad()
def _zoom(solver, *, t_low: torch.Tensor, f_low: torch.Tensor,
          t_high: torch.Tensor, f0: torch.Tensor, g0d: torch.Tensor):
    ws, obj = solver.ws, solver.obj
    dev, dtype = ws.primary_device(), ws.dtype_r
    c1 = torch.tensor(getattr(solver, "c1", 1e-4), device=dev, dtype=dtype)
    c2 = torch.tensor(getattr(solver, "c2", 0.9),  device=dev, dtype=dtype)
    itmax = int(getattr(solver, "ls_max_iter", 20))

    for _ in range(itmax):
        t_mid = 0.5 * (t_low + t_high)
        f_mid, gdot_mid = obj.f_g_tensor(ws, t_mid)
        if (f_mid > f0 + c1 * t_mid * g0d) or (f_mid >= f_low):
            t_high = t_mid
        else:
            if gdot_mid.abs() <= (-c2 * g0d):
                return torch.ones_like(t_mid), t_mid, f_mid, gdot_mid
            if gdot_mid * (t_high - t_low) >= 0:
                t_high = t_low
            t_low, f_low = t_mid, f_mid

    # Return the best Armijo point we had (t_low), with a consistent gdot
    _, gdot_low = obj.f_g_tensor(ws, t_low)
    return torch.ones_like(t_low), t_low, f_low, gdot_low
