#%%
# graspcg/numerics/line_search.py
from __future__ import annotations
from typing import Tuple
import torch


@torch.no_grad()
def search(solver) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unified line-search entry point.
    Returns (ok, t, f_t, gdot_t) as 0‑D REAL tensors on the primary device.
    """
    name = str(getattr(solver, "ls_name", "armijo")).lower()
    if name == "wolfe":
        return _wolfe(solver)
    return _armijo(solver)


# ──────────────────────────────────────────────────────────────────────
# Armijo backtracking (tensor‑native, device‑consistent)
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _armijo(solver):
    ws, obj = solver.ws, solver.obj
    dev = getattr(ws, "primary_device", None) or solver.y.device
    dtype = getattr(ws, "dtype_r", solver.y.real.dtype)

    c1   = torch.tensor(getattr(solver, "c1", 1e-4), device=dev, dtype=dtype)
    beta = float(getattr(solver, "ls_beta", 0.5))

    sb = getattr(ws, "stats", None)

    obj.begin_linesearch(ws)
    try:
        # Scalars on the primary device
        t  = torch.ones((), device=dev, dtype=dtype)
        t0 = torch.zeros_like(t)

        # Pre‑step objective + slope (fills ws.g at x)
        f0, g0d = obj.f_g_tensor(ws, t0)

        # Record pre‑step stats for history (in LS scope)
        if sb is not None:
            sb.begin_scope("ls", activate=True, clear=True)
            sb.scalar_slot("f_base", dev, dtype).add_(f0)
            sb.scalar_slot("gdot0",  dev, dtype).add_(g0d)

        # If not a descent direction (or not finite), reject
        if (g0d >= 0).logical_or(torch.isnan(g0d)).logical_or(torch.isinf(g0d)):
            if sb is not None:
                sb.abort_scope("ls", deactivate=True, clear=True)
            return torch.zeros_like(t), t * 0, f0, g0d

        itmax = int(getattr(solver, "ls_max_iter", 20))

        for _ in range(itmax):
            f_t, gdot_t = obj.f_g_tensor(ws, t)

            # Armijo: f(t) ≤ f(0) + c1 * t * g'(0)
            rhs = f0 + c1 * t * g0d
            if f_t <= rhs:
                # overwrite LS-scoped 'gdot' with the slope at the accepted step
                if sb is not None:
                    slot = sb.scalar_slot("gdot", dev, dtype)
                    slot.zero_().add_(gdot_t)
                    sb.scalar_slot("step_len", dev, dtype).add_(t)
                    sb.commit_scope("ls", replace=True, deactivate=True, clear=True)
                return torch.ones_like(t), t, f_t, gdot_t

            # backtrack
            t.mul_(beta)

        # failed to satisfy Armijo
        if sb is not None:
            sb.abort_scope("ls", deactivate=True, clear=True)
        return torch.zeros_like(t), t, f_t, gdot_t
    finally:
        obj.end_linesearch(ws)


# ──────────────────────────────────────────────────────────────────────
# Strong Wolfe with simple zoom (tensor‑native)
# Conditions:
#   Armijo:          f(t) ≤ f(0) + c1 * t * g'(0)
#   Curvature (|.|): |g'(t)| ≤ c2 * |g'(0)|
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _wolfe(solver):
    ws, obj = solver.ws, solver.obj
    dev = getattr(ws, "primary_device", None) or solver.y.device
    dtype = getattr(ws, "dtype_r", solver.y.real.dtype)

    c1   = torch.tensor(getattr(solver, "c1", 1e-4), device=dev, dtype=dtype)
    c2   = torch.tensor(getattr(solver, "c2", 0.9),  device=dev, dtype=dtype)
    tmax = torch.tensor(1e6, device=dev, dtype=dtype)
    sb   = getattr(ws, "stats", None)

    obj.begin_linesearch(ws)
    try:
        t  = torch.ones((), device=dev, dtype=dtype)
        t0 = torch.zeros_like(t)

        f0, g0d = obj.f_g_tensor(ws, t0)

        if sb is not None:
            sb.begin_scope("ls", activate=True, clear=True)
            sb.scalar_slot("f_base", dev, dtype).add_(f0)
            sb.scalar_slot("gdot0",  dev, dtype).add_(g0d)

        # Reject if not descent / not finite
        if (g0d >= 0).logical_or(torch.isnan(g0d)).logical_or(torch.isinf(g0d)):
            if sb is not None:
                sb.abort_scope("ls", deactivate=True, clear=True)
            return torch.zeros_like(t), t * 0, f0, g0d

        t_prev = torch.zeros_like(t)
        f_prev, g_prev = f0, g0d

        itmax = int(getattr(solver, "ls_max_iter", 20))
        for i in range(itmax):
            f_t, gdot_t = obj.f_g_tensor(ws, t)

            # Armijo violation or non‑improving -> zoom
            if (f_t > f0 + c1 * t * g0d) or (i > 0 and f_t >= f_prev):
                return _zoom_commit(solver, t_low=t_prev, f_low=f_prev,
                                    t_high=t, f0=f0, g0d=g0d, sb=sb)

            # Curvature (strong): |g'(t)| ≤ c2 * |g'(0)|
            if gdot_t.abs() <= (c2 * g0d.abs()):
                if sb is not None:
                    slot = sb.scalar_slot("gdot", dev, dtype)
                    slot.zero_().add_(gdot_t)
                    sb.scalar_slot("step_len", dev, dtype).add_(t)
                    sb.commit_scope("ls", replace=True, deactivate=True, clear=True)
                return torch.ones_like(t), t, f_t, gdot_t

            # Positive slope -> zoom between t_prev and t
            if gdot_t >= 0:
                return _zoom_commit(solver, t_low=t, f_low=f_t,
                                    t_high=t_prev, f0=f0, g0d=g0d, sb=sb)

            # else: increase step and continue
            t_prev, f_prev, g_prev = t, f_t, gdot_t
            t = torch.minimum(2.0 * t, tmax)

        # If we ran out of iterations, treat as failure
        if sb is not None:
            sb.abort_scope("ls", deactivate=True, clear=True)
        return torch.zeros_like(t), t_prev, f_prev, g_prev
    finally:
        obj.end_linesearch(ws)


@torch.no_grad()
def _zoom_commit(solver,
                 *,
                 t_low: torch.Tensor,
                 f_low: torch.Tensor,
                 t_high: torch.Tensor,
                 f0: torch.Tensor,
                 g0d: torch.Tensor,
                 sb):
    """
    Simple bisection‑zoom that honors strong Wolfe.
    Commits LS scope on success; aborts on failure.
    """
    ws, obj = solver.ws, solver.obj
    dev   = t_low.device
    dtype = t_low.dtype
    c1    = torch.tensor(getattr(solver, "c1", 1e-4), device=dev, dtype=dtype)
    c2    = torch.tensor(getattr(solver, "c2", 0.9),  device=dev, dtype=dtype)
    itmax = int(getattr(solver, "ls_max_iter", 20))

    t_lo = t_low.clone()
    t_hi = t_high.clone()
    f_lo = f_low.clone()

    for _ in range(itmax):
        t_mid = 0.5 * (t_lo + t_hi)
        f_mid, g_mid = obj.f_g_tensor(ws, t_mid)

        if (f_mid > f0 + c1 * t_mid * g0d) or (f_mid >= f_lo):
            t_hi = t_mid
        else:
            if g_mid.abs() <= (c2 * g0d.abs()):
                if sb is not None:
                    slot = sb.scalar_slot("gdot", dev, dtype)
                    slot.zero_().add_(g_mid)
                    sb.scalar_slot("step_len", dev, dtype).add_(t_mid)
                    sb.commit_scope("ls", replace=True, deactivate=True, clear=True)
                return torch.ones_like(t_mid), t_mid, f_mid, g_mid
            if g_mid * (t_hi - t_lo) >= 0:
                t_hi = t_lo
            t_lo, f_lo = t_mid, f_mid

    # Fallback: best Armijo point we had (t_lo)
    f_lo, g_lo = obj.f_g_tensor(ws, t_lo)
    if sb is not None:
        slot = sb.scalar_slot("gdot", dev, dtype)
        slot.zero_().add_(g_lo)
        sb.scalar_slot("step_len", dev, dtype).add_(t_lo)
        sb.commit_scope("ls", replace=True, deactivate=True, clear=True)
    return torch.ones_like(t_lo), t_lo, f_lo, g_lo