# graspcg/solver/utils.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
import torch
from ..regularization.stats_board import StatsBoard
from ..numerics.utils import dot_real0, dot_precond0

@torch.no_grad()
def xnorm0(ws) -> torch.Tensor:
    acc = None
    for sh, i in ws.iter_shards():
        v = dot_real0(ws.get("x", i), ws.get("x", i))
        acc = v if acc is None else (acc + v.to(acc.device))
    dev  = getattr(ws, "primary_device", None) or next(ws.iter_shards())[0].device
    dtype = getattr(ws, "dtype_r", torch.float32)
    return torch.sqrt((acc if acc is not None else torch.zeros((), device=dev, dtype=dtype)).clamp_min_(0))

@torch.no_grad()
def stepnorm0(ws) -> torch.Tensor:
    acc = None
    for sh, i in ws.iter_shards():
        v = dot_real0(ws.get("dx", i), ws.get("dx", i))
        acc = v if acc is None else (acc + v.to(acc.device))
    dev  = getattr(ws, "primary_device", None) or next(ws.iter_shards())[0].device
    dtype = getattr(ws, "dtype_r", torch.float32)
    return torch.sqrt((acc if acc is not None else torch.zeros((), device=dev, dtype=dtype)).clamp_min_(0))

@torch.no_grad()
def gnorm_precond0(ws) -> torch.Tensor:
    acc = None
    for sh, i in ws.iter_shards():
        g, D = ws.bind(i, "g", "diag")
        v = dot_precond0(g, g, D, arena=getattr(ws, "arena", None))
        acc = v if acc is None else (acc + v.to(acc.device))
    dev  = getattr(ws, "primary_device", None) or next(ws.iter_shards())[0].device
    dtype = getattr(ws, "dtype_r", torch.float32)
    return torch.sqrt((acc if acc is not None else torch.zeros((), device=dev, dtype=dtype)).clamp_min_(0))

@torch.no_grad()
def converged(ws, *, tol_g: float, tol_step: float) -> bool:
    x = xnorm0(ws); s = stepnorm0(ws); g = gnorm_precond0(ws)
    t_x = 1.0 + x
    return bool(((g <= (t_x * tol_g)) & (s <= (t_x * tol_step))).item())

@torch.no_grad()
def preallocate_stats(sb: StatsBoard, regm, arena, y, *, extra_keys=("gdot0","f_base","step_len","xnorm","gnorm","stepnorm")) -> None:
    devs = []
    try: devs = list(arena.cuda_devices())
    except Exception: pass
    if not devs:
        try: devs = [arena.compute_device()]
        except Exception: devs = [y.device]
    rdt  = y.real.dtype
    for dev in devs:
        for k in ("E_data","E_reg_total","gdot","f_total"):
            sb.scalar_slot(k, dev, rdt)
        for k in extra_keys:
            sb.scalar_slot(k, dev, rdt)
    for reg in getattr(regm, "_regs", []):
        for dev in devs:
            sb.scalar_slot(f"E_reg/{reg.name}", dev, rdt)
            sb.scalar_slot(f"tv_q/{reg.name}", dev, rdt)

def history_from_stats(sb: StatsBoard) -> List[Dict[str, float]]:
    f_hist = sb.read_history("f_base")
    if not f_hist:
        f_hist = sb.read_history("f_total")
    step_h   = sb.read_history("step_len")
    g0_hist  = sb.read_history("gdot0")
    ga_hist  = sb.read_history("gdot")
    xn_hist  = sb.read_history("xnorm")
    gn_hist  = sb.read_history("gnorm")
    sn_hist  = sb.read_history("stepnorm")
    n = max(map(len, (f_hist, step_h, g0_hist, ga_hist, xn_hist, gn_hist, sn_hist)), default=0)
    out = []
    for k in range(n):
        out.append({
            "iter": k,
            "f": f_hist[k] if k < len(f_hist) else 0.0,
            "gdot": g0_hist[k] if k < len(g0_hist) else 0.0,
            "gdot_at_step": ga_hist[k] if k < len(ga_hist) else 0.0,
            "step": step_h[k] if k < len(step_h) else 0.0,
            "xnorm": xn_hist[k] if k < len(xn_hist) else 0.0,
            "gnorm": gn_hist[k] if k < len(gn_hist) else 0.0,
            "stepnorm": sn_hist[k] if k < len(sn_hist) else 0.0,
        })
    return out

def ls_begin(sb: Optional[StatsBoard], g0d: torch.Tensor) -> None:
    if sb is None: return
    sb.begin_scope("ls", activate=True, clear=True)
    sb.scalar_slot("gdot0", g0d.device, g0d.dtype).add_(g0d)

def ls_accept(sb: Optional[StatsBoard], t: torch.Tensor) -> None:
    if sb is None: return
    sb.scalar_slot("step_len", t.device, t.dtype).add_(t)
    sb.commit_scope("ls", replace=True, deactivate=True, clear=True)

def ls_abort(sb: Optional[StatsBoard]) -> None:
    if sb is None: return
    sb.abort_scope("ls", deactivate=True, clear=True)