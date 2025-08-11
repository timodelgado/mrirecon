# graspcg/ops/init_scaling.py
from __future__ import annotations
import math, torch
from typing import Mapping
from graspcg.utils.operations import dot_chunked

@torch.no_grad()
def initial_backproj_and_scaling(
    ws,
    regm,
    *,
    xfactor: float = 1.0,
    # per‑reg policy: {"tv_t": {"percentile":0.9,"eps_floor":1e-6,"kappa":1.0}, ...}
    stats_cfg: Mapping[str, Mapping[str, float | bool]] | None = None,
    verbose: bool = True,
) -> dict:
    """
    Matched filter back‑projection and robust scaling. Then, for each
    registered regulariser (that has a policy entry), estimate (ε,σ) and set λ=κ·σ.
    """
    # 1) back‑projection  A^H y -> x  (out=ws.x; no allocation)
    ws.nufft_op.AH(ws.y, out=ws.x)

    # 2) energy‑normalise via single forward pass into arena scratch
    E_data = dot_chunked(ws.y, ws.y, arena=ws.arena)
    kbuf = ws.arena.request(ws.y.numel(), ws.y.dtype, anchor=ws.y).view_as(ws.y)
    ws.nufft_op.A(ws.x, out=kbuf)
    E_est = dot_chunked(kbuf, kbuf, arena=ws.arena) + 1e-30
    ws.arena.release(kbuf)

    ws.x.mul_(math.sqrt(xfactor * E_data / E_est))

    # 3) pilot for stats = x (each regulariser applies its own scale internally)
    xs = ws.x

    chosen = {}
    if stats_cfg:
        chosen = regm.estimate_from_pilot(ws, xs, stats_cfg, verbose=verbose)

    return {"E_data": float(E_data), "reg_cfgs": chosen}
