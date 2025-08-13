from __future__ import annotations
import math, torch
from typing import Mapping
from graspcg.utils.operations import dot_chunked

@torch.no_grad()
def initial_backproj_and_scaling(ws, regm, *, xfactor: float = 1.0,
                                 stats_cfg=None, verbose: bool = True) -> dict:
    # 1) x := A^H y (shard-wise, in place)
    b_off = 0
    for sh, _ in ws.iter_shards():
        B = sh.x.shape[0]
        ws.nufft_op.AH(ws.y[b_off:b_off+B], out=sh.x)
        b_off += B

    # 2) Energy-normalise via a single forward into arena scratch
    E_data = dot_chunked(ws.y, ws.y, arena=ws.arena)
    kbuf = ws.arena.request(ws.y.numel(), ws.y.dtype, anchor=ws.y).view_as(ws.y)
    kbuf.zero_()
    b_off = 0
    for sh, _ in ws.iter_shards():
        B = sh.x.shape[0]
        kpart = ws.arena.request(sh.x.numel(), ws.y.dtype, anchor=ws.y).view_as(sh.x)
        ws.nufft_op.A(sh.x, out=kpart)
        kbuf[b_off:b_off+B].add_(kpart)
        ws.arena.release(kpart)
        b_off += B
    E_est = dot_chunked(kbuf, kbuf, arena=ws.arena) + 1e-30
    ws.arena.release(kbuf)

    # scale all shards uniformly
    scale = math.sqrt(xfactor * E_data / E_est)
    for sh, _ in ws.iter_shards():
        sh.x.mul_(scale)

    xs = next(ws.iter_shards())[0].x  # a shard-level pilot is sufficient

    chosen = {}
    if stats_cfg and hasattr(regm, "estimate_from_pilot"):
        chosen = regm.estimate_from_pilot(ws, xs, stats_cfg, verbose=verbose)

    return {"E_data": float(E_data), "reg_cfgs": chosen}