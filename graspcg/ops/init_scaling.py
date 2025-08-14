from __future__ import annotations
import math
from types import SimpleNamespace
from typing import Mapping, Optional

import torch

# Optional: use your project dot if available; otherwise fall back
try:
    from ..numerics.dot import dot_chunked  # preferred location
except Exception:
    try:
        from ..ops.dot import dot_chunked
    except Exception:
        def dot_chunked(a: torch.Tensor, b: torch.Tensor, *, arena=None):
            return (a.conj() * b).real.sum()


@torch.no_grad()
def initial_backproj_and_scaling(
    ws,
    regm,
    *,
    xfactor: float = 1.0,
    stats_cfg: Optional[Mapping] = None,
    verbose: bool = True
) -> dict:
    """
    1) x <- A^H y (per shard, in-place).
    2) Compute scale s = sqrt( xfactor * ||y||^2 / ||A(x)||^2 ) (per-shard forward, no global kbuf).
    3) x <- s * x  (per shard).
    4) Attach ws.scale.divide_inplace(t) so continuation can use a normalised pilot.
    5) Optionally ask regm.estimate_from_pilot(...) for stats/policies.
    """
    # 1) Backproject per shard
    b_off = 0
    for sh, i in ws.iter_shards():
        x = ws.get("x", i)
        B = int(x.shape[0])
        y_slice = ws.y[b_off:b_off + B]
        if y_slice.device != x.device:
            y_slice = y_slice.to(x.device, non_blocking=True)
        ws.nufft_op.AH(y_slice, out=x)
        b_off += B

    # 2) Energies: E_data = ||y||^2 ; E_est = ||A(x)||^2 (sum per shard)
    E_data = float(dot_chunked(ws.y, ws.y, arena=ws.arena))
    E_est = 0.0
    b_off = 0
    for sh, i in ws.iter_shards():
        x = ws.get("x", i)
        B = int(x.shape[0])
        y_slice = ws.y[b_off:b_off + B]
        # tmp has *k-space* slice shape, resident on x.device
        ks_shape = y_slice.shape
        tmp = ws.arena.request(int(torch.tensor(ks_shape).prod().item()), ws.y.dtype, device=x.device).view(*ks_shape)
        ws.nufft_op.A(x, out=tmp)
        E_est += float(dot_chunked(tmp, tmp, arena=ws.arena))
        ws.arena.release(tmp)
        b_off += B
    E_est = max(E_est, 1e-30)  # numerical guard

    # 3) Uniform scale across shards
    scale = math.sqrt(float(xfactor) * E_data / E_est)
    for sh, i in ws.iter_shards():
        x = ws.get("x", i)
        x.mul_(scale)

    # 4) Provide ws.scale.divide_inplace(...) for continuation logic
    ws.scale = SimpleNamespace(divide_inplace=lambda t, s=float(scale): t.mul_(1.0 / s))

    # 5) Optional pilot-driven reg policy estimation
    # A single shard pilot is sufficient
    pilot = None
    for sh, i in ws.iter_shards():
        pilot = ws.get("x", i)
        break

    chosen = {}
    if stats_cfg and hasattr(regm, "estimate_from_pilot") and callable(regm.estimate_from_pilot):
        chosen = regm.estimate_from_pilot(ws, pilot, stats_cfg, verbose=verbose) or {}

    return {"E_data": float(E_data), "scale": float(scale), "reg_cfgs": chosen}