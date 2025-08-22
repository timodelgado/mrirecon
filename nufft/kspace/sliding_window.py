from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import torch

@dataclass
class SlidingWindowSpec:
    spokes_per_frame: int
    hop: Optional[int] = None
    keep_tail: bool = False

def sliding_window_sort(y: torch.Tensor, om: torch.Tensor, dcf: Optional[torch.Tensor], spec: SlidingWindowSpec) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns (yB, omB, dcfB) with shapes:
      yB : (B, C, K),   omB : (B, K, nd),   dcfB : (B, K) or None
    where B = number of windows and K = spokes_per_frame * samples_per_spoke.
    Assumes spokes are already ordered (e.g., golden-angle list).
    """
    S = int(om.shape[0])
    spp = int(om.shape[1])
    nd = int(om.shape[2])
    hop = spec.hop or spec.spokes_per_frame
    B = (S - spec.spokes_per_frame) // hop + 1
    if spec.keep_tail and (S - spec.spokes_per_frame) % hop != 0:
        B += 1
    C = int(y.shape[1])
    K = spec.spokes_per_frame * spp
    y_out = torch.empty((B, C, K), dtype=y.dtype, device=y.device)
    om_out = torch.empty((B, K, nd), dtype=om.dtype, device=om.device)
    dcf_out = None if dcf is None else torch.empty((B, K), dtype=dcf.dtype, device=dcf.device)
    b = 0
    t = 0
    while t + spec.spokes_per_frame <= S:
        sel = slice(t, t + spec.spokes_per_frame)
        y_b = y[sel].permute(1, 0, 2).reshape(C, K)
        om_b = om[sel].reshape(K, nd)
        y_out[b] = y_b
        om_out[b] = om_b
        if dcf is not None:
            dcf_out[b] = dcf[sel].reshape(K)
        b += 1
        t += hop
        if b >= B:
            break
    return (y_out, om_out, dcf_out)