# -----------------------------------------------------------------------------
# graspcg/nufft/specs.py
# -----------------------------------------------------------------------------
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Literal
import torch

@dataclass
class ProblemSpec:
    ndim: Literal[2,3]
    traj_units: Literal['norm','rad'] = 'norm'
    default_backend: Literal['auto','torchkb','cufi'] = 'auto'
    axis: Optional[object] = None              # AxisSpec if you expose layouts
    vectorize_like: bool = True

@dataclass
class Shard:
    device: torch.device
    frames: slice                  # [start:stop)
    backend: Literal['torchkb','cufi']
    like_mode: Literal['fold_into_batch','explicit_axis']  # TorchKb vs CUFI
    like_prod: int                 # C×L_other for CUFI (n_trans)

@dataclass
class DistributionPlan:
    B_total: int
    shards: List[Shard]
    streams: Optional[Dict[torch.device, torch.cuda.Stream]] = None
