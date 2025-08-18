# graspcg/regularization/wrappers.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import torch
from .base import Regularizer, RegContext
from .mapping import MappedRegularizer, LinOpAdapter, Op

@dataclass
class Transformed(Regularizer):
    """
    Wrap a Regularizer with a mapping:
      • If 'op' implements the unified Op API (forward_apply/adjoint_apply), we use it directly.
      • Else, if 'op' provides .fwd(x)->y and .adj(y)->x, we adapt it via LinOpAdapter.
    """
    base: Regularizer
    op: Any                    # Op or object with fwd/adj
    name: str
    params: Any                # optional passthrough of base.params for policies
    _mr: MappedRegularizer = field(init=False)

    def __post_init__(self):
        # Unified mapping first
        if hasattr(self.op, "forward_apply") and hasattr(self.op, "adjoint_apply"):
            op_u = self.op  # already a unified Op
        else:
            # Legacy fwd/adj -> wrap
            op_u = LinOpAdapter(self.op)
        self._mr = MappedRegularizer(name=self.name, inner=self.base, op=op_u)

    def halo(self, roles):
        try:
            return self._mr.halo(roles)
        except Exception:
            return {}

    def energy_grad(self, ctx: RegContext) -> torch.Tensor:
        return self._mr.energy_grad(ctx)

    # Optional pass-throughs
    def add_diag(self, ctx: RegContext) -> None:
        try:
            return self._mr.add_diag(ctx)
        except Exception:
            return None



    def continuation_update(self, stats):
        try:
            return bool(self.base.continuation_update(stats))
        except Exception:
            return False

    def scaling_policy(self, ctx):  # deprecated in unified mapping
        return None

    def prox_inplace(self, ctx, step: float):
        pass

    def majorizer_diag(self, ctx):
        try:
            return self.base.majorizer_diag(ctx)
        except Exception:
            return None
