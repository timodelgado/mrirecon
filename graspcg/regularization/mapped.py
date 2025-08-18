# graspcg/regularization/mapped.py
from __future__ import annotations
# Re-export unified mapping API
from .mapping import (
    Op, IdentityOp, ScaleOp, TemporalBasisOp, ComposeOp, LinOpAdapter,
    MappedRegularizer,
)
__all__ = [
    "Op", "IdentityOp", "ScaleOp", "TemporalBasisOp", "ComposeOp", "LinOpAdapter",
    "MappedRegularizer",
]
