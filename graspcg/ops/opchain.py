# graspcg/ops/opchain.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Protocol, Mapping, Optional
import torch

class LinOp(Protocol):
    def fwd(self, x: torch.Tensor) -> torch.Tensor: ...
    def adj(self, y: torch.Tensor) -> torch.Tensor: ...
    def halo(self, roles) -> Mapping[int, int]:  # optional halo request
        return {}

@dataclass
class ScaleOp(LinOp):
    inv_s: torch.Tensor  # broadcastable (e.g., (B,1,...,1))
    def fwd(self, x): return x * self.inv_s
    def adj(self, y): return y * self.inv_s  # self-adjoint

@dataclass
class MatmulTemporalBasis(LinOp):
    U: torch.Tensor          # (T, K), real or complex OK
    # Expects V of shape (B, K, *spatial)  ->  X of shape (B, T, *spatial)
    def fwd(self, V): return torch.einsum("tk,bk...->bt...", self.U, V)
    def adj(self, X): return torch.einsum("tk,bt...->bk...", self.U.conj(), X)

@dataclass
class Chain(LinOp):
    ops: Sequence[LinOp]
    def fwd(self, x):
        for op in self.ops: x = op.fwd(x)
        return x
    def adj(self, y):
        for op in reversed(self.ops): y = op.adj(y)
        return y
    def halo(self, roles):
        h = {}
        for op in self.ops:
            for ax, k in op.halo(roles).items():
                h[ax] = max(h.get(ax, 0), k)
        return h