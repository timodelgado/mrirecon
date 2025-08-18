from __future__ import annotations
from typing import Callable, Iterator, Optional, Tuple
import torch

# Optional: use arena-aware dot if available
try:
    from graspcg.ops.dot import dot_chunked
except Exception:
    def dot_chunked(a, b, *, diag=None, arena=None):
        z = (a.conj() * b).real
        return z.sum() if diag is None else (z / diag).sum()


class Sharded:
    """
    Lightweight view over a Workspace buffer `name` with per-shard ops.
    Control-flow only (not compiled); heavy math stays in kernels.
    """
    def __init__(self, ws, name: str):
        self.ws = ws
        self.name = name

    # ---------- iteration ----------
    def shards(self) -> Iterator[Tuple[int, torch.Tensor]]:
        for _, i in self.ws.iter_shards():
            yield i, self.ws.get(self.name, i)

    # ---------- in-place transforms ----------
    @torch.no_grad()
    def zero_(self):
        for i, t in self.shards():
            t.zero_()
        return self

    @torch.no_grad()
    def fill_(self, val: complex | float):
        for i, t in self.shards():
            t.fill_(val)
        return self

    @torch.no_grad()
    def copy_from(self, other: "Sharded"):
        for i, t in self.shards():
            t.copy_(other.ws.get(other.name, i))
        return self

    @torch.no_grad()
    def axpby_(self, a: complex | float, x: "Sharded", b: complex | float = 1.0):
        """y <- a*x + b*y    (this object is y)"""
        for i, y in self.shards():
            xi = x.ws.get(x.name, i)
            if b == 0:
                y.copy_(xi).mul_(a)
            else:
                y.mul_(b).add_(xi, alpha=a)
        return self

    @torch.no_grad()
    def addcdiv_(self, num: "Sharded", den: "Sharded", value: float = 1.0):
        """y <- y + value * num / den"""
        for i, y in self.shards():
            y.addcdiv_(num.ws.get(num.name, i), den.ws.get(den.name, i), value=value)
        return self

    @torch.no_grad()
    def map_(self, fn: Callable[[torch.Tensor], None]):
        """Apply an in-place lambda to each shard tensor."""
        for i, t in self.shards():
            fn(t)
        return self

    # ---------- reductions ----------
    @torch.no_grad()
    def dot(self, other: "Sharded", *, diag: Optional["Sharded"] = None) -> float:
        s = 0.0
        for i, a in self.shards():
            b = other.ws.get(other.name, i)
            d = None if diag is None else diag.ws.get(diag.name, i)
            s += float(dot_chunked(a, b, diag=d, arena=self.ws.arena))
        return s

    @torch.no_grad()
    def norm2(self) -> float:
        s = 0.0
        for i, t in self.shards():
            s += float(dot_chunked(t, t, arena=self.ws.arena))
        return s ** 0.5

    # ---------- convenience ----------
    @staticmethod
    def of(ws, name: str) -> "Sharded":
        return Sharded(ws, name)
