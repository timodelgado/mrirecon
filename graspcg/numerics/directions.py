from __future__ import annotations
from typing import Optional, Any

import torch

# Try to import dot_chunked; fall back to simple impl
try:
    from .dot import dot_chunked
except Exception:
    try:
        from ..ops.dot import dot_chunked
    except Exception:
        def dot_chunked(a, b, *, diag=None, arena=None):
            z = (a.conj() * b).real
            return z.sum() if diag is None else (z / diag).sum()


def _sum_over_shards(ws, fn) -> float:
    s = 0.0
    for sh, i in ws.iter_shards():
        s += float(fn(i))
    return s


def _g_dot_d(ws) -> float:
    return _sum_over_shards(ws, lambda i:
        dot_chunked(ws.get("g", i), ws.get("dx", i), arena=ws.arena))


class _BaseDir:
    def __init__(self, ws):
        self.ws = ws
        self._rho_prev: Optional[float] = None

    @torch.no_grad()
    def init_state(self, g0: Optional[torch.Tensor] = None) -> float:
        """Return g·d at t=0 after seeding any needed state."""
        return _g_dot_d(self.ws)

    @torch.no_grad()
    def update_inplace(self, ws) -> float:
        raise NotImplementedError


class DirPRPlus(_BaseDir):
    @torch.no_grad()
    def init_state(self, g0: Optional[torch.Tensor] = None) -> float:
        # Seed g_prev from current g (manifest must include 'g_prev')
        for sh, i in self.ws.iter_shards():
            g, gp = self.ws.bind(i, "g", "g_prev")
            gp.copy_(g)
        return _g_dot_d(self.ws)

    @torch.no_grad()
    def update_inplace(self, ws) -> float:
        # β_PR+ = (gᵀM^{-1}g − g_prevᵀM^{-1}g) / (g_prevᵀM^{-1}g_prev)
        num = _sum_over_shards(ws, lambda i:
            dot_chunked(ws.get("g", i), ws.get("g", i),
                        diag=ws.get("diag", i), arena=ws.arena)
            - dot_chunked(ws.get("g_prev", i), ws.get("g", i),
                          diag=ws.get("diag", i), arena=ws.arena))
        den = _sum_over_shards(ws, lambda i:
            dot_chunked(ws.get("g_prev", i), ws.get("g_prev", i),
                        diag=ws.get("diag", i), arena=ws.arena))
        β = max(0.0, float(num / max(den, 1e-20)))

        for sh, i in ws.iter_shards():
            g, d, D = ws.bind(i, "g", "dx", "diag")
            d.mul_(β)
            d.addcdiv_(g, D, value=-1.0)
        for sh, i in ws.iter_shards():
            g, gp = ws.bind(i, "g", "g_prev")
            gp.copy_(g)
        return β


class DirDY(_BaseDir):
    @torch.no_grad()
    def init_state(self, g0: Optional[torch.Tensor] = None) -> float:
        for sh, i in self.ws.iter_shards():
            g, gp = self.ws.bind(i, "g", "g_prev")
            gp.copy_(g)
        return _g_dot_d(self.ws)

    @torch.no_grad()
    def update_inplace(self, ws) -> float:
        # β_DY = (gᵀM^{-1}g) / (dᵀ(g − g_prev))
        num = _sum_over_shards(ws, lambda i:
            dot_chunked(ws.get("g", i), ws.get("g", i),
                        diag=ws.get("diag", i), arena=ws.arena))
        den = _sum_over_shards(ws, lambda i:
            dot_chunked(ws.get("dx", i), ws.get("g", i), arena=ws.arena)
            - dot_chunked(ws.get("dx", i), ws.get("g_prev", i), arena=ws.arena))
        β = float(num / max(den, 1e-20))

        for sh, i in ws.iter_shards():
            g, d, D = ws.bind(i, "g", "dx", "diag")
            d.mul_(β)
            d.addcdiv_(g, D, value=-1.0)
        for sh, i in ws.iter_shards():
            g, gp = ws.bind(i, "g", "g_prev")
            gp.copy_(g)
        return β


class DirFR(_BaseDir):
    @torch.no_grad()
    def update_inplace(self, ws) -> float:
        # β_FR = (gᵀM^{-1}g) / (g_prevᵀM^{-1}g_prev)  (with rho cache)
        if self._rho_prev is None:
            self._rho_prev = _sum_over_shards(ws, lambda i:
                dot_chunked(ws.get("g", i), ws.get("g", i),
                            diag=ws.get("diag", i), arena=ws.arena))
        rho_k = _sum_over_shards(ws, lambda i:
            dot_chunked(ws.get("g", i), ws.get("g", i),
                        diag=ws.get("diag", i), arena=ws.arena))
        β = float(rho_k / max(self._rho_prev, 1e-20))
        self._rho_prev = rho_k

        for sh, i in ws.iter_shards():
            g, d, D = ws.bind(i, "g", "dx", "diag")
            d.mul_(β)
            d.addcdiv_(g, D, value=-1.0)
        return β