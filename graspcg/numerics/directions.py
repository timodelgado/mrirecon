"""
Direction‑update classes that cooperate with CGWorkspace / UnifiedArena.
All scratch tensors are requested once from `ws.arena` and re‑used; nothing
is cloned each iteration.
"""
from __future__ import annotations
import torch
from typing import Dict, Callable
from graspcg.utils.operations import dot_chunked

# --------------------------------------------------------------------- helpers
def _alloc_like(ws, ref, name):
    """Allocate (once) a tensor with same shape/device/dtype via arena."""
    buf = getattr(ws, name, None)
    if buf is None or buf.shape != ref.shape:
        buf = ws.arena.request(ref.numel(), ref.dtype,
                               anchor=ref).view_as(ref)
        setattr(ws, name, buf)
    return buf

# --------------------------------------------------------------------- base
class _BaseDir:
    def __init__(self, ws):
        self.ws = ws
    def init_state(self, g0):
        self._init_buffers(g0)
        return dot_chunked(g0, self.ws.dx, arena=self.ws.arena)
    def _init_buffers(self, g0):  pass
    def _beta(self, g):  raise NotImplementedError
    def update_inplace(self, g_new):
        β = self._beta(g_new)
        self.ws.dx.mul_(β).sub_(g_new.div(self.ws.diag))
        return dot_chunked(g_new, self.ws.dx, arena=self.ws.arena)

# ------------------------------------------------------------------ PR⁺
# numerics/directions.py  – replace _beta implementations
class DirPRPlus(_BaseDir):
    def _init_buffers(self, g0):
        self.g_prev = self.ws.g_prev
        self.g_prev.copy_(g0)
    def _beta(self, g):
        ws, diag = self.ws, self.ws.diag
        num = dot_chunked(g      , g      , diag=diag, arena=ws.arena) \
            - dot_chunked(self.g_prev, g , diag=diag, arena=ws.arena)
        den = dot_chunked(self.g_prev, self.g_prev, diag=diag,
                          arena=ws.arena)
        self.g_prev.copy_(g)
        return max(num / max(den, 1e-15), 0.0)


# ------------------------------------------------------------------ DY
class DirDY(_BaseDir):
    def _init_buffers(self, g0):
        self.g_prev = self.ws.g_prev
        self.g_prev.copy_(g0)

    def _beta(self, g):
        ws = self.ws
        # numerator  d_k^T M^{-1} g_k  (unchanged)
        num = dot_chunked(g, g, diag=ws.diag, arena=ws.arena)

        # denominator  d_k^T (g_k - g_{k-1})  without forming y
        den = dot_chunked(ws.dx, g, arena=ws.arena) \
            - dot_chunked(ws.dx, self.g_prev, arena=ws.arena)
        den = max(den, 1e-15)

        self.g_prev.copy_(g)
        return max(num / den, 0.0)


# ------------------------------------------------------------------ FR⁺
class DirFR(_BaseDir):
    def _init_buffers(self, g0):
        self._rho_prev = dot_chunked(g0, g0, diag=self.ws.diag, arena=self.ws.arena)
    def _beta(self, g):
        rho_k = dot_chunked(g, g, diag=self.ws.diag, arena=self.ws.arena)
        β     = max(rho_k / self._rho_prev, 0.0)
        self._rho_prev = rho_k
        return β

# ------------------------------------------------------------------ factory
_FACTORY: Dict[str, Callable] = {
    "prplus": DirPRPlus,
    "dy":     DirDY,
    "fr":     DirFR,
}
def build_direction(name: str, ws):
    try:   return _FACTORY[name](ws)
    except KeyError: raise ValueError(f"unknown direction '{name}'")
