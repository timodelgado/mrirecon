# graspcg/numerics/directions.py
"""
Shard‑aware direction‑update classes that cooperate with CGWorkspace /
UnifiedArena.  No global ws.dx/diag assumptions—everything is per‑shard.

All large buffers are avoided; we only keep per‑shard g_prev (allocated
once via the arena) and tiny scalars.  Reductions use dot_chunked().
"""
from __future__ import annotations
import torch
from typing import Dict, Callable, Optional
from graspcg.utils.operations import dot_chunked


# ------------------------------- helpers ------------------------------------
def _ensure_shard_like(ws, sh, attr: str, like: torch.Tensor) -> torch.Tensor:
    """
    Ensure shard `sh` has a tensor attribute `attr` shaped like `like`.
    Allocate once via the arena (anchored on `like`) if missing/mismatch.
    """
    buf = getattr(sh, attr, None)
    if (buf is None) or (buf.shape != like.shape) or (buf.dtype != like.dtype) \
       or (buf.device != like.device):
        buf = ws.arena.request(like.numel(), like.dtype, anchor=like).view_as(like)
        setattr(sh, attr, buf)
    return buf


def _sum_over_shards(ws, fn) -> float:
    """
    Call `fn(sh)` for each shard, sum the scalar results (Python float).
    Guaranteed not to allocate large temporaries.
    """
    s = 0.0
    for sh, _ in ws.iter_shards():
        s += float(fn(sh))
    return s


def _g_dot_d(ws) -> float:
    """Compute Σ_sh ⟨g, d⟩ (unweighted) via dot_chunked per shard."""
    return _sum_over_shards(ws, lambda sh: dot_chunked(sh.g, sh.dx, arena=ws.arena))


# --------------------------------- base --------------------------------------
class _BaseDir:
    """
    Base class with shard‑aware lifecycle. Subclasses implement `_beta(ws)`.
    """
    def __init__(self, ws):
        self.ws = ws

    def init_state(self, g0: Optional[torch.Tensor] = None) -> float:
        """
        Prepare per‑shard state once. Returns current gᵀd to seed line search.
        """
        # Default: nothing required; subclasses may override.
        return _g_dot_d(self.ws)

    def _beta(self, ws) -> float:
        raise NotImplementedError

    @torch.no_grad()
    def update_inplace(self, ws) -> float:
        """
        Compute β from current shard gradients, update every shard’s d in‑place:
            d ← β d − M^{-1} g
        Return new gᵀd for the line search.
        """
        β = max(self._beta(ws), 0.0)

        # Update each shard’s direction: d = β d − M^{-1} g  (no big temps)
        for sh, _ in ws.iter_shards():
            # d *= β
            sh.dx.mul_(β)
            # d += - (g / diag)  (complex / real) without allocating big temporaries
            # addcdiv_: self += value * (tensor1 / tensor2)
            sh.dx.addcdiv_(sh.g, sh.diag, value=-1.0)

        # return the fresh directional derivative
        return _g_dot_d(ws)


# ----------------------------- Polak–Ribière+ --------------------------------
class DirPRPlus(_BaseDir):
    @torch.no_grad()
    def init_state(self, g0: Optional[torch.Tensor] = None) -> float:
        # Ensure and seed per‑shard g_prev
        for sh, _ in self.ws.iter_shards():
            gp = _ensure_shard_like(self.ws, sh, "g_prev", sh.g)
            gp.copy_(sh.g)
        return _g_dot_d(self.ws)

    def _beta(self, ws) -> float:
        tiny = 1e-15
        # β = max( (gᵀM^{-1}g − g_prevᵀM^{-1}g) / (g_prevᵀM^{-1}g_prev), 0 )
        num = _sum_over_shards(ws, lambda sh:
            dot_chunked(sh.g, sh.g, diag=sh.diag, arena=ws.arena)
            - dot_chunked(sh.g_prev, sh.g, diag=sh.diag, arena=ws.arena))

        den = _sum_over_shards(ws, lambda sh:
            dot_chunked(sh.g_prev, sh.g_prev, diag=sh.diag, arena=ws.arena))

        β = num / max(den, tiny)

        # Update g_prev for next iteration (after β computed)
        for sh, _ in ws.iter_shards():
            sh.g_prev.copy_(sh.g)
        return β


# ----------------------------- Dai–Yuan (PCG) --------------------------------
class DirDY(_BaseDir):
    @torch.no_grad()
    def init_state(self, g0: Optional[torch.Tensor] = None) -> float:
        for sh, _ in self.ws.iter_shards():
            gp = _ensure_shard_like(self.ws, sh, "g_prev", sh.g)
            gp.copy_(sh.g)
        return _g_dot_d(self.ws)

    def _beta(self, ws) -> float:
        tiny = 1e-15
        # β = (gᵀM^{-1}g) / (dᵀ(g − g_prev))
        num = _sum_over_shards(ws, lambda sh:
            dot_chunked(sh.g, sh.g, diag=sh.diag, arena=ws.arena))

        den = _sum_over_shards(ws, lambda sh:
            dot_chunked(sh.dx, sh.g, arena=ws.arena)
            - dot_chunked(sh.dx, sh.g_prev, arena=ws.arena))

        β = num / max(den, tiny)

        for sh, _ in ws.iter_shards():
            sh.g_prev.copy_(sh.g)
        return β


# --------------------------- Fletcher–Reeves (PCG) ---------------------------
class DirFR(_BaseDir):
    def __init__(self, ws):
        super().__init__(ws)
        self._rho_prev = None  # scalar

    @torch.no_grad()
    def init_state(self, g0: Optional[torch.Tensor] = None) -> float:
        # ρ₀ = Σ_sh g₀ᵀM^{-1}g₀
        self._rho_prev = _sum_over_shards(self.ws, lambda sh:
            dot_chunked(sh.g, sh.g, diag=sh.diag, arena=self.ws.arena))
        return _g_dot_d(self.ws)

    def _beta(self, ws) -> float:
        tiny = 1e-15
        rho_k = _sum_over_shards(ws, lambda sh:
            dot_chunked(sh.g, sh.g, diag=sh.diag, arena=ws.arena))
        if self._rho_prev is None:
            β = 0.0
        else:
            β = rho_k / max(self._rho_prev, tiny)
        self._rho_prev = rho_k
        return β


# -------------------------------- factory -----------------------------------
_FACTORY: Dict[str, Callable[[object], _BaseDir]] = {
    "prplus": DirPRPlus,
    "dy":     DirDY,
    "fr":     DirFR,
}

def build_direction(name: str, ws):
    try:
        return _FACTORY[name](ws)
    except KeyError:
        raise ValueError(f"unknown direction '{name}'")