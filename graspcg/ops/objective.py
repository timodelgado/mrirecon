# graspcg/ops/objective.py
from __future__ import annotations
import torch
from typing import Optional, Tuple
from ..utils.operations import dot_chunked
from ..ops.reg_registry import REG_HANDLERS
from ..ops.reg_manager import RegManager
class Objective:
    """
    Memory‑aware objective with arena‑backed scratch and NUFFT caches.

    API
    ---
    begin_linesearch(ws):  precompute Ax, Adx once (scratch in arena)
    f_g(ws, t) -> (f_total, gdot):  compute φ(t) and φ'(t)=g(t)^T d
    end_linesearch(ws):    release scratch

    Notes
    -----
    • No big persistent buffers; all scratch is via ws.arena.
    • Data term uses linearity: Ax, Adx cached -> residual r(t) without re‑NUFFT.
    • Regularisers are evaluated at x+t·d by temporarily pushing x and popping back.
    """

    def __init__(self, nufft_op, y: torch.Tensor, regm: RegManager):
        self.nufft = nufft_op
        self.y     = y
        self.regm  = regm
        # transient handles to arena scratch (set by begin_linesearch)
        self._Ax   : Optional[torch.Tensor] = None
        self._Adx  : Optional[torch.Tensor] = None
        self._r    : Optional[torch.Tensor] = None

    # ──────────────────────────────────────────────────────────────────────
    # LS lifecycle
    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def begin_linesearch(self, ws) -> None:
        """
        Prepare NUFFT caches: Ax and Adx, plus one residual buffer r(t).
        All allocations go through the arena and are released in end_linesearch.
        """
        arena = ws.arena
        y     = self.y

        # k-space scratch
        self._Ax  = arena.request(y.numel(), y.dtype, anchor=y).view_as(y)
        self._Adx = arena.request(y.numel(), y.dtype, anchor=y).view_as(y)
        self._r   = arena.request(y.numel(), y.dtype, anchor=y).view_as(y)

        # Ax into scratch
        self.nufft.A(ws.x, out=self._Ax)

        # Adx: forward of the direction (already preconditioned)
        self.nufft.A(ws.dx, out=self._Adx)

    @torch.no_grad()
    def end_linesearch(self, ws) -> None:
        """Release LS scratch back to arena."""
        arena = ws.arena
        if self._Ax is not None:  arena.release(self._Ax);  self._Ax  = None
        if self._Adx is not None: arena.release(self._Adx); self._Adx = None
        if self._r is not None:   arena.release(self._r);   self._r   = None

    # ──────────────────────────────────────────────────────────────────────
    # Objective at x + t·d
    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def f_g(self, ws, t: float) -> Tuple[float, float]:
        """
        Evaluate total objective φ(t) and φ'(t)=g(t)^T d at x + t·d.
        Fills ws.g with the full gradient (data + regs) at x + t·d.
        """
        assert self._Ax is not None and self._Adx is not None and self._r is not None, \
               "call begin_linesearch() before f_g"

        # 0) clear gradient
        for sh,_ in ws.iter_shards():
            sh.g.zero_()

        # 1) Data term via caches
        r = self._r
        # r(t) = Ax + t·Adx − y   (in place on scratch)
        r.copy_(self._Ax).add_(self._Adx, alpha=float(t)).sub_(self.y)

        # f_data = 0.5 * ||r||^2   (chunked dot via arena)
        f_data = 0.5 * float(dot_chunked(r, r, arena=ws.arena))

        # g_data = A^H r(t)  (accumulate directly into ws.g sharded)
        # We map frame ranges shard-by-shard; assume leading dim ~ frames.
        b_off = 0
        for sh,_ in ws.iter_shards():
            B = sh.g.shape[0]
            self.nufft.AH(r[b_off:b_off+B], out=sh.g)
            b_off += B

        # 2) Regularisers at x + t·d
        #    Push x in-place, accumulate reg energy+grad, then pop back.
        for sh,_ in ws.iter_shards():
            sh.x.add_(sh.dx, alpha=float(t))

        f_reg = float(self.regm.energy_and_grad(ws))

        for sh,_ in ws.iter_shards():
            sh.x.add_(sh.dx, alpha=-float(t))  # pop

        # 3) total f and directional derivative
        f_total = f_data + f_reg

        gdot = 0.0
        for sh,_ in ws.iter_shards():
            # local inner product Re⟨g, d⟩
            gdot += (sh.g.conj() * sh.dx).real.sum().item()

        return f_total, float(gdot)
