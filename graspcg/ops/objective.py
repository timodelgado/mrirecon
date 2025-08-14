from __future__ import annotations
from typing import Optional, Tuple, Any

import torch

# Try to import dot_chunked from your repo; fall back to a simple impl.
try:
    from ..numerics.dot import dot_chunked  # if you have this
except Exception:
    try:
        from ..ops.dot import dot_chunked    # or this path
    except Exception:
        def dot_chunked(a, b, *, diag=None, arena=None):
            z = (a.conj() * b).real
            return z.sum() if diag is None else (z / diag).sum()


class Objective:
    """
    Uses per-shard k-space caches if present in Workspace:
      - 'Ax_sh', 'Ad_sh', 'r_sh'  (preferred; no P2P)
    Else falls back to a global scratch anchored on y.device.

    Regularizer contract:
      regm.energy_and_grad(ws): add reg energy to return, and accumulate into 'g' buffers.
    """
    def __init__(self, nufft_op, y: torch.Tensor, regm):
        self.nufft = nufft_op
        self.y     = y
        self.regm  = regm

        self._Ax   : Optional[torch.Tensor] = None
        self._Adx  : Optional[torch.Tensor] = None
        self._r    : Optional[torch.Tensor] = None
        self._use_sharded: bool = False

    def begin_linesearch(self, ws) -> None:
        """
        Prepare caches. If per-shard k-space caches ('Ax_sh','Ad_sh','r_sh') exist
        in the workspace manifest, use them. Otherwise fall back to a global scratch.
        """
        arena = ws.arena
        y     = self.y

        # Prefer sharded k-space caches if present
        have_sharded = all(n in ws.list_bufs() for n in ("Ax_sh", "Ad_sh", "r_sh"))
        self._use_sharded = bool(have_sharded)

        if self._use_sharded:
            for sh, i in ws.iter_shards():
                x, dx        = ws.bind(i, "x", "dx")
                Ax_sh, Ad_sh = ws.bind(i, "Ax_sh", "Ad_sh")
                self.nufft.A(x,  out=Ax_sh)
                self.nufft.A(dx, out=Ad_sh)
            self._Ax = self._Adx = self._r = None
            return

        # Global scratch path (anchored on y.device), careful with cross-device compute
        self._Ax  = arena.request(y.numel(), y.dtype, anchor=y).view_as(y)
        self._Adx = arena.request(y.numel(), y.dtype, anchor=y).view_as(y)
        self._r   = arena.request(y.numel(), y.dtype, anchor=y).view_as(y)

        b_off = 0
        for sh, i in ws.iter_shards():
            x  = ws.get("x", i)
            B  = int(x.shape[0])
            ys = y[b_off:b_off+B]
            tmp_local = arena.request(ys.numel(), y.dtype, device=x.device).view_as(ys)
            self.nufft.A(x, out=tmp_local)
            self._Ax[b_off:b_off+B].copy_(tmp_local.to(self._Ax.device, non_blocking=True))
            arena.release(tmp_local)
            b_off += B

        b_off = 0
        for sh, i in ws.iter_shards():
            dx = ws.get("dx", i)
            B  = int(dx.shape[0])
            ys = y[b_off:b_off+B]
            tmp_local = arena.request(ys.numel(), y.dtype, device=dx.device).view_as(ys)
            self.nufft.A(dx, out=tmp_local)
            self._Adx[b_off:b_off+B].copy_(tmp_local.to(self._Adx.device, non_blocking=True))
            arena.release(tmp_local)
            b_off += B

    def end_linesearch(self, ws) -> None:
        """Release LS scratch. No-op when using per-shard caches."""
        if self._use_sharded:
            return
        arena = ws.arena
        if self._Ax is not None:  arena.release(self._Ax);  self._Ax  = None
        if self._Adx is not None: arena.release(self._Adx); self._Adx = None
        if self._r is not None:   arena.release(self._r);   self._r   = None

    @torch.no_grad()
    def f_g(self, ws, t: float) -> Tuple[float, float]:
        """
        Evaluate total objective φ(t) and φ'(t)=g(t)^T d at x + t·d.
        Fills per-shard 'g' with data+reg gradients at x + t·d.
        """
        # 0) clear gradient
        for sh, i in ws.iter_shards():
            ws.get("g", i).zero_()

        # 1) Data term via caches
        if self._use_sharded:
            f_data = 0.0
            b_off = 0
            for sh, i in ws.iter_shards():
                Ax, Ad, r = ws.bind(i, "Ax_sh", "Ad_sh", "r_sh")
                B = int(Ax.shape[0])
                r.copy_(Ax).add_(Ad, alpha=float(t))
                y_slice = self.y[b_off:b_off+B]
                if y_slice.device != r.device:
                    y_slice = y_slice.to(r.device, non_blocking=True)
                r.sub_(y_slice)
                f_data += float(dot_chunked(r, r, arena=ws.arena))
                # data-grad
                g = ws.get("g", i)
                self.nufft.AH(r, out=g)
                b_off += B
            f_data *= 0.5
        else:
            r = self._r
            r.copy_(self._Ax).add_(self._Adx, alpha=float(t)).sub_(self.y)
            f_data = 0.5 * float(dot_chunked(r, r, arena=ws.arena))
            b_off = 0
            for sh, i in ws.iter_shards():
                g = ws.get("g", i)
                B = int(g.shape[0])
                self.nufft.AH(r[b_off:b_off+B], out=g)
                b_off += B

        # 2) Regularization at x + t·dx (push/pop)
        for sh, i in ws.iter_shards():
            x, dx = ws.bind(i, "x", "dx")
            x.add_(dx, alpha=float(t))
        f_reg = float(self.regm.energy_and_grad(ws))
        for sh, i in ws.iter_shards():
            x, dx = ws.bind(i, "x", "dx")
            x.add_(dx, alpha=-float(t))

        # 3) g·d
        gdot = 0.0
        for sh, i in ws.iter_shards():
            g, d = ws.bind(i, "g", "dx")
            gdot += float(dot_chunked(g, d, arena=ws.arena))
        return f_data + f_reg, float(gdot)