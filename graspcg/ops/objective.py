from __future__ import annotations
from typing import Optional, Tuple

import torch


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    """Map complex -> real counterpart, else pass through."""
    if dtype is torch.complex64:
        return torch.float32
    if dtype is torch.complex128:
        return torch.float64
    return dtype


def _dot_real(a: torch.Tensor,
              b: torch.Tensor,
              diag: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Vectorized complex-real dot: Re⟨a, b⟩ or Re⟨a, b/diag⟩.
    Returns a 0‑D real tensor on a.device, compiler‑friendly.
    """
    if diag is None:
        z = a.conj() * b
    else:
        z = a.conj() * (b / diag)
    return z.real.sum()  # 0‑D tensor


class Objective:
    """
    Compilation‑friendly objective evaluation.

    Contract:
      - begin_linesearch(ws): populate caches (per‑shard preferred).
      - f_g_tensor(ws, t): returns (f_total, gdot) as 0‑D REAL tensors; fills ws.g.
      - f_g(ws, t_float): compatibility wrapper that returns Python floats.
      - end_linesearch(ws): release global scratch if used.

    Regularizers:
      regm.energy_and_grad(ws) must return a 0‑D REAL tensor and accumulate into ws.g.
    """

    def __init__(self, nufft_op, y: torch.Tensor, regm):
        self.nufft = nufft_op
        self.y     = y
        self.regm  = regm

        self._Ax   : Optional[torch.Tensor] = None
        self._Adx  : Optional[torch.Tensor] = None
        self._r    : Optional[torch.Tensor] = None
        self._use_sharded: bool = False

    @torch.no_grad()
    def begin_linesearch(self, ws) -> None:
        """
        Prepare caches. If per‑shard k‑space caches ('Ax_sh','Ad_sh','r_sh') exist,
        use them; otherwise fall back to a global scratch anchored on y.device.
        """
        arena = ws.arena
        y     = self.y

        have_sharded = all(n in ws.list_bufs() for n in ("Ax_sh", "Ad_sh", "r_sh"))
        self._use_sharded = bool(have_sharded)

        if self._use_sharded:
            # Populate per‑shard forward caches from current x and dx
            for sh, i in ws.iter_shards():
                x, dx        = ws.bind(i, "x", "dx")
                Ax_sh, Ad_sh = ws.bind(i, "Ax_sh", "Ad_sh")
                self.nufft.A(x,  out=Ax_sh)
                self.nufft.A(dx, out=Ad_sh)
            self._Ax = self._Adx = self._r = None
            return

        # Global scratch path (anchored on y.device), careful with cross‑device compute
        self._Ax  = arena.request(y.numel(), y.dtype, anchor=y).view_as(y)
        self._Adx = arena.request(y.numel(), y.dtype, anchor=y).view_as(y)
        self._r   = arena.request(y.numel(), y.dtype, anchor=y).view_as(y)

        # Fill Ax
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

        # Fill Adx
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

    @torch.no_grad()
    def end_linesearch(self, ws) -> None:
        """Release global scratch (no‑op for per‑shard caches)."""
        if self._use_sharded:
            return
        arena = ws.arena
        if self._Ax is not None:  arena.release(self._Ax);  self._Ax  = None
        if self._Adx is not None: arena.release(self._Adx); self._Adx = None
        if self._r is not None:   arena.release(self._r);   self._r   = None

    @torch.no_grad()
    def f_g_tensor(self, ws, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate φ(t) and φ'(t)=g(t)^T d at x + t·d.
        Returns (f_total, gdot) **as 0‑D REAL tensors**; fills per‑shard 'g'.
        """
        # 0) clear gradient
        for sh, i in ws.iter_shards():
            ws.get("g", i).zero_()

        dtype_r = _real_dtype(self.y.dtype)

        # --- 1) Data term via caches (per‑device accum to avoid repeated host syncs)
        f_data_dev: dict[torch.device, torch.Tensor] = {}
        gdot_dev  : dict[torch.device, torch.Tensor] = {}

        if self._use_sharded:
            b_off = 0
            for sh, i in ws.iter_shards():
                Ax, Ad, r = ws.bind(i, "Ax_sh", "Ad_sh", "r_sh")
                B = int(Ax.shape[0])
                dev = Ax.device

                # r = Ax + t*Ad − y_slice   (without alpha=float(..))
                y_slice = self.y[b_off:b_off+B]
                if y_slice.device != dev:
                    y_slice = y_slice.to(dev, non_blocking=True)
                t_dev = t.to(dev, dtype=dtype_r)

                r.copy_(Ad)       # r = Ad
                r.mul_(t_dev)     # r = t*Ad
                r.add_(Ax)        # r = Ax + t*Ad
                r.sub_(y_slice)   # r = Ax + t*Ad − y

                # f_data accum on dev
                e = _dot_real(r, r)  # 0‑D real on dev
                if dev in f_data_dev:
                    f_data_dev[dev].add_(e)
                else:
                    f_data_dev[dev] = e

                # data‑grad: g_i = A^H r_i
                g = ws.get("g", i)
                self.nufft.AH(r, out=g)

                # g·d accum on dev
                d = ws.get("dx", i)
                gd = _dot_real(g, d)
                if dev in gdot_dev:
                    gdot_dev[dev].add_(gd)
                else:
                    gdot_dev[dev] = gd

                b_off += B

            # Reduce dev accumulators to a primary device
            if f_data_dev:
                primary = next(iter(f_data_dev.keys()))
                f_data = torch.zeros((), device=primary, dtype=dtype_r)
                for dev, val in f_data_dev.items():
                    f_data.add_(val if dev == primary else val.to(primary, non_blocking=True))
                f_data.mul_(0.5)
            else:
                # degenerate (no shards)
                f_data = torch.zeros((), device=self.y.device, dtype=dtype_r)
            ws.stats.scalar_slot("E_data", f_data.device, f_data.dtype).add_(f_data)

            if gdot_dev:
                primary_g = next(iter(gdot_dev.keys()))
                gdot = torch.zeros((), device=primary_g, dtype=dtype_r)
                for dev, val in gdot_dev.items():
                    gdot.add_(val if dev == primary_g else val.to(primary_g, non_blocking=True))
            else:
                gdot = torch.zeros((), device=self.y.device, dtype=dtype_r)
            ws.stats.scalar_slot("gdot", gdot.device, gdot.dtype).add_(gdot)
        else:
            # Global scratch on y.device
            r = self._r
            assert r is not None and self._Ax is not None and self._Adx is not None
            t_y = t.to(r.device, dtype=dtype_r)

            r.copy_(self._Adx)   # r = Adx
            r.mul_(t_y)          # r = t*Adx
            r.add_(self._Ax)     # r = Ax + t*Adx
            r.sub_(self.y)       # r = Ax + t*Adx − y

            f_data = _dot_real(r, r)
            f_data.mul_(0.5)
            ws.stats.scalar_slot("E_data", f_data.device, f_data.dtype).add_(f_data)

            # data‑grad per shard
            b_off = 0
            for sh, i in ws.iter_shards():
                g = ws.get("g", i)
                B = int(g.shape[0])
                # AH must accept input slice on r.device and write to g.device
                self.nufft.AH(r[b_off:b_off+B], out=g)
                b_off += B

            # g·d (reduce to y.device as primary)
            gdot = torch.zeros((), device=r.device, dtype=dtype_r)
            for sh, i in ws.iter_shards():
                g, d = ws.bind(i, "g", "dx")
                gd = _dot_real(g, d)
                gdot.add_(gd if gd.device == gdot.device else gd.to(gdot.device, non_blocking=True))
            ws.stats.scalar_slot("gdot", gdot.device, gdot.dtype).add_(gdot)
        # --- 2) Regularization at x + t·dx (push/pop)
        for sh, i in ws.iter_shards():
            x, dx = ws.bind(i, "x", "dx")
            t_dev = t.to(x.device, dtype=dtype_r)
            x.add_(dx * t_dev)

        f_reg = self.regm.energy_and_grad(ws)  # 0‑D tensor (may be on another device)

        for sh, i in ws.iter_shards():
            x, dx = ws.bind(i, "x", "dx")
            t_dev = t.to(x.device, dtype=dtype_r)
            x.add_(dx * (-t_dev))

        # --- 3) Total objective and return (keep tensors, no .item() here)
        f_total = f_data + (f_reg if f_reg.device == f_data.device else f_reg.to(f_data.device, non_blocking=True))
        ws.stats.scalar_slot("E_reg_total", f_total.device, f_total.dtype).add_(f_reg)

        return f_total, gdot

    @torch.no_grad()
    def f_g(self, ws, t: float) -> Tuple[float, float]:
        """
        Compatibility wrapper: calls f_g_tensor and returns Python floats.
        Keep this outside compiled regions (only converts at the boundary).
        """
        dtype_r = _real_dtype(self.y.dtype)
        t_t = torch.tensor(t, device=self.y.device, dtype=dtype_r)
        f_t, gd_t = self.f_g_tensor(ws, t_t)
        return float(f_t.item()), float(gd_t.item())