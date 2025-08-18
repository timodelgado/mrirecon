# graspcg/ops/objective.py
from __future__ import annotations
from typing import Optional, Tuple, Dict

import contextlib
import torch


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    """Map complex -> real counterpart, else pass through."""
    if dtype is torch.complex64:
        return torch.float32
    if dtype is torch.complex128:
        return torch.float64
    return dtype


# --- small, compile‑friendly kernels (defined at module scope) ----------------
def _ker_residual_energy_(r: torch.Tensor,
                          Ax: torch.Tensor,
                          Ad: torch.Tensor,
                          y: torch.Tensor,
                          t: torch.Tensor) -> torch.Tensor:
    """
    In‑place residual + energy on device:
      r = Ax + t*Ad − y
      e = ||r||^2 (REAL 0‑D)
    """
    r.copy_(Ad)
    r.mul_(t)
    r.add_(Ax)
    r.sub_(y)
    return (r.conj() * r).real.sum()


def _ker_dot_real(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """REAL dot product (0‑D) on device: Re⟨a, b⟩"""
    return (a.conj() * b).real.sum()


def _maybe_compile(fn, enable: bool):
    if not enable or not hasattr(torch, "compile"):
        return fn
    try:
        return torch.compile(fn, fullgraph=False, dynamic=False)  # shapes per shard are static
    except Exception:
        return fn


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

    def __init__(self,
                 nufft_op,
                 y: torch.Tensor,
                 regm,
                 *,
                 compile_data_kernels: bool = True):
        self.nufft = nufft_op
        self.y     = y
        self.regm  = regm

        self._Ax   : Optional[torch.Tensor] = None   # global path
        self._Adx  : Optional[torch.Tensor] = None
        self._r    : Optional[torch.Tensor] = None
        self._use_sharded: bool = False

        # Pre‑compile the tiny pure‑tensor kernels (safe with torchkbnufft backend)
        self._ker_residual_energy = _maybe_compile(_ker_residual_energy_, compile_data_kernels)
        self._ker_dot_real        = _maybe_compile(_ker_dot_real,        compile_data_kernels)

    # ---------------------------------------------------------------- caches --
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

                # enqueue on the shard's device stream for overlap
                dev = x.device
                stream = ws.arena.stream_for(dev) if dev.type == "cuda" else None
                ctx_mgr = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
                with ctx_mgr:
                    self.nufft.A(x,  out=Ax_sh)
                    self.nufft.A(dx, out=Ad_sh)

            # ensure prior enqueues finish before LS uses them
            for sh, _ in ws.iter_shards():
                if sh.device.type == "cuda":
                    torch.cuda.synchronize(sh.device)

            self._Ax = self._Adx = self._r = None
            return

        # Global scratch path (anchored on y.device), careful with cross‑device compute
        self._Ax  = arena.request(y.numel(), y.dtype, anchor=y).view_as(y)
        self._Adx = arena.request(y.numel(), y.dtype, anchor=y).view_as(y)
        self._r   = arena.request(y.numel(), y.dtype, anchor=y).view_as(y)

        # Fill Ax/Adx in parallel across shard devices, gather to y.device
        b_off = 0
        for sh, i in ws.iter_shards():
            x  = ws.get("x", i)
            dx = ws.get("dx", i)
            B  = int(x.shape[0])
            ys = y[b_off:b_off+B]

            stream = ws.arena.stream_for(x.device) if x.device.type == "cuda" else None
            ctx_mgr = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
            with ctx_mgr:
                tmpA = arena.request(ys.numel(), y.dtype, device=x.device).view_as(ys)
                tmpD = arena.request(ys.numel(), y.dtype, device=dx.device).view_as(ys)
                self.nufft.A(x,  out=tmpA)
                self.nufft.A(dx, out=tmpD)
                self._Ax[b_off:b_off+B].copy_(tmpA.to(self._Ax.device,  non_blocking=True))
                self._Adx[b_off:b_off+B].copy_(tmpD.to(self._Adx.device, non_blocking=True))
                arena.release(tmpA); arena.release(tmpD)
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

    # --------------------------------------------------------------- f,g eval --
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

        # --- 1) Data term via caches (per‑device accum to avoid host syncs)
        f_data_dev: Dict[torch.device, torch.Tensor] = {}

        if self._use_sharded:
            b_off = 0
            for sh, i in ws.iter_shards():
                Ax, Ad, r = ws.bind(i, "Ax_sh", "Ad_sh", "r_sh")
                B = int(Ax.shape[0])
                dev = Ax.device

                # overlap via device stream
                stream = ws.arena.stream_for(dev) if dev.type == "cuda" else None
                ctx_mgr = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
                with ctx_mgr:
                    y_slice = self.y[b_off:b_off+B]
                    if y_slice.device != dev:
                        y_slice = y_slice.to(dev, non_blocking=True)
                    t_dev = t.to(dev, dtype=dtype_r)

                    # residual + energy (compiled)
                    e = self._ker_residual_energy(r, Ax, Ad, y_slice, t_dev)  # 0‑D real on dev

                    # device accum
                    if dev in f_data_dev:
                        f_data_dev[dev].add_(e)
                    else:
                        f_data_dev[dev] = e

                    # data‑grad: g_i = A^H r_i
                    g = ws.get("g", i)
                    self.nufft.AH(r, out=g)

                    b_off += B

            # reduce to a primary device
            f_data = self._reduce0d(f_data_dev, prefer_device=self.y.device, dtype=dtype_r)
            f_data.mul_(0.5)
            ws.stats.scalar_slot("E_data", f_data.device, f_data.dtype).add_(f_data)
        else:
            # Global scratch on y.device
            r = self._r
            assert r is not None and self._Ax is not None and self._Adx is not None
            t_y = t.to(r.device, dtype=dtype_r)

            e = self._ker_residual_energy(r, self._Ax, self._Adx, self.y, t_y)
            f_data = e.mul(0.5)
            ws.stats.scalar_slot("E_data", f_data.device, f_data.dtype).add_(f_data)

            # data‑grad per shard (allow per‑device streams)
            b_off = 0
            for sh, i in ws.iter_shards():
                g = ws.get("g", i)
                B = int(g.shape[0])
                dev_g = g.device
                stream = ws.arena.stream_for(dev_g) if dev_g.type == "cuda" else None
                ctx_mgr = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
                with ctx_mgr:
                    # AH must accept input slice on r.device and write to g.device
                    self.nufft.AH(r[b_off:b_off+B], out=g)
                b_off += B

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

        # --- 3) g·d including reg gradient (compute after reg added into g)
        gdot_dev: Dict[torch.device, torch.Tensor] = {}
        for sh, i in ws.iter_shards():
            g, d = ws.bind(i, "g", "dx")
            dev = g.device
            gd = self._ker_dot_real(g, d)
            if dev in gdot_dev:
                gdot_dev[dev].add_(gd)
            else:
                gdot_dev[dev] = gd
        gdot = self._reduce0d(gdot_dev, prefer_device=self.y.device, dtype=dtype_r)
        ws.stats.scalar_slot("gdot", gdot.device, gdot.dtype).add_(gdot)

        # --- 4) Total objective and return (keep tensors, no .item() here)
        f_total = f_data + (f_reg if f_reg.device == f_data.device else f_reg.to(f_data.device, non_blocking=True))
        # (E_reg_total is recorded inside RegManager)
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

    # --------------------------------------------------------------- helpers --
    @staticmethod
    def _reduce0d(devmap: Dict[torch.device, torch.Tensor],
                  *,
                  prefer_device: torch.device,
                  dtype: torch.dtype) -> torch.Tensor:
        """Reduce a map of 0‑D tensors across devices to a single 0‑D on prefer_device."""
        if not devmap:
            return torch.zeros((), device=prefer_device, dtype=dtype)
        primary = next(iter(devmap.keys()))
        if prefer_device.type == "cuda":
            primary = prefer_device
        out = torch.zeros((), device=primary, dtype=dtype)
        for dev, val in devmap.items():
            out.add_(val if dev == primary else val.to(primary, non_blocking=True))
        return out