from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import contextlib
import torch

# small helpers already in your tree
from ..numerics.utils import reduce0d, dot_real0  # 0-D reductions & real dot

def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype is torch.complex64:  return torch.float32
    if dtype is torch.complex128: return torch.float64
    return dtype


@dataclass
class ObjectiveConfig:
    """
    Solver-agnostic data objective (L2 term) config.

    • use_sharded_caches : prefer Workspace per-shard k-space buffers
                           ('fwd_var_sh', 'fwd_dir_sh', 'resid_sh').
    • compile_kernels    : try torch.compile on tiny residual/energy kernels.
    """
    use_sharded_caches: bool = True
    compile_kernels: bool = True


# --------------------- tiny, compile-friendly kernels -------------------------

def _ker_residual_energy_(r: torch.Tensor,
                          Ax: torch.Tensor,
                          Ad: torch.Tensor,
                          y: torch.Tensor,
                          t: torch.Tensor) -> torch.Tensor:
    """
    In-place residual + energy on device.
      r = Ax + t*Ad − y     (shape (Bsh, C, K))
      return ||r||^2  (0-D REAL tensor on r.device)
    """
    r.copy_(Ad)
    r.mul_(t)
    r.add_(Ax)
    r.sub_(y)
    return (r.conj() * r).real.sum()

def _maybe_compile(fn, enable: bool):
    if not enable or not hasattr(torch, "compile"):
        return fn
    try:
        return torch.compile(fn, fullgraph=False, dynamic=False)
    except Exception:
        return fn


# -------------------------------- Objective -----------------------------------

class Objective:
    """
    L2 data term with sharded fast path and generic stats.

    Lifecycle:
      begin_linesearch(ws)   -> precompute Ax, Ad into caches
      f_g_tensor(ws, t)      -> evaluate φ(t) and slope; accumulates grad in ws
      end_linesearch(ws)     -> release global scratch (if allocated)

    Contracts
    ---------
    • Operator provides A/AH with out= (NUFFT or wrapper).
    • Workspace may provide per‑shard caches:
        'fwd_var_sh' (Ax), 'fwd_dir_sh' (Ad), 'resid_sh' (r).
    • Buffers: primary 'var/dir/grad' with legacy fallback 'x/dx/g'.
    • Stats: writes both generic and legacy keys.
    """

    _BX = "fwd_var_sh"   # Ax
    _BD = "fwd_dir_sh"   # Ad
    _BR = "resid_sh"     # r

    def __init__(self,
                 nufft_op,
                 y: torch.Tensor,
                 regm,
                 cfg: Optional[ObjectiveConfig] = None):
        self.nufft = nufft_op
        self.y     = y
        self.regm  = regm
        self.cfg   = cfg or ObjectiveConfig()

        # global scratch (used only if sharded caches not present)
        self._Ax_g  : Optional[torch.Tensor] = None
        self._Ad_g  : Optional[torch.Tensor] = None
        self._r_g   : Optional[torch.Tensor] = None
        self._use_sharded: bool = False

        # compile the tiny pure-tensor kernel
        self._ker_residual_energy = _maybe_compile(_ker_residual_energy_, self.cfg.compile_kernels)

    # Legacy hook removed intentionally
    def set_weight(self, *_args, **_kw):
        raise NotImplementedError("Residual weighting was removed. Fold DCF/noise into A/AH.")

    # ---------------------------- line-search hooks ----------------------------

    def _get(self, ws, i, primary: str, legacy: str):
        if hasattr(ws, "has") and ws.has(primary):
            return ws.get(primary, i)
        return ws.get(legacy, i)

    def _bind_pair(self, ws, i, p1: str, l1: str, p2: str, l2: str):
        return self._get(ws, i, p1, l1), self._get(ws, i, p2, l2)

    @torch.no_grad()
    def begin_linesearch(self, ws) -> None:
        """
        Precompute Ax and Ad into k-space caches. Prefer per-shard caches
        supplied by the Workspace; otherwise use a global scratch anchored on
        y.device and gather per-shard results.
        """
        arena = ws.arena
        y     = self.y

        have_sharded = bool(self.cfg.use_sharded_caches) and all(
            hasattr(ws, "list_bufs") and (name in ws.list_bufs()) for name in (self._BX, self._BD, self._BR)
        )
        self._use_sharded = bool(have_sharded)

        if self._use_sharded:
            # Populate per-shard forward caches from current var and dir
            for sh, i in ws.iter_shards():
                x, dx        = self._bind_pair(ws, i, "var", "x", "dir", "dx")
                Ax_sh, Ad_sh = ws.bind(i, self._BX, self._BD)

                dev = x.device
                stream = ws.arena.stream_for(dev) if dev.type == "cuda" else None
                ctx_mgr = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
                with ctx_mgr:
                    self.nufft.A(x,  out=Ax_sh)
                    self.nufft.A(dx, out=Ad_sh)

            for sh, _ in ws.iter_shards():
                if sh.device.type == "cuda":
                    torch.cuda.synchronize(sh.device)

            self._Ax_g = self._Ad_g = self._r_g = None
            return

        # ---- global scratch fallback ----
        self._Ax_g = arena.request(y.numel(), y.dtype, anchor=y).view_as(y)
        self._Ad_g = arena.request(y.numel(), y.dtype, anchor=y).view_as(y)
        self._r_g  = arena.request(y.numel(), y.dtype, anchor=y).view_as(y)

        b_off = 0
        for sh, i in ws.iter_shards():
            x, dx = self._bind_pair(ws, i, "var", "x", "dir", "dx")
            B  = int(x.shape[0])
            ys = y[b_off:b_off+B]

            stream = ws.arena.stream_for(x.device) if x.device.type == "cuda" else None
            ctx_mgr = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
            with ctx_mgr:
                tmpA = arena.request(ys.numel(), y.dtype, device=x.device).view_as(ys)
                tmpD = arena.request(ys.numel(), y.dtype, device=dx.device).view_as(ys)
                self.nufft.A(x,  out=tmpA)
                self.nufft.A(dx, out=tmpD)
                self._Ax_g[b_off:b_off+B].copy_(tmpA.to(self._Ax_g.device,  non_blocking=True))
                self._Ad_g[b_off:b_off+B].copy_(tmpD.to(self._Ad_g.device, non_blocking=True))
                arena.release(tmpA); arena.release(tmpD)
            b_off += B

    @torch.no_grad()
    def end_linesearch(self, ws) -> None:
        """Release global scratch (no‑op for sharded fast‑path)."""
        if self._use_sharded:
            return
        arena = ws.arena
        if self._Ax_g is not None:  arena.release(self._Ax_g);  self._Ax_g  = None
        if self._Ad_g is not None:  arena.release(self._Ad_g);  self._Ad_g  = None
        if self._r_g is not None:   arena.release(self._r_g);   self._r_g   = None

    # ---------------------------- objective & slope ----------------------------

    @torch.no_grad()
    def f_g_tensor(self, ws, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate φ(t) and φ'(t) at x + t·dx (var + t·dir).
        Returns (f_total, gdot) as 0‑D REAL tensors; fills per‑shard grad.
        """
        # 0) clear gradient (both new/legacy when present)
        for _, i in ws.iter_shards():
            if hasattr(ws, "has") and ws.has("grad"):
                ws.get("grad", i).zero_()
            if hasattr(ws, "has") and ws.has("g"):
                ws.get("g", i).zero_()

        dtype_r = _real_dtype(self.y.dtype)
        sb = getattr(ws, "stats", None)

        # --- 1) Data term
        f_data_dev: Dict[torch.device, torch.Tensor] = {}

        if self._use_sharded:
            b_off = 0
            for sh, i in ws.iter_shards():
                Ax, Ad, r = ws.bind(i, self._BX, self._BD, self._BR)
                B = int(Ax.shape[0])
                dev = Ax.device

                stream = ws.arena.stream_for(dev) if dev.type == "cuda" else None
                ctx_mgr = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
                with ctx_mgr:
                    y_slice = self.y[b_off:b_off+B]
                    if y_slice.device != dev:
                        y_slice = y_slice.to(dev, non_blocking=True)
                    t_dev = t.to(dev, dtype=dtype_r)

                    e = self._ker_residual_energy(r, Ax, Ad, y_slice, t_dev)
                    if dev in f_data_dev:
                        f_data_dev[dev].add_(e)
                    else:
                        f_data_dev[dev] = e

                    # gradient of data term: grad_i = A^H r_i
                    g_buf = ws.get("grad", i) if (hasattr(ws, "has") and ws.has("grad")) else ws.get("g", i)
                    self.nufft.AH(r, out=g_buf)

                    b_off += B

            f_data = reduce0d(f_data_dev, prefer_device=t.device, dtype=dtype_r).mul_(0.5)
        else:
            # Global scratch (y.device)
            r = self._r_g
            assert r is not None and self._Ax_g is not None and self._Ad_g is not None
            t_y = t.to(r.device, dtype=dtype_r)
            f_data = self._ker_residual_energy(r, self._Ax_g, self._Ad_g, self.y, t_y).mul(0.5)

            # data-grad per shard
            b_off = 0
            for sh, i in ws.iter_shards():
                g_buf = ws.get("grad", i) if (hasattr(ws, "has") and ws.has("grad")) else ws.get("g", i)
                B = int(g_buf.shape[0])
                dev_g = g_buf.device
                stream = ws.arena.stream_for(dev_g) if dev_g.type == "cuda" else None
                ctx_mgr = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
                with ctx_mgr:
                    self.nufft.AH(r[b_off:b_off+B], out=g_buf)
                b_off += B

        # stats (generic + legacy)
        if sb is not None:
            slot = sb.scalar_slot
            slot("term/data/energy", f_data.device, f_data.dtype).add_(f_data)
            slot("E_data",           f_data.device, f_data.dtype).add_(f_data)  # legacy

        # --- 2) Regularization at x + t·dx
        for _, i in ws.iter_shards():
            x, d = self._bind_pair(ws, i, "var", "x", "dir", "dx")
            t_dev = t.to(x.device, dtype=dtype_r)
            x.add_(d * t_dev)

        f_reg = self.regm.energy_and_grad(ws)  # regs add into grad/g

        for _, i in ws.iter_shards():
            x, d = self._bind_pair(ws, i, "var", "x", "dir", "dx")
            t_dev = t.to(x.device, dtype=dtype_r)
            x.add_(d * (-t_dev))

        # --- 3) g·d after reg added into grad
        gdot_dev: Dict[torch.device, torch.Tensor] = {}
        for _, i in ws.iter_shards():
            g = self._get(ws, i, "grad", "g")
            d = self._get(ws, i, "dir",  "dx")
            dev = g.device
            gd  = dot_real0(g, d)   # 0‑D REAL on dev
            if dev in gdot_dev: gdot_dev[dev].add_(gd)
            else:               gdot_dev[dev] = gd
        gdot = reduce0d(gdot_dev, prefer_device=ws.primary_device, dtype=dtype_r)

        if sb is not None:
            slot = sb.scalar_slot
            slot("slope/dir", gdot.device, gdot.dtype).add_(gdot)
            slot("gdot",      gdot.device, gdot.dtype).add_(gdot)  # legacy

        # --- 4) total objective
        f_total = f_data + (f_reg if f_reg.device == f_data.device else f_reg.to(f_data.device, non_blocking=True))
        if sb is not None:
            slot = sb.scalar_slot
            slot("obj/total", f_total.device, f_total.dtype).add_(f_total)
            slot("f_total",   f_total.device, f_total.dtype).add_(f_total)  # legacy

        return f_total.to(t.device), gdot.to(t.device)

    @torch.no_grad()
    def f_g(self, ws, t: float) -> Tuple[float, float]:
        """Compatibility wrapper returning Python floats (kept outside compiled regions)."""
        dtype_r = _real_dtype(self.y.dtype)
        t_t = torch.tensor(t, device=self.y.device, dtype=dtype_r)
        f_t, gd_t = self.f_g_tensor(ws, t_t)
        return float(f_t.item()), float(gd_t.item())
