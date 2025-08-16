from __future__ import annotations
import math
from typing import Optional, Sequence

import torch

from ..workspace.unified_arena import DeviceArena
from ..workspace.workspace     import Workspace, BufSpec
from ..ops.objective           import Objective
from ..regularization.manager  import RegManager
from ..numerics.directions     import DirPRPlus, DirDY, DirFR
from ..numerics.line_search    import search as linesearch

class CGSolver:
    def __init__(self,
                 y: torch.Tensor,
                 nufft_op,
                 regm: RegManager,
                 devices: Optional[Sequence[str | int]] = None,
                 direction: str = "prplus",
                 *,
                 ls_name: str = "wolfe",
                 c1: float = 1e-4,
                 c2: float = 0.9,
                 ls_max_iter: int = 25,
                 ls_zoom: bool = True):
        # Arena
        if devices is None:
            compute_spec, helpers = "cuda", None
        elif isinstance(devices, (list, tuple)):
            compute_spec, helpers = devices[0], list(devices[1:])
        else:
            compute_spec, helpers = devices, None
        try:
            self.arena = DeviceArena(compute=compute_spec, helpers=helpers)
        except TypeError:
            self.arena = DeviceArena(compute=compute_spec)

        self.y = y
        self.nufft_op = nufft_op
        self.regm = regm

        # Line search knobs read by numerics/line_search
        self.ls_name = ls_name
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.ls_max_iter = int(ls_max_iter)
        self.ls_zoom = bool(ls_zoom)

        CPLX = y.dtype if y.is_complex() else torch.complex64
        REAL = torch.float32

        specs = [
            BufSpec("x",    "image",  "per_shard", "image",   CPLX, init="zeros"),
            BufSpec("g",    "image",  "per_shard", "image",   CPLX, init="zeros"),
            BufSpec("dx",   "image",  "per_shard", "image",   CPLX, init="zeros"),
            BufSpec("diag", "image",  "per_shard", "spatial", REAL, init="ones"),
            # preferred per-shard line-search caches (Obj will use these if present)
            BufSpec("Ax_sh","kspace","per_shard","kspace", CPLX, init="zeros", lifetime="ls"),
            BufSpec("Ad_sh","kspace","per_shard","kspace", CPLX, init="zeros", lifetime="ls"),
            BufSpec("r_sh", "kspace","per_shard","kspace", CPLX, init="zeros", lifetime="ls"),
        ]
        if direction in ("prplus", "dy"):
            specs.append(BufSpec("g_prev", "image", "per_shard", "image", CPLX, init="zeros"))

        self.ws  = Workspace(y, nufft_op, arena=self.arena, buf_specs=specs, kspace_mode="sharded")
        self.obj = Objective(nufft_op, y, regm)

        # directions
        if direction == "prplus":
            self.dir = DirPRPlus(self.ws)
        elif direction == "dy":
            self.dir = DirDY(self.ws)
        elif direction == "fr":
            self.dir = DirFR(self.ws)
        else:
            raise ValueError(f"Unknown direction '{direction}'")

    # ---- helpers ----
    @staticmethod
    def _dot0(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a.conj() * b).real.sum()  # 0‑D REAL tensor

    def _g_dot_d0(self) -> torch.Tensor:
        out = None
        for sh, i in self.ws.iter_shards():
            v = self._dot0(self.ws.get("g", i), self.ws.get("dx", i))
            out = v if out is None else (out + v.to(out.device))
        return out if out is not None else torch.zeros((), device=self.y.device, dtype=self.y.real.dtype)

    def _x_norm0(self) -> torch.Tensor:
        out = None
        for sh, i in self.ws.iter_shards():
            v = self._dot0(self.ws.get("x", i), self.ws.get("x", i))
            out = v if out is None else (out + v.to(out.device))
        return torch.sqrt(out.clamp_min_(0)) if out is not None else torch.zeros((), device=self.y.device, dtype=self.y.real.dtype)

    def _step_norm0(self) -> torch.Tensor:
        out = None
        for sh, i in self.ws.iter_shards():
            v = self._dot0(self.ws.get("dx", i), self.ws.get("dx", i))
            out = v if out is None else (out + v.to(out.device))
        return torch.sqrt(out.clamp_min_(0)) if out is not None else torch.zeros((), device=self.y.device, dtype=self.y.real.dtype)

    @torch.no_grad()
    def _check_converged(self, *, tol_g: float, tol_step: float) -> bool:
        xnorm  = self._x_norm0()
        gdotgd = None
        for sh, i in self.ws.iter_shards():
            g = self.ws.get("g", i); D = self.ws.get("diag", i)
            v = (g.conj() * (g / D)).real.sum()
            gdotgd = v if gdotgd is None else (gdotgd + v.to(gdotgd.device))
        gnorm = torch.sqrt(gdotgd.clamp_min_(0)) if gdotgd is not None else torch.zeros_like(xnorm)

        snorm = self._step_norm0()
        # 0‑D tensor checks keep us compile-friendly
        cond_g    = (gnorm <= (tol_g   * (xnorm + 1e-12)))
        cond_step = (snorm <= (tol_step * (xnorm + 1e-12)))
        return bool((cond_g & cond_step).item())

    # ---- main loop ----
    @torch.no_grad()
    def run(self, max_iters: int = 50, tol_g: float = 1e-6, tol_step: float = 1e-6):
        ws = self.ws
        # initial gradient and preconditioned steepest descent
        self.obj.begin_linesearch(ws)
        f0, _ = self.obj.f_g(ws, t=0.0)  # fills g at current x

        for sh, i in ws.iter_shards():
            g, dx, D = ws.bind(i, "g", "dx", "diag")
            dx.copy_(g).div_(D).neg_()

        # seed direction state (e.g., g_prev for PR+/DY)
        self.dir.init_state()

        k = 0
        while k < max_iters:
            # scalars for LS
            g0d = float(self._g_dot_d0().item())
            ok, t, f_t, gdot_t = linesearch(self, f0=float(f0), g0d=g0d)  # uses numerics/line_search

            # accept step: x += t * dx (gradient at new point is already in ws.g)
            for sh, i in ws.iter_shards():
                x, d = ws.bind(i, "x", "dx")
                x.add_(d, alpha=t)

            # record a minimal history
            if getattr(ws, "stats", None) is not None:
                ws.stats.scalar_slot("f", ws.plan.shards[0].device, ws.dtype_r).add_(torch.tensor(f_t, device=ws.plan.shards[0].device, dtype=ws.dtype_r))
                ws.stats.push_history(["E_data", "E_reg_total", "gdot", "f"])

            # convergence (0‑D tensor logic)
            if self._check_converged(tol_g=tol_g, tol_step=tol_step):
                break

            # prepare next direction: caches + gradient at t=0 of new x
            self.obj.begin_linesearch(ws)
            f0, _ = self.obj.f_g(ws, t=0.0)

            # β update and new search direction
            self.dir.update_inplace(ws)

            k += 1

        self.obj.end_linesearch(ws)

    def result(self) -> torch.Tensor:
        return self.ws.concat("x").detach()
