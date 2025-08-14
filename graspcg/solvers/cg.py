from __future__ import annotations
import math
from typing import Optional, Sequence

import torch

from ..workspace.unified_arena import DeviceArena
from ..workspace.workspace     import Workspace, BufSpec
from ..ops.objective           import Objective
from ..regularization.reg_manager import RegManager
# If you have named direction classes, import them here


class CGSolver:
    def __init__(self,
                 y: torch.Tensor,
                 nufft_op,
                 regm: RegManager,
                 devices: Optional[Sequence[str | int]] = None,
                 direction: str = "prplus"):
        # DeviceArena: accept None | single | list
        if devices is None:
            compute_spec, helpers = "cuda", None
        elif isinstance(devices, (list, tuple)):
            compute_spec = devices[0]
            helpers      = list(devices[1:])
        else:
            compute_spec, helpers = devices, None
        try:
            self.arena = DeviceArena(compute=compute_spec, helpers=helpers)
        except TypeError:
            self.arena = DeviceArena(compute=compute_spec)

        self.y = y
        self.nufft_op = nufft_op
        self.regm = regm
        self.direction = direction

        CPLX = y.dtype if y.is_complex() else torch.complex64
        REAL = torch.float32

        specs = [
            BufSpec("x",      "image",  "per_shard", "image",   CPLX, init="zeros"),
            BufSpec("g",      "image",  "per_shard", "image",   CPLX, init="zeros"),
            BufSpec("dx",     "image",  "per_shard", "image",   CPLX, init="zeros"),
            BufSpec("diag",   "image",  "per_shard", "spatial", REAL, init="ones"),
            # Per-shard k-space caches (preferred; Objective will use these)
            BufSpec("Ax_sh","kspace","per_shard","kspace", CPLX, init="zeros", lifetime="ls"),
            BufSpec("Ad_sh","kspace","per_shard","kspace", CPLX, init="zeros", lifetime="ls"),
            BufSpec("r_sh", "kspace","per_shard","kspace", CPLX, init="zeros", lifetime="ls"),
        ]
        if direction in ("prplus", "dy"):
            specs.append(BufSpec("g_prev", "image", "per_shard", "image", CPLX, init="zeros"))

        self.ws  = Workspace(y, nufft_op, arena=self.arena, buf_specs=specs, kspace_mode="sharded")
        self.obj = Objective(nufft_op, y, regm)

    # --- helpers used by tests/solver bookkeeping ---
    def _g_dot_d(self) -> float:
        s = 0.0
        for sh, i in self.ws.iter_shards():
            g, dx = self.ws.bind(i, "g", "dx")
            s += float(self._dot(g, dx))
        return s

    def _x_norm(self) -> float:
        s = 0.0
        for sh, i in self.ws.iter_shards():
            x = self.ws.get("x", i)
            s += float(self._dot(x, x))
        return math.sqrt(s)

    def _step_norm(self) -> float:
        s = 0.0
        for sh, i in self.ws.iter_shards():
            dx = self.ws.get("dx", i)
            s += float(self._dot(dx, dx))
        return math.sqrt(s)

    @staticmethod
    def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a.conj() * b).real.sum()

    # --- one-iteration skeleton (adapt to your loop/line-search) ---
    def run_one(self):
        ws = self.ws
        self.obj.begin_linesearch(ws)
        f0, _ = self.obj.f_g(ws, t=0.0)  # fills g at current x

        # preconditioned steepest descent initial direction
        for sh, i in ws.iter_shards():
            g, dx, D = ws.bind(i, "g", "dx", "diag")
            dx.copy_(g).div_(D).neg_()

        # ... line-search + accept step would go here ...
        self.obj.end_linesearch(ws)

    def result(self) -> torch.Tensor:
        return self.ws.concat("x").detach()