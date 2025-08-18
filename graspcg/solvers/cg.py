# graspcg/solvers/cg.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch

# Workspace / arena
from ..workspace.unified_arena import DeviceArena
from ..workspace.workspace import Workspace, BufSpec

# Objective / regs / policy / stats
from ..ops.objective import Objective
from ..regularization.manager import RegManager
from ..regularization.policy import RegPolicy, RegPolicyConfig   # shim or real path
from ..regularization.stats_board import StatsBoard

# Directions + line search (compile‑friendly numerics live outside the solver)
from ..numerics.directions import DirPRPlus, DirDY, DirFR
from ..numerics.line_search import search as line_search  # takes only (solver)

# ---------------------------------------------------------------------------

@dataclass
class CGConfig:
    # devices: compute + helpers specification accepted by DeviceArena
    devices: Optional[torch.device | str | int | list | tuple] = None
    # direction: "fr" (default), "prplus", or "dy"
    direction: str = "fr"
    # line-search
    ls_name: str = "armijo"   # "armijo" (default) or "wolfe"
    c1: float = 1e-4
    c2: float = 0.9
    ls_max_iter: int = 20
    ls_zoom: bool = True
    # iterations / tolerances
    max_iter: int = 50
    tol_g: float = 1e-4
    tol_step: float = 1e-9
    # init
    init_mode: str = "keep"   # "keep", "zero", "backproj"
    backproj_xfactor: float = 1.0
    # logging
    record_history: bool = True
    verbose: bool = False

# ---------------------------------------------------------------------------

class CGSolver:
    """
    Preconditioned non‑linear CG solver (North‑Star).

    Orchestration only:
      • Compiled numerics live in Objective (data) and RegManager/Regularizers (reg).
      • Continuation (ε/λ) is solver‑agnostic via RegPolicy with scalar‑only StatsBoard.
      • Multi‑device friendly; reductions kept as 0‑D tensors until history() is read.
    """

    def __init__(self,
                 y: torch.Tensor,
                 nufft_op,
                 regm: RegManager,
                 cfg: Optional[CGConfig] = None):
        self.cfg = cfg or CGConfig()
        self.y = y
        self.nufft_op = nufft_op
        self.regm = regm

        # ------------------ arena ------------------
        devices = self.cfg.devices
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

        # ------------------ manifest ------------------
        CPLX = y.dtype if y.is_complex() else torch.complex64
        REAL = torch.float32
        specs = [
            BufSpec("x",    "image",  "per_shard", "image",   CPLX, init="zeros"),
            BufSpec("g",    "image",  "per_shard", "image",   CPLX, init="zeros"),
            BufSpec("dx",   "image",  "per_shard", "image",   CPLX, init="zeros"),
            BufSpec("diag", "image",  "per_shard", "spatial", REAL, init="ones"),
            # Preferred per‑shard line‑search caches (Objective will use them)
            BufSpec("Ax_sh","kspace","per_shard","kspace", CPLX, init="zeros", lifetime="ls"),
            BufSpec("Ad_sh","kspace","per_shard","kspace", CPLX, init="zeros", lifetime="ls"),
            BufSpec("r_sh", "kspace","per_shard","kspace", CPLX, init="zeros", lifetime="ls"),  # fixed dist
        ]
        # PR+ / DY need g_prev
        if self.cfg.direction.lower() in ("prplus", "dy"):
            specs.append(BufSpec("g_prev", "image", "per_shard", "image", CPLX, init="zeros"))

        self.ws = Workspace(y, nufft_op, arena=self.arena, buf_specs=specs, kspace_mode="sharded")

        # Ensure a StatsBoard so Objective/RegManager can log scalar slots
        if not hasattr(self.ws, "stats"):
            self.ws.stats = StatsBoard()

        # Objective and policy
        self.obj = Objective(nufft_op, y, regm)  # uses per‑shard caches if present
        self.policy = RegPolicy(RegPolicyConfig())

        # Direction object
        dir_name = self.cfg.direction.lower()
        self.dir = {"fr": DirFR, "prplus": DirPRPlus, "dy": DirDY}[dir_name](self.ws)

        # State
        self.iter = 0
        self._g0_norm: Optional[float] = None
        self._history: list[Dict[str, Any]] = []

        # Line‑search knobs read by numerics/line_search.search
        self.ls_name     = self.cfg.ls_name
        self.c1          = float(self.cfg.c1)
        self.c2          = float(self.cfg.c2)
        self.ls_zoom     = bool(self.cfg.ls_zoom)
        self.ls_max_iter = int(self.cfg.ls_max_iter)

        # Preconditioner seed from regularizers (diagonal contribution)
        try:
            self.regm.add_diag(self.ws)  # if available, adds into ws.diag
        except Exception:
            pass

        # Optional pilot / scaling
        if self.cfg.init_mode == "backproj":
            try:
                from ..ops.init_scaling import initial_backproj_and_scaling
                initial_backproj_and_scaling(
                    self.ws, self.regm, xfactor=self.cfg.backproj_xfactor,
                    verbose=self.cfg.verbose
                )
            except Exception:
                # fall back to zeros/keep
                pass
        elif self.cfg.init_mode == "zero":
            for sh, i in self.ws.iter_shards():
                self.ws.get("x", i).zero_()
        # "keep" -> assume caller filled ws.x

        # Pre‑allocate scalar slots (stable for compile & avoids first‑use overhead)
        try:
            q = getattr(self.policy.cfg, "eps_percentile", 0.90)
            self._preallocate_stats(percentiles=(q,))
        except Exception:
            pass

    # ------------------ helpers ------------------

    @staticmethod
    def _dot0(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """0‑D REAL, device of a."""
        return (a.conj() * b).real.sum()

    def _g_dot_d0(self) -> torch.Tensor:
        out = None
        for sh, i in self.ws.iter_shards():
            v = self._dot0(self.ws.get("g", i), self.ws.get("dx", i))
            out = v if out is None else (out + v.to(out.device))
        if out is None:
            out = torch.zeros((), device=self.y.device, dtype=self.y.real.dtype)
        return out

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
        """
        Uses 0‑D torch tensors for the tests; returns a Python bool for loop control.
        """
        # ||x||, ||dx||
        xnorm  = self._x_norm0()
        snorm  = self._step_norm0()

        # ||g||_{M^{-1}} = sqrt(sum Re( g* (g / D) ))
        gsum = None
        for sh, i in self.ws.iter_shards():
            g = self.ws.get("g", i); D = self.ws.get("diag", i)
            v = (g.conj() * (g / D)).real.sum()
            gsum = v if gsum is None else (gsum + v.to(gsum.device))
        gnorm = torch.sqrt(gsum.clamp_min_(0)) if gsum is not None else torch.zeros_like(xnorm)

        # thresholds
        t_x = 1.0 + xnorm
        ok_g    = (gnorm <= (t_x * tol_g))
        ok_step = (snorm <= (t_x * tol_step))
        return bool((ok_g & ok_step).item())

    # ------------------ main loop ------------------

    @torch.no_grad()
    def run(self,
            max_iter: Optional[int] = None,
            tol_g: Optional[float] = None,
            tol_step: Optional[float] = None) -> "CGSolver":
        """
        Execute the CG iterations with Armijo (default) or Wolfe line‑search.
        Continuation is applied after accepting a step, on stats collected at the new iterate.
        """
        max_iter = int(max_iter or self.cfg.max_iter)
        tol_g    = float(tol_g if tol_g is not None else self.cfg.tol_g)
        tol_step = float(tol_step if tol_step is not None else self.cfg.tol_step)

        ws, obj = self.ws, self.obj
        dtype_r = self.y.real.dtype

        # --- prime gradient at t=0 and set steepest‑descent direction
        obj.begin_linesearch(ws)  # per‑shard caches preferred
        _f0, _g0d = obj.f_g_tensor(ws, torch.zeros((), device=self.y.device, dtype=dtype_r))  # fills ws.g

        for sh, i in ws.iter_shards():
            g, d, D = ws.bind(i, "g", "dx", "diag")
            d.copy_(g).div_(D).neg_()

        # Seed direction state (e.g., g_prev for PR⁺/DY)
        try:
            self.dir.init_state()
        except Exception:
            pass
        obj.end_linesearch(ws)

        # ----------------- iteration loop -----------------
        for k in range(max_iter):
            self.iter = k

            # ---- line‑search (objective keeps heavy ops on device) ----
            ok, t, f_t, gdot_t = line_search(self)  # ok may be bool or 0‑D tensor
            ok_bool = bool(ok.item()) if isinstance(ok, torch.Tensor) else bool(ok)

            if not ok_bool:
                if self.cfg.verbose:
                    print(f"[cg] line‑search failed at iter {k}")
                self._record_history_step(k, f_t, gdot_t, t)
                break

            # Accept step: x <- x + t·dx
            for sh, i in ws.iter_shards():
                x, d = ws.bind(i, "x", "dx")
                x.add_(d * t.to(x.device, dtype=d.real.dtype))

            # >>> Enable stats collection for the new iterate (TV quantile, per‑reg energies, etc.)
            try:
                self.policy.prepare_collection(ws, self.regm)
            except Exception:
                pass

            # Refresh gradient at new x for convergence and next direction (collectors are now on)
            obj.begin_linesearch(ws)
            _f0, _g0d = obj.f_g_tensor(ws, torch.zeros((), device=self.y.device, dtype=dtype_r))  # fills ws.g again
            obj.end_linesearch(ws)

            # >>> Apply continuation updates based on the just‑collected stats; refresh diag if needed
            try:
                changed = self.policy.update_from_stats(ws, self.regm)
                if changed:
                    self.regm.add_diag(ws)  # cheap refresh (adds tiny scalars per shard)
            except Exception:
                pass

            # Convergence check (0‑D tensors)
            if self._check_converged(tol_g=tol_g, tol_step=tol_step):
                if self.cfg.verbose:
                    print(f"[cg] converged at iter {k}")
                self._record_history_step(k, f_t, gdot_t, t)
                break

            # Direction update (in‑place on ws.dx)
            try:
                self.dir.update_inplace(ws)  # uses ws.g/ws.diag (and ws.g_prev if needed)
            except Exception:
                # fallback: steepest descent if direction update fails
                for sh, i in ws.iter_shards():
                    g, d, D = ws.bind(i, "g", "dx", "diag")
                    d.copy_(g).div_(D).neg_()

            # Record stats/history
            self._record_history_step(k, f_t, gdot_t, t)
            if hasattr(self.ws.stats, "record_step"):
                try:
                    self.ws.stats.record_step()
                except Exception:
                    pass

        return self

    # ------------------ results / history ------------------

    @torch.no_grad()
    def result(self) -> torch.Tensor:
        """Concatenate shards of x."""
        return self.ws.concat("x").detach()

    @torch.no_grad()
    def history(self) -> list[Dict[str, float]]:
        """Return history as Python floats (conversion happens here, not during the run)."""
        out: list[Dict[str, float]] = []
        for rec in self._history:
            out.append({k: (float(v.item()) if isinstance(v, torch.Tensor) else float(v))
                        for k, v in rec.items()})
        return out

    # ------------------ logging ------------------
    # call from CGSolver.__init__ after self.ws, self.regm exist
    def _preallocate_stats(self, percentiles=(0.90,)):
        sb   = self.ws.stats
        # Prefer explicit list of devices from arena if available
        devs = []
        try:
            devs = list(self.arena.cuda_devices())
        except Exception:
            pass
        if not devs:
            try:
                devs = [self.arena.compute_device()]
            except Exception:
                devs = [self.y.device]
        rdt  = getattr(self.ws, "dtype_r", self.y.real.dtype)

        # Always-used scalars
        base_keys = ["E_data", "E_reg_total", "gdot", "f_total"]
        for dev in devs:
            for k in base_keys:
                sb.scalar_slot(k, dev, rdt)

        # Per-regularizer energy and TV quantiles (single scalar per reg)
        for reg in getattr(self.regm, "_regs", []):
            for dev in devs:
                sb.scalar_slot(f"E_reg/{reg.name}", dev, rdt)  # optional per-reg energy
                sb.scalar_slot(f"tv_q/{reg.name}",   dev, rdt)  # percentile scalar used by policy

    @torch.no_grad()
    def _record_history_step(self, k: int, f_t: torch.Tensor, gdot_t: torch.Tensor, t: torch.Tensor) -> None:
        # Optional local bookkeeping (stay as 0‑D tensors)
        try:
            xnorm  = self._x_norm0()
            snorm  = self._step_norm0()
            gsum = None
            for sh, i in self.ws.iter_shards():
                g = self.ws.get("g", i); D = self.ws.get("diag", i)
                v = (g.conj() * (g / D)).real.sum()
                gsum = v if gsum is None else (gsum + v.to(gsum.device))
            gnorm = torch.sqrt(gsum.clamp_min_(0)) if gsum is not None else torch.zeros_like(xnorm)
        except Exception:
            xnorm = torch.tensor(0.0, device=self.y.device, dtype=self.y.real.dtype)
            snorm = torch.tensor(0.0, device=self.y.device, dtype=self.y.real.dtype)
            gnorm = torch.tensor(0.0, device=self.y.device, dtype=self.y.real.dtype)

        rec = dict(iter=k, f=f_t, gdot=gdot_t, step=t, xnorm=xnorm, gnorm=gnorm, stepnorm=snorm)
        if self.cfg.record_history:
            self._history.append(rec)

        # StatsBoard history (if available). Keep on‑device until the user reads it.
        if getattr(self.ws, "stats", None) is not None:
            try:
                self.ws.stats.scalar_slot("f_total", f_t.device, f_t.dtype).add_(f_t)
            except Exception:
                pass