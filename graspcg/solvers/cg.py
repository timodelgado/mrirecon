# graspcg/solvers/cg.py
from __future__ import annotations
import math, torch
from typing import Sequence, Optional, Mapping

from ..workspace.device_cfg    import DeviceCfg
from ..workspace.cg_workspace  import CGWorkspace
from ..ops.objective           import Objective
from ..ops.preconditioner      import build_precond_diag
from ..ops.init_scaling        import initial_backproj_and_scaling
from ..numerics                import line_search, directions
from ..numerics.continuation   import ContinuationManager, ContinuationConfig
from ..utils.operations        import dot_chunked


class CGSolver:
    """
    Conjugate‑Gradient driver cohesive with the minimal‑buffer workspace,
    arena, objective, RegManager, and continuation policy.

    Parameters
    ----------
    nufft_op : NUFFT operator exposing .A(image[, out]) and .AH(kspace[, out])
    y        : measured k-space (already sharded by frames)
    regm     : RegManager instance (owns regularisers, policies)
    devices  : None | int|str | sequence[int|str]
    line_search : "wolfe" | "armijo"
    direction   : "prplus" | "dy" | "fr"
    max_iter    : int
    tol_rel     : relative step tolerance
    tol_abs     : absolute step tolerance
    c1, c2      : Wolfe/Armijo parameters (c2 ignored for armijo)
    ls_max_iter : max bracket/zoom iterations
    ls_zoom     : enable Wolfe zoom phase
    xfactor     : initial matched‑filter scaling factor
    stats_cfg   : per‑regulariser stats policy for init (see init_scaling.py)
    continuation: ContinuationManager (optional)
    verbose     : bool
    """

    def __init__(self,
                 nufft_op,
                 y: torch.Tensor,
                 *,
                 regm,
                 devices: int|str|Sequence[int|str]|None = None,
                 line_search: str = "wolfe",
                 direction: str   = "prplus",
                 max_iter: int    = 30,
                 tol_rel: float   = 1e-6,
                 tol_abs: float   = 0.0,
                 c1: float        = 1e-4,
                 c2: float        = 0.9,
                 ls_max_iter: int = 20,
                 ls_zoom: bool    = True,
                 xfactor: float   = 1.0,
                 stats_cfg: Mapping[str, Mapping[str, float|bool]] | None = None,
                 continuation: ContinuationManager | None = None,
                 verbose: bool    = False):

        # Device config + workspace
        if devices is None:
            device_cfg = DeviceCfg()
        else:
            if isinstance(devices, (int, str)):
                devices = [devices]
            device_cfg = DeviceCfg(compute=devices[0], helpers=list(devices[1:]))

        self.ws   = CGWorkspace(y, nufft_op, device_cfg=device_cfg)
        self.nuf  = nufft_op
        self.y    = y
        self.regm = regm

        # Objective (stateless; uses ws + regm)
        self.obj      = Objective(nufft_op, y, self.regm)
        self.dir_obj  = directions.build_direction(direction, self.ws)

        # Line‑search config
        self.ls_name     = line_search
        self.c1          = float(c1)
        self.c2          = 0.0 if line_search == "armijo" else float(c2)
        self.ls_max_iter = int(ls_max_iter)
        self.ls_zoom     = bool(ls_zoom)
        self.t_init      = 1.0

        # Stopping & reporting
        self.max_iter = int(max_iter)
        self.tol_rel  = float(tol_rel)
        self.tol_abs  = float(tol_abs)
        self.verbose  = bool(verbose)

        # Init scaling knobs
        self.xfactor   = float(xfactor)
        self.stats_cfg = stats_cfg or {}

        # Continuation
        self.cont = continuation  # may be None

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def run(self) -> torch.Tensor:
        ws = self.ws

        # 0) Initial back‑projection & robust scaling (in place; no big allocs)
        initial_backproj_and_scaling(
            ws, self.regm,
            xfactor=self.xfactor,
            stats_cfg=self.stats_cfg,
            verbose=self.verbose,
        )

        # 1) Preconditioner diag (data term + reg terms) in place per shard
        #    NOTE: build_precond_diag must accept `regm` and query it for add_diag.
        build_precond_diag(ws, regm=self.regm, mode="full")

        # 2) first f, g, and initial search direction d0 = -M^{-1} g0
        #    Objective uses workspace buffers; no extra large allocations.
        self.obj.begin_linesearch(ws)           # set up cached residual & A(dx)
        f0, _ = self.obj.f_g(ws, t=0.0)        # fills ws.g at current x
        for sh, _ in ws.iter_shards():
            sh.dx.copy_(sh.g).div_(sh.diag).neg_()   # preconditioned steepest descent
        g0d = self._g_dot_d()
        self.obj.end_linesearch(ws)             # keep memory tidy

        # Optional: let Objective reuse ws.dx as its direction buffer
        if hasattr(self.obj, "bind_direction"):
            self.obj.bind_direction(ws.dx, ws)

        # 3) Main CG loop ----------------------------------------------------
        for k in range(self.max_iter):
            # Evaluate along d via line‑search (objective provides f,g via caches)
            self.obj.begin_linesearch(ws)
            ok, t, f_new, _ = line_search.search(self, f0, g0d)
            self.obj.end_linesearch(ws)

            if not ok:
                if self.verbose:
                    print("[CG] line‑search failed; stopping.")
                break

            # Accept step: x <- x + t d (in place per shard)
            for sh, _ in ws.iter_shards():
                sh.x.add_(sh.dx, alpha=t)
            f0 = f_new
            self.t_init = t  # good next guess

            # Continuation: update (eps, lambda) using current x/s if configured
            if self.cont is not None:
                changed = False
                # Prefer (ws, regm, k_iter) signature; fall back to old if needed
                try:
                    changed = self.cont.maybe_update(ws, self.regm, k_iter=k+1)
                except TypeError:
                    changed = self.cont.maybe_update(ws, k_iter=k+1)
                if changed:
                    build_precond_diag(ws, regm=self.regm, mode="update")

            # New direction (PR+/DY/FR) — mutate ws.dx in place
            g0d = self.dir_obj.update_inplace(ws)

            # Stopping on step size
            if self._step_norm() <= max(self.tol_abs, self.tol_rel * self._x_norm()):
                if self.verbose:
                    print(f"[CG] converged at iter {k+1}")
                break

        # Return reconstructed image (single‑tensor or first shard for tests)
        try:
            return ws.x.detach()
        except AttributeError:
            sh, _ = next(ws.iter_shards())
            return sh.x.detach()

    # ──────────────────────────────────────────────────────────────────────
    # Small helpers: all use arena‑aware dot to avoid transient megatensors
    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _g_dot_d(self) -> float:
        s = 0.0
        for sh, _ in self.ws.iter_shards():
            # complex‑safe inner product; real scalar
            val = dot_chunked(sh.g, sh.dx, arena=self.ws.arena)
            s += float(val.real.item())
        return s

    @torch.no_grad()
    def _x_norm(self) -> float:
        s = 0.0
        for sh, _ in self.ws.iter_shards():
            s += float(dot_chunked(sh.x, sh.x, arena=self.ws.arena).real.item())
        return math.sqrt(s)

    @torch.no_grad()
    def _step_norm(self) -> float:
        s = 0.0
        for sh, _ in self.ws.iter_shards():
            s += float(dot_chunked(sh.dx, sh.dx, arena=self.ws.arena).real.item())
        return math.sqrt(s)
