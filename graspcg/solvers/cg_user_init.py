# graspcg/cg_user_init.py
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch

# Core building blocks from your codebase
from graspcg.workspace.unified_arena import UnifiedArena
from graspcg.ops.objective import Objective
from graspcg.ops.init_scaling import initial_backproj_and_scaling
from graspcg.regularization.preconditioner import build_precond_diag
from graspcg.numerics.directions import build_direction
from graspcg.numerics import line_search


# ───────────────────────────────────────────────────────────────────────────────
# Public config with safe, sensible defaults
# ───────────────────────────────────────────────────────────────────────────────

@dataclass
class CGConfig:
    # Iteration control
    max_iter: int = 50
    tol_rel: float = 1e-4            # relative step tolerance ||t·d|| / (||x|| + 1e-12)
    tol_abs: float = 0.0             # absolute step tolerance (optional)
    log_every: int = 1               # iteration logging cadence

    # Direction & line search
    dir_name: str = "prplus"         # {"prplus", "dy", "fr"} from graspcg.numerics.directions
    ls_name: str = "wolfe"           # {"wolfe", "armijo"}
    c1: float = 1e-4                 # Armijo
    c2: float = 0.9                  # Strong Wolfe (ignored by Armijo)
    ls_max_iter: int = 25
    ls_zoom: bool = True

    # Initialization
    xfactor: float = 1.0             # scale so ||A x||^2 ≈ xfactor * ||y||^2
    precond_mode: str = "full"       # {"full", "update"}
    use_nufft_norm: bool = True      # use op.scale_emp if available for data diag
    nufft_diag_val: float = 1.0      # fallback constant for data diag

    # Workspace & numeric types
    device: Optional[torch.device | str | int] = None  # default: y.device
    dtype: Optional[torch.dtype] = None                # default: infer from AH(y)
    shard_B: Optional[int] = None     # frames per shard (None ⇒ single shard)

    # Misc
    verbose: bool = True
    seed: Optional[int] = None


# ───────────────────────────────────────────────────────────────────────────────
# Minimal, user-friendly workspace wrapper
# ───────────────────────────────────────────────────────────────────────────────

class _UnitScale:
    """Identity scale field: leaves values unchanged."""
    @torch.no_grad()
    def inv_s2_for_shard(self, sh, anchor=None):
        shape = [sh.x.shape[0]] + [1] * (sh.x.ndim - 1)
        return torch.ones(shape, dtype=torch.float32, device=sh.x.device)

    @torch.no_grad()
    def inv_s_for_shard(self, sh, anchor=None):
        return torch.ones_like(self.inv_s2_for_shard(sh, anchor)).sqrt()


class _Shard:
    def __init__(self, shape, *, device, dtype):
        self.x   = torch.zeros(shape, dtype=dtype, device=device)
        self.g   = torch.zeros_like(self.x)
        self.dx  = torch.zeros_like(self.x)
        self.diag = torch.ones_like(self.x.real, dtype=torch.float32)


class CGUserWorkspace:
    """
    Thin workspace wrapper that matches the minimal interface used by Objective,
    preconditioner, directions, and init_scaling.
    """
    def __init__(self, y: torch.Tensor, nufft_op, *, x_shape: Tuple[int, ...],
                 device=None, dtype=None, shard_B: Optional[int] = None):
        self.y = y
        self.nufft_op = nufft_op
        self.arena = UnifiedArena()
        self.scale = _UnitScale()
        self._shards: List[Tuple[_Shard, int]] = []

        B_total = x_shape[0]
        device = device or y.device
        dtype = dtype or y.dtype

        # Shard along the leading (frame) axis if requested
        if shard_B is None or shard_B >= B_total:
            self._shards.append((_Shard(x_shape, device=device, dtype=dtype), 0))
        else:
            b0 = 0
            k = 0
            while b0 < B_total:
                b1 = min(b0 + shard_B, B_total)
                sh_shape = (b1 - b0,) + tuple(x_shape[1:])
                self._shards.append((_Shard(sh_shape, device=device, dtype=dtype), k))
                b0, k = b1, k + 1

    def iter_shards(self) -> Iterable[Tuple[_Shard, int]]:
        yield from self._shards

    @torch.no_grad()
    def find_shard_by_tensor(self, tensor: torch.Tensor, attr: str = "diag"):
        for sh, _ in self.iter_shards():
            t = getattr(sh, attr, None)
            if t is not None and t.data_ptr() == tensor.data_ptr():
                return sh
        return next(self.iter_shards())[0]


# ───────────────────────────────────────────────────────────────────────────────
# Default "no‑op" regularizer manager for ease of use
# ───────────────────────────────────────────────────────────────────────────────

class NullRegManager:
    """Provides the minimal API expected by Objective/preconditioner when regs are disabled."""
    @torch.no_grad()
    def energy_and_grad(self, ws) -> float:
        return 0.0

    @torch.no_grad()
    def add_diag_shard(self, ws, sh, diag: torch.Tensor) -> None:
        # no-op: leaves data-term diagonal as is
        return

    def estimate_from_pilot(self, ws, xs, cfg, verbose: bool = False):
        return {}


# ───────────────────────────────────────────────────────────────────────────────
# Runner
# ───────────────────────────────────────────────────────────────────────────────

@dataclass
class CGState:
    it: int
    f: float
    gdot: float
    step: float
    x_norm: float

class CGRunner:
    """
    Self-contained CG driver with:
    - Objective caching (Ax, Adx)
    - Preconditioning via full diag
    - PR+/DY/FR direction updates
    - Armijo/Strong-Wolfe line search
    """
    def __init__(self,
                 y: torch.Tensor,
                 nufft_op,
                 cfg: Optional[CGConfig] = None,
                 regm=None):
        self.cfg = cfg or CGConfig()
        if self.cfg.seed is not None:
            torch.manual_seed(int(self.cfg.seed))

        # 1) Probe AH(y) to infer image shape & dtype if not provided
        with torch.no_grad():
            x0 = nufft_op.AH(y)
        dtype = self.cfg.dtype or x0.dtype
        device = self.cfg.device or y.device

        # 2) Build workspace and objective
        self.ws = CGUserWorkspace(y.to(device), nufft_op, x_shape=tuple(x0.shape),
                                  device=device, dtype=dtype, shard_B=self.cfg.shard_B)

        self.regm = regm if regm is not None else NullRegManager()
        self.obj = Objective(nufft_op, self.ws.y, self.regm)

        # Solver attributes required by line_search.search(...)
        self.ls_name = self.cfg.ls_name
        self.c1 = self.cfg.c1
        self.c2 = self.cfg.c2 if self.cfg.ls_name != "armijo" else 0.0
        self.ls_max_iter = self.cfg.ls_max_iter
        self.ls_zoom = self.cfg.ls_zoom

        # Direction
        self.dir_obj = build_direction(self.cfg.dir_name, self.ws)
        if hasattr(self.dir_obj, "init_state"):
            self.dir_obj.init_state()

        # Bookkeeping
        self.history: List[CGState] = []

    # --- helpers -------------------------------------------------------------

    @torch.no_grad()
    def _x_norm(self) -> float:
        s = 0.0
        for sh,_ in self.ws.iter_shards():
            s += float(torch.real(sh.x.conj() * sh.x).sum())
        return math.sqrt(max(s, 0.0))

    @torch.no_grad()
    def _step_norm(self) -> float:
        s = 0.0
        for sh,_ in self.ws.iter_shards():
            s += float(torch.real(sh.dx.conj() * sh.dx).sum())
        return math.sqrt(max(s, 0.0))

    @torch.no_grad()
    def _accept_step(self, t: float):
        for sh,_ in self.ws.iter_shards():
            sh.x.add_(sh.dx, alpha=float(t))

    # --- main entry ----------------------------------------------------------

    @torch.no_grad()
    def run(self,
            callbacks: Optional[List[Callable[[CGState, "CGRunner"], None]]] = None
           ) -> Tuple[torch.Tensor, List[CGState]]:
        cfg = self.cfg
        ws = self.ws

        # 0) Initial backproj + scaling (fills sh.x), then preconditioner
        initial_backproj_and_scaling(ws, self.regm, xfactor=cfg.xfactor,
                                     stats_cfg=None, verbose=cfg.verbose)
        build_precond_diag(ws, self.regm,
                           mode=cfg.precond_mode, nufft_diag_val=cfg.nufft_diag_val,
                           use_nufft_norm=cfg.use_nufft_norm)

        # 1) First gradient at x (with dx=0); set initial direction = -M^{-1} g
        self.obj.begin_linesearch(ws)
        f0, _ = self.obj.f_g(ws, t=0.0)      # ws.g now holds ∇ at x
        self.obj.end_linesearch(ws)

        # Make initial direction
        self.dir_obj.update_inplace(ws)       # for first iter, becomes -g/diag

        # iterations
        for it in range(cfg.max_iter):
            # Rebuild caches for current (x, dx)
            self.obj.begin_linesearch(ws)
            f0, g0d = self.obj.f_g(ws, t=0.0)

            # If direction degenerate, refresh to -M^{-1} g
            if abs(g0d) < 1e-30 or self._step_norm() == 0.0:
                self.dir_obj.update_inplace(ws)
                # refresh caches for new d
                self.obj.end_linesearch(ws)
                self.obj.begin_linesearch(ws)
                f0, g0d = self.obj.f_g(ws, t=0.0)

            # 2) Line search along d
            ok, t, f_new, gdot = line_search.search(self, f0=f0, g0d=g0d)
            self.obj.end_linesearch(ws)

            # 3) Accept step if OK; otherwise bail out
            if not ok:
                if cfg.verbose:
                    print(f"[CG] it={it}: line search failed; stopping.")
                break

            self._accept_step(t)

            # 4) Convergence checks on step size
            step_norm = abs(t) * self._step_norm()
            x_norm = self._x_norm()
            rel = step_norm / (x_norm + 1e-12)

            state = CGState(it=it, f=float(f_new), gdot=float(gdot),
                            step=float(step_norm), x_norm=float(x_norm))
            self.history.append(state)
            if cfg.verbose and (it % cfg.log_every == 0):
                print(f"[CG] it={it:03d}  f={state.f:.6e}  |step|={state.step:.3e}  rel={rel:.3e}")

            if step_norm <= cfg.tol_abs or rel <= cfg.tol_rel:
                if cfg.verbose:
                    print(f"[CG] converged at it={it} (rel={rel:.3e}, abs={step_norm:.3e}).")
                break

            # 5) Update direction for next iter (uses current ws.g which was set at last LS call)
            self.dir_obj.update_inplace(ws)

            # Optional: refresh/update preconditioner here if you support "update" mode
            if cfg.precond_mode == "update":
                build_precond_diag(ws, self.regm,
                                   mode="update", nufft_diag_val=cfg.nufft_diag_val,
                                   use_nufft_norm=cfg.use_nufft_norm)

            # Callbacks
            if callbacks:
                for cb in callbacks:
                    try:
                        cb(state, self)
                    except Exception as e:
                        if cfg.verbose:
                            print(f"[CG] callback error ignored: {e}")

        # Return final x stacked over shards (leading axis concat)
        x_out = torch.cat([sh.x for sh,_ in ws.iter_shards()], dim=0)
        return x_out, self.history


# ───────────────────────────────────────────────────────────────────────────────
# Top-level convenience helpers
# ───────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def quick_cg(y: torch.Tensor,
             nufft_op,
             *,
             regm=None,
             cfg: Optional[CGConfig] = None,
             callbacks: Optional[List[Callable[[CGState, CGRunner], None]]] = None
            ) -> Tuple[torch.Tensor, List[CGState]]:
    """
    Easiest path: run CG with sensible defaults.
    Example:
        x, hist = quick_cg(y, nufft)
    """
    runner = CGRunner(y, nufft_op, cfg=cfg, regm=regm)
    return runner.run(callbacks=callbacks)


def build_runner(y: torch.Tensor,
                 nufft_op,
                 *,
                 regm=None,
                 cfg: Optional[CGConfig] = None) -> CGRunner:
    """
    Build a CGRunner you can customize (e.g., attach callbacks, inspect state).
    """
    return CGRunner(y, nufft_op, cfg=cfg, regm=regm)