# graspcg/solvers/cg_user_init.py
from __future__ import annotations
from typing import Sequence, Mapping, Optional
import torch

from ..workspace.device_cfg    import DeviceCfg
from ..workspace.unified_arena import UnifiedArena
from ..workspace.device_pool   import DevicePool
from ..workspace.cg_workspace  import CGWorkspace

from ..ops.reg_manager         import RegManager
from ..ops.init_scaling        import initial_backproj_and_scaling
from ..ops.preconditioner      import build_precond_diag

from ..solvers.cg              import CGSolver
from ..numerics.continuation   import ContinuationConfig, ContinuationManager


# ─────────────────────────────────────────────────────────────────────────────
# 1) Workspace builder
# ─────────────────────────────────────────────────────────────────────────────
def build_workspace(nufft_op,
                    y: torch.Tensor,
                    devices: int | str | Sequence[int|str] | None = None,
                    *,
                    arena: UnifiedArena | None = None) -> CGWorkspace:
    """
    Create a sharded workspace on the chosen device(s).
    """
    if devices is None:
        device_cfg = DeviceCfg()
    else:
        if isinstance(devices, (int, str)):
            devices = [devices]
        device_cfg = DeviceCfg(compute=devices[0], helpers=list(devices[1:]))

    arena = arena or UnifiedArena(DevicePool())
    ws = CGWorkspace(y, nufft_op, device_cfg=device_cfg, arena=arena)
    return ws


# ─────────────────────────────────────────────────────────────────────────────
# 2) RegManager builder
# ─────────────────────────────────────────────────────────────────────────────
def build_regmanager(regs: Sequence[str] | Mapping[str, Mapping] | None,
                     *,
                     voxel_size: tuple[float,float,float] | None = None,
                     defaults: Mapping[str, Mapping] | None = None) -> RegManager:
    """
    Accepts:
      • Sequence[str]: e.g. ("tv_t","tv_s")
      • Mapping[str,dict]: {"tv_t": {...}, "tv_s": {...}}
      • None: build empty manager (no regularisation)

    Fills minimal sane defaults; weights/eps are typically set by
    `initial_backproj_and_scaling`.
    """
    rm = RegManager()

    # resolve candidate configs
    cfg_map: dict[str, dict] = {}
    if regs is None:
        cfg_map = {}
    elif isinstance(regs, Mapping):
        cfg_map = {k: dict(v) for k, v in regs.items()}
    else:
        cfg_map = {k: {} for k in regs}

    # defaults per reg (user can override)
    defaults = dict(defaults or {})
    # TV‑T default
    defaults.setdefault("tv_t", {
        "weight": 0.0,         # will be set by init stats
        "eps":    1e-3,
        "tile":   None,
        "apply_scale": True,    # measure on u = x/s
        "huber_percentile": 0.90
    })
    # TV‑S default
    defaults.setdefault("tv_s", {
        "weight": 0.0,
        "eps":    1e-3,
        "tile_s1": None,
        "apply_scale": True,    # measure on u = x/s
        "voxel_size": (1.0, 1.0, 1.0) if voxel_size is None else tuple(voxel_size),
        "huber_percentile": 0.90
    })

    # merge user cfgs over defaults
    for k, base in list(defaults.items()):
        user = cfg_map.get(k, {})
        merged = dict(base); merged.update(user)
        rm.add(k, **merged)

    # include any extra custom regs the user asked for
    for k, user in cfg_map.items():
        if k not in rm.regs:
            rm.add(k, **user)

    return rm


# ─────────────────────────────────────────────────────────────────────────────
# 3) One‑shot initialisation: back‑projection, stats → λ/ε, preconditioner
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def initialise_problem(ws: CGWorkspace,
                       regm: RegManager,
                       *,
                       xfactor: float = 1.0,
                       stats_cfg: Mapping[str, Mapping[str, float | bool]] | None = None,
                       verbose: bool = True) -> dict:
    """
    • AH(y) → x, energy‑normalise by matched‑filter ratio
    • Compute reg‑specific ε (percentile) & σ (MAD), set λ = κ·σ
    • Build preconditioner diagonal (data + regs)
    """
    # stats_cfg lets users override percentile/eps_floor/kappa/apply_scale per reg
    out = initial_backproj_and_scaling(
        ws, regm,
        xfactor=xfactor,
        stats_cfg=stats_cfg or {},
        verbose=verbose,
    )
    build_precond_diag(ws, mode="full")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4) Continuation setup (optional but convenient)
# ─────────────────────────────────────────────────────────────────────────────
def make_continuation(*,
                      every: int = 3,
                      alpha: float = 0.6,
                      percentile: float = 0.90,
                      eps_floor: float = 1e-6,
                      kappa: Mapping[str, float] | None = None,
                      update_diag: bool = True) -> ContinuationManager:
    cfg = ContinuationConfig(
        every=every, alpha=alpha,
        percentile=percentile, eps_floor=eps_floor,
        kappa=dict(kappa or {}),
        update_diag=update_diag
    )
    return ContinuationManager(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# 5) “Quick CG”: ergonomic one‑call runner
# ─────────────────────────────────────────────────────────────────────────────
def quick_cg(nufft_op,
             y: torch.Tensor,
             *,
             regs: Sequence[str] | Mapping[str, Mapping] | None = ("tv_t","tv_s"),
             devices: int | str | Sequence[int|str] | None = None,
             voxel_size: tuple[float,float,float] | None = None,
             xfactor: float = 1.0,
             line_search: str = "wolfe",
             direction: str   = "prplus",
             max_iter: int = 30,
             tol_rel: float = 1e-6,
             tol_abs: float = 0.0,
             continuation: ContinuationManager | None = None,
             verbose: bool = False) -> tuple[torch.Tensor, dict]:
    """
    End‑to‑end convenience:
      ws = build_workspace(...)
      regm = build_regmanager(...)
      initialise_problem(...)
      CGSolver(...).run()
    """
    # 1) workspace + reg manager
    ws   = build_workspace(nufft_op, y, devices)
    regm = build_regmanager(regs, voxel_size=voxel_size)

    # 2) back‑proj + stats + preconditioner
    init_info = initialise_problem(ws, regm, xfactor=xfactor, verbose=verbose)

    # 3) solver
    solver = CGSolver(
        nufft_op, y,
        regm=regm,
        devices=devices,
        line_search=line_search,
        direction=direction,
        max_iter=max_iter,
        tol_rel=tol_rel, tol_abs=tol_abs,
        verbose=verbose,
        continuation=continuation,
    )
    x_rec = solver.run()
    return x_rec, init_info
