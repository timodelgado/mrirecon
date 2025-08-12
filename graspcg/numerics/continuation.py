# graspcg/numerics/continuation.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from graspcg.ops.reg_registry   import STATS_HELPERS  # (ws, xs, percentile, eps_floor)->(eps,sigma)
from graspcg.ops.preconditioner import build_precond_diag


@dataclass
class ContinuationConfig:
    """
    Global defaults. Individual regs in regm.regs[key] can override any field:
      {"alpha":..., "percentile":..., "eps_floor":..., "kappa":..., "apply_scale_to_data":...}
    """
    every      : int   = 3       # update cadence (iterations)
    alpha      : float = 0.6     # EMA smoothing factor
    percentile : float = 0.90    # Huber percentile
    eps_floor  : float = 1e-6    # ε floor
    kappa      : Dict[str, float] = field(default_factory=dict)  # fallback κ per regularizer id
    update_diag: bool  = True    # refresh preconditioner diag after updates


class ContinuationManager:
    """
    Operates on an external regularizer manager (regm) rather than ws.regs.
    Expects: regm.regs: Dict[str, Dict]  (config blob per reg id).
    """
    def __init__(self, cfg: ContinuationConfig, regm):
        self.cfg  = cfg
        self.regm = regm

    @torch.no_grad()
    def maybe_update(self, ws, k_iter: int) -> bool:
        """
        Run continuation at the end of an accepted step k_iter.
        Returns True if any reg changed (λ or ε), and optionally rebuilds diag.
        """
        if self.cfg.every <= 0 or (k_iter % self.cfg.every) != 0:
            return False

        # If any reg requests scale-normalised stats, prepare a pilot u = x/s
        any_scaled = any(bool(cfg.get("apply_scale_to_data", False))
                         for cfg in self.regm.regs.values())

        if any_scaled:
            # place scaled pilot in ws.dx shard-by-shard to avoid allocations
            for sh, _ in ws.iter_shards():
                sh.dx.copy_(sh.x)
                ws.scale.divide_inplace(sh.dx)

        changed = False

        # Update each registered regularizer
        for key, cfg in self.regm.regs.items():
            helper = STATS_HELPERS.get(key)
            if helper is None:
                # no stats helper: skip (e.g., custom reg without continuation)
                continue

            # resolve policy with per‑reg override -> global default
            alpha      = float(cfg.get("alpha",      self.cfg.alpha))
            percentile = float(cfg.get("percentile", self.cfg.percentile))
            eps_floor  = float(cfg.get("eps_floor",  self.cfg.eps_floor))
            kappa      = float(cfg.get("kappa",      self.cfg.kappa.get(key, 1.0)))
            apply_scale= bool(cfg.get("apply_scale_to_data", False))

            # Aggregate stats across shards without materialising big tensors
            # We take a simple mean across shards (streaming). This is robust and cheap.
            eps_acc, sig_acc, n_shards = 0.0, 0.0, 0
            for sh, _ in ws.iter_shards():
                xs = sh.dx if apply_scale and any_scaled else sh.x
                eps_i, sigma_i = helper(ws, xs,
                                        percentile=percentile,
                                        eps_floor=eps_floor)
                eps_acc += float(eps_i)
                sig_acc += float(sigma_i)
                n_shards += 1

            if n_shards == 0:
                continue

            eps_new = max(eps_acc / n_shards, eps_floor)
            lam_new = kappa * (sig_acc / n_shards)

            # current values (create if missing)
            lam_old = float(cfg.get("weight", 0.0))
            eps_old = float(cfg.get("eps",    0.0))

            # EMA smoothing
            lam = alpha * lam_old + (1.0 - alpha) * lam_new
            eps = alpha * eps_old + (1.0 - alpha) * eps_new

            # write back if changed
            if (abs(lam - lam_old) > 0.0) or (abs(eps - eps_old) > 0.0):
                cfg["weight"] = float(lam)
                cfg["eps"]    = float(eps)
                changed = True

        if changed and self.cfg.update_diag:
            build_precond_diag(ws, regm=self.regm, mode="update")


        return changed
