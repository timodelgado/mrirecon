# graspcg/policies/reg_policies.py
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Optional, Mapping

import torch

from ..regularization.tv_nd import TVND  # detect TV regs for ε updates


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
@dataclass
class RegPolicyConfig:
    # ---- Which stats to collect ----
    # Primary path: histogram-based TV magnitude collection
    tv_hist_key: str = "tv_hist"
    # Per-regularizer energy collection (E_reg/<name>); E_reg_total/E_data are always useful
    reg_energy_key: str = "reg_energy"

    # ---- ε (Huber knee) from percentile of TV magnitude ----
    enable_eps_from_percentile: bool = True
    eps_percentile: float = 0.90
    eps_floor: float = 1e-6
    eps_ema_alpha: float = 0.6  # new = α*old + (1-α)*measured

    # ---- λ from relative energy ratio E_reg/E_data ----
    enable_lambda_from_ratio: bool = True
    # If many regs and only ratio_target_global is set, we skip per-reg updates (to avoid arbitrary apportioning).
    ratio_target_global: Optional[float] = 0.10
    ratio_target_per_reg: Optional[Mapping[str, float]] = None  # name -> target
    ratio_eta: float = 0.5             # multiplicative damping exponent
    lambda_ema_alpha: float = 0.6

    # ---- Preconditioner (diag) management ----
    manage_preconditioner: bool = True          # let the policy rebuild diag
    update_diag_after_change: bool = True       # rebuild right after a parameter change
    precond_refresh_every: Optional[int] = None # also rebuild every N updates (accepted steps); None disables
    precond_base: float = 1.0                   # base fill value before reg contributions

    # ---- Bookkeeping ----
    clear_after_read: bool = True               # clear consumed stats to keep them per-iter


# ---------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------
class RegPolicy:
    """
    Reads stats from ws.stats, updates regularizer parameters (ε, λ),
    and (optionally) rebuilds the preconditioner (diag) via RegManager.

    Usage:
        policy = RegPolicy(RegPolicyConfig(...))
        policy.prepare_collection(ws, regm)   # before an evaluation you want stats for
        ...
        changed = policy.update_from_stats(ws, regm)  # after accepted iterate
    """

    def __init__(self, cfg: Optional[RegPolicyConfig] = None):
        self.cfg = cfg or RegPolicyConfig()
        self._tick: int = 0  # counts calls to update_from_stats (i.e., accepted steps)

    # -----------------------------------------------------------------
    # Orchestration
    # -----------------------------------------------------------------
    def prepare_collection(self, ws, regm) -> None:
        """
        Enable just the stats we need for the next evaluation.
        """
        sb = getattr(ws, "stats", None)
        if sb is None:
            return

        # TV histogram needed if updating ε from percentile and there is a TV reg
        need_tv_hist = self.cfg.enable_eps_from_percentile and any(isinstance(r, TVND) for r in getattr(regm, "_regs", []))
        self._sb_enable(sb, self.cfg.tv_hist_key, need_tv_hist)

        # Per-reg energies if we might do per-reg λ updates or have multiple regs with a global target
        need_reg_energy = False
        if self.cfg.enable_lambda_from_ratio:
            if self.cfg.ratio_target_per_reg and len(self.cfg.ratio_target_per_reg) > 0:
                need_reg_energy = True
            elif len(getattr(regm, "_regs", [])) > 1 and (self.cfg.ratio_target_global is not None):
                need_reg_energy = True
        self._sb_enable(sb, self.cfg.reg_energy_key, need_reg_energy)

        # Optionally clear relevant stats for clean per-iter values
        if self.cfg.clear_after_read:
            if need_tv_hist:
                for r in getattr(regm, "_regs", []):
                    if isinstance(r, TVND):
                        self._reset_tv_hist(sb, r.name)
            if need_reg_energy:
                for r in getattr(regm, "_regs", []):
                    self._reset_scalar(sb, f"E_reg/{r.name}")
            self._reset_scalar(sb, "E_data")
            self._reset_scalar(sb, "E_reg_total")
            self._reset_scalar(sb, "gdot")

    def prime_from_stats(self, ws, regm) -> bool:
        """
        Optional one-time init pass after a pilot evaluation populated stats.
        """
        return self._update_from_stats_impl(ws, regm, use_ema=True)

    def update_from_stats(self, ws, regm) -> bool:
        """
        Continuation update after an accepted step.
        """
        self._tick += 1
        changed = self._update_from_stats_impl(ws, regm, use_ema=True)

        # Periodic preconditioner refresh even if nothing changed (optional)
        if self.cfg.manage_preconditioner and self.cfg.precond_refresh_every:
            if (self._tick % max(1, int(self.cfg.precond_refresh_every))) == 0:
                self._rebuild_preconditioner(ws, regm)

        return changed

    # -----------------------------------------------------------------
    # Core updates
    # -----------------------------------------------------------------
    def _update_from_stats_impl(self, ws, regm, *, use_ema: bool) -> bool:
        sb = getattr(ws, "stats", None)
        if sb is None:
            return False

        changed_any = False

        # -------- ε (TV) via percentile from histogram (primary), with fallbacks --------
        if self.cfg.enable_eps_from_percentile:
            q = float(self.cfg.eps_percentile)
            for reg in getattr(regm, "_regs", []):
                if not isinstance(reg, TVND):
                    continue

                eps_est = 0.0

                # Preferred: histogram-per-reg present in StatsBoard
                hist = self._sb_read_reg_hist(sb, reg.name)
                if hist is not None:
                    try:
                        eps_est = float(self._sb_hist_percentile(sb, hist, q))
                    except Exception:
                        eps_est = 0.0

                # Fallback 1: direct quantile scalar emitted by manager
                if eps_est <= 0.0:
                    try:
                        eps_est = float(self._sb_read_scalar(sb, f"tv_q/{reg.name}"))
                    except Exception:
                        eps_est = 0.0

                # Fallback 2: legacy (sum, n) per percentile
                if eps_est <= 0.0:
                    tag = f"p{int(round(q*100))}"
                    s = self._sb_read_scalar(sb, f"tv/{tag}/sum/{reg.name}")
                    n = self._sb_read_scalar(sb, f"tv/{tag}/n/{reg.name}")
                    if n > 0.0:
                        eps_est = s / n

                if eps_est <= 0.0:
                    continue

                eps_est = max(eps_est, float(self.cfg.eps_floor))
                eps_old = float(getattr(reg.params, "eps", 0.0))
                eps_new = self._ema(eps_old, eps_est, self.cfg.eps_ema_alpha) if use_ema else eps_est

                if eps_new != eps_old:
                    reg.params = replace(reg.params, eps=eps_new)
                    changed_any = True

        # -------- λ via E_reg/E_data --------
        if self.cfg.enable_lambda_from_ratio:
            E_data = self._sb_read_scalar(sb, "E_data")
            if E_data <= 0.0:
                E_data = 1e-30  # guard

            many = len(getattr(regm, "_regs", [])) > 1
            for reg in getattr(regm, "_regs", []):
                # choose target
                rho_t: Optional[float] = None
                if self.cfg.ratio_target_per_reg and reg.name in self.cfg.ratio_target_per_reg:
                    rho_t = float(self.cfg.ratio_target_per_reg[reg.name])
                elif (self.cfg.ratio_target_global is not None) and not many:
                    rho_t = float(self.cfg.ratio_target_global)

                if rho_t is None:
                    # If many regs and only a single global target exists, we skip per-reg updates
                    continue

                # E_reg: prefer per-reg if collected; else fallback when only one reg exists
                E_reg = 0.0
                if self._sb_enabled(sb, self.cfg.reg_energy_key):
                    E_reg = self._sb_read_scalar(sb, f"E_reg/{reg.name}")
                if (E_reg <= 0.0) and not many:
                    E_reg = self._sb_read_scalar(sb, "E_reg_total")

                if E_reg <= 0.0:
                    continue

                rho = E_reg / max(E_data, 1e-30)
                lam_old = float(getattr(reg.params, "weight", 0.0))
                mult = (rho_t / max(rho, 1e-30)) ** float(self.cfg.ratio_eta)
                lam_est = lam_old * mult
                lam_new = self._ema(lam_old, lam_est, self.cfg.lambda_ema_alpha) if use_ema else lam_est

                if lam_new != lam_old:
                    reg.params = replace(reg.params, weight=lam_new)
                    changed_any = True

        # -------- optional preconditioner rebuild --------
        if changed_any and self.cfg.manage_preconditioner and self.cfg.update_diag_after_change:
            self._rebuild_preconditioner(ws, regm)

        # -------- housekeeping --------
        if self.cfg.clear_after_read:
            self._clear_consumed(sb, regm)

        return changed_any

    # -----------------------------------------------------------------
    # Preconditioner rebuild
    # -----------------------------------------------------------------
    def _rebuild_preconditioner(self, ws, regm) -> None:
        """
        Reset diag to base and re-add contributions from regs.
        Avoids cumulative double-add when parameters change.
        """
        try:
            for _, i in ws.iter_shards():
                D = ws.get("diag", i)
                if D is None:
                    continue
                # fill base (usually 1.0)
                base = torch.as_tensor(self.cfg.precond_base, device=D.device, dtype=D.dtype)
                D.fill_(base)
            regm.add_diag(ws)
        except Exception:
            # Preconditioner is an acceleration feature—never hard-fail the solver
            pass

    # -----------------------------------------------------------------
    # Small helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _ema(old: float, new: float, alpha: float) -> float:
        a = float(alpha)
        a = 1.0 if a >= 1.0 else (0.0 if a <= 0.0 else a)
        return a * float(old) + (1.0 - a) * float(new)

    @staticmethod
    def _sb_enable(sb, key: str, value: bool) -> None:
        try:
            sb.enable(key, bool(value))
        except Exception:
            pass

    @staticmethod
    def _sb_enabled(sb, key: str) -> bool:
        try:
            en = getattr(sb, "enabled", {})
            return bool(en.get(key, False))
        except Exception:
            return False

    @staticmethod
    def _sb_read_scalar(sb, key: str) -> float:
        try:
            return float(sb.read_scalar(key))
        except Exception:
            return 0.0

    @staticmethod
    def _reset_scalar(sb, key: str) -> None:
        try:
            slots = getattr(sb, "scalars", {}).get(key, {})
            for t in slots.values():
                try:
                    t.zero_()
                except Exception:
                    pass
        except Exception:
            pass

    @staticmethod
    def _sb_read_reg_hist(sb, reg_name: str):
        """
        Expected StatsBoard API: read_reg_hist(name) -> {"edges": 1D Tensor, "counts": 1D Tensor}
        Return None if unavailable.
        """
        try:
            return sb.read_reg_hist(reg_name)
        except Exception:
            return None

    @staticmethod
    def _sb_hist_percentile(sb, hist: dict, q: float) -> float:
        """
        Expected StatsBoard API: hist_percentile(edges, counts, q) -> float
        """
        return float(sb.hist_percentile(hist["edges"], hist["counts"], float(q)))

    @staticmethod
    def _reset_tv_hist(sb, reg_name: str) -> None:
        """
        Zero per-device histogram counts for a given reg if present.
        """
        # Prefer a direct API if StatsBoard provides one
        try:
            if hasattr(sb, "reset_reg_hist"):
                sb.reset_reg_hist(reg_name)
                return
        except Exception:
            pass

        # Fallback: mutate internal storage defensively
        try:
            dev_map = getattr(sb, "reg_hists", {}).get(reg_name, {})
            for slots in dev_map.values():
                try:
                    cnt = slots.get("counts", None)
                    if torch.is_tensor(cnt):
                        cnt.zero_()
                except Exception:
                    pass
        except Exception:
            pass

    def _clear_consumed(self, sb, regm) -> None:
        self._reset_scalar(sb, "E_data")
        self._reset_scalar(sb, "E_reg_total")
        self._reset_scalar(sb, "gdot")
        if self._sb_enabled(sb, self.cfg.reg_energy_key):
            for r in getattr(regm, "_regs", []):
                self._reset_scalar(sb, f"E_reg/{r.name}")
        if self._sb_enabled(sb, self.cfg.tv_hist_key):
            for r in getattr(regm, "_regs", []):
                if isinstance(r, TVND):
                    self._reset_tv_hist(sb, r.name)
