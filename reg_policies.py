from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Optional, Mapping

import torch

# ---------------------------------------------------------------------
# Configuration (generic)
# ---------------------------------------------------------------------
@dataclass
class RegPolicyConfig:
    # ---- Which stats to collect (generic toggles) ----
    probe_quantile_key: str = "probe_quantile"
    probe_mad_key: str      = "probe_mad"
    term_energy_key: str    = "term_reg_energy"

    # ---- ε (Huber knee) updates ----
    enable_eps_from_percentile: bool = True
    eps_percentile: float = 0.90
    enable_eps_from_mad: bool = True
    mad_to_eps_scale: float = 1.4826
    eps_floor: float = 1e-6
    eps_ema_alpha: float = 0.6  # new = α*old + (1-α)*measured

    # ---- λ from relative energy ratio E_reg/E_data ----
    enable_lambda_from_ratio: bool = True
    # If many regs and only ratio_target_global is set, skip per‑reg updates.
    ratio_target_global: Optional[float] = 0.10
    ratio_target_per_reg: Optional[Mapping[str, float]] = None  # name -> target
    ratio_eta: float = 0.5
    lambda_ema_alpha: float = 0.6

    # ---- Preconditioner (diag) management ----
    manage_preconditioner: bool = True
    update_diag_after_change: bool = True
    precond_refresh_every: Optional[int] = None
    precond_base: float = 1.0

    # ---- Bookkeeping ----
    clear_after_read: bool = True


# ---------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------
class RegPolicy:
    """
    Reads stats from ws.stats, updates regularizer parameters (ε, λ),
    and (optionally) rebuilds the preconditioner (diag) via RegManager.

    Usage:
        policy = RegPolicy(RegPolicyConfig(...))
        policy.prepare_collection(ws, regm)
        ...
        changed = policy.update_from_stats(ws, regm)
    """

    def __init__(self, cfg: Optional[RegPolicyConfig] = None):
        self.cfg = cfg or RegPolicyConfig()
        self._tick: int = 0  # accepted steps

    # -----------------------------------------------------------------
    # Orchestration
    # -----------------------------------------------------------------
    def prepare_collection(self, ws, regm) -> None:
        sb = getattr(ws, "stats", None)
        if sb is None:
            return

        regs = list(getattr(regm, "_regs", []))
        has_eps = any(hasattr(getattr(r, "params", None), "eps") for r in regs)
        need_probes = has_eps and (self.cfg.enable_eps_from_percentile or self.cfg.enable_eps_from_mad)
        self._sb_enable(sb, self.cfg.probe_quantile_key, need_probes)
        self._sb_enable(sb, self.cfg.probe_mad_key,      need_probes)

        need_term_energy = False
        if self.cfg.enable_lambda_from_ratio:
            if self.cfg.ratio_target_per_reg and len(self.cfg.ratio_target_per_reg) > 0:
                need_term_energy = True
            elif len(regs) > 1 and (self.cfg.ratio_target_global is not None):
                need_term_energy = True
        self._sb_enable(sb, self.cfg.term_energy_key, need_term_energy)

        # clear per‑iter values if requested
        if self.cfg.clear_after_read:
            if need_probes:
                for r in regs:
                    self._reset_scalar(sb, f"probe/q/{r.name}")
                    self._reset_scalar(sb, f"probe/mad/{r.name}")
            if need_term_energy:
                for r in regs:
                    self._reset_scalar(sb, f"term/reg/{r.name}/energy")
            for k in ("term/data/energy", "term/reg/total/energy", "slope/dir", "obj/total"):
                self._reset_scalar(sb, k)

    def prime_from_stats(self, ws, regm) -> bool:
        return self._update_from_stats_impl(ws, regm, use_ema=True)

    def update_from_stats(self, ws, regm) -> bool:
        self._tick += 1
        changed = self._update_from_stats_impl(ws, regm, use_ema=True)
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

        # ε via probe quantile and/or MAD
        if self.cfg.enable_eps_from_percentile or self.cfg.enable_eps_from_mad:
            regs = list(getattr(regm, "_regs", []))
            for reg in regs:
                params = getattr(reg, "params", None)
                if not hasattr(params, "eps"):
                    continue

                eps_est = 0.0
                if self.cfg.enable_eps_from_percentile:
                    eps_est = float(self._sb_read_scalar(sb, f"probe/q/{reg.name}"))
                if (eps_est <= 0.0) and self.cfg.enable_eps_from_mad:
                    mad = float(self._sb_read_scalar(sb, f"probe/mad/{reg.name}"))
                    if mad > 0.0:
                        eps_est = mad * float(self.cfg.mad_to_eps_scale)
                if eps_est <= 0.0:
                    continue

                eps_est = max(eps_est, float(self.cfg.eps_floor))
                eps_old = float(getattr(params, "eps", 0.0))
                eps_new = self._ema(eps_old, eps_est, self.cfg.eps_ema_alpha) if use_ema else eps_est
                if eps_new != eps_old:
                    reg.params = replace(params, eps=eps_new)
                    changed_any = True

        # λ via energy ratio
        if self.cfg.enable_lambda_from_ratio:
            E_data = self._sb_read_scalar(sb, "term/data/energy")
            if E_data <= 0.0:
                E_data = 1e-30
            regs = list(getattr(regm, "_regs", []))
            many = len(regs) > 1
            for reg in regs:
                rho_t: Optional[float] = None
                if self.cfg.ratio_target_per_reg and reg.name in self.cfg.ratio_target_per_reg:
                    rho_t = float(self.cfg.ratio_target_per_reg[reg.name])
                elif (self.cfg.ratio_target_global is not None) and not many:
                    rho_t = float(self.cfg.ratio_target_global)
                if rho_t is None:
                    continue

                E_reg = 0.0
                if self._sb_enabled(sb, self.cfg.term_energy_key):
                    E_reg = self._sb_read_scalar(sb, f"term/reg/{reg.name}/energy")
                if (E_reg <= 0.0) and not many:
                    E_reg = self._sb_read_scalar(sb, "term/reg/total/energy")
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

        if changed_any and self.cfg.manage_preconditioner and self.cfg.update_diag_after_change:
            self._rebuild_preconditioner(ws, regm)

        if self.cfg.clear_after_read:
            self._clear_consumed(sb, regm)

        return changed_any

    # -----------------------------------------------------------------
    # Preconditioner rebuild
    # -----------------------------------------------------------------
    def _rebuild_preconditioner(self, ws, regm) -> None:
        try:
            for _, i in ws.iter_shards():
                D = ws.get("diag", i)
                if D is None:
                    continue
                base = torch.as_tensor(self.cfg.precond_base, device=D.device, dtype=D.dtype)
                D.fill_(base)
            regm.add_diag(ws)
        except Exception:
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

    def _clear_consumed(self, sb, regm) -> None:
        for k in ("term/data/energy", "term/reg/total/energy", "slope/dir", "obj/total"):
            self._reset_scalar(sb, k)
        if self._sb_enabled(sb, self.cfg.term_energy_key):
            for r in getattr(regm, "_regs", []):
                self._reset_scalar(sb, f"term/reg/{r.name}/energy")
        if self._sb_enabled(sb, self.cfg.probe_quantile_key) or self._sb_enabled(sb, self.cfg.probe_mad_key):
            for r in getattr(regm, "_regs", []):
                self._reset_scalar(sb, f"probe/q/{r.name}")
                self._reset_scalar(sb, f"probe/mad/{r.name}")
