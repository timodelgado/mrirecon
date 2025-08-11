# graspcg/ops/reg_manager.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterable
import torch

# ---- Per-regularizer policy -----------------------------------------------
@dataclass
class RegPolicy:
    # How the scale field should be used by the regularizer
    #   "none" : ignore scale field
    #   "inv_s": weight gradients by 1/s (classic frame scaling)
    #   "inv_s2": weight quadratics/diagonals by 1/s^2
    scale_kind: str = "inv_s"     # {"none","inv_s","inv_s2"}

    # Where to apply the scale weights
    # e.g. {"grad","diag","stats"} – energy is usually unaffected
    apply_to: set[str] = field(default_factory=lambda: {"grad","diag","stats"})

    # Huber / stats knobs (per-regularizer)
    percentile: float = 0.90
    eps_floor: float = 1e-6
    # λ = κ · σ  (initialisation / continuation target)
    kappa: float = 1.0
    # Optional EMA for dynamic updates (0 = no smoothing)
    ema: float = 0.0


# ---- Registry entry kept by the manager -----------------------------------
@dataclass
class RegEntry:
    name: str
    obj: object               # regularizer object (class with the methods below)
    weight: float = 0.0
    eps: float = 1e-3
    policy: RegPolicy = field(default_factory=RegPolicy)
    energy_last: float = 0.0

    def push_to_obj(self):
        # Keep the object in sync with manager’s truth.
        if hasattr(self.obj, "weight"): self.obj.weight = float(self.weight)
        if hasattr(self.obj, "eps"):    self.obj.eps    = float(self.eps)
        if hasattr(self.obj, "policy"): self.obj.policy = self.policy

@dataclass
class RegManager:
    """Owns instantiated regularizer modules; single source of truth."""
    regs: Dict[str, Any] = field(default_factory=dict)   # name -> reg object
    ledger: Dict[str, float] = field(default_factory=dict)

    def add(self, name: str, reg_obj) -> None:
        self.regs[name] = reg_obj
        self.ledger[name] = 0.0

    def keys(self) -> Iterable[str]:
        return self.regs.keys()

    def get(self, name: str):
        return self.regs[name]

    @torch.no_grad()
    def energy_and_grad(self, ws) -> float:
        total = 0.0
        for name, reg in self.regs.items():
            e = reg.energy_and_grad(ws)      # must write into ws.g in-place
            self.ledger[name] = float(e)
            total += float(e)
        return float(total)

    @torch.no_grad()
    def add_diag(self, ws, diag: torch.Tensor) -> None:
        for _, reg in self.regs.items():
            if hasattr(reg, "add_diag"):
                reg.add_diag(ws, diag)
    @torch.no_grad()
    def add_diag_shard(self, ws, sh, diag: torch.Tensor) -> None:
        for name, reg in self.regs.items():
            if hasattr(reg, "add_diag_shard"):
                reg.add_diag_shard(ws, sh, diag)
            elif hasattr(reg, "add_diag"):
                reg.add_diag(ws, diag)
            else:
                # Try functional handlers by name
                fn = DIAG_SHARD_HELPERS.get(name) or DIAG_HELPERS.get(name)
                if fn is not None:
                    # shard‑explicit helper gets (ws, sh, diag); the older helper gets (ws, diag)
                    try:
                        fn(ws, sh, diag)
                    except TypeError:
                        fn(ws, diag)
    @torch.no_grad()
    def estimate_from_pilot(
        self,
        ws,
        xs: torch.Tensor,
        policies: Mapping[str, Mapping[str, float | bool]],
        *,
        verbose: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        For each reg with a policy:
          - compute (eps, sigma) with reg.estimate_stats(...)
          - set weight = kappa * sigma
          - update reg params in-place
        Returns a dict of the chosen params for logging / reproducibility.
        """
        out: Dict[str, Dict[str, float]] = {}
        for name, reg in self.regs.items():
            pol = dict(policies.get(name, {}))
            if not pol:
                continue
            q = float(pol.get("percentile", 0.90))
            eps_floor = float(pol.get("eps_floor", 1e-6))
            kappa     = float(pol.get("kappa", 1.0))
            eps, sigma = reg.estimate_stats(ws, xs, percentile=q, eps_floor=eps_floor)
            lam = kappa * sigma

            reg.set_params(weight=float(lam), eps=float(eps))

            out[name] = {"weight": float(lam), "eps": float(eps)}
            if verbose:
                print(f"[reg.init] {name}: λ={lam:.3g}, ε={eps:.3g} (κ={kappa}, q={q})")
        return out
