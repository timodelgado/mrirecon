# graspcg/regularization/stats_board.py
from __future__ import annotations
from typing import Dict, Optional, List, Iterator, Tuple
import contextlib
import torch


class StatsBoard:
    """
    Small, compile‑friendly stats hub (scalar‑only) with *scopes*.

    Core ideas
    ----------
    • Live slots are 0‑D *device* tensors updated in-place (no host syncs).
    • A *scope* is a per-iteration (or per-line-search) scratch map where writes
      are redirected until you `commit_scope`, at which point the scope’s values
      are atomically copied into the live slots (replace or add semantics).
    • Scopes avoid recomputation: you can evaluate multiple trials, keep only
      the accepted trial’s stats (commit), and discard the rest (abort/clear).
    • All operations are tensor-native (no .item()) in the hot path.

    API (existing)
    --------------
    - scalar_slot(key, device, dtype) -> 0-D device tensor (in-place .add_())
    - enable(key, on)
    - record_step()
    - read_scalar(key) -> float (post-run)
    - read_history(key) -> list[float] (post-run)
    - reset_scalar(key)  # base/live slots only
    - clear_all()

    New scope API
    -------------
    - begin_scope(name: str, *, activate=True, clear=True)
        Create (or reuse) a scope; set as active (stack) and optionally zero it.
    - use_scope(name: str)  # switch active to an existing scope (stack push)
    - reset_scope(name: str | None = None)  # zero tensors in scope (cheap)
    - commit_scope(name: str | None = None, *, replace=True, deactivate=True, clear=True)
        Copy scope values into the *live* slots. If replace=True, overwrite live
        values with scope values (per-device). If replace=False, add into live.
        Optionally pop/deactivate and zero the scope for reuse.
    - abort_scope(name: str | None = None, *, deactivate=True, clear=True)
        Drop scope values (no changes to live).
    - scope_active -> Optional[str]
    - scoped(name: str, *, replace=True) -> context manager that commits on success
      and aborts on exception.
    """

    def __init__(self):
        # Live per-device scalars
        self.scalars: Dict[str, Dict[torch.device, torch.Tensor]] = {}
        # Optional knobs that policies/manager may consult (with sane defaults)
        self.probe_percentile: float = 0.90
        self.probe_sample_K: int = 4096

        # Feature toggles (e.g., "reg_energy", "tv_quantile")
        self.enabled: Dict[str, bool] = {}
        # History: key -> list of {device: 0-D clone}
        self._history: Dict[str, List[Dict[torch.device, torch.Tensor]]] = {}
        # Scopes: name -> (key -> {device: 0-D tensor})
        self._scopes: Dict[str, Dict[str, Dict[torch.device, torch.Tensor]]] = {}
        # Stack of active scopes (top of stack is the current active route)
        self._scope_stack: List[str] = []

    # ───────────────── toggles ─────────────────

    def enable(self, key: str, on: bool = True) -> None:
        self.enabled[key] = bool(on)

    # ──────────────── scope management ────────────────

    @property
    def scope_active(self) -> Optional[str]:
        return self._scope_stack[-1] if self._scope_stack else None

    def begin_scope(self, name: str, *, activate: bool = True, clear: bool = True) -> None:
        """Create (or reuse) a scope, optionally clear it, and activate it."""
        scope = self._scopes.setdefault(name, {})
        if clear:
            for dev_map in scope.values():
                for t in dev_map.values():
                    t.zero_()
        if activate:
            self._scope_stack.append(name)

    def use_scope(self, name: str) -> None:
        """Activate an existing scope without clearing."""
        if name not in self._scopes:
            raise KeyError(f"scope '{name}' does not exist; call begin_scope first")
        self._scope_stack.append(name)

    def reset_scope(self, name: Optional[str] = None) -> None:
        """Zero all tensors in the given scope (or the active one)."""
        name = name or self.scope_active
        if not name:
            return
        scope = self._scopes.get(name, {})
        for dev_map in scope.values():
            for t in dev_map.values():
                t.zero_()

    def commit_scope(self,
                     name: Optional[str] = None,
                     *,
                     replace: bool = True,
                     deactivate: bool = True,
                     clear: bool = True) -> None:
        """
        Copy scope values into *live* slots. If replace=True, overwrite live
        per-device values; else add into them. Optionally pop and zero the scope.
        """
        name = name or self.scope_active
        if not name:
            return
        scope = self._scopes.get(name)
        if not scope:
            # nothing to commit
            if deactivate and self._scope_stack and self._scope_stack[-1] == name:
                self._scope_stack.pop()
            return

        for key, dev_map in scope.items():
            for dev, sval in dev_map.items():
                base = self._base_slot(key, dev, sval.dtype)
                if replace:
                    base.zero_().add_(sval)
                else:
                    base.add_(sval)

        if deactivate and self._scope_stack and self._scope_stack[-1] == name:
            self._scope_stack.pop()
        if clear:
            for dev_map in scope.values():
                for t in dev_map.values():
                    t.zero_()

    def abort_scope(self,
                    name: Optional[str] = None,
                    *,
                    deactivate: bool = True,
                    clear: bool = True) -> None:
        """Drop scope values (no changes to live)."""
        name = name or self.scope_active
        if not name:
            return
        scope = self._scopes.get(name, {})
        if clear:
            for dev_map in scope.values():
                for t in dev_map.values():
                    t.zero_()
        if deactivate and self._scope_stack and self._scope_stack[-1] == name:
            self._scope_stack.pop()

    @contextlib.contextmanager
    def scoped(self, name: str, *, replace: bool = True) -> Iterator["StatsBoard"]:
        """
        Context manager for common pattern:
            with sb.scoped("ls"):
                ... eval trials ...
            # on exception -> abort; on success -> commit(replace=...)
        """
        self.begin_scope(name, activate=True, clear=True)
        try:
            yield self
        except Exception:
            self.abort_scope(name, deactivate=True, clear=True)
            raise
        else:
            self.commit_scope(name, replace=replace, deactivate=True, clear=True)

    # ─────────── live/scoped slots (device tensors) ───────────

    def scalar_slot(self, key: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Return a 0‑D *device* tensor for in-place accumulation (.add_()).
        If a scope is active, return the scope-local slot; otherwise live slot.
        """
        active = self.scope_active
        if active is not None:
            scope = self._scopes.setdefault(active, {})
            dev_map = scope.setdefault(key, {})
            t = dev_map.get(device)
            if t is None:
                t = torch.zeros((), device=device, dtype=dtype)
                dev_map[device] = t
            return t
        # default: live
        return self._base_slot(key, device, dtype)

    def _base_slot(self, key: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Live slot (ignores scope)."""
        d = self.scalars.setdefault(key, {})
        t = d.get(device)
        if t is None:
            t = torch.zeros((), device=device, dtype=dtype)
            d[device] = t
        return t

    # ───────────── history capture (keeps tensors until final readout) ─────────────

    @torch.no_grad()
    def record_step(self) -> None:
        """
        Snapshot current *live* per-device scalars into history
        (0‑D clones, still on device). Scopes are ignored; commit first.
        """
        for key, dev_map in self.scalars.items():
            snap: Dict[torch.device, torch.Tensor] = {dev: t.detach().clone() for dev, t in dev_map.items()}
            self._history.setdefault(key, []).append(snap)

    # ───────────── CPU-side consumers (safe after run) ─────────────

    def read_scalar(self, key: str) -> float:
        """Aggregate current live scalars across devices to a Python float."""
        return sum(float(t.detach().cpu()) for t in self.scalars.get(key, {}).values())

    def read_history(self, key: str) -> List[float]:
        """Aggregate history snapshots across devices per step -> list of floats."""
        out: List[float] = []
        for snap in self._history.get(key, []):
            out.append(sum(float(t.detach().cpu()) for t in snap.values()))
        return out

    # ───────────── resets / clear ─────────────

    @torch.no_grad()
    def reset_scalar(self, key: str) -> None:
        """Zero all *live* per-device scalar slots for `key` (keeps slots alive)."""
        for t in self.scalars.get(key, {}).values():
            t.zero_()

    @torch.no_grad()
    def clear_all(self) -> None:
        """Hard clear (rare). Prefer scope reset/commit and live resets to keep graphs stable."""
        self.scalars.clear()
        self.enabled.clear()
        self._history.clear()
        # Clear scopes but keep names to avoid reallocation if reused soon
        for scope in self._scopes.values():
            for dev_map in scope.values():
                for t in dev_map.values():
                    t.zero_()
        self._scope_stack.clear()