# graspcg/workspace/scale.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch

@dataclass
class ScaleField:
    """
    General per‑batch scale with tiny memory footprint.

    Canonical storage shape: (B, 1, ..., 1)
      B = product of all batch axes (e.g., time × echo × TI ...),
      inner_ndim = number of inner axes (e.g., Z, X, Y).

    • Default: ones  → scaling disabled at zero cost.
    • inv_s2_for_shard(sh, anchor) returns (B_loc, 1, ..., 1) on the shard device.
    • Small per‑device caches to avoid repeated host→device copies.
    """
    B: int
    inner_ndim: int
    device_master: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32

    # internal state
    _val: torch.Tensor = field(init=False, repr=False)         # (B,1,...,1) on CPU
    _inv_s2: torch.Tensor = field(init=False, repr=False)      # (B,1,...,1) on CPU
    _cache_inv_s2: Dict[torch.device, torch.Tensor] = field(default_factory=dict, repr=False)
    _dirty: bool = field(default=False, repr=False)

    def __post_init__(self):
        shape = (self.B,) + tuple(1 for _ in range(self.inner_ndim))
        self._val = torch.ones(shape, dtype=self.dtype, device=self.device_master)
        self._inv_s2 = torch.ones_like(self._val)
        self._cache_inv_s2.clear()
        self._dirty = False

    # ---------------------------- public API ----------------------------
    @torch.no_grad()
    def set(self, value: torch.Tensor | float | int):
        """
        Set the scale. Accepts:
          • scalar,
          • (B,) vector,
          • broadcastable to (B, 1, ..., 1),
          • or already in canonical shape (B, 1, ..., 1).
        """
        if not torch.is_tensor(value):
            v = torch.as_tensor(value, dtype=self.dtype, device=self.device_master)
        else:
            v = value.to(self.dtype, non_blocking=True).to(self.device_master)

        # reshape/broadcast to canonical (B, 1, ..., 1)
        if v.ndim == 1 and v.shape[0] == self.B:
            v = v.view(self.B, *([1] * self.inner_ndim))
        elif v.ndim == 0:
            v = v.view(1, *([1] * self.inner_ndim)).expand(self.B, *([1] * self.inner_ndim))
        else:
            v = v.expand(self.B, *([1] * self.inner_ndim))

        v = v.contiguous()
        v.clamp_(min=1e-12)  # avoid zero / negative
        self._val = v
        self._recompute_inv_s2()

    @torch.no_grad()
    def as_tensor(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
        t = self._val
        if device is not None:
            t = t.to(device, non_blocking=True)
        if dtype is not None:
            t = t.to(dtype)
        return t

    @torch.no_grad()
    def inv_s2_for_shard(self, sh, anchor: torch.Tensor | None) -> torch.Tensor:
        """
        Broadcast 1/s^2 to the shard's batch length as (B_loc, 1, 1, ...).
        Uses a per‑device full‑B cache and returns a view/slice.
        """
        dev = anchor.device if (anchor is not None) else torch.device("cpu")
        inv_s2_full = self._get_or_build_device_cache(dev, anchor=anchor)
        B0, B1 = sh.b_slice.start, sh.b_slice.stop
        return inv_s2_full[B0:B1]

    @torch.no_grad()
    def inv_s_for_shard(self, sh, anchor: torch.Tensor | None) -> torch.Tensor:
        """
        Return 1/s on shard device (B_loc, 1, ..., 1).
        """
        inv_s2 = self.inv_s2_for_shard(sh, anchor)
        return torch.sqrt(inv_s2)

    @torch.no_grad()
    def update_ema(self, new_vals: torch.Tensor | float, alpha: float = 0.9):
        """
        Exponential moving average for per‑batch scale values.
        new_vals can be scalar or (B,) or (B,1,...,1) on any device.
        """
        if not torch.is_tensor(new_vals):
            v = torch.tensor(new_vals, dtype=self._val.dtype, device=self._val.device)
        else:
            v = new_vals.to(self._val.dtype, non_blocking=True)
            if v.ndim == 1 and v.shape[0] == self.B:
                v = v.view(self.B, *([1] * self.inner_ndim))
            elif v.ndim == 0:
                v = v.view(1, *([1]*self.inner_ndim)).expand_as(self._val)
            else:
                v = v.expand_as(self._val)
        # EMA in place on CPU master
        self._val.mul_(alpha).add_(v, alpha=(1.0 - alpha)).clamp_(min=1e-12)
        self._recompute_inv_s2()
    # graspcg/workspace/scale.py  (inside ScaleField)
    def s_for_shard(self, sh, *, anchor: torch.Tensor | None = None) -> torch.Tensor:
        """Return per-batch scale (B_loc,1,1,...) on shard's device."""
        dev = anchor.device if anchor is not None else sh.x.device
        B_loc = sh.b_slice.stop - sh.b_slice.start
        view = self.s[sh.b_slice]                 # (B_loc,)
        return view.to(dev).view(B_loc, *([1]*self.inner_ndim))
    # alias (sometimes nicer to call)
    ema_update = update_ema

    @torch.no_grad()
    def invalidate(self):
        """Clear device caches (call if you updated internals externally)."""
        self._cache_inv_s2.clear()
        self._dirty = False

    # ------------------------- internal helpers -------------------------
    @torch.no_grad()
    def _recompute_inv_s2(self):
        self._val.clamp_(min=1e-12)
        self._inv_s2 = (1.0 / self._val).square_()
        self._cache_inv_s2.clear()
        self._dirty = False

    @torch.no_grad()
    def _get_or_build_device_cache(self, dev: torch.device, *, anchor: Optional[torch.Tensor]):
        if self._dirty or dev not in self._cache_inv_s2:
            # materialize full‑B inv_s2 on device once, keep it tiny
            if anchor is not None:
                inv = self._inv_s2.to(anchor.device, non_blocking=True)
            else:
                inv = self._inv_s2.to(dev, non_blocking=True)
            inv = inv.contiguous()  # contiguous so views are cheap
            self._cache_inv_s2[inv.device] = inv  # key by actual device
            self._dirty = False
        return self._cache_inv_s2[dev]
    