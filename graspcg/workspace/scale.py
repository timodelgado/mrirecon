# graspcg/workspace/scale.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch

@dataclass
class ScaleField:
    """
    Lightweight per‑frame scale (stores 1/s on CPU; caches by device lazily).

    Canonical storage: inv = 1/s with shape (B, 1, ..., 1) on CPU.
    - If you don't want any scaling, simply don't attach this to ws.
    - Per‑device cache is materialised only on demand and shard returns are views.
    """
    B: int
    inner_ndim: int
    device_master: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32
    pin_cpu: bool = False  # optional: pin CPU buffer for faster H2D non_blocking copies

    _inv: torch.Tensor = field(init=False, repr=False)              # (B,1,...,1) on CPU
    _cache_inv: Dict[torch.device, torch.Tensor] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        shape = (self.B,) + (1,) * self.inner_ndim
        self._inv = torch.ones(shape, dtype=self.dtype, device=self.device_master)
        if self.pin_cpu and self._inv.device.type == "cpu":
            # Optional pin to speed up .to(..., non_blocking=True)
            try:
                self._inv = self._inv.pin_memory()
            except RuntimeError:
                pass
        self._cache_inv.clear()

    # ---------------------------- public API ----------------------------
    @torch.no_grad()
    def set(self, s: torch.Tensor | float | int):
        """
        Set s (scalar, (B,), broadcastable to (B,1,...,1), or already canonical).
        We store 1/s with clipping for numerical safety.
        """
        if not torch.is_tensor(s):
            s_t = torch.as_tensor(s, dtype=self.dtype, device=self.device_master)
        else:
            s_t = s.to(self.dtype, non_blocking=True).to(self.device_master)

        if s_t.ndim == 1 and s_t.shape[0] == self.B:
            s_t = s_t.view(self.B, *([1] * self.inner_ndim))
        elif s_t.ndim == 0:
            s_t = s_t.view(1, *([1] * self.inner_ndim)).expand(self.B, *([1] * self.inner_ndim))
        else:
            s_t = s_t.expand(self.B, *([1] * self.inner_ndim))

        s_t = s_t.contiguous().clamp_min_(1e-12)
        inv = (1.0 / s_t)
        if self.pin_cpu and inv.device.type == "cpu":
            try:
                inv = inv.pin_memory()
            except RuntimeError:
                pass
        self._inv = inv
        self._cache_inv.clear()

    @torch.no_grad()
    def inv_for_shard(self, sh, anchor: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Return 1/s as (B_loc,1,...,1) *on the shard device*. Uses a per‑device
        cache of the full‑B tensor and returns a view slice (no extra VRAM when unused).
        """
        dev = anchor.device if (anchor is not None) else self.device_master
        inv_full = self._get_or_build_device_cache(dev, anchor=anchor)
        B0, B1 = sh.b_start, sh.b_stop
        return inv_full[B0:B1]

    # Backwards/alternative spelling used by RegManager
    inv_s_for_shard = inv_for_shard

    @torch.no_grad()
    def inv2_for_shard(self, sh, anchor: Optional[torch.Tensor]) -> torch.Tensor:
        inv = self.inv_for_shard(sh, anchor)
        return inv * inv

    # Backwards/alternative spelling
    inv_s2_for_shard = inv2_for_shard

    @torch.no_grad()
    def invalidate(self):
        self._cache_inv.clear()

    # ------------------------- internal helpers -------------------------
    @torch.no_grad()
    def _get_or_build_device_cache(self, dev: torch.device, *, anchor: Optional[torch.Tensor]):
        if dev not in self._cache_inv:
            target = anchor.device if anchor is not None else dev
            inv = self._inv.to(target, non_blocking=True).contiguous()
            self._cache_inv[dev] = inv  # key by requested device for robustness
        return self._cache_inv[dev]