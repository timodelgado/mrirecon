# graspcg/workspace/unified_arena.py — unified device+arena manager
"""
DeviceArena: unified device/stream configuration + scratch allocator.

This class consolidates the roles of the old DeviceCfg, DevicePool, and
UnifiedArena into a single, shareable object.

Highlights
---------
✓ Per‑device *and* per‑dtype reuse pools (reduces fragmentation)
✓ Device enumeration (primary + helpers) and CUDA stream management
✓ Smart device selection with soft CPU fallback based on free memory
✓ Idle‑trim of unused slabs and optional per‑dtype byte caps
✓ Backwards compatibility: `UnifiedArena = DeviceArena`
"""
from __future__ import annotations

import math
import time
import bisect
import collections
from typing import Dict, List, Tuple, Union

import torch
import psutil


# ── dummy context for CPU scopes ────────────────────────────────────────────
class _NullCtx:
    def __enter__(self): ...
    def __exit__(self, *exc): ...


class DeviceArena:
    """
    Unified device manager and scratch allocator.

    Public API (stable)
    -------------------
    # Device/config:
      • compute_device() -> torch.device
      • cuda_devices()   -> list[torch.device]
      • stream_for(dev)  -> torch.cuda.Stream
      • use(dev)         -> context manager for active CUDA device

    # Scratch allocation:
      • request(numel, dtype, *, anchor=None, device=None, reserve_frac=None)
      • release(tensor)
      • trim(max_idle=None)
      • free_elems(dtype, device)
    """

    _Pool = collections.namedtuple("_Pool", "sizes bufs")   # per (dev,dtype)

    # ......................................................................
    def __init__(self,
                 *,
                 compute: Union[str, int] = "cuda",
                 helpers: List[Union[str, int]] | None = None,
                 pin_memory: bool = True,
                 reserve_frac: float = 1.3,
                 max_idle: float = 120.0,
                 max_bytes: int | None = None):
        # device/config
        self.compute = compute
        self.pin_memory = bool(pin_memory)
        self._streams: Dict[int, torch.cuda.Stream] = {}

        # normalise helper list similar to DeviceCfg
        if helpers is None:
            if self.compute_device().type == "cuda":
                all_ids = list(range(torch.cuda.device_count()))
                helpers = [i for i in all_ids
                           if i != self.compute_device().index]
            else:
                helpers = []
        self.helpers: List[Union[str, int]] = helpers

        # allocator knobs
        self.reserve_frac = float(reserve_frac)
        self.max_idle     = float(max_idle)
        self.max_bytes    = max_bytes

        # (device,dtype) → _Pool(sizes=[int], bufs=[Tensor])
        self._pools: Dict[Tuple[torch.device, torch.dtype], DeviceArena._Pool] = (
            collections.defaultdict(lambda: DeviceArena._Pool([], []))
        )
        # slab‑id → last‑access timestamp
        self._touch: Dict[int, float] = {}

    # ----------------------------------------------------------------- config
    def compute_device(self) -> torch.device:
        """Return primary compute device (alias: torch_device)."""
        d = self.compute
        if d == "cpu":
            return torch.device("cpu")
        if d in ("cuda", "gpu"):
            return torch.device("cuda", torch.cuda.current_device())
        if isinstance(d, str) and d.startswith("cuda"):
            return torch.device(d)
        return torch.device("cuda", int(d))

    # legacy alias
    torch_device = compute_device

    def cuda_devices(self) -> List[torch.device]:
        """Primary + helper CUDA devices, deduplicated and ordered."""
        p = self.compute_device()
        helpers = [torch.device(f"cuda:{h}") if not isinstance(h, torch.device) else h
                   for h in (self.helpers or [])]
        uniq, seen = [], set()
        for d in [p] + helpers:
            if d.type == "cuda" and d.index not in seen:
                uniq.append(d); seen.add(d.index)
        return uniq

    def stream_for(self,
                   dev: Union[int, str, torch.device, None] = None
                   ) -> torch.cuda.Stream:
        """Return a (lazily created) CUDA stream for the chosen device."""
        d = self.compute_device() if dev is None else torch.device(dev)
        if d.type != "cuda":
            raise ValueError("Streams are only meaningful on CUDA devices.")
        if d.index not in self._streams:
            self._streams[d.index] = torch.cuda.Stream(device=d)
        return self._streams[d.index]

    def use(self, dev: Union[int, str, torch.device, None] = None):
        """Context manager that sets the active CUDA device; no‑op for CPU."""
        target = self.compute_device() if dev is None else torch.device(dev)
        return torch.cuda.device(target) if target.type == "cuda" else _NullCtx()

    # ------------------------------------------------------------ alloc helper
    def _mem_free_bytes(self, dev: torch.device) -> int:
        if dev.type == "cuda":
            free_phys, _ = torch.cuda.mem_get_info(dev)
            cached = torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(dev)
            return int(free_phys + max(0, cached))
        return int(psutil.virtual_memory().available)

    def _select_device(self, bytes_needed: int, prefer: torch.device | None) -> torch.device:
        # 1) prefer if large enough
        if prefer is not None and self._mem_free_bytes(prefer) > bytes_needed:
            return prefer
        # 2) first CUDA device with room
        for d in self.cuda_devices():
            if self._mem_free_bytes(d) > bytes_needed:
                return d
        # 3) CPU fallback
        return torch.device("cpu")

    # --------------------------------------------------------------- allocation
    @torch.no_grad()
    def request(self,
                numel: int,
                dtype: torch.dtype,
                *,
                anchor: torch.Tensor | None = None,
                device: torch.device | str | int | None = None,
                reserve_frac: float | None = None) -> torch.Tensor:
        """
        Get a 1‑D scratch tensor with at least `numel` elements.

        Placement priority
        ------------------
        1. if `anchor` given  ⇒ same device as anchor
        2. elif `device`      ⇒ that device
        3. else               ⇒ best device selected by free memory

        Returned tensor is a *view* of the underlying slab and can be reshaped
        freely by the caller.
        """
        if anchor is not None and device is not None:
            raise ValueError("Pass either 'anchor' or 'device', not both.")

        # decide device
        if anchor is not None:
            dev = anchor.device
        elif device is not None:
            dev = device if isinstance(device, torch.device) else torch.device(device)
        else:
            bytes_needed = numel * torch.empty((), dtype=dtype).element_size()
            dev = self._select_device(bytes_needed, prefer=None)

        key = (dev, dtype)
        sizes, bufs = self._pools[key]

        # fast O(log k) lookup
        idx = bisect.bisect_left(sizes, numel)
        if idx < len(sizes):
            buf = bufs.pop(idx); sizes.pop(idx)
            self._touch[id(buf)] = time.time()
            return buf[:numel]

        # need new slab
        n_alloc = math.ceil(numel * (reserve_frac or self.reserve_frac))
        try_on_dev = (dev.type == "cuda")

        # safety VRAM check
        if try_on_dev:
            free_phys, _ = torch.cuda.mem_get_info(dev)
            cached = torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(dev)
            eff_free = int(free_phys + max(0, cached))
            bytes_alloc = n_alloc * torch.empty((), dtype=dtype).element_size()
            if bytes_alloc > eff_free:
                try_on_dev = False

        loc = dev if try_on_dev else torch.device("cpu")
        buf = torch.empty(
            n_alloc, dtype=dtype, device=loc,
            pin_memory=(loc.type == "cpu" and self.pin_memory)
        )

        # Do NOT register freshly allocated slabs in the free list. The slab
        # becomes available to the pool when the caller releases the view.
        self._touch[id(buf)] = time.time()
        return buf[:numel]

    # ......................................................................
    def release(self, tensor: torch.Tensor):
        key = (tensor.device, tensor.dtype)
        sizes, stor, bufs = self._pools[key]
        view_elems = tensor.numel()
        storage_elems = tensor.storage().size()  # true slab
        idx = bisect.bisect_left(sizes, view_elems)
        sizes.insert(idx, view_elems)
        stor.insert(idx, storage_elems)
        bufs.insert(idx, tensor)
        self._pools[key] = self._Pool(sizes, stor, bufs)
        self._touch[id(tensor)] = time.time()
        self._maybe_evict(key)
    def _maybe_evict(self, key):
        if self.max_bytes is None:
            return
        sizes, stor, bufs = self._pools[key]
        total_bytes = sum(se * bufs[0].element_size() for se in stor)  # all slabs' storage
        while total_bytes > self.max_bytes and bufs:
            # evict oldest by last‑touch
            oldest_ix = min(range(len(bufs)), key=lambda j: self._touch.get(id(bufs[j]), 0))
            total_bytes -= stor[oldest_ix] * bufs[oldest_ix].element_size()
            for arr in (sizes, stor, bufs):
                arr.pop(oldest_ix)
        if not bufs:
            del self._pools[key]
        torch.cuda.empty_cache()
    # ......................................................................
    def trim(self, *, max_idle: float | None = None):
        """Drop slabs idle for longer than `max_idle` (default: self.max_idle)."""
        if max_idle is None:
            max_idle = self.max_idle
        cutoff = time.time() - max_idle
        for key in list(self._pools):
            sizes, bufs = self._pools[key]
            keep_sizes, keep_bufs = [], []
            for s, b in zip(sizes, bufs):
                if self._touch.get(id(b), 0) > cutoff:
                    keep_sizes.append(s); keep_bufs.append(b)
            if keep_bufs:
                self._pools[key] = self._Pool(keep_sizes, keep_bufs)
            else:
                del self._pools[key]
        torch.cuda.empty_cache()

    # ......................................................................
    def free_elems(self, dtype: torch.dtype,
                   device: torch.device | int | str = "cuda") -> int:
        """How many *additional* elements fit into the current slab (0 if none)."""
        dev = (device if isinstance(device, torch.device) else torch.device(device))
        key = (dev, dtype)
        sizes, _ = self._pools.get(key, ([], []))
        return max(sizes) if sizes else 0

    # ......................................................................
    # internal helpers
    # ......................................................................
    def _evict_oldest(self, key):
        """Remove oldest slab for (device,dtype) until under byte cap."""
        sizes, bufs = self._pools[key]
        limit = self.max_bytes if self.max_bytes is not None else float("inf")
        while sizes and sizes[-1] * bufs[-1].element_size() > limit:
            sizes.pop(-1); bufs.pop(-1)
        if not sizes:
            del self._pools[key]
        torch.cuda.empty_cache()


# Backwards compatibility: allow importing the old name
UnifiedArena = DeviceArena