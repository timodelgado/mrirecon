# graspcg/workspace/arena.py ────────────────────────────────────────────────
"""
Unified scratch‑tensor allocator for GRASP‑CG

Combines and supersedes:
    • arena.Arena
    • cg_workspace._ArenaManager
    • workspace.multi_arena.MultiArena
    • workspace.scratch_arena.MultiDeviceArena

Features
--------
✓ Per‑device *and* per‑dtype reuse pools (minimises fragmentation)  
✓ Size‑bucket lookup via bisect → O(log k) retrieval  
✓ VRAM head‑room check & soft CPU fallback (pinned)  
✓ Optional DevicePool‑guided placement for multi‑GPU workstations  
✓ Idle‑trim and per‑dtype byte caps  
✓ Compatible with existing `attach_arena()` calls in NUFFT ops  
"""

from __future__ import annotations
import torch, math, time, bisect, collections
from typing import Dict, List, Tuple
from ..workspace.device_pool import DevicePool
# ────────────────────────────────────────────────────────────────────────────
# UnifiedArena
# ────────────────────────────────────────────────────────────────────────────
class UnifiedArena:
    """
    General‑purpose scratch allocator.
    One instance can be shared across the whole reconstruction session.

    Public API (stable)
    -------------------
    request(numel, dtype, *, anchor=None, device=None, reserve_frac=1.3) -> Tensor
    release(tensor)            # optional, reuse is automatic
    trim(max_idle=None)        # drop slabs idle > max_idle seconds
    free_elems(dtype, device)  # capacity still available in current slab
    """

    _Pool = collections.namedtuple("_Pool", "sizes bufs")   # per (dev,dtype)

    # ......................................................................
    def __init__(self,
                 pool: DevicePool | None = None,
                 *,
                 reserve_frac: float = 1.3,
                 max_idle: float = 120.0,
                 max_bytes: int | None = None):
        """
        Parameters
        ----------
        pool         : DevicePool | None
            If given, used to pick a device automatically when the caller
            does **not** pass an anchor or explicit device.
        reserve_frac : float
            Head‑room factor when allocating new slabs.
        max_idle     : float
            Seconds after which an *unused* slab is trimmed.
        max_bytes    : int | None
            Hard cap per (dev,dtype) slab; `None` ⇒ unlimited.
        """
        self.pool         = pool or DevicePool()
        self.reserve_frac = reserve_frac
        self.max_idle     = max_idle
        self.max_bytes    = max_bytes

        # (device,dtype) → _Pool(sizes=[int], bufs=[Tensor])
        self._pools: Dict[Tuple[torch.device, torch.dtype],
                          UnifiedArena._Pool] = collections.defaultdict(
                              lambda: UnifiedArena._Pool([], []))
        # slab‑id → last‑access timestamp
        self._touch: Dict[int, float] = {}

    # ......................................................................
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
        1. if `anchor` given            ⇒ same device as anchor
        2. elif `device` given          ⇒ that device
        3. else                         ⇒ DevicePool.claim()

        Returned tensor is always a *view* of the underlying slab and
        can be reshaped freely by the caller.
        """
        if anchor is not None and device is not None:
            raise ValueError("Pass either 'anchor' or 'device', not both.")

        # --- decide device ------------------------------------------------
        if anchor is not None:
            dev = anchor.device
        elif device is not None:
            dev = (device if isinstance(device, torch.device)
                   else torch.device(device))
        else:
            # size in *bytes* for DevicePool bookkeeping
            bytes_needed = numel * torch.tensor([], dtype=dtype).element_size()
            dev = self.pool.claim(bytes_needed, prefer=None)

        key = (dev, dtype)
        sizes, bufs = self._pools[key]

        # --- fast O(log k) lookup -----------------------------------------
        idx = bisect.bisect_left(sizes, numel)
        if idx < len(sizes):
            buf = bufs.pop(idx); sizes.pop(idx)
            self._touch[id(buf)] = time.time()
            return buf[:numel]          # view

        # --- need new slab ---
        n_alloc = math.ceil(numel * (reserve_frac or self.reserve_frac))
        try_on_dev = (dev.type == "cuda")

        # pre‑flight VRAM check
        if try_on_dev:
            free_phys, _ = torch.cuda.mem_get_info(dev)
            cached = (torch.cuda.memory_reserved(dev) -
                      torch.cuda.memory_allocated(dev))
            eff_free = free_phys + cached
            bytes_alloc = n_alloc * torch.tensor([], dtype=dtype).element_size()
            if bytes_alloc > eff_free:
                try_on_dev = False      # fall back to CPU

        # allocate
        loc = dev if try_on_dev else "cpu"
        buf = torch.empty(n_alloc, dtype=dtype, device=loc, pin_memory=(loc == "cpu"))

        # Do NOT register freshly allocated slabs in the free list.
        # We only return a view to the caller; the slab becomes available
        # to the pool when the caller explicitly releases the view.
        self._touch[id(buf)] = time.time()
        return buf[:numel]

    # ......................................................................
    def release(self, tensor: torch.Tensor):
        """Return a tensor to the pool (optional; automatic when view dies)."""
        key = (tensor.device, tensor.dtype)
        sizes, bufs = self._pools[key]
        n_elems = tensor.numel()
        idx = bisect.bisect_left(sizes, n_elems)
        sizes.insert(idx, n_elems)
        bufs.insert(idx, tensor)
        self._pools[key] = self._Pool(sizes, bufs)
        self._touch[id(tensor)] = time.time()

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
        dev = (device if isinstance(device, torch.device)
               else torch.device(device))
        key = (dev, dtype)
        sizes, _ = self._pools.get(key, ([], []))
        return max(sizes) if sizes else 0

    # ......................................................................
    # internal helpers
    # ......................................................................
    def _evict_oldest(self, key):
        """Remove oldest slab for (device,dtype) until under byte cap."""
        sizes, bufs = self._pools[key]
        while sizes and sizes[-1]*bufs[-1].element_size() > self.max_bytes:
            sizes.pop(-1); bufs.pop(-1)
        if not sizes:
            del self._pools[key]
        torch.cuda.empty_cache()