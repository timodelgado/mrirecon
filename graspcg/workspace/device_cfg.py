# device_cfg.py ────────────────────────────────────────────────────────────
"""
Runtime device configuration with multi‑GPU awareness.

Key points
----------
* `compute`  – primary device where solver state tensors live.
* `helpers`  – optional list of extra CUDA devices for operators that can
               distribute work (e.g. NUFFT frame sharding).
* `stream_for(dev)` – returns a dedicated CUDA stream, created lazily.
* Back‑compat: the old `.device`, `.torch_device()`, and `.use()` still work.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import contextlib, torch
from typing import List, Dict, Union


# ── dummy context for CPU scopes ────────────────────────────────────────────
class _NullCtx:
    def __enter__(self): ...
    def __exit__(self, *exc): ...


# ── main dataclass ─────────────────────────────────────────────────────────
@dataclass(kw_only=True)
class DeviceCfg:
    # ------------------------------------------------------------------ API
    compute : Union[str, int] = "cuda"          # e.g. "cpu", 0, "cuda:1"
    helpers : List[Union[str, int]] | None = None
    pin_memory : bool = True
    streams : Dict[int, torch.cuda.Stream] = field(default_factory=dict)

    # ------------------------------------------------------------------ init
    def __post_init__(self):
        # keep legacy attribute in sync
        object.__setattr__(self, "device", self.compute)

        # normalise helper list
        if self.helpers is None:
            if self.compute_device().type == "cuda":
                all_ids = list(range(torch.cuda.device_count()))
                self.helpers = [i for i in all_ids
                                if i != self.compute_device().index]
            else:
                self.helpers = []

    # ---------------------------------------------------------------- helpers
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

    # legacy name
    torch_device = compute_device

    # ................................................................. streams
    def stream_for(self,
                   dev: Union[int, str, torch.device, None] = None
                   ) -> torch.cuda.Stream:
        """Return a (lazily created) CUDA stream for the chosen device."""
        d = self.compute_device() if dev is None else torch.device(dev)
        if d.type != "cuda":
            raise ValueError("Streams are only meaningful on CUDA devices.")
        if d.index not in self.streams:
            self.streams[d.index] = torch.cuda.Stream(device=d)
        return self.streams[d.index]

    # ................................................................. helpers
    def cuda_devices(self) -> List[torch.device]:
        """Primary + helper CUDA devices, deduplicated and ordered."""
        p = self.compute_device()
        helpers = [torch.device(f"cuda:{h}") if not isinstance(h, torch.device)
                   else h for h in self.helpers]
        uniq, seen = [], set()
        for d in [p] + helpers:
            if d.type == "cuda" and d.index not in seen:
                uniq.append(d); seen.add(d.index)
        return uniq

    # ................................................................. context
    def use(self,
            dev: Union[int, str, torch.device, None] = None):
        """
        Context manager that sets the active CUDA device; no‑op for CPU.

        If `dev` is None, the primary compute device is used.
        """
        target = self.compute_device() if dev is None else torch.device(dev)
        return torch.cuda.device(target) if target.type == "cuda" else _NullCtx()