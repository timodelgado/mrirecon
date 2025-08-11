# device_pool.py ─────────────────────────────────────────────────────────
import torch, psutil
from typing import List
# ────────────────────────────────────────────────────────────────────────────
# Optional helper for smart device selection
# (kept local to avoid import cycles; drop‑in identical to old workspace.device_pool)
# ────────────────────────────────────────────────────────────────────────────
class _Target(torch.nn.Module):
    __slots__ = ("dev", "free")
    def __init__(self, dev: torch.device):
        super().__init__()
        self.dev, self.free = dev, 0


class DevicePool:
    """
    Ordered list of CUDA GPUs followed by an implicit CPU “device”.
    Querying free memory is cheap (<50 µs) so we update live.
    """
    def __init__(self, gpu_ids: List[int] | None = None):
        self.targets: List[_Target] = []
        if gpu_ids is None:
            gpu_ids = list(range(torch.cuda.device_count()))
        for i in gpu_ids:
            self.targets.append(_Target(torch.device("cuda", i)))
        self.targets.append(_Target(torch.device("cpu")))         # always last

    # ......................................................................
    def refresh(self):
        for t in self.targets:
            if t.dev.type == "cuda":
                t.free, _ = torch.cuda.mem_get_info(t.dev)
            else:
                import psutil
                t.free = psutil.virtual_memory().available

    # ......................................................................
    def claim(self,
              bytes_needed: int,
              prefer: torch.device | None = None
              ) -> torch.device:
        """
        Select a device that can currently fit `bytes_needed`:
            1) `prefer`      – if large enough
            2) first GPU     – with room
            3) CPU           – always succeeds
        """
        self.refresh()
        # 1)
        if prefer is not None:
            for t in self.targets:
                if t.dev == prefer and t.free > bytes_needed:
                    return t.dev
        # 2)
        for t in self.targets:
            if t.dev.type == "cuda" and t.free > bytes_needed:
                return t.dev
        # 3)
        return self.targets[-1].dev
