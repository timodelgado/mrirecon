# graspcg/utils/device.py
from dataclasses import dataclass
import torch, re

def _canonical(d):
    return torch.device(
        d if isinstance(d, str) else ("cuda:"+str(d) if isinstance(d,int) else d)
    )

@dataclass(frozen=True)
class DeviceConfig:
    compute: torch.device = _canonical("cuda:0")  # where x,g,dx live
    scratch: torch.device = _canonical("cuda:0")  # arena slabs

    @classmethod
    def from_args(cls,
                  device: str | int | torch.device | None = None,
                  scratch_device: str | int | torch.device | None = None):
        return cls(
            compute=_canonical(device or "cuda:0"),
            scratch=_canonical(scratch_device or device or "cuda:0"))
        
