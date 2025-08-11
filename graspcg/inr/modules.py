from __future__ import annotations
import torch, torch.nn as nn

class INRModule(nn.Module):
    """
    coords [..., D] -> values [..., C]
    If x is complex, set out_complex=True and output C=2 (Re, Im).
    """
    def __init__(self, in_dim: int = 4, out_complex: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_complex = out_complex

    def to_device(self, device: torch.device) -> "INRModule":
        return self.to(device)

class SIREN(INRModule):
    def __init__(self, in_dim=4, hidden=128, depth=4, out_complex=True):
        super().__init__(in_dim=in_dim, out_complex=out_complex)
        def lin(i, o):
            l = nn.Linear(i, o); nn.init.kaiming_uniform_(l.weight, a=math.sqrt(5))
            if l.bias is not None: nn.init.zeros_(l.bias)
            return l
        import math
        layers = [lin(in_dim, hidden), nn.SiLU()]
        for _ in range(depth-1): layers += [lin(hidden, hidden), nn.SiLU()]
        out_dim = 2 if out_complex else 1
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)  # [..., 2] or [...,1]
