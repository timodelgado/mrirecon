# graspcg/ops/basic.py
import torch, math
from typing import Tuple, Callable
from ..workspace.unified_arena import UnifiedArena

@torch.no_grad()
def forward_diff(x: torch.Tensor, dim: int, out: torch.Tensor) -> torch.Tensor:
    # out shape must match x with size(dim)-1 along dim
    s = [slice(None)]*x.ndim
    s1, s2 = s.copy(), s.copy()
    s1[dim] = slice(1, None)
    s2[dim] = slice(0, -1)
    torch.sub(x[tuple(s1)], x[tuple(s2)], out=out)
    return out

@torch.no_grad()
def huber_den_sq(z_real: torch.Tensor, z_imag: torch.Tensor, eps: float, out: torch.Tensor):
    # out = sqrt(zr^2 + zi^2 + eps^2)  (use float buffer)
    torch.mul(z_real, z_real, out=out)
    out.addcmul_(z_imag, z_imag).add_(eps*eps).sqrt_()
    return out

@torch.no_grad()
def suggest_tile(per_slice_vox: int, arena: UnifiedArena, dtype, device, *,
                 user_cap: int | None = None, safety: float = 0.9) -> int:
    if device.type == "cpu":
        return user_cap or (1<<30)
    elem = torch.tensor([], dtype=dtype).element_size()
    free = arena.free_elems(dtype, device=device) * elem
    if free == 0:
        free, _ = torch.cuda.mem_get_info(device)
    cap = int(free * safety // max(1, per_slice_vox * elem))
    tile = 1 << (cap.bit_length()-1) if cap > 0 else 1
    return min(tile, user_cap) if user_cap else tile
