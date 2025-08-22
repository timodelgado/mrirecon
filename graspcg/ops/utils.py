# graspcg/ops/utils.py
import torch
from ..workspace.unified_arena import UnifiedArena

@torch.no_grad()
def like(ref: torch.Tensor, *size, dtype=None):
    """Return empty tensor with same device/dtype as `ref` (unless overridden)."""
    return torch.empty(size if size else ref.shape,
                       dtype=dtype or ref.dtype,
                       device=ref.device)

@torch.no_grad()
def quantile(t, q):             # tiny helper
    k = int((t.numel()-1)*q)
    return t.flatten().kthvalue(k+1).values

@torch.no_grad()
def suggest_tile(target_elems: int,
                  arena: UnifiedArena,
                  dtype,
                  dev: torch.device,
                  safety: float = 0.9,
                  user_default: int | None = None) -> int:
    if dev.type == "cpu":
        return user_default or (1 << 30)
    elem_size = torch.tensor([], dtype=dtype).element_size()
    free_elems = arena.free_elems(dtype, device=dev)
    free_bytes = free_elems * elem_size
    if free_bytes == 0:
        free_bytes, _ = torch.cuda.mem_get_info(dev)
    cap = int(free_bytes * safety // (target_elems * elem_size))
    tile = 1 << (cap.bit_length() - 1) if cap > 0 else 1
    if user_default:
        tile = min(tile, user_default)
    return max(1, tile)