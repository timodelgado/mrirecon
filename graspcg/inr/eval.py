from __future__ import annotations
import torch
from graspcg.inr.coords import coords_tile_abs_t

@torch.no_grad()
def inr_predict_into(*, ws, sh, inr, t_abs0: int,
                     z0: int, z1: int, x0: int, x1: int, y0: int, y1: int,
                     out_tile: torch.Tensor,         # view into arena scratch, dtype like x
                     out_complex: bool = True):
    """
    Fill `out_tile[T,Z,X,Y]` with INR predictions **in x-units** (not x/s_t):
      pred_x = s_t * f_theta(coords)
    Assumes out_tile is allocated on sh.device & matches sh.x dtype.
    """
    dev = sh.device
    T = sh.x.shape[0]  # local frames in shard
    coords = coords_tile_abs_t(t_abs0=t_abs0, t_abs1=t_abs0+T,
                               z0=z0, z1=z1, x0=x0, x1=x1, y0=y0, y1=y1,
                               dims=ws.dims, device=dev, dtype=torch.float32)
    # chunk along flattened voxels to limit peak mem
    N = coords.numel() // 4
    # heuristic chunk size from arena free elems
    max_elems = max(1, ws.arena.free_elems(torch.float32, device=dev) // 16)
    step = max(1, min(N, max_elems))

    cview = coords.view(N, 4)
    oview = out_tile.view(N) if out_complex else out_tile.view(N)
    s_t   = ws.s_t[sh.t_slice].to(dev).view(T,1,1,1)  # [T,1,1,1]
    i = 0
    while i < N:
        j = min(N, i+step)
        chunk = inr.to_device(dev)(cview[i:j])           # [M,2] or [M,1], float
        if out_complex:
            re = chunk[..., 0].to(out_tile.real.dtype)
            im = chunk[..., 1].to(out_tile.real.dtype)
            out = torch.complex(re, im)                  # [M]
        else:
            out = chunk[..., 0].to(out_tile.dtype)
        # scale to x-units: multiply by s_t (broadcast via reshape)
        out = out.view(T, z1-z0, x1-x0, y1-y0)
        out.mul_(s_t)
        oview[i:j] = out.reshape(-1).to(out_tile.dtype)
        i = j
