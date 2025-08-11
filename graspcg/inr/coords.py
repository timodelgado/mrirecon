from __future__ import annotations
import torch
from typing import Tuple

@torch.no_grad()
def coords_tile_abs_t(*,
                      t_abs0: int, t_abs1: int,
                      z0: int, z1: int, x0: int, x1: int, y0: int, y1: int,
                      dims: Tuple[int,int,int,int],  # (Nt,Nz,Nx,Ny)
                      device, dtype=torch.float32) -> torch.Tensor:
    """
    Normalised coords in [-1,1] for the absolute time range [t_abs0:t_abs1)
    and spatial box. Returned shape [T,Z,X,Y,4] with order (t,z,x,y).
    """
    Nt,Nz,Nx,Ny = dims
    tt = torch.arange(t_abs0, t_abs1, device=device, dtype=dtype).view(-1,1,1,1)
    zz = torch.arange(z0, z1,          device=device, dtype=dtype).view(1,-1,1,1)
    xx = torch.arange(x0, x1,          device=device, dtype=dtype).view(1,1,-1,1)
    yy = torch.arange(y0, y1,          device=device, dtype=dtype).view(1,1,1,-1)
    tt = 2*tt/(Nt-1) - 1
    zz = 2*zz/(Nz-1) - 1
    xx = 2*xx/(Nx-1) - 1
    yy = 2*yy/(Ny-1) - 1
    T, Z, X, Y = (t_abs1 - t_abs0), (z1 - z0), (x1 - x0), (y1 - y0)
    t = tt.expand(T,Z,X,Y); z = zz.expand(T,Z,X,Y)
    x = xx.expand(T,Z,X,Y); y = yy.expand(T,Z,X,Y)
    return torch.stack([t, z, x, y], dim=-1)  # [T,Z,X,Y,4]
