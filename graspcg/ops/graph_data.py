# graspcg/ops/graph_data.py
from __future__ import annotations
import torch
from dataclasses import dataclass

@dataclass
class GraphCOO:
    """
    Sparse, undirected edge list:
      num_nodes = Nt*Nz*Nx*Ny  (global, full volume)
      src, dst: int64 indices in [0, num_nodes)
      w       : float32 weights (>=0)
    In practice, build this once (e.g. kNN over patches / temporal neighbours).
    """
    num_nodes: int
    src: torch.Tensor  # [M] int64 (CPU)
    dst: torch.Tensor  # [M] int64 (CPU)
    w:   torch.Tensor  # [M] float32 (CPU)

    def degree(self) -> torch.Tensor:
        """Return degree per node (CPU tensor, float32)."""
        deg = torch.zeros(self.num_nodes, dtype=torch.float32)
        deg.index_add_(0, self.src, self.w)
        deg.index_add_(0, self.dst, self.w)
        return deg

    def subgraph_for_tslice_spatial(template: GraphCOO, dims, t_slice: slice):
        Nt,Nz,Nx,Ny = dims
        nodes_per_t = Nz*Nx*Ny
        Tloc = t_slice.stop - t_slice.start
        base_src, base_dst, w = template.src, template.dst, template.w
        # replicate per local frame
        src_loc = []
        dst_loc = []
        for τ in range(Tloc):
            off = τ * nodes_per_t
            src_loc.append(base_src + off)
            dst_loc.append(base_dst + off)
        return torch.cat(src_loc), torch.cat(dst_loc), w.repeat(Tloc)



