# graspcg/ops/reg_graph.py
from __future__ import annotations
import torch
from typing import Dict, Tuple
from graspcg.ops.reg_registry import register, register_diag, register_stats
from graspcg.ops.graph_data import GraphCOO

# ------------------------ utilities ------------------------
def _flatten_views(sh):
    """Return flattened views of (x, g) for a shard."""
    x_flat = sh.x.view(-1)
    g_flat = sh.g.view(-1)
    return x_flat, g_flat

def _inv_s2_flat(ws, sh):
    """Return inv_s2 expanded to shard and flattened: (Tloc*Nz*Nx*Ny,)."""
    Tloc = sh.x.shape[0]
    inv_s2 = (1.0 / ws.s_t[sh.t_slice].to(sh.x.device)).square().view(Tloc,1,1,1)
    return inv_s2.expand_as(sh.x).reshape(-1)  # broadcast then flatten

def _edge_chunks(total_edges: int, elems_free: int) -> int:
    """
    Decide chunk length M for edges based on arena capacity.
    We need a handful of edge-sized temporaries; be conservative.
    """
    if elems_free <= 0:
        return max(4096, min(total_edges, 1<<15))
    # assume ~8 scalars per edge scratch budget
    M = max(4096, min(total_edges, elems_free // 8))
    return M

# ============================================================
#  Graph Laplacian (L2 on edges)
# ============================================================
@register("graph_l2")
@torch.no_grad()
def reg_graph_l2(ws) -> torch.Tensor:
    """
    E += (λ/2) Σ w_ij |x_i - x_j|^2 (with symmetric s_t weighting)
    g += λ Σ w_ij (inv_s2_i + inv_s2_j)/2 * (x_i - x_j) at i, and the opposite at j
    Expected config:
      ws.regs["graph_l2"] = {
         "graph": GraphCOO(...),   # host-side COO
         "weight": float,
         "prepared": { shard_id: (src_loc, dst_loc, w) }  # auto-filled on first call
      }
    """
    cfg = ws.regs.get("graph_l2", {})
    lam = float(cfg.get("weight", 0.0))
    if lam == 0.0:
        return torch.tensor(0.0, device=ws.shards[0].device)

    graph: GraphCOO = cfg["graph"]
    if "prepared" not in cfg:
        # prepare per-shard subgraphs once
        prepared = {}
        for sid, (sh, _) in enumerate(ws.iter_shards()):
            prepared[sid] = graph.subgraph_for_tslice(ws.dims, sh.t_slice)
        cfg["prepared"] = prepared
        ws.regs["graph_l2"] = cfg

    E = 0.0
    Nt,Nz,Nx,Ny = ws.dims
    nodes_per_t = Nz*Nx*Ny

    for sid, (sh, _) in enumerate(ws.iter_shards()):
        dev = sh.x.device
        x_flat, g_flat = _flatten_views(sh)
        invs2 = _inv_s2_flat(ws, sh)   # [N_nodes_loc]
        src_loc, dst_loc, w = cfg["prepared"][sid]

        if src_loc.numel() == 0:
            continue

        # chunk policy from arena
        elems_free = ws.arena.free_elems(torch.float32, device=dev)
        M = _edge_chunks(src_loc.numel(), elems_free)

        # process in chunks
        for i0 in range(0, src_loc.numel(), M):
            i1 = min(src_loc.numel(), i0+M)
            s_idx = src_loc[i0:i1].to(dev, non_blocking=True)
            d_idx = dst_loc[i0:i1].to(dev, non_blocking=True)
            ww    = w[i0:i1].to(dev, non_blocking=True)

            xi = x_flat.index_select(0, s_idx)
            xj = x_flat.index_select(0, d_idx)
            diff = xi - xj  # complex ok

            # symmetric per-edge coefficient c = w * 0.5*(inv_s2_i + inv_s2_j)
            ci = invs2.index_select(0, s_idx)   # real
            cj = invs2.index_select(0, d_idx)
            c  = ww * 0.5 * (ci + cj)           # [M] real

            # energy increment: 0.5*λ * sum c * |diff|^2
            abs2 = (diff.real*diff.real + diff.imag*diff.imag)
            E += float(0.5 * lam * (c * abs2).sum().item())

            # gradient increment
            gi = c * diff                      # complex scalar per edge
            g_flat.index_add_(0, s_idx, lam * gi)
            g_flat.index_add_(0, d_idx, -lam * gi)

    return torch.tensor(E, device=ws.shards[0].device)

@register_diag("graph_l2")
@torch.no_grad()
def diag_graph_l2(ws, diag: torch.Tensor):
    """
    Add λ * 0.5*(deg_i + deg_i) = λ*deg_i, with inv_s2 symmetry:
    diag += λ * inv_s2_i * deg_i          (simple, fast)
    (You can also use inv_s2_i + inv_s2_j averaging, but that needs edge pass.)
    """
    cfg = ws.regs.get("graph_l2", {})
    lam = float(cfg.get("weight", 0.0))
    if lam == 0.0: return
    graph: GraphCOO = cfg["graph"]
    deg = graph.degree()  # CPU
    for sh,_ in ws.iter_shards():
        dev = sh.x.device
        # local degree slice for this shard
        Nt_loc = sh.x.shape[0]
        nodes_per_t = ws.dims[1]*ws.dims[2]*ws.dims[3]
        offset = sh.t_slice.start * nodes_per_t
        loc_deg = deg[offset: offset + Nt_loc*nodes_per_t].to(dev, non_blocking=True)
        invs2   = _inv_s2_flat(ws, sh)  # [N_loc]
        # add λ * inv_s2 * deg per voxel
        diag.view(-1).add_(lam * invs2 * loc_deg.to(invs2.dtype))

@register_stats("graph_l2")
@torch.no_grad()
def stats_graph_l2(ws, xs, *, percentile: float = 0.9, eps_floor: float = 1e-6):
    """
    Suggest a σ from MAD of edge differences on a *pilot* subset of edges.
    Caller can set λ = κ * σ as in your existing scheme. (ε not used.)
    """
    cfg = ws.regs.get("graph_l2", {})
    graph: GraphCOO = cfg["graph"]
    M = min(200_000, graph.src.numel())
    if M == 0: return eps_floor, 0.0
    idx = torch.randint(0, graph.src.numel(), (M,))
    # sample on CPU; move small blocks to device if needed
    nodes_per_t = ws.dims[1]*ws.dims[2]*ws.dims[3]
    # Take the first shard for sampling
    sh,_ = next(ws.iter_shards())
    x_flat = sh.x.view(-1).detach().cpu()
    xi = x_flat[graph.src[idx]]
    xj = x_flat[graph.dst[idx]]
    d  = (xi - xj).abs().float()
    med = torch.quantile(d, 0.5).item()
    mad = torch.quantile((d - med).abs(), 0.5).item() / 0.6745
    return eps_floor, max(mad, eps_floor)


@register("graph_tv")
@torch.no_grad()
def reg_graph_tv(ws) -> torch.Tensor:
    """
    E += λ Σ w_ij * ( sqrt(|δ|^2 + ε^2) - ε ) * 0.5*(inv_s2_i + inv_s2_j)
    g += λ Σ w_ij * ( δ / sqrt(|δ|^2 + ε^2) ) * 0.5*(inv_s2_i + inv_s2_j)
    """
    cfg = ws.regs.get("graph_tv", {})
    lam = float(cfg.get("weight", 0.0))
    eps = float(cfg.get("eps", 1e-3))
    if lam == 0.0: return torch.tensor(0.0, device=ws.shards[0].device)

    graph: GraphCOO = cfg["graph"]
    if "prepared" not in cfg:
        prepared = {}
        for sid, (sh,_) in enumerate(ws.iter_shards()):
            prepared[sid] = graph.subgraph_for_tslice(ws.dims, sh.t_slice)
        cfg["prepared"] = prepared; ws.regs["graph_tv"] = cfg

    E = 0.0
    for sid, (sh,_) in enumerate(ws.iter_shards()):
        dev = sh.x.device
        x_flat, g_flat = _flatten_views(sh)
        invs2 = _inv_s2_flat(ws, sh)
        src_loc, dst_loc, w = cfg["prepared"][sid]
        if src_loc.numel() == 0: continue

        elems_free = ws.arena.free_elems(torch.float32, device=dev)
        M = _edge_chunks(src_loc.numel(), elems_free)

        for i0 in range(0, src_loc.numel(), M):
            i1 = min(src_loc.numel(), i0+M)
            s_idx = src_loc[i0:i1].to(dev, non_blocking=True)
            d_idx = dst_loc[i0:i1].to(dev, non_blocking=True)
            ww    = w[i0:i1].to(dev, non_blocking=True)

            xi = x_flat.index_select(0, s_idx)
            xj = x_flat.index_select(0, d_idx)
            diff = xi - xj
            abs2 = (diff.real*diff.real + diff.imag*diff.imag)
            denom = (abs2 + eps*eps).sqrt_() + 1e-12

            ci = invs2.index_select(0, s_idx)
            cj = invs2.index_select(0, d_idx)
            c  = ww * 0.5 * (ci + cj)

            E += float(lam * (c * (denom - eps)).sum().item())

            scale = c / denom
            gi = scale * diff
            g_flat.index_add_(0, s_idx, lam * gi)
            g_flat.index_add_(0, d_idx, -lam * gi)

    return torch.tensor(E, device=ws.shards[0].device)

@register_diag("graph_tv")
@torch.no_grad()
def diag_graph_tv(ws, diag: torch.Tensor):
    """
    Safe bound: diag += λ * deg / max(ε, eps_floor)
    (Set cfg["diag_mode"]="live" to recompute a tighter value per-iter if desired.)
    """
    cfg = ws.regs.get("graph_tv", {})
    lam = float(cfg.get("weight", 0.0))
    eps = float(cfg.get("eps", 1e-3))
    if lam == 0.0: return
    graph: GraphCOO = cfg["graph"]
    deg = graph.degree()  # CPU
    for sh,_ in ws.iter_shards():
        dev = sh.x.device
        Nt_loc = sh.x.shape[0]
        nodes_per_t = ws.dims[1]*ws.dims[2]*ws.dims[3]
        offset = sh.t_slice.start * nodes_per_t
        loc_deg = deg[offset: offset + Nt_loc*nodes_per_t].to(dev, non_blocking=True)
        invs2   = _inv_s2_flat(ws, sh)
        diag.view(-1).add_( lam * invs2 * (loc_deg / max(eps, 1e-9)) )


def build_spatial_corr_graph(x_low: torch.Tensor,   # [T,Z,X,Y] (real or complex)
                             mask: torch.Tensor | None,
                             K: int = 6,
                             r_xy: int = 3, r_z: int = 0,
                             lag: int = 0, lag_penalty: float = 0.0,
                             tile_xy: int = 32) -> GraphCOO:
    """
    Returns a *spatial template* graph over Nsp = Z*X*Y nodes on CPU:
      src/dst in [0, Nsp), w >= 0
    Later, the handler will replicate per-time when constructing shard subgraphs.
    """
    with torch.no_grad():
        T,Z,X,Y = x_low.shape
        # 1) real-valued curves
        curves = x_low.real if torch.is_complex(x_low) else x_low
        curves = curves.to(torch.float32)

        # 2) z-score over time
        c = curves.view(T, -1).clone()
        m = c.mean(0, keepdim=True); s = c.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
        zc = (c - m) / s                                               # [T, Nsp]

        # 3) optional mask of valid spatial nodes
        if mask is not None:
            mask_flat = mask.view(-1).bool().cpu()
        else:
            mask_flat = torch.ones(Z*X*Y, dtype=torch.bool)

        # 4) iterate tiles in (x,y) for each z
        src_all, dst_all, w_all = [], [], []
        for z in range(Z):
            for x0 in range(0, X, tile_xy):
                x1 = min(X, x0+tile_xy)
                for y0 in range(0, Y, tile_xy):
                    y1 = min(Y, y0+tile_xy)
                    # centres in this tile
                    px = torch.arange(x0, x1); py = torch.arange(y0, y1)
                    P = [(xx,yy) for xx in px for yy in py]
                    if not P: continue
                    # candidate window bounds
                    wx0 = max(0, x0 - r_xy); wx1 = min(X, x1 + r_xy)
                    wy0 = max(0, y0 - r_xy); wy1 = min(Y, y1 + r_xy)
                    # flatten indices helpers
                    def idx(z, x, y): return (z*X + x)*Y + y

                    # gather indices
                    centres = torch.tensor([idx(z,xx,yy) for xx,yy in P])
                    centres = centres[mask_flat[centres]]
                    if centres.numel() == 0: continue

                    cand_xy = [(xx,yy) for xx in range(wx0,wx1) for yy in range(wy0,wy1)]
                    cands = torch.tensor([idx(z,xx,yy) for xx,yy in cand_xy])
                    cands = cands[mask_flat[cands]]
                    if cands.numel() == 0: continue

                    # time matrices
                    B = zc[:, centres]       # [T, B]
                    C = zc[:, cands]         # [T, C]

                    # small-lag cross-corr
                    best = None
                    for L in range(-lag, lag+1):
                        if L == 0:
                            BC = C.T @ B                      # [C, B]
                        elif L > 0:
                            BC = C[L:,:].T @ B[:-L,:]
                        else:
                            BC = C[:L,:].T @ B[-L:,:]
                        if lag_penalty:
                            BC = BC - lag_penalty*abs(L)
                        best = BC if best is None else torch.maximum(best, BC)

                    sim = best if best is not None else (C.T @ B)   # [C, B]
                    sim = torch.clamp(sim / float(T), min=0.0, max=1.0)  # ρ in [0,1]

                    # top-K per centre
                    Keff = min(K, sim.shape[0])
                    vals, idxs = torch.topk(sim, Keff, dim=0)        # [Keff, B]
                    dst = centres.repeat(Keff)                       # B times
                    src = cands[idxs.reshape(-1)]
                    w   = vals.reshape(-1)

                    src_all.append(src.cpu()); dst_all.append(dst.cpu()); w_all.append(w.cpu())

        src = torch.cat(src_all) if src_all else torch.empty(0, dtype=torch.int64)
        dst = torch.cat(dst_all) if dst_all else torch.empty(0, dtype=torch.int64)
        w   = torch.cat(w_all)   if w_all   else torch.empty(0, dtype=torch.float32)

        # symmetrise (optional): keep max weight in both directions
        key = src * (Z*X*Y) + dst
        rev = dst * (Z*X*Y) + src
        take = w >= 0
        src, dst, w = src[take], dst[take], w[take]
        return GraphCOO(num_nodes=Z*X*Y, src=src.to(torch.int64), dst=dst.to(torch.int64), w=w.to(torch.float32))
