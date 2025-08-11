from __future__ import annotations
import torch
from graspcg.ops.reg_registry import register, register_diag, register_stats
from graspcg.inr.eval import inr_predict_into

@register("inr_l2")
@torch.no_grad()
def reg_inr_l2(ws) -> torch.Tensor:
    """
    E += (λ/2) * Σ |x - s_t fθ|^2 / s_t^2
    g += λ * (x - s_t fθ) / s_t^2
    Expect ws.regs["inr_l2"] = {"module": inr, "weight": λ, "tile": {"z":16,"x":64,"y":64}, "out_complex": True}
    """
    cfg = ws.regs.get("inr_l2", {})
    lam = float(cfg.get("weight", 0.0))
    if lam == 0.0:
        return torch.tensor(0.0, device=ws.shards[0].device if ws.shards else ws.x.device)

    inr = cfg["module"]
    out_complex = bool(cfg.get("out_complex", True))
    # tile sizes (defaults are conservative)
    tz = int(cfg.get("tile", {}).get("z", 16))
    tx = int(cfg.get("tile", {}).get("x", 64))
    ty = int(cfg.get("tile", {}).get("y", 64))

    E_total = 0.0
    Nt, Nz, Nx, Ny = ws.dims

    for sh, _ in ws.iter_shards():
        dev = sh.device
        Tloc = sh.x.shape[0]
        # scratch buffer for predictions, re-used for diff to avoid extra alloc
        pred = ws.arena.request(sh.x.numel(), sh.x.dtype, anchor=sh.x).view_as(sh.x)

        t_abs0 = sh.t_slice.start
        for z0 in range(0, Nz, tz):
            z1 = min(Nz, z0+tz)
            for x0 in range(0, Nx, tx):
                x1 = min(Nx, x0+tx)
                for y0 in range(0, Ny, ty):
                    y1 = min(Ny, y0+ty)

                    # 1) predict x-units into pred-tile
                    tile = pred[:, z0:z1, x0:x1, y0:y1]
                    inr_predict_into(ws=ws, sh=sh, inr=inr, t_abs0=t_abs0,
                                     z0=z0, z1=z1, x0=x0, x1=x1, y0=y0, y1=y1,
                                     out_tile=tile, out_complex=out_complex)

                    # 2) diff := x - pred  (re-use pred as the diff buffer)
                    tile.neg_().add_(sh.x[:, z0:z1, x0:x1, y0:y1])

                    # 3) energy & grad with inv_s2 = (1/s_t)^2
                    inv_s2 = (1.0 / ws.s_t[sh.t_slice].to(dev)).square_().view(Tloc,1,1,1)
                    # energy: sum(|diff|^2 * inv_s2)
                    # compute |diff|^2 in-place using real/imag views (no extra alloc)
                    abs2 = (tile.real**2 + tile.imag**2)
                    abs2.mul_(inv_s2)
                    E_total += float(0.5 * lam * abs2.sum().item())

                    # grad: g += lam * diff * inv_s2
                    tile.mul_(inv_s2)
                    sh.g[:, z0:z1, x0:x1, y0:y1].add_(tile, alpha=lam)

        ws.arena.release(pred)

    return torch.tensor(E_total, device=ws.shards[0].device if ws.shards else ws.x.device)

@register_diag("inr_l2")
@torch.no_grad()
def diag_inr_l2(ws, diag: torch.Tensor):
    cfg = ws.regs.get("inr_l2", {})
    lam = float(cfg.get("weight", 0.0))
    if lam == 0.0:
        return
    inv_s2 = (1.0 / ws.s_t.squeeze()).square().view(ws.dims[0],1,1,1).to(diag.device)
    # add λ/s_t^2 voxelwise (broadcast-safe scalar or tensor)
    diag.add_(lam * inv_s2)

@register_stats("inr_l2")
@torch.no_grad()
def stats_inr_l2(ws, xs, *, percentile: float = 0.9, eps_floor: float = 1e-6):
    """
    Suggest σ from MAD of |x/s_t| (amplitude scale); ε not used here.
    """
    x_scaled = xs / ws.s_t.to(xs.device)
    flat = x_scaled.abs().reshape(-1)
    if flat.numel() == 0:
        return eps_floor, 0.0
    med = torch.quantile(flat, 0.5).item()
    mad = torch.quantile((flat - med).abs(), 0.5).item() / 0.6745
    return eps_floor, max(mad, eps_floor)

# somewhere central (helper)
def add_inr_l2(ws, module, weight: float, out_complex: bool = True,
               tile: dict | None = None):
    ws.regs["inr_l2"] = {
        "module": module,
        "weight": float(weight),
        "out_complex": bool(out_complex),
        "tile": dict(tile or {})
    }