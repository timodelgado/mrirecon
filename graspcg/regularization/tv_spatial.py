from __future__ import annotations
import math, torch
from typing import Tuple, Optional
from ..workspace.cg_workspace   import CGWorkspace
from ..workspace.unified_arena  import UnifiedArena
from .reg_registry         import register, register_diag, register_stats, register_diag_shard
from ..utils.operations         import quantile

class SpatialTV:
    """
    3D anisotropic spatial TV with Huber smoothing.
    Works per shard, tiles along the first inner spatial axis S1.

    Config in ws.regs["tv_s"]:
      weight       : float              λ
      eps          : float              ε (Huber)
      voxel_size   : (d1, d2, d3)       spacing for (S1,S2,S3)
      tile_s1      : Optional[int]      slices per tile along S1 (auto if None)
      apply_scale  : bool               if True, TV is on u=x/s (default True)
    """
    key = "tv_s"

    @classmethod
    def from_ws(cls, ws: CGWorkspace) -> "SpatialTV":
        cfg = ws.regs.get(cls.key, {})
        return cls(weight=float(cfg["weight"]),
                   eps=float(cfg["eps"]),
                   voxel_size=tuple(cfg["voxel_size"]),
                   tile_s1=cfg.get("tile_s1"),
                   apply_scale=bool(cfg.get("apply_scale", True)))

    def __init__(self,
                 weight: float,
                 eps: float,
                 voxel_size: Tuple[float, float, float],
                 tile_s1: Optional[int] = None,
                 apply_scale: bool = True):
        self.lam = float(weight)
        self.eps = float(eps)
        self.d1, self.d2, self.d3 = [float(v) for v in voxel_size]
        self.tile_s1 = tile_s1
        self.apply_scale = bool(apply_scale)

    @torch.no_grad()
    def energy_and_grad(self, ws: CGWorkspace) -> float:
        E = 0.0
        for sh,_ in ws.iter_shards():
            E += self._eval_shard(ws, sh)
        return float(E)

    @torch.no_grad()
    def add_diag(self, ws: CGWorkspace, diag: torch.Tensor) -> None:
        # identify shard by storage pointer using data_ptr()
        sh = None
        diag_ptr = diag.data_ptr()
        for s,_ in ws.iter_shards():
            if hasattr(s, "diag") and s.diag.data_ptr() == diag_ptr:
                sh = s
                break
        if sh is None:
            sh,_ = next(ws.iter_shards())
        self.add_diag_shard(ws, sh, diag)

    @torch.no_grad()
    def add_diag_shard(self, ws: CGWorkspace, sh, diag: torch.Tensor) -> None:
        lam = self.lam
        d1, d2, d3 = self.d1, self.d2, self.d3
        _, S1, S2, S3 = sh.x.shape[:4]

        inv_d1_2, inv_d2_2, inv_d3_2 = (1/d1)**2, (1/d2)**2, (1/d3)**2
        deg_xy = torch.full((S2, S3), 4., dtype=diag.dtype, device=diag.device)
        deg_xy[[0,-1],:] -= 1; deg_xy[:,[0,-1]] -= 1
        diag_xy = deg_xy * (inv_d2_2 + inv_d3_2)           # (S2,S3)
        diag_xy = diag_xy.unsqueeze(0).repeat(S1,1,1)      # (S1,S2,S3)

        lap_s1 = torch.ones((S1,), dtype=diag.dtype, device=diag.device)
        if S1 >= 3: lap_s1[1:-1] = 2.0
        diag_s1 = lap_s1.view(S1,1,1).expand_as(diag_xy) * inv_d1_2

        if self.apply_scale:
            inv_s2 = ws.scale.inv_s2_for_shard(sh, anchor=sh.x)      # (B,1,1,1…)
        else:
            inv_s2 = 1.0

        term = (diag_xy + diag_s1).unsqueeze(0) * inv_s2 * (lam / 3.0)
        diag.add_(term)

    # -------------------------- internals ---------------------------
    @torch.no_grad()
    def _eval_shard(self, ws: CGWorkspace, sh) -> float:
        """
        Spatial TV over inner dims (S1,S2,S3) for shard x:(B,S1,S2,S3,*),
        tiled along S1 to manage memory.
        Uses correct chain rule:
          if apply_scale=True (TV on u=x/s) then
            denom = sqrt( (1/s^2)*|Δx|^2 + ε^2 )
            ∂E/∂x gets an extra (1/s^2) factor.
        """
        x     = sh.x
        arena = ws.arena
        dev   = x.device
        assert x.ndim >= 4, "Spatial TV expects (B, S1, S2, S3, ...)"

        B, S1, S2, S3 = x.shape[:4]
        if min(S1, S2, S3) < 1 or B < 1:
            return 0.0

        # choose tile along S1 (skip last slice)
        base = B * S2 * S3
        tile_s1 = self.tile_s1 if self.tile_s1 is not None else _suggest_tile(
            base, arena, x.dtype, dev, user_default=max(1, S1-1)
        )

        max_s1 = min(tile_s1, max(1, S1-1))
        max_shp = (B, max_s1, S2, S3)
        d_buf   = arena.request(math.prod(max_shp), x.dtype,       anchor=x).view(max_shp)
        den_buf = arena.request(math.prod(max_shp), torch.float32, anchor=x).view(max_shp)

        # per‑batch (1/s^2) for denom and gradient
        inv_s2 = ws.scale.inv_s2_for_shard(sh, anchor=x) if self.apply_scale else None

        lam_div3 = self.lam / 3.0
        eps2     = self.eps * self.eps
        E        = 0.0

        for s10 in range(0, S1-1, tile_s1):
            span  = min(tile_s1, S1-1-s10)
            sl_s1 = slice(0, span)

            # ---- ΔS2 -------------------------------------------------
            dx = d_buf[:, sl_s1, :S2-1, :]
            torch.sub(x[:, s10:s10+span, 1:, :],
                      x[:, s10:s10+span, :-1, :],
                      out=dx)
            dx.div_(self.d2)

            den = den_buf[:, sl_s1, :S2-1, :]
            torch.mul(dx.real, dx.real, out=den)
            den.addcmul_(dx.imag, dx.imag)
            if inv_s2 is not None:
                den.mul_(inv_s2)  # broadcast (B,1,1,1)
            den.add_(eps2).sqrt_()

            E += float((den.sum() - den.numel()*self.eps) * lam_div3)

            # sign(d) = d / denom
            dx.div_(den)
            _accum_spatial_grad(sh, dx, axis=2, s10=s10,
                                spacing=self.d2, lam_div3=lam_div3, inv_s2=inv_s2)

            # ---- ΔS3 -------------------------------------------------
            dy = d_buf[:, sl_s1, :, :S3-1]
            torch.sub(x[:, s10:s10+span, :, 1:],
                      x[:, s10:s10+span, :, :-1],
                      out=dy)
            dy.div_(self.d3)

            den = den_buf[:, sl_s1, :, :S3-1]
            torch.mul(dy.real, dy.real, out=den)
            den.addcmul_(dy.imag, dy.imag)
            if inv_s2 is not None:
                den.mul_(inv_s2)
            den.add_(eps2).sqrt_()

            E += float((den.sum() - den.numel()*self.eps) * lam_div3)

            dy.div_(den)
            _accum_spatial_grad(sh, dy, axis=3, s10=s10,
                                spacing=self.d3, lam_div3=lam_div3, inv_s2=inv_s2)

            # ---- ΔS1 -------------------------------------------------
            dz = d_buf[:, sl_s1, :, :]
            torch.sub(x[:, s10+1:s10+1+span, :, :],
                      x[:, s10    :s10    +span, :, :],
                      out=dz)
            dz.div_(self.d1)

            den = den_buf[:, sl_s1, :, :]
            torch.mul(dz.real, dz.real, out=den)
            den.addcmul_(dz.imag, dz.imag)
            if inv_s2 is not None:
                den.mul_(inv_s2)
            den.add_(eps2).sqrt_()

            E += float((den.sum() - den.numel()*self.eps) * lam_div3)

            dz.div_(den)
            _accum_spatial_grad(sh, dz, axis=1, s10=s10,
                                spacing=self.d1, lam_div3=lam_div3, inv_s2=inv_s2)

        arena.release(d_buf); arena.release(den_buf)
        return E

# --------------------------- registry glue ----------------------------
@register("tv_s")
@torch.no_grad()
def tv_spatial(ws: CGWorkspace) -> float:
    return SpatialTV.from_ws(ws).energy_and_grad(ws)

@register_diag("tv_s")
@torch.no_grad()
def diag_tv_spatial(ws: CGWorkspace, diag: torch.Tensor):
    SpatialTV.from_ws(ws).add_diag(ws, diag)

@register_diag_shard("tv_s")
@torch.no_grad()
def diag_tv_spatial_shard(ws: CGWorkspace, sh, diag: torch.Tensor):
    tv = SpatialTV.from_ws(ws)
    tv.add_diag_shard(ws, sh, diag)

@register_stats("tv_s")
@torch.no_grad()
def stats_tv_spatial(ws: CGWorkspace, xs: torch.Tensor, *, percentile: float, eps_floor: float):
    """
    Percentile/MAD on spatial diffs. If tv_s.apply_scale=True, stats are
    computed on u = x/s (per‑batch division). Includes voxel anisotropy.
    """
    if xs.ndim < 4:
        return max(eps_floor, 1e-6), 0.0

    cfg  = ws.regs.get("tv_s", {})
    d1, d2, d3 = map(float, cfg.get("voxel_size", (1.0, 1.0, 1.0)))
    if bool(cfg.get("apply_scale", True)):
        s = ws.scale.as_tensor().to(xs.real.dtype).to(xs.device)
        s = s[: xs.shape[0]].view(xs.shape[0], *([1] * (xs.ndim - 1)))
        xs = xs / s

    g1 = xs.diff(dim=1).abs().div_(d1)
    g2 = xs.diff(dim=2).abs().div_(d2)
    g3 = xs.diff(dim=3).abs().div_(d3)
    d_abs = torch.cat([g1.reshape(-1), g2.reshape(-1), g3.reshape(-1)])
    eps   = max(quantile(d_abs, percentile).item(), eps_floor)
    med   = quantile(d_abs, 0.5).item()
    sigma = quantile((d_abs - med).abs(), 0.5).item() / 0.6745
    return eps, sigma

# ------------------------------ helpers -------------------------------
def _suggest_tile(target_elems: int,
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

@torch.no_grad()
def _accum_spatial_grad(sh,
                        sign_d: torch.Tensor,
                        *,
                        axis: int,      # 1 (S1), 2 (S2), 3 (S3)
                        s10: int,
                        spacing: float,
                        lam_div3: float,
                        inv_s2: torch.Tensor | None):
    """
    Accumulate ∇ for a spatial difference field within current S1 tile.
    If inv_s2 is not None, multiply sign_d in place by 1/s^2 (chain rule).
    """
    g    = sh.g
    coef = lam_div3 / spacing

    # apply 1/s^2 once, then use alpha=±coef to avoid temp tensors
    if inv_s2 is not None:
        sign_d.mul_(inv_s2)

    if axis == 1:  # S1 (z-like)
        g[:, s10      , :, :].add_(sign_d[:, 0], alpha=-coef)
        if sign_d.shape[1] > 1:
            g[:, s10+1 : s10+sign_d.shape[1], :, :].add_(sign_d[:, :-1] - sign_d[:, 1:], alpha=coef)
        g[:, s10 + sign_d.shape[1], :, :].add_(sign_d[:, -1], alpha=coef)

    elif axis == 2:  # S2 (x-like)
        g_slice = g[:, s10:s10+sign_d.shape[1], :, :]
        g_slice[:, :, :-1, :].add_(sign_d, alpha=-coef)
        g_slice[:, :,  1:, :].add_(sign_d, alpha= coef)

    elif axis == 3:  # S3 (y-like)
        g_slice = g[:, s10:s10+sign_d.shape[1], :, :]
        g_slice[:, :, :, :-1].add_(sign_d, alpha=-coef)
        g_slice[:, :, :,  1:].add_(sign_d, alpha= coef)

    else:
        raise ValueError("axis must be 1, 2 or 3")
