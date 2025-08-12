# graspcg/ops/tvt.py
from __future__ import annotations
import math, torch
from typing import Optional

from ..workspace.cg_workspace   import CGWorkspace
from ..workspace.unified_arena  import UnifiedArena
from .reg_registry         import (
    register, register_diag, register_diag_shard, register_stats
)
from graspcg.utils.operations         import quantile


class TemporalTV:
    """
    1D Huber-TV along a shard's leading axis (typically time).
    - Energy and ∇ computed in-place (tiled; scratch via arena).
    - Optional per-frame scaling via ws.scale (chain-rule weights 1/s and 1/s^2).
    """

    key = "tv_t"

    # -------------------------- construction --------------------------
    @classmethod
    def from_ws(cls, ws: CGWorkspace) -> "TemporalTV":
        cfg = ws.regs.get(cls.key, {})
        return cls(
            weight     = float(cfg.get("weight", 0.0)),
            eps        = float(cfg.get("eps", 1e-3)),
            tile       = cfg.get("tile", None),
            apply_scale= bool(cfg.get("apply_scale", True)),
        )

    def __init__(self, *, weight: float, eps: float,
                 tile: Optional[int] = None, apply_scale: bool = True):
        self.weight      = float(weight)
        self.eps         = float(eps)
        self.tile        = tile
        self.apply_scale = bool(apply_scale)

    # -------------------------- public API ----------------------------
    @torch.no_grad()
    def energy_and_grad(self, ws: CGWorkspace) -> float:
        E = 0.0
        for sh, _ in ws.iter_shards():
            E += self._eval_shard(ws, sh)
        return float(E)

    @torch.no_grad()
    def add_diag_shard(self, ws: CGWorkspace, sh, diag: torch.Tensor) -> None:
        """
        Add this TV's diagonal contribution for a *specific shard*.

        diag shape: same spatial shape as sh.diag (real, per-voxel)
        """
        lam = self.weight
        B_loc = sh.x.shape[0]
        # 1D Laplacian degrees along the leading axis
        lap = torch.ones((B_loc,), dtype=diag.dtype, device=diag.device)
        if B_loc >= 3:
            lap[1:-1] = 2.0

        # chain rule for u = x/s ⇒ Hessian-ish scales as 1/s^2
        if self.apply_scale and hasattr(ws, "scale"):
            inv_s2 = ws.scale.inv_s2_for_shard(sh, anchor=sh.x)  # (B_loc,1,1,...)
            term = (lam * lap.view(B_loc, *([1]*(diag.ndim-1)))) * inv_s2
        else:
            term = lam * lap.view(B_loc, *([1]*(diag.ndim-1)))

        diag.add_(term)

    @torch.no_grad()
    def add_diag(self, ws: CGWorkspace, diag: torch.Tensor) -> None:
        """
        Convenience wrapper: resolve which shard owns `diag`, then delegate
        to add_diag_shard(ws, sh, diag).
        """
        sh = None
        for s, _ in ws.iter_shards():
            if getattr(s, "diag", None) is not None and s.diag.data_ptr() == diag.data_ptr():
                sh = s
                break
        if sh is None:
            # Fallback: assume single shard
            sh, _ = next(ws.iter_shards())
        self.add_diag_shard(ws, sh, diag)

    # For initialisation / continuation
    @staticmethod
    @torch.no_grad()
    def estimate_stats(ws: CGWorkspace, xs: torch.Tensor, *, percentile: float, eps_floor: float):
        """
        Returns (eps, sigma) estimated on a shard-like pilot xs: (B_loc, *inner).
        """
        if xs.ndim < 2 or xs.shape[0] < 2:
            return max(eps_floor, 1e-6), 0.0
        dt    = xs.diff(dim=0).abs()
        eps   = max(quantile(dt, percentile).item(), eps_floor)
        med   = quantile(dt, 0.5).item()
        sigma = quantile((dt - med).abs(), 0.5).item() / 0.6745
        return eps, sigma

    # -------------------------- internals -----------------------------
    @torch.no_grad()
    def _eval_shard(self, ws: CGWorkspace, sh) -> float:
        """
        Computes energy and accumulates gradient into sh.g (in-place).
        Tiled along the leading axis; no large temporary allocations.
        """
        x     = sh.x
        arena = ws.arena
        dev   = x.device

        if x.ndim < 2:
            return 0.0

        B_loc = x.shape[0]
        inner = x.shape[1:]
        vox   = math.prod(inner)

        # Choose tile (# of frame-differences per sweep)
        tile = self.tile
        if tile is None:
            tile = _suggest_tile(vox, arena, x.dtype, dev,
                                 user_default=max(1, B_loc - 1))

        # Scratch sized to the largest possible span this iter
        max_span = min(tile, max(1, B_loc - 1))
        dt  = arena.request(max_span * vox, x.dtype,      anchor=x).view(max_span, *inner)
        den = arena.request(max_span * vox, torch.float32, anchor=x).view(max_span, *inner)

        # Small per-batch scales
        if self.apply_scale and hasattr(ws, "scale"):
            inv_s2 = ws.scale.inv_s2_for_shard(sh, anchor=x)  # (B_loc,1,1,…)
            inv_s  = torch.sqrt(inv_s2)
        else:
            inv_s  = None  # no scaling

        eps2 = self.eps * self.eps
        E    = 0.0

        for b0 in range(0, B_loc - 1, tile):
            span  = min(tile, B_loc - 1 - b0)
            dt_s  = dt[:span]
            den_s = den[:span]

            if inv_s is not None:
                # dt_s = x[b+1]/s[b+1] - x[b]/s[b]
                torch.mul(x[b0+1:b0+1+span], inv_s[b0+1:b0+1+span], out=dt_s)
                dt_s.addcmul_(x[b0:b0+span], inv_s[b0:b0+span], value=-1.0)
            else:
                torch.sub(x[b0+1:b0+1+span], x[b0:b0+span], out=dt_s)

            # denom & energy
            torch.mul(dt_s.real.float(), dt_s.real.float(), out=den_s)
            den_s.addcmul_(dt_s.imag.float(), dt_s.imag.float()).add_(eps2).sqrt_()
            E += float(den_s.sum().item() - den_s.numel()*self.eps)

            # sign(dt) and ∇
            dt_s.div_(den_s)
            _accum_1d_grad(sh.g, dt_s, b0, self.weight, inv_s)

        arena.release(dt); arena.release(den)
        return self.weight * E


# ---------------------------- registry glue ----------------------------
@register("tv_t")
@torch.no_grad()
def reg_tv_t(ws: CGWorkspace) -> float:
    """Legacy registry handler: energy + ∇ in place."""
    return TemporalTV.from_ws(ws).energy_and_grad(ws)

@register_diag("tv_t")
@torch.no_grad()
def reg_tv_t_diag(ws: CGWorkspace, diag: torch.Tensor) -> None:
    """Legacy registry diag helper: resolve shard and add its contribution."""
    TemporalTV.from_ws(ws).add_diag(ws, diag)

@register_diag_shard("tv_t")
@torch.no_grad()
def reg_tv_t_diag_shard(ws: CGWorkspace, sh, diag: torch.Tensor) -> None:
    """Shard‑explicit diag helper; preferred when your preconditioner iterates shards."""
    TemporalTV.from_ws(ws).add_diag_shard(ws, sh, diag)

# REPLACE the whole reg_tv_t_stats() in graspcg/ops/tvt.py with this:

@register_stats("tv_t")
@torch.no_grad()
def reg_tv_t_stats(ws: CGWorkspace,
                   xs: torch.Tensor,
                   *,
                   percentile: float,
                   eps_floor: float):
    """
    Temporal‑TV stats (ε, σ) computed on the provided pilot `xs`.

    NOTE:
      • This helper is intentionally SCALE‑AGNOSTIC.
      • If your policy requires stats on u = x/s, scale `xs` *before*
        calling this function (e.g., via ws.scale.divide_inplace(xs)).
    """
    return TemporalTV.estimate_stats(ws, xs,
                                     percentile=percentile,
                                     eps_floor=eps_floor)


# ------------------------------ helpers --------------------------------
def _accum_1d_grad(g: torch.Tensor,
                   sign_dt: torch.Tensor,   # (span, *inner)
                   b0: int,
                   lam: float,
                   inv_s: torch.Tensor | None) -> None:
    """
    Accumulate ∇ for a tile of forward differences.
    If inv_s is provided, applies chain‑rule weights 1/s per frame.
    """
    span = sign_dt.shape[0]

    if inv_s is None:
        g[b0].add_( sign_dt[0].mul(-lam) )
        if span > 1:
            g[b0+1:b0+span].add_( (sign_dt[:-1] - sign_dt[1:]).mul_(lam) )
        g[b0+span].add_( sign_dt[-1].mul(lam) )
        return

    # With scaling: weights per frame (span+1,1,1,…)
    w = inv_s[b0 : b0 + span + 1]
    g[b0].add_( sign_dt[0].mul(-lam).mul_(w[0]) )
    if span > 1:
        g[b0+1:b0+span].add_( (sign_dt[:-1] - sign_dt[1:]).mul_(lam).mul_(w[1:-1]) )
    g[b0+span].add_( sign_dt[-1].mul(lam).mul_(w[-1]) )


def _suggest_tile(target_elems: int,
                  arena: UnifiedArena,
                  dtype,
                  dev: torch.device,
                  safety: float = 0.9,
                  user_default: int | None = None) -> int:
    """
    Pick a power‑of‑two tile that fits free scratch; fall back to user_default.
    """
    if dev.type == "cpu":
        return user_default or (1 << 30)
    elem_size = torch.tensor([], dtype=dtype).element_size()
    free_elems= arena.free_elems(dtype, device=dev)
    free_bytes= free_elems * elem_size
    if free_bytes == 0:
        free_bytes, _ = torch.cuda.mem_get_info(dev)
    cap  = int(free_bytes * safety // (target_elems * elem_size))
    tile = 1 << (cap.bit_length()-1) if cap > 0 else 1
    if user_default:
        tile = min(tile, user_default)
    return max(1, tile)