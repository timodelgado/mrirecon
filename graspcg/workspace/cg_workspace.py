# graspcg/workspace/cg_workspace.py (essentials only)
from __future__ import annotations
import math, torch
from dataclasses import dataclass
from typing import Optional, Iterator, Tuple
from .unified_arena import UnifiedArena
from .device_cfg import DeviceCfg

@dataclass
class _FwdCache:
    # k-space shaped cache on y.device
    r0   : Optional[torch.Tensor] = None   # r0 = A(x) - y
    Ad   : Optional[torch.Tensor] = None   # A(dx)
    # optional: image-shaped adjoint cache for accepted-step updates
    Ahd  : Optional[torch.Tensor] = None   # AH(Ad)  (cleared each iter)
    # scalars for fast φ_data and φ'_data
    dot_Ad_Ad : float = 0.0
    dot_Ad_r0 : float = 0.0
    f_data0   : float = 0.0   # ½‖r0‖²

class CGWorkspace:
    def __init__(self, y: torch.Tensor, nufft_op,
                 device_cfg: DeviceCfg,
                 arena: UnifiedArena,
                 *,
                 direction: str = "prplus",
                 dtype_c = torch.complex64,
                 dtype_r = torch.float32):

        self.nufft_op = nufft_op
        self.y        = y
        self.arena    = arena
        self.device   = y.device
        self.dtype_c  = dtype_c
        self.dtype_r  = dtype_r
        self.direction = direction

        # shards: construct from your existing layout/sharding planner
        self.shards = self._plan_shards(device_cfg)

        # forward cache (k-space)
        self.cache = _FwdCache()

    # ---------------- shard iteration ----------------
    def iter_shards(self) -> Iterator[Tuple[object, int]]:
        # your existing shard iterator (return (shard, idx))
        yield from self.shards

    # ---------------- allocation helpers ----------------
    def _plan_shards(self, device_cfg) -> list[Tuple[object,int]]:
        # allocate x, g, dx, diag; optionally g_prev depending on direction
        shards = []
        need_gprev = self.direction in ("prplus","dy")
        for sh,_ in self._internal_make_shards(device_cfg):
            sh.x   = torch.zeros(sh.shape, dtype=self.dtype_c, device=sh.device)
            sh.g   = torch.zeros_like(sh.x)
            sh.dx  = torch.zeros_like(sh.x)
            sh.diag= torch.ones( sh.shape[1:], dtype=self.dtype_r, device=sh.device)  # per-voxel
            if need_gprev:
                sh.g_prev = torch.zeros_like(sh.x)
            shards.append((sh,_))
        return shards

    # ======================================================
    # Forward-cache lifecycle (used by line-search/objective)
    # ======================================================
    @torch.no_grad()
    def refresh_r0(self) -> None:
        """
        r0 = A(x) - y, sets f_data0 = 0.5 * ||r0||^2.
        Iterates shards and accumulates into a k-space buffer on y.device.
        """
        if self.cache.r0 is None:
            self.cache.r0 = torch.empty_like(self.y)
        r0 = self.cache.r0.zero_()
        # accumulate A(x) from shards into r0
        for sh,_ in self.iter_shards():
            self.nufft_op.A(sh.x, out=r0, accumulate=True)  # implement accumulate=True in op or do r0.add_(Ax_part)
        r0.sub_(self.y)
        self.cache.f_data0 = 0.5 * (r0.real.square().sum() + r0.imag.square().sum()).item()

    @torch.no_grad()
    def prepare_direction(self) -> None:
        """
        Compute Ad = A(dx) and cheap scalars for φ_data(t) and φ'_data(t).
        Also (optional) compute Ahd = AH(Ad) once for accept-step updates.
        """
        # Ad
        if self.cache.Ad is None:
            self.cache.Ad = torch.empty_like(self.y)
        Ad = self.cache.Ad.zero_()
        for sh,_ in self.iter_shards():
            self.nufft_op.A(sh.dx, out=Ad, accumulate=True)

        # dot scalars
        r0 = self.cache.r0;  assert r0 is not None, "call refresh_r0 first"
        dot_Ad_r0 = (Ad.conj() * r0).real.sum().item()
        dot_Ad_Ad = (Ad.conj() * Ad).real.sum().item()
        self.cache.dot_Ad_r0 = dot_Ad_r0
        self.cache.dot_Ad_Ad = dot_Ad_Ad

        # optional Ahd for zero-adjoint accept
        # allocate scratch image, then scatter to shards if desired
        Ahd_full = None
        if hasattr(self.nufft_op, "AH"):
            Ahd_full = self.nufft_op.AH(Ad)   # one adjoint per iteration
        self.cache.Ahd = Ahd_full  # can be None if you want to save memory

    # -------- data φ(t) and φ'(t) without NUFFT ----------
    @torch.no_grad()
    def phi_data(self, t: float) -> float:
        c = self.cache
        return c.f_data0 + t*c.dot_Ad_r0 + 0.5*t*t*c.dot_Ad_Ad

    @torch.no_grad()
    def phi_prime_data(self, t: float) -> float:
        c = self.cache
        return c.dot_Ad_r0 + t*c.dot_Ad_Ad

    # -------- accept step: update x and caches ----------
    @torch.no_grad()
    def accept_step(self, t: float) -> None:
        # x ← x + t dx (all shards)
        for sh,_ in self.iter_shards():
            sh.x.add_(sh.dx, alpha=t)

        # r0 ← r0 + t Ad, and update f_data0, dot_Ad_r0 for next iter seed
        c = self.cache; r0, Ad = c.r0, c.Ad
        r0.add_(Ad, alpha=t)
        # new f_data0 = 0.5||r0||^2  (cheap to recompute)
        c.f_data0 = 0.5 * (r0.real.square().sum() + r0.imag.square().sum()).item()

        # If Ahd was computed: data-grad update g_data += t * Ahd (optional)
        if c.Ahd is not None:
            # scatter Ahd into shard grads as the data term part
            off = 0
            for sh,_ in self.iter_shards():
                # assume batch-major sharding along leading axis:
                span = sh.x.shape[0]
                sh.g.copy_(c.Ahd[off:off+span])  # seed g with data grad
                off += span
        else:
            # If you skip Ahd, the next objective call will AH(r0) once.
            for sh,_ in self.iter_shards():
                sh.g.zero_()

        # Clear per-iteration direction cache; next iter will recompute
        c.Ad = None; c.Ahd = None
        c.dot_Ad_Ad = 0.0; c.dot_Ad_r0 = 0.0
