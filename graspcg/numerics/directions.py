from __future__ import annotations
from typing import Optional, Any

import torch

# Try to import dot_chunked; fall back to simple impl
from ..ops.dot import dot_chunked


def _reduce0d_add_(acc: Optional[torch.Tensor], v: torch.Tensor) -> torch.Tensor:
    return v if acc is None else (acc + v.to(acc.device))

def _sum0_over_shards(ws, fn) -> torch.Tensor:
    """
    Sum a 0‑D REAL over shards. Keeps everything on device; no host syncs.
    """
    acc = None
    primary = None
    for sh, i in ws.iter_shards():
        v = fn(i)  # expected 0‑D REAL on shard device
        if not isinstance(v, torch.Tensor):
            # adapt legacy dot_chunked that might return float (rare)
            g = ws.get("g", i)
            v = torch.as_tensor(float(v), device=g.device, dtype=g.real.dtype)
        if acc is None:
            primary = v.device
            acc = v
        else:
            acc = acc + v.to(primary, non_blocking=True)
    if acc is None:
        # fallback to compute device if no shards
        dev = ws.primary_device if hasattr(ws, "primary_device") else torch.device("cpu")
        dt  = ws.dtype_r if hasattr(ws, "dtype_r") else torch.float32
        acc = torch.zeros((), device=dev, dtype=dt)
    return acc

def _g_dot_d0(ws) -> torch.Tensor:
    return _sum0_over_shards(ws, lambda i:
        dot_chunked(ws.get("g", i), ws.get("dx", i), arena=ws.arena))

class _BaseDir:
    def __init__(self, ws):
        self.ws = ws
        self._rho_prev: Optional[torch.Tensor] = None  # 0‑D REAL

    @torch.no_grad()
    def init_state(self, g0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return g·d at t=0 after seeding any needed state, as 0‑D REAL tensor."""
        return _g_dot_d0(self.ws)

    @torch.no_grad()
    def update_inplace(self, ws) -> torch.Tensor:
        raise NotImplementedError
# --- helper to make per‑spatial 'diag' match per‑frame tensors -------------
def _diag_like(a: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """
    Expand a per‑spatial diagonal `D` (e.g., (C,H,W)) to match a full tensor
    `a` (e.g., (B,C,H,W)) so that flattening/element‑wise ops are aligned.
    Also ensures device compatibility.
    """
    if D.device != a.device:
        D = D.to(a.device)
    if D.shape == a.shape:
        return D
    # Common case: a == (B, C, H, W) and D == (C, H, W)
    if D.dim() + 1 == a.dim() and D.shape == a.shape[1:]:
        return D.unsqueeze(0).expand_as(a)
    # Generic broadcast via leading singleton dims
    try:
        view_shape = (1,) * (a.dim() - D.dim()) + tuple(D.shape)
        return D.view(view_shape).expand_as(a)
    except Exception:
        raise ValueError(f"Cannot broadcast diag of shape {tuple(D.shape)} to {tuple(a.shape)}")
    
class DirPRPlus(_BaseDir):
    @torch.no_grad()
    def init_state(self, g0: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Seed g_prev from current g (manifest must include 'g_prev')
        for sh, i in self.ws.iter_shards():
            g, gp = self.ws.bind(i, "g", "g_prev")
            gp.copy_(g)
        return _g_dot_d0(self.ws)

    @torch.no_grad()
    def update_inplace(self, ws) -> torch.Tensor:
        # β_PR+ = (gᵀM^{-1}g − g_prevᵀM^{-1}g) / (g_prevᵀM^{-1}g_prev), clamped ≥ 0
        num = _sum0_over_shards(ws, lambda i: (
            dot_chunked(ws.get("g", i), ws.get("g", i),
                        diag=_diag_like(ws.get("g", i), ws.get("diag", i)),
                        arena=ws.arena)
            - dot_chunked(ws.get("g_prev", i), ws.get("g", i),
                          diag=_diag_like(ws.get("g", i), ws.get("diag", i)),
                          arena=ws.arena)
        ))
        den = _sum0_over_shards(ws, lambda i:
            dot_chunked(ws.get("g_prev", i), ws.get("g_prev", i),
                        diag=_diag_like(ws.get("g_prev", i), ws.get("diag", i)),
                        arena=ws.arena))
        eps = torch.as_tensor(1e-20, device=num.device, dtype=num.dtype)
        beta = torch.clamp(num / torch.clamp(den, min=eps), min=0)

        for sh, i in ws.iter_shards():
            g, d, D = ws.bind(i, "g", "dx", "diag")
            d.mul_(beta.to(d.device, dtype=d.real.dtype))
            d.addcdiv_(g, D, value=-1.0)
        for sh, i in ws.iter_shards():
            g, gp = ws.bind(i, "g", "g_prev")
            gp.copy_(g)
        return beta

class DirDY(_BaseDir):
    @torch.no_grad()
    def init_state(self, g0: Optional[torch.Tensor] = None) -> torch.Tensor:
        for sh, i in self.ws.iter_shards():
            g, gp = self.ws.bind(i, "g", "g_prev")
            gp.copy_(g)
        return _g_dot_d0(self.ws)

    @torch.no_grad()
    def update_inplace(self, ws) -> torch.Tensor:
        # β_DY = (gᵀM^{-1}g) / (dᵀ(g − g_prev))
        num = _sum0_over_shards(ws, lambda i:
            dot_chunked(ws.get("g", i), ws.get("g", i),
                        diag=_diag_like(ws.get("g", i), ws.get("diag", i)),
                        arena=ws.arena))
        den = _sum0_over_shards(ws, lambda i:
            dot_chunked(ws.get("dx", i), ws.get("g", i), arena=ws.arena)
            - dot_chunked(ws.get("dx", i), ws.get("g_prev", i), arena=ws.arena))
        eps = torch.as_tensor(1e-20, device=num.device, dtype=num.dtype)
        beta = num / torch.clamp(den, min=eps)

        for sh, i in ws.iter_shards():
            g, d, D = ws.bind(i, "g", "dx", "diag")
            d.mul_(beta.to(d.device, dtype=d.real.dtype))
            d.addcdiv_(g, D, value=-1.0)
        for sh, i in ws.iter_shards():
            g, gp = ws.bind(i, "g", "g_prev")
            gp.copy_(g)
        return beta

class DirFR(_BaseDir):
    @torch.no_grad()
    def update_inplace(self, ws) -> torch.Tensor:
        # β_FR = (gᵀM^{-1}g) / (g_prevᵀM^{-1}g_prev)  (ρ cache)
        if self._rho_prev is None:
            self._rho_prev = _sum0_over_shards(ws, lambda i:
                dot_chunked(ws.get("g", i), ws.get("g", i),
                            diag=_diag_like(ws.get("g", i), ws.get("diag", i)),
                            arena=ws.arena))
        rho_k = _sum0_over_shards(ws, lambda i:
            dot_chunked(ws.get("g", i), ws.get("g", i),
                        diag=_diag_like(ws.get("g", i), ws.get("diag", i)),
                        arena=ws.arena))
        eps = torch.as_tensor(1e-20, device=rho_k.device, dtype=rho_k.dtype)
        beta = rho_k / torch.clamp(self._rho_prev, min=eps)
        self._rho_prev = rho_k

        for sh, i in ws.iter_shards():
            g, d, D = ws.bind(i, "g", "dx", "diag")
            d.mul_(beta.to(d.device, dtype=d.real.dtype))
            d.addcdiv_(g, D, value=-1.0)
        return beta
