# graspcg/nufft/backends/cufi.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Literal
import torch
try:
    from cufinufft import Plan as _CuFiPlan
    _HAVE_CUFI = True
except Exception:
    _HAVE_CUFI = False



# Mark CUFI call-sites as non-traceable for torch.compile
try:
    from torch._dynamo import disable as _dynamo_disable
except Exception:
    def _dynamo_disable(fn):
        return fn
from collections import OrderedDict

@dataclass
class CuFiNUFFTAdapter:
    """
    CUFINUFFT backend adapter (vectorized across like dims; tiny LRU by n_trans).

    Contract:
      • prepare(..., like_prod): n_trans = like_prod = C × L_other
      • ensure_like_prod(n): switch or build cached plans without full re-prepare
      • A(x):  (B,C,sp) or (B,L,C,sp) -> (B,C,K) or (B,L,C,K)
      • AH(y): (B,C,K) or (B,L,C,K)   -> (B,1,sp) or (B,L,1,sp)

    DCF handling:
      dcf_mode='standard' -> apply DCF in adjoint (and optionally in forward)
      dcf_mode='balanced' -> use sqrt(DCF) in both directions
      dcf_mode='none'     -> never apply DCF
    """
    ndim: int
    backend_name: str = 'cufinufft'
    traj_units: Literal['rad', 'norm'] = 'rad'
    dcf_mode: Literal['standard', 'balanced', 'none'] = 'balanced'
    apply_dcf_in_fwd: bool = False
    apply_dcf_in_adj: bool = True
    eps: float = 1e-6
    isign: int = -1
    # cache control
    max_cache: int = 2

    # prepared state
    _im_shape: Optional[Tuple[int, ...]] = None
    _maps: Optional[torch.Tensor] = None
    _traj_BndK: Optional[torch.Tensor] = None
    _dcf_BK: Optional[torch.Tensor] = None
    _K: int = 0
    _dev: Optional[torch.device] = None
    _dtype: torch.dtype = torch.complex64
    _n_trans: int = 1
    # active plans
    _plans_t2: list | None = None
    _plans_t1: list | None = None
    # LRU cache: like_prod -> (plans_t2, plans_t1)
    _plan_cache: "OrderedDict[int, tuple[list, list]]" | None = None

    _alpha_scalar: Optional[float] = None
    _alpha_profile: Optional[torch.Tensor] = None

    # -------- utility --------
    def _check(self):
        if not _HAVE_CUFI: raise ImportError("cufinufft not available.")
        if self.ndim not in (2, 3): raise ValueError("CuFiNUFFTAdapter supports ndim ∈ {2,3}.")
        if self.max_cache < 1: self.max_cache = 1

    def _spatial(self) -> Tuple[int, ...]:
        assert self._im_shape is not None
        return tuple(int(s) for s in self._im_shape[1:])

    def _to_BndK(self, traj: torch.Tensor) -> torch.Tensor:
        # normalize to (B, nd, K) then scale to radians
        if traj.ndim == 2:
            t = traj
            if t.shape[0] != self.ndim and t.shape[1] == self.ndim: t = t.transpose(0, 1).contiguous()
            if t.shape[0] != self.ndim: raise ValueError("traj must be (nd,K) or (K,nd) when 2D.")
            t = t.unsqueeze(0)
        elif traj.ndim == 3:
            t = traj
            if t.shape[1] == self.ndim: pass
            elif t.shape[2] == self.ndim: t = t.transpose(1, 2).contiguous()
            else: raise ValueError("traj must be (B,nd,K) or (B,K,nd).")
        else:
            raise ValueError("traj must have 2 or 3 dims.")
        if self.traj_units == 'rad':  return t.to(torch.float32)
        if self.traj_units == 'norm': return (2.0 * torch.pi) * t.to(torch.float32)
        raise ValueError("traj_units must be 'rad' or 'norm'.")

    def k_per_frame(self) -> int:
        return int(self._K)

    # -------- prepare & cache --------
    def prepare(self, im_shape: Tuple[int, ...], traj: torch.Tensor, dcf: Optional[torch.Tensor],
                maps: torch.Tensor, dtype: torch.dtype, device: torch.device, *, like_prod: int) -> None:
        """
        Build plans for the given like_prod and seed the LRU. Keeps state for future cache hits.
        """
        self._check()
        if device.type != 'cuda':
            raise RuntimeError("CUFINUFFT requires CUDA device; got CPU.")
        self._im_shape = tuple(int(s) for s in im_shape)
        self._dev = device
        self._dtype = dtype
        self._maps = maps.to(device=device, dtype=dtype, non_blocking=True).contiguous()
        t = self._to_BndK(traj); self._traj_BndK = t.to(device=device, non_blocking=True).contiguous()
        self._K = int(self._traj_BndK.shape[-1])
        # dcf to (B,K) on device
        if dcf is None:
            self._dcf_BK = None
        else:
            d = dcf
            if d.ndim == 1: d = d.view(1, -1)
            d = d.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
            if int(d.shape[0]) == 1 and int(d.shape[1]) == self._K:
                d = d.expand(int(self._traj_BndK.shape[0]), -1)
            self._dcf_BK = d
        # init cache and build active plans
        self._plan_cache = OrderedDict()
        self._build_and_activate_plans(int(like_prod))
        self._alpha_scalar = None; self._alpha_profile = None

    def _build_and_activate_plans(self, like_prod: int) -> None:
        """(Re)build plans for a given like_prod and activate them; manage LRU."""
        assert self._traj_BndK is not None and self._im_shape is not None and self._dev is not None
        spatial = self._spatial()
        dtype_str = 'complex128' if (self._dtype == torch.complex128) else 'complex64'
        dev_id = self._dev.index or 0
        B = int(self._traj_BndK.shape[0])

        plans_t2, plans_t1 = [], []
        for b in range(B):
            coords = self._traj_BndK[b]
            if self.ndim == 2:
                xj, yj = (coords[0].contiguous(), coords[1].contiguous())
                p2 = _CuFiPlan(2, spatial, n_trans=like_prod, eps=self.eps, isign=self.isign, dtype=dtype_str, gpu_device_id=dev_id); p2.setpts(xj, yj)
                p1 = _CuFiPlan(1, spatial, n_trans=like_prod, eps=self.eps, isign=self.isign, dtype=dtype_str, gpu_device_id=dev_id); p1.setpts(xj, yj)
            else:
                xj, yj, zj = (coords[0].contiguous(), coords[1].contiguous(), coords[2].contiguous())
                p2 = _CuFiPlan(2, spatial, n_trans=like_prod, eps=self.eps, isign=self.isign, dtype=dtype_str, gpu_device_id=dev_id); p2.setpts(xj, yj, zj)
                p1 = _CuFiPlan(1, spatial, n_trans=like_prod, eps=self.eps, isign=self.isign, dtype=dtype_str, gpu_device_id=dev_id); p1.setpts(xj, yj, zj)
            plans_t2.append(p2); plans_t1.append(p1)

        # activate
        self._plans_t2, self._plans_t1 = plans_t2, plans_t1
        self._n_trans = int(like_prod)
        # LRU insert
        assert self._plan_cache is not None
        self._plan_cache[like_prod] = (plans_t2, plans_t1)
        self._plan_cache.move_to_end(like_prod)
        while len(self._plan_cache) > int(self.max_cache):
            old_like, (pt2, pt1) = self._plan_cache.popitem(last=False)
            try:
                for p in pt2: p.destroy()
                for p in pt1: p.destroy()
            except Exception:
                pass

    def ensure_like_prod(self, like_prod: int) -> None:
        """Activate cached plans or build new ones for the requested like_prod."""
        like_prod = int(like_prod)
        if like_prod == int(self._n_trans):
            return
        assert self._plan_cache is not None
        if like_prod in self._plan_cache:
            self._plans_t2, self._plans_t1 = self._plan_cache[like_prod]
            self._plan_cache.move_to_end(like_prod)
            self._n_trans = like_prod
            return
        self._build_and_activate_plans(like_prod)


    # -------------------------------- forward ----------------------------------
    @_dynamo_disable
    @torch.no_grad()
    def A(self, x: torch.Tensor, out: Optional[torch.Tensor] = None, *, scratch: Optional[dict] = None) -> torch.Tensor:
        """
        x: (B,C,sp) or (B,L,C,sp) → y: (B,C,K) or (B,L,C,K)
        Honors 'out' (no final allocation) and optional scratch buffers: keys "fk", "ck".
        """
        assert (self._plans_t2 is not None) and (self._maps is not None) and (self._traj_BndK is not None)
        spatial = self._spatial(); K = self._K; maps = self._maps

        # Parse shapes  (correct ranks: (2+ndim) for (B,Cin,*S), (3+ndim) for (B,L,Cin,*S))
        if x.ndim == (2 + self.ndim):         # (B,Cin,spatial)
            B, Cin = int(x.shape[0]), int(x.shape[1])
            L = 1
            x_view = x
            need = (B, Cmaps, K)
        elif x.ndim == (3 + self.ndim):       # (B,L,Cin,spatial)
            B, L, Cin = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
            x_view = x
            need = (B, L, Cmaps, K)
        else:
            raise ValueError(f"x must be (B,C,spatial) or (B,L,C,spatial); got shape {tuple(x.shape)} for ndim={self.ndim}")


        ntr = int(self._n_trans)
        if ntr != C * L:
            raise ValueError(f"Adapter prepared with like_prod={ntr}, but call requires C×L={C*L}.")

        # allocate/validate output
        if L == 1:
            if out is None: y = torch.empty((B, C, K), device=x.device, dtype=x.dtype)
            else:
                if out.shape != (B, C, K): raise ValueError(f"out shape {tuple(out.shape)} must be (B,C,K).")
                y = out
        else:
            if out is None: y = torch.empty((B, L, C, K), device=x.device, dtype=x.dtype)
            else:
                if out.shape != (B, L, C, K): raise ValueError(f"out shape {tuple(out.shape)} must be (B,L,C,K).")
                y = out

        # staging (fk, ck)
        ntr = int(self._n_trans)
        fk = ck = None
        if scratch is not None:
            fk = scratch.get("fk", None); ck = scratch.get("ck", None)
        if not (isinstance(fk, torch.Tensor) and tuple(fk.shape) == ((ntr,) + spatial) and fk.device == x.device and fk.dtype == x.dtype):
            fk = torch.empty((ntr,) + spatial, device=x.device, dtype=x.dtype)
        if not (isinstance(ck, torch.Tensor) and tuple(ck.shape) == (ntr, K) and ck.device == x.device and ck.dtype == x.dtype):
            ck = torch.empty((ntr, K), device=x.device, dtype=x.dtype)

        # execute per frame
        for b in range(B):
            plan2 = self._plans_t2[b % len(self._plans_t2)]
            if L == 1:
                fk.copy_(x_view[b] * maps)     # (C,sp)->(ntr=C,sp)
                plan2.execute(fk, ck)
                y_b = ck
                # DCF in forward
                if self.dcf_mode == 'standard':
                    if (self._dcf_BK is not None) and self.apply_dcf_in_fwd: y_b = y_b * self._dcf_BK[b].view(1, K)
                elif self.dcf_mode == 'balanced':
                    if self._dcf_BK is not None: y_b = y_b * torch.sqrt(self._dcf_BK[b].clamp_min(0)).view(1, K)
                y[b].copy_(y_b)
            else:
                fk.view(L, C, *spatial).copy_(x_view[b] * maps.view(1, C, *spatial))
                plan2.execute(fk, ck)  # (L*C, K)
                y_b = ck.view(L, C, K)
                if self.dcf_mode == 'standard':
                    if (self._dcf_BK is not None) and self.apply_dcf_in_fwd: y_b = y_b * self._dcf_BK[b].view(1, 1, K)
                elif self.dcf_mode == 'balanced':
                    if self._dcf_BK is not None: y_b = y_b * torch.sqrt(self._dcf_BK[b].clamp_min(0)).view(1, 1, K)
                y[b].copy_(y_b)

        return y

    # --------------------------------- adjoint ---------------------------------
    @_dynamo_disable
    @torch.no_grad()
    def AH(self, y: torch.Tensor, out: Optional[torch.Tensor] = None, *, scratch: Optional[dict] = None) -> torch.Tensor:
        """
        y: (B,C,K) or (B,L,C,K) → x: (B,1,sp) or (B,L,1,sp)
        Honors 'out' (no final allocation) and optional scratch buffers: keys "grid", "cnu".
        """
        assert (self._plans_t1 is not None) and (self._maps is not None) and (self._traj_BndK is not None)
        spatial = self._spatial(); K = self._K; maps = self._maps

        if y.ndim == 3:
            B, C = int(y.shape[0]), int(y.shape[1]); L = 1; y_view = y
        elif y.ndim == 4:
            B, L, C = int(y.shape[0]), int(y.shape[1]), int(y.shape[2]); y_view = y
        else:
            raise ValueError("y must be (B,C,K) or (B,L,C,K)")

        ntr = int(self._n_trans)
        if ntr != C * L:
            raise ValueError(f"Adapter prepared with like_prod={ntr}, but call requires C×L={C*L}.")

        if L == 1:
            if out is None: x_out = torch.empty((B, 1) + spatial, device=y.device, dtype=y.dtype)
            else:
                if out.shape != (B, 1) + spatial: raise ValueError(f"out shape {tuple(out.shape)} must be (B,1,spatial).")
                x_out = out
        else:
            if out is None: x_out = torch.empty((B, L, 1) + spatial, device=y.device, dtype=y.dtype)
            else:
                if out.shape != (B, L, 1) + spatial: raise ValueError(f"out shape {tuple(out.shape)} must be (B,L,1,spatial).")
                x_out = out

        # staging
        grid = cnu = None
        if scratch is not None:
            grid = scratch.get("grid", None); cnu = scratch.get("cnu", None)
        if not (isinstance(grid, torch.Tensor) and tuple(grid.shape) == ((ntr,) + spatial) and grid.device == y.device and grid.dtype == y.dtype):
            grid = torch.empty((ntr,) + spatial, device=y.device, dtype=y.dtype)
        if not (isinstance(cnu, torch.Tensor) and tuple(cnu.shape) == (ntr, K) and cnu.device == y.device and cnu.dtype == y.dtype):
            cnu = torch.empty((ntr, K), device=y.device, dtype=y.dtype)

        for b in range(B):
            plan1 = self._plans_t1[b % len(self._plans_t1)]
            if L == 1:
                yw = y_view[b]
                if self.dcf_mode == 'standard':
                    if (self._dcf_BK is not None) and self.apply_dcf_in_adj: yw = yw * self._dcf_BK[b].view(1, K)
                elif self.dcf_mode == 'balanced':
                    if self._dcf_BK is not None: yw = yw * torch.sqrt(self._dcf_BK[b].clamp_min(0)).view(1, K)
                cnu.copy_(yw)
                plan1.execute(cnu, grid)
                x_b = (grid * torch.conj(maps)).sum(dim=0, keepdim=True)
                x_out[b].copy_(x_b)
            else:
                yw = y_view[b]  # (L,C,K)
                if self.dcf_mode == 'standard':
                    if (self._dcf_BK is not None) and self.apply_dcf_in_adj: yw = yw * self._dcf_BK[b].view(1, 1, K)
                elif self.dcf_mode == 'balanced':
                    if self._dcf_BK is not None: yw = yw * torch.sqrt(self._dcf_BK[b].clamp_min(0)).view(1, 1, K)
                cnu.view(L, C, K).copy_(yw)
                plan1.execute(cnu, grid)
                grid_v = grid.view(L, C, *spatial)
                x_like = (grid_v * torch.conj(maps).view(1, C, *spatial)).sum(dim=1)  # (L,spatial)
                x_out[b].copy_(x_like.view(L, 1, *spatial))

        return x_out


    # -------------------------------- calibration ------------------------------

    @torch.no_grad()
    def diag_AHA_profile(self) -> torch.Tensor:
        """Per‑frame proxy α_b = sum_k dcf[b,k] if DCF is set, else α_b = K."""
        assert self._traj_BndK is not None
        B = int(self._traj_BndK.shape[0])
        if self._dcf_BK is None:
            return torch.full((B,), float(self._K), device=self._traj_BndK.device, dtype=torch.float32)
        return self._dcf_BK.sum(dim=1).to(torch.float32)

    @torch.no_grad()
    def diag_AHA_scalar(self) -> float:
        """
        Exact scalar via delta: α = AᴴA(δ_center) / Σ_c |S_c|²(center). Cached after first call.
        """
        if self._alpha_scalar is not None:
            return float(self._alpha_scalar)
        assert (self._im_shape is not None) and (self._dev is not None) and (self._maps is not None)
        spatial = self._spatial()
        ctr = tuple(s // 2 for s in spatial)
        C = int(self._im_shape[0])
        x0 = torch.zeros((1, C) + spatial, device=self._dev, dtype=self._dtype)
        x0[0, 0][ctr] = 1.0 + 0.0j
        y = self.A(x0)     # (1,C,K)
        z = self.AH(y)     # (1,1,spatial)
        s2 = (self._maps.abs() ** 2).sum(dim=0)[ctr].clamp_min(1e-20).item()
        alpha = float(z[0, 0][ctr].real.item()) / float(s2)
        self._alpha_scalar = alpha
        return alpha

    def __del__(self):
        try:
            if self._plan_cache is not None:
                for _, (pt2, pt1) in list(self._plan_cache.items()):
                    for p in pt2:
                        try: p.destroy()
                        except Exception: pass
                    for p in pt1:
                        try: p.destroy()
                        except Exception: pass
            else:
                # fallback: destroy active
                if self._plans_t2 is not None:
                    for p in self._plans_t2:
                        try: p.destroy()
                        except Exception: pass
                if self._plans_t1 is not None:
                    for p in self._plans_t1:
                        try: p.destroy()
                        except Exception: pass
        except Exception:
            pass
        finally:
            self._plans_t1 = None; self._plans_t2 = None; self._plan_cache = None
