from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Dict, Any
from collections import OrderedDict
import torch

# Optional CUFINUFFT dependency
try:
    from cufinufft import Plan as _CuFiPlan
    _HAVE_CUFI = True
except Exception:
    _HAVE_CUFI = False

# Keep CUFI calls out of torch.compile graphs (front-end pack/unpack can still compile)
try:
    from torch._dynamo import disable as _dynamo_disable
except Exception:
    def _dynamo_disable(fn):
        return fn


@dataclass
class CuFiNUFFTAdapter:
    """
    CUFINUFFT backend adapter (North-Star):
      • Vectorizes across Like via n_trans = C × L
      • Minimal scratch; writes directly into `out` when possible
      • Small LRU of plans keyed by n_trans
      • DCF policy supports: 'balanced' (default), 'standard', 'none'

    Adapter shapes (canonical, i.e., backend order):
      Forward A:
        x: (B, C, H, W[,D])      or (B, L, C, H, W[,D])
        y: (B, C, K)             or (B, L, C, K)
      Adjoint AH:
        y: (B, C, K)             or (B, L, C, K)
        x: (B, 1, H, W[,D])      or (B, L, 1, H, W[,D])

    Minimal scratch contract:
      • A  : fk  (n_trans, *spatial)  [REQUIRED]
             ck  (n_trans, K)         [OPTIONAL] — aliased to out-slice when possible
      • AH : grid(n_trans, *spatial)  [REQUIRED]
             cnu (n_trans, K)         [OPTIONAL] — alias y-view when possible/no weighting
    """
    # Config
    ndim: int
    backend_name: str = 'cufinufft'
    traj_units: Literal['rad', 'norm'] = 'rad'
    dcf_mode: Literal['balanced', 'standard', 'none'] = 'balanced'
    apply_dcf_in_fwd: bool = False    # used only for 'standard'
    apply_dcf_in_adj: bool = True     # used only for 'standard'
    eps: float = 1e-6
    isign: int = -1
    max_cache: int = 2                # LRU size for plans keyed by n_trans

    # Prepared state
    _im_shape: Optional[Tuple[int, ...]] = None          # (C, H, W[,D])
    _maps: Optional[torch.Tensor] = None                 # (C, H, W[,D]) complex on device
    _traj_BndK: Optional[torch.Tensor] = None            # (B, nd, K) float32 (radians)
    _dcf_BK: Optional[torch.Tensor] = None               # (B, K) float32 or None
    _K: int = 0
    _dev: Optional[torch.device] = None
    _dtype: torch.dtype = torch.complex64
    _n_trans: int = 1

    # Active plans (per B frame)
    _plans_t2: list | None = None     # type-2 (image->k)
    _plans_t1: list | None = None     # type-1 (k->image)

    # LRU cache: like_prod -> (plans_t2, plans_t1)
    _plan_cache: "OrderedDict[int, tuple[list, list]]" | None = None

    # Calibration caches (optional helpers)
    _alpha_scalar: Optional[float] = None
    _alpha_profile: Optional[torch.Tensor] = None

    # ------------------------------- Utilities --------------------------------

    def _check(self):
        if not _HAVE_CUFI:
            raise ImportError("cufinufft not available. Install it or select backend='torchkb'.")
        if self.ndim not in (2, 3):
            raise ValueError("CuFiNUFFTAdapter supports ndim ∈ {2,3} only.")

    def _spatial(self) -> Tuple[int, ...]:
        assert self._im_shape is not None
        return tuple(int(s) for s in self._im_shape[1:])

    def _to_BndK(self, traj: torch.Tensor) -> torch.Tensor:
        """
        Normalize trajectory to (B, nd, K) and cast to float32; scale to radians if needed.
        Accepts (B,nd,K), (B,K,nd), (nd,K) or (K,nd).
        """
        if traj.ndim == 2:
            t = traj
            if t.shape[0] != self.ndim and t.shape[1] == self.ndim:
                t = t.transpose(0, 1).contiguous()
            if t.shape[0] != self.ndim:
                raise ValueError("traj must be (nd,K) or (K,nd) when 2D/3D")
            t = t.unsqueeze(0)
        elif traj.ndim == 3:
            t = traj
            if t.shape[1] == self.ndim:
                pass
            elif t.shape[2] == self.ndim:
                t = t.transpose(1, 2).contiguous()
            else:
                raise ValueError("traj must be (B,nd,K) or (B,K,nd)")
        else:
            raise ValueError("traj must have 2 or 3 dims.")
        t = t.to(torch.float32)
        if self.traj_units == 'rad':
            return t
        if self.traj_units == 'norm':
            return (2.0 * torch.pi) * t
        raise ValueError("traj_units must be 'rad' or 'norm'.")

    def k_per_frame(self) -> int:
        return int(self._K)

    # --------------------------- Prepare & Plan LRU ----------------------------

    def prepare(self,
                im_shape: Tuple[int, ...],
                traj: torch.Tensor,
                dcf: Optional[torch.Tensor],
                maps: torch.Tensor,
                dtype: torch.dtype,
                device: torch.device,
                *, like_prod: int) -> None:
        """
        Build/update CUFINUFFT plans with n_trans = like_prod = C × L (vectorize across Like).
        Seeds/refreshes the tiny LRU keyed by like_prod.
        """
        self._check()
        if device.type != 'cuda':
            raise RuntimeError("CUFINUFFT requires CUDA device; got CPU.")

        self._im_shape = tuple(int(s) for s in im_shape)            # (C, H, W[,D])
        self._dev = device
        self._dtype = dtype
        self._maps = maps.to(device=device, dtype=dtype, non_blocking=True).contiguous()

        t = self._to_BndK(traj)
        B = int(t.shape[0]); K = int(t.shape[-1])
        self._traj_BndK = t.to(device=device, non_blocking=True).contiguous()
        self._K = K

        if dcf is None:
            self._dcf_BK = None
        else:
            d = dcf
            if d.ndim == 1:
                d = d.view(1, -1)
            d = d.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
            if int(d.shape[0]) == 1 and int(d.shape[1]) == K:
                d = d.expand(B, -1)
            self._dcf_BK = d

        # Initialize LRU and build active plans for like_prod
        self._plan_cache = OrderedDict()
        self._build_and_activate_plans(int(like_prod))

        # Invalidate calibration caches
        self._alpha_scalar = None
        self._alpha_profile = None

    def _build_and_activate_plans(self, like_prod: int) -> None:
        """(Re)build per-frame plans for like_prod and activate; maintain LRU."""
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
                p2 = _CuFiPlan(2, spatial, n_trans=int(like_prod), eps=self.eps, isign=self.isign, dtype=dtype_str, gpu_device_id=dev_id); p2.setpts(xj, yj)
                p1 = _CuFiPlan(1, spatial, n_trans=int(like_prod), eps=self.eps, isign=self.isign, dtype=dtype_str, gpu_device_id=dev_id); p1.setpts(xj, yj)
            else:
                xj, yj, zj = (coords[0].contiguous(), coords[1].contiguous(), coords[2].contiguous())
                p2 = _CuFiPlan(2, spatial, n_trans=int(like_prod), eps=self.eps, isign=self.isign, dtype=dtype_str, gpu_device_id=dev_id); p2.setpts(xj, yj, zj)
                p1 = _CuFiPlan(1, spatial, n_trans=int(like_prod), eps=self.eps, isign=self.isign, dtype=dtype_str, gpu_device_id=dev_id); p1.setpts(xj, yj, zj)
            plans_t2.append(p2); plans_t1.append(p1)

        self._plans_t2, self._plans_t1 = plans_t2, plans_t1
        self._n_trans = int(like_prod)

        # LRU: keep most-recent like_prod entries up to max_cache
        if self._plan_cache is None:
            self._plan_cache = OrderedDict()
        self._plan_cache[int(like_prod)] = (plans_t2, plans_t1)
        self._plan_cache.move_to_end(int(like_prod))
        while len(self._plan_cache) > int(max(1, self.max_cache)):
            _, (pt2, pt1) = self._plan_cache.popitem(last=False)
            for p in pt2:
                try: p.destroy()
                except Exception: pass
            for p in pt1:
                try: p.destroy()
                except Exception: pass

    def ensure_like_prod(self, like_prod: int) -> None:
        """Activate cached plans or build new ones for requested like_prod=C×L."""
        like_prod = int(like_prod)
        if like_prod == int(self._n_trans):
            return
        if (self._plan_cache is not None) and (like_prod in self._plan_cache):
            self._plans_t2, self._plans_t1 = self._plan_cache[like_prod]
            self._plan_cache.move_to_end(like_prod)
            self._n_trans = like_prod
            return
        self._build_and_activate_plans(like_prod)

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

    # -------------------------------- Forward ---------------------------------

    @_dynamo_disable
    @torch.no_grad()
    def A(self,
        x: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        *,
        scratch: Optional[Dict[str, torch.Tensor]] = None,
        arena: Optional[Any] = None) -> torch.Tensor:
        """
        Forward NUFFT (type-2):
        x: (B,Cin,spatial)   or (B,L,Cin,spatial) with Cin∈{1, Cmaps}
        y: (B,C,K)           or (B,L,C,K)         with C = Cmaps (from maps)
        """
        assert (self._plans_t2 is not None) and (self._maps is not None) and (self._traj_BndK is not None)
        spatial = self._spatial(); K = self._K; maps = self._maps
        Cmaps = int(self._im_shape[0])

        # Parse shapes
        if x.ndim == (3 + self.ndim):         # (B,Cin,spatial)
            B, Cin = int(x.shape[0]), int(x.shape[1]); L = 1
            x_view = x
            need = (B, Cmaps, K)
        elif x.ndim == (4 + self.ndim):       # (B,L,Cin,spatial)
            B, L, Cin = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
            x_view = x
            need = (B, L, Cmaps, K)
        else:
            raise ValueError("x must be (B,C,spatial) or (B,L,C,spatial)")
        if Cin not in (1, Cmaps):
            raise ValueError(f"image coil slot must be 1 (coil‑combined) or C={Cmaps}; got {Cin}")

        # OUT (final result)
        if out is None:
            y = torch.empty(need, device=x.device, dtype=x.dtype)
        else:
            if tuple(out.shape) != need:
                raise ValueError(f"out shape {tuple(out.shape)} must be {need}.")
            y = out

        # Ensure plans for this Like extent
        self.ensure_like_prod(Cmaps * L)
        ntr = int(self._n_trans)

        # fk required by CUFI
        fk = None
        if scratch is not None:
            fk = scratch.get("fk", None)
        if fk is None and arena is not None:
            numel_fk = ntr
            for s in spatial: numel_fk *= int(s)
            fk = arena.request(numel_fk, dtype=x.dtype, anchor=y).reshape((ntr,) + spatial)  # type: ignore[attr-defined]
        if not (isinstance(fk, torch.Tensor) and tuple(fk.shape) == ((ntr,) + spatial) and fk.device == x.device and fk.dtype == x.dtype):
            fk = torch.empty((ntr,) + spatial, device=x.device, dtype=x.dtype)

        # Execute per frame, aliasing ck to out-view when possible
        for b in range(B):
            plan2 = self._plans_t2[b % len(self._plans_t2)]

            if L == 1:
                # (ntr=Cmaps,spatial) <- either x[b] (Cin=Cmaps) or maps * x[b,0] (Cin=1)
                if Cin == 1:
                    fk.copy_(maps * x_view[b, 0])
                else:
                    fk.copy_(x_view[b])

                y_b = y[b]                              # (Cmaps,K)
                ck = y_b if y_b.is_contiguous() else None
                if ck is None:
                    if scratch is not None:
                        tmp = scratch.get("ck", None)
                        if isinstance(tmp, torch.Tensor) and tuple(tmp.shape) == (ntr, K) and tmp.device == x.device and tmp.dtype == x.dtype:
                            ck = tmp
                if ck is None and arena is not None:
                    ck = arena.request(ntr * K, dtype=x.dtype, anchor=y).reshape(ntr, K)  # type: ignore[attr-defined]
                if ck is None:
                    ck = torch.empty((ntr, K), device=x.device, dtype=x.dtype)

                plan2.execute(fk, ck)

                # DCF weighting (forward)
                if self._dcf_BK is not None:
                    if self.dcf_mode == 'standard' and self.apply_dcf_in_fwd:
                        ck.mul_(self._dcf_BK[b].view(1, K).expand(ntr, K))
                    elif self.dcf_mode == 'balanced':
                        ck.mul_(torch.sqrt(self._dcf_BK[b].clamp_min(0)).view(1, K).expand(ntr, K))

                if ck.data_ptr() != y_b.data_ptr():
                    y_b.copy_(ck)

            else:
                # L > 1
                fk_v = fk.view(L, Cmaps, *spatial)
                if Cin == 1:
                    # fk_v <- maps (broadcast to L) * x[b,l,0]
                    fk_v.copy_(maps.view(1, Cmaps, *spatial).expand(L, Cmaps, *spatial))
                    fk_v.mul_(x_view[b].view(L, 1, *spatial))
                else:
                    fk_v.copy_(x_view[b])

                y_b = y[b]                                # (L,Cmaps,K)
                y_b_contig = y_b.is_contiguous()
                ck = y_b.view(ntr, K) if y_b_contig else None
                if ck is None:
                    if scratch is not None:
                        tmp = scratch.get("ck", None)
                        if isinstance(tmp, torch.Tensor) and tuple(tmp.shape) == (ntr, K) and tmp.device == x.device and tmp.dtype == x.dtype:
                            ck = tmp
                if ck is None and arena is not None:
                    ck = arena.request(ntr * K, dtype=x.dtype, anchor=y).reshape(ntr, K)  # type: ignore[attr-defined]
                if ck is None:
                    ck = torch.empty((ntr, K), device=x.device, dtype=x.dtype)

                plan2.execute(fk, ck)

                # DCF weighting (forward)
                if self._dcf_BK is not None:
                    if self.dcf_mode == 'standard' and self.apply_dcf_in_fwd:
                        ck.mul_(self._dcf_BK[b].view(1, K).expand(ntr, K))
                    elif self.dcf_mode == 'balanced':
                        ck.mul_(torch.sqrt(self._dcf_BK[b].clamp_min(0)).view(1, K).expand(ntr, K))

                if (not y_b_contig) or (ck.data_ptr() != y_b.view(ntr, K).data_ptr()):
                    y_b.copy_(ck.view(L, Cmaps, K))

        return y


    # -------------------------------- Adjoint ---------------------------------
    @_dynamo_disable
    @torch.no_grad()
    def AH(self,
        y: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        *,
        scratch: Optional[Dict[str, torch.Tensor]] = None,
        arena: Optional[Any] = None) -> torch.Tensor:
        """
        Adjoint NUFFT (type-1):
        y: (B,C,K) or (B,L,C,K) with C=Cmaps
        x: (B,1,spatial) or (B,L,1,spatial) (coil‑combined inside)
        """
        assert (self._plans_t1 is not None) and (self._maps is not None) and (self._traj_BndK is not None)
        spatial = self._spatial(); K = self._K; maps = self._maps
        Cmaps = int(self._im_shape[0])

        if y.ndim == 3:
            B, C = int(y.shape[0]), int(y.shape[1]); L = 1
            if C != Cmaps:
                raise ValueError(f"y has C={C} but maps have C={Cmaps}")
            need = (B, 1) + spatial
        elif y.ndim == 4:
            B, L, C = int(y.shape[0]), int(y.shape[1]), int(y.shape[2])
            if C != Cmaps:
                raise ValueError(f"y has C={C} but maps have C={Cmaps}")
            need = (B, L, 1) + spatial
        else:
            raise ValueError("y must be (B,C,K) or (B,L,C,K)")

        # OUT
        if out is None:
            x_out = torch.empty(need, device=y.device, dtype=y.dtype)
        else:
            if tuple(out.shape) != need:
                raise ValueError(f"out shape {tuple(out.shape)} must be {need}.")
            x_out = out

        # Ensure plans for this Like extent
        self.ensure_like_prod(Cmaps * L)
        ntr = int(self._n_trans)

        # grid scratch
        grid = None
        if scratch is not None:
            grid = scratch.get("grid", None)
        if grid is None and arena is not None:
            numel_grid = ntr
            for s in spatial: numel_grid *= int(s)
            grid = arena.request(numel_grid, dtype=y.dtype, anchor=x_out).reshape((ntr,) + spatial)  # type: ignore[attr-defined]
        if not (isinstance(grid, torch.Tensor) and tuple(grid.shape) == ((ntr,) + spatial) and grid.device == y.device and grid.dtype == y.dtype):
            grid = torch.empty((ntr,) + spatial, device=y.device, dtype=y.dtype)

        def _get_cnu() -> torch.Tensor:
            cnu = None
            if scratch is not None:
                cnu = scratch.get("cnu", None)
                if isinstance(cnu, torch.Tensor) and tuple(cnu.shape) == (ntr, K) and cnu.device == y.device and cnu.dtype == y.dtype:
                    return cnu
            if arena is not None:
                return arena.request(ntr * K, dtype=y.dtype, anchor=x_out).reshape(ntr, K)  # type: ignore[attr-defined]
            return torch.empty((ntr, K), device=y.device, dtype=y.dtype)

        # Execute per frame
        for b in range(B):
            plan1 = self._plans_t1[b % len(self._plans_t1)]

            if L == 1:
                yb = y[b]  # (C,K)
                # AH weighting
                if self._dcf_BK is not None:
                    if self.dcf_mode == 'standard' and self.apply_dcf_in_adj:
                        cnu = _get_cnu(); cnu.copy_(yb); cnu.mul_(self._dcf_BK[b].view(1, K).expand(Cmaps, K))
                    elif self.dcf_mode == 'balanced':
                        cnu = _get_cnu(); cnu.copy_(yb); cnu.mul_(torch.sqrt(self._dcf_BK[b].clamp_min(0)).view(1, K).expand(Cmaps, K))
                    else:
                        cnu = yb if yb.is_contiguous() else _get_cnu(); 
                        if cnu is not yb: cnu.copy_(yb)
                else:
                    cnu = yb if yb.is_contiguous() else _get_cnu()
                    if cnu is not yb: cnu.copy_(yb)

                plan1.execute(cnu, grid)  # (C,spatial)
                x_b = (grid * torch.conj(maps)).sum(dim=0, keepdim=True)  # (1,spatial)
                x_out[b].copy_(x_b)

            else:
                yb = y[b]  # (L,C,K)
                if self._dcf_BK is not None:
                    if self.dcf_mode == 'standard' and self.apply_dcf_in_adj:
                        cnu = _get_cnu(); cnu.view(L, Cmaps, K).copy_(yb); cnu.mul_(self._dcf_BK[b].view(1, 1, K).expand(L, Cmaps, K))
                    elif self.dcf_mode == 'balanced':
                        cnu = _get_cnu(); cnu.view(L, Cmaps, K).copy_(yb); cnu.mul_(torch.sqrt(self._dcf_BK[b].clamp_min(0)).view(1, 1, K).expand(L, Cmaps, K))
                    else:
                        if yb.is_contiguous():
                            cnu = yb.view(ntr, K)
                        else:
                            cnu = _get_cnu(); cnu.view(L, Cmaps, K).copy_(yb)
                else:
                    if yb.is_contiguous():
                        cnu = yb.view(ntr, K)
                    else:
                        cnu = _get_cnu(); cnu.view(L, Cmaps, K).copy_(yb)

                plan1.execute(cnu, grid)  # (L*C, spatial)
                grid_v = grid.view(L, Cmaps, *spatial)
                x_like = (grid_v * torch.conj(maps).view(1, Cmaps, *spatial)).sum(dim=1)  # (L,spatial)
                x_out[b].copy_(x_like.view(L, 1, *spatial))

        return x_out


    # ------------------------------- Calibration ------------------------------

    @torch.no_grad()
    def diag_AHA_profile(self) -> torch.Tensor:
        """α_b = sum_k dcf[b,k] if DCF set, else α_b = K."""
        assert self._traj_BndK is not None
        B = int(self._traj_BndK.shape[0])
        if self._dcf_BK is None:
            return torch.full((B,), float(self._K), device=self._traj_BndK.device, dtype=torch.float32)
        return self._dcf_BK.sum(dim=1).to(torch.float32)

    @torch.no_grad()
    def diag_AHA_scalar(self) -> float:
        """Exact α via delta; cached after first call."""
        if self._alpha_scalar is not None:
            return float(self._alpha_scalar)
        assert (self._im_shape is not None) and (self._dev is not None) and (self._maps is not None)
        spatial = self._spatial()
        ctr = tuple(s // 2 for s in spatial)
        C = int(self._im_shape[0])

        x0 = torch.zeros((1, C) + spatial, device=self._dev, dtype=self._dtype)
        x0[0, 0][ctr] = 1.0 + 0.0j
        y = self.A(x0)      # (1,C,K)
        z = self.AH(y)      # (1,1,spatial)

        sos = (self._maps.abs() ** 2).sum(dim=0)[ctr].clamp_min(1e-20).item()
        alpha = float(z[0, 0][ctr].real.item()) / float(sos)
        self._alpha_scalar = alpha
        return alpha
