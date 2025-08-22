from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Dict, Any, List
import math
import torch

@dataclass
class PTNUFFTAdapter:
    """
    Pure‑Torch NUFFT (gridding/degridding + FFT) that:
      • Vectorizes across Like×Coil and accepts per‑frame trajectories (loops B)
      • Writes directly into caller‑supplied `out` (no y-staging/ck)
      • Uses minimal scratch: a single `grid` buffer (n_trans, *grid_spatial)
      • Default DCF policy = 'balanced' (√DCF on both A and AH)
    Shapes (canonical, backend order):
      A:  x -> y
        x: (B,C,H,W[,D]) or (B,L,C,H,W[,D])   complex
        y: (B,C,K)       or (B,L,C,K)         complex
      AH: y -> x
        y: (B,C,K)       or (B,L,C,K)
        x: (B,1,H,W[,D]) or (B,L,1,H,W[,D])
    """

    # --- config ---------------------------------------------------------------
    ndim: int
    backend_name: str = "ptnufft"
    traj_units: Literal["rad", "norm"] = "rad"
    dcf_mode: Literal["balanced", "standard", "none"] = "balanced"
    apply_dcf_in_fwd: bool = False    # used only when dcf_mode == 'standard'
    apply_dcf_in_adj: bool = True     # used only when dcf_mode == 'standard'
    osf: float = 2.0                  # oversampling factor
    kwidth: int = 4                   # kernel width per dim (M = kwidth^nd)
    chunk_k: Optional[int] = None     # chunk K to cap peak gather/scatter memory
    fft_norm: Literal["backward","ortho","forward"] = "backward"

    # --- prepared state -------------------------------------------------------
    _im_shape: Optional[Tuple[int, ...]] = None      # (C,H,W[,D])
    _maps: Optional[torch.Tensor] = None             # (C,H,W[,D]) complex on device
    _traj_BndK: Optional[torch.Tensor] = None        # (B,nd,K) float32 radians
    _dcf_BK: Optional[torch.Tensor] = None           # (B,K) float32 or None
    _dev: Optional[torch.device] = None
    _dtype: torch.dtype = torch.complex64
    _K: int = 0
    _grid_shape: Optional[Tuple[int, ...]] = None    # (H',W'[,D'])
    # per-frame interpolation tables (indices & weights)
    _idx_B: List[torch.Tensor] | None = None         # list of (K,M) long on device
    _w_B:   List[torch.Tensor] | None = None         # list of (K,M) float32 on device
    # cached de/apodization
    _deapo_cache: Optional[Tuple[torch.Tensor, ...]] = None  # per-dim deapod 1D tensors

    # ------------------------------ utilities ---------------------------------
    def _to_BndK(self, traj: torch.Tensor) -> torch.Tensor:
        # Normalize to (B, nd, K), then scale → radians
        if traj.ndim == 2:
            t = traj
            if t.shape[0] != self.ndim and t.shape[1] == self.ndim:
                t = t.transpose(0, 1).contiguous()
            if t.shape[0] != self.ndim:
                raise ValueError("traj must be (nd,K) or (K,nd)")
            t = t.unsqueeze(0)
        elif traj.ndim == 3:
            t = traj
            if t.shape[1] == self.ndim: pass
            elif t.shape[2] == self.ndim: t = t.transpose(1, 2).contiguous()
            else: raise ValueError("traj must be (B,nd,K) or (B,K,nd)")
        else:
            raise ValueError("traj must have 2 or 3 dims.")
        t = t.to(torch.float32)
        return t if self.traj_units == "rad" else (2.0 * math.pi) * t

    def _grid_spatial(self, spatial: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(int(math.ceil(self.osf * s)) for s in spatial)

    def _center_slices(self, big: Tuple[int, ...], small: Tuple[int, ...]):
        # returns slices to paste/crop centered: big[...] <-> small[...]
        starts = [ (B - S)//2 for B, S in zip(big, small) ]
        return tuple(slice(st, st+S) for st, S in zip(starts, small))

    def _kb1d_weight(self, dx: torch.Tensor) -> torch.Tensor:
        """
        Kaiser‑Bessel weight φ(dx) with width=W=self.kwidth and beta=self._beta,
        evaluated at distance to *cell center* in grid units.
        φ(dx) = I0(β * sqrt(1 - (2|dx|/W)^2)) / I0(β),   |dx| < W/2
                = 0,                                       otherwise
        """
        # inside support: 1 - (2|dx|/W)^2  ∈ [0, 1]
        arg = 1.0 - (2.0 * dx.abs() / float(self.kwidth))**2
        arg_clamped = torch.clamp(arg, min=0.0)
        num = torch.special.i0(self._beta * torch.sqrt(arg_clamped))
        return (num / self._i0_beta_den).to(dx.dtype)


    def _deapo(self, spatial: Tuple[int, ...]) -> Tuple[torch.Tensor, ...]:
        # per-dim deapodization (triangle kernel has mild rolloff; keep =1.0 as placeholder)
        # replace with Kaiser‑Bessel deapod for production
        outs = []
        for s in spatial:
            outs.append(torch.ones((s,), device=self._dev, dtype=torch.float32))
        return tuple(outs)

    # ------------------------------ planning -----------------------------------
    def prepare(self,
                im_shape: Tuple[int, ...],
                traj: torch.Tensor,
                dcf: Optional[torch.Tensor],
                maps: torch.Tensor,
                dtype: torch.dtype,
                device: torch.device,
                *, like_prod: int = 1) -> None:
        """
        Precompute per‑frame interpolation tables; cache DCF and maps on device.
        `like_prod` is accepted for API parity; this backend vectorizes over L×C at call time.
        """
        if self.ndim not in (2, 3):
            raise ValueError("PTNUFFTAdapter supports ndim ∈ {2,3}")
        if device.type != "cuda" and device.type != "cpu":
            raise RuntimeError(f"Unsupported device type: {device.type}")

        self._im_shape = tuple(int(s) for s in im_shape)   # (C, H, W[,D])
        C, *spatial = self._im_shape
        self._grid_shape = self._grid_spatial(tuple(spatial))
        self._dev = device
        self._dtype = dtype
        # Kaiser‑Bessel beta (Beatty 2005 / Fessler): α=osf, W=kwidth
        self._beta: float = math.pi * math.sqrt(((self.kwidth / float(self.osf)) * (float(self.osf) - 0.5))**2 - 0.8)
        # Denominator I0(β) (device tensor to match torch ops)
        self._i0_beta_den: torch.Tensor = torch.special.i0(
            torch.tensor(self._beta, device=device, dtype=torch.float32)
            )

        self._maps = maps.to(device=device, dtype=dtype, non_blocking=True).contiguous()
        self._traj_BndK = self._to_BndK(traj).to(device=device, non_blocking=True).contiguous()
        B, nd, K = self._traj_BndK.shape
        self._K = int(K)

        if dcf is None:
            self._dcf_BK = None
        else:
            d = dcf
            if d.ndim == 1: d = d.view(1, -1)
            if d.shape[-1] != K: raise ValueError("dcf must have K samples")
            self._dcf_BK = d.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
            if int(self._dcf_BK.shape[0]) == 1: self._dcf_BK = self._dcf_BK.expand(B, -1)

        # precompute indices & weights per frame
        self._idx_B, self._w_B = [], []
        G = self._grid_shape
        for b in range(B):
            coords = self._traj_BndK[b]  # (nd,K) radians, in [-π, π)
            # map to grid coordinates in [0, Gd)
            u = (coords / (2.0 * math.pi) + 0.5)
            if self.ndim == 2:
                gx = (u[0] * G[0]).clamp(0, G[0]-1e-6)
                gy = (u[1] * G[1]).clamp(0, G[1]-1e-6)
                # neighbor centers
                r = self.kwidth // 2
                offs_x = torch.arange(-r, -r + self.kwidth, device=device)
                offs_y = torch.arange(-r, -r + self.kwidth, device=device)
                nx = (gx.unsqueeze(-1) + offs_x).floor().long()           # integer cell indices
                ny = (gy.unsqueeze(-1) + offs_y).floor().long()
                nx = nx.remainder(G[0]); ny = ny.remainder(G[1])
                # distance to *cell centers* (nx + 0.5, ny + 0.5)
                cx = nx.to(gx.dtype) + 0.5
                cy = ny.to(gy.dtype) + 0.5
                dx = gx.unsqueeze(-1) - cx
                dy = gy.unsqueeze(-1) - cy
                wx = self._kb1d_weight(dx)                                # (K, kw)
                wy = self._kb1d_weight(dy)                                # (K, kw)
                # combine separably: (K, kw^2)
                idx = (nx.unsqueeze(-1) + G[0] * ny.unsqueeze(-2)).reshape(K, self.kwidth * self.kwidth)
                w   = (wx.unsqueeze(-1) * wy.unsqueeze(-2)).reshape(K, self.kwidth * self.kwidth)
            else:
                gx = (u[0] * G[0]).clamp(0, G[0]-1e-6)
                gy = (u[1] * G[1]).clamp(0, G[1]-1e-6)
                gz = (u[2] * G[2]).clamp(0, G[2]-1e-6)
                r = self.kwidth // 2
                offs = torch.arange(-r, -r + self.kwidth, device=device)
                nx = (gx.unsqueeze(-1) + offs).floor().long()
                ny = (gy.unsqueeze(-1) + offs).floor().long()
                nz = (gz.unsqueeze(-1) + offs).floor().long()
                nx = nx.remainder(G[0]); ny = ny.remainder(G[1]); nz = nz.remainder(G[2])
                # distances to centers
                cx = nx.to(gx.dtype) + 0.5
                cy = ny.to(gy.dtype) + 0.5
                cz = nz.to(gz.dtype) + 0.5
                dx = gx.unsqueeze(-1) - cx
                dy = gy.unsqueeze(-1) - cy
                dz = gz.unsqueeze(-1) - cz
                wx = self._kb1d_weight(dx)
                wy = self._kb1d_weight(dy)
                wz = self._kb1d_weight(dz)
                idx = (nx.unsqueeze(-1).unsqueeze(-1)
                    + G[0] * ny.unsqueeze(-2).unsqueeze(-1)
                    + (G[0]*G[1]) * nz.unsqueeze(-2).unsqueeze(-2))
                idx = idx.reshape(K, self.kwidth**3)
                w   = (wx.unsqueeze(-1).unsqueeze(-1)
                    * wy.unsqueeze(-2).unsqueeze(-1)
                    * wz.unsqueeze(-2).unsqueeze(-2)).reshape(K, self.kwidth**3)

            self._idx_B.append(idx.to(torch.long))
            self._w_B.append(w.to(torch.float32))
        # Build per-dimension deapodization (separable) on image sizes
        spatial = tuple(int(s) for s in self._im_shape[1:])
        grid_spatial = tuple(int(s) for s in self._grid_shape)
        deapo_1d = []
        apod_1d  = []
        for Sd, Gd in zip(spatial, grid_spatial):
            d1 = self._compute_deapo_1d(Sd, Gd)           # (Sd,)
            a1 = (1.0 / d1.clamp_min(1e-8))               # apodization = rolloff crop
            deapo_1d.append(d1)
            apod_1d.append(a1.to(d1.dtype))
        self._deapo_1d: Tuple[torch.Tensor, ...] = tuple(deapo_1d)
        self._apod_1d:  Tuple[torch.Tensor, ...] = tuple(apod_1d)

        # cache deapod
        self._deapo_cache = self._deapo(tuple(spatial))

    def k_per_frame(self) -> int:
        return int(self._K)

    # ------------------------------ forward (A) --------------------------------
    @torch.no_grad()
    def A(self,
          x: torch.Tensor,
          out: Optional[torch.Tensor] = None,
          *,
          scratch: Optional[Dict[str, torch.Tensor]] = None,
          arena: Optional[Any] = None) -> torch.Tensor:
        """
        Forward NUFFT (Type‑2): image -> k-space.
        Writes directly into `out` per frame (no ck when out is contiguous).
        scratch: expects optional 'grid' of shape (n_trans, *grid_spatial)
        """
        assert self._im_shape is not None and self._grid_shape is not None
        C, *spatial = self._im_shape
        G = self._grid_shape
        B = int(self._traj_BndK.shape[0]); K = self._K

        # Parse shapes & get/alloc OUT
        if x.ndim == (3 + self.ndim):
            Bx, Cx = int(x.shape[0]), int(x.shape[1]); L = 1
            if Cx != C: raise ValueError("x coil dim mismatches maps")
            need = (B, C, K)
        elif x.ndim == (4 + self.ndim):
            Bx, L, Cx = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
            if Cx != C: raise ValueError("x coil dim mismatches maps")
            need = (B, L, C, K)
        else:
            raise ValueError("x must be (B,C,spatial) or (B,L,C,spatial)")
        if Bx != B: raise ValueError("x batch B must match traj batch B")

        if out is None:
            y = torch.empty(need, device=x.device, dtype=x.dtype)
        else:
            if tuple(out.shape) != need: raise ValueError(f"out has shape {tuple(out.shape)} but expected {need}")
            y = out

        # optional FFT deapodization/padding factors (triangle: set to 1)
        # deap = self._deapo_cache   # not used here for triangle kernel

        # Work per frame b; vectorize across n_trans = (L×C) inside
        for b in range(B):
            idx = self._idx_B[b]; w = self._w_B[b]          # (K,M), float32
            M = int(w.shape[1])
            # n_trans and canonical views
            if L == 1:
                ntr = C
                x_b = x[b]                                  # (C, *S)
                y_b = y[b].view(ntr, K)                     # (C,K) as (ntr,K)
            else:
                ntr = L * C
                x_b = x[b].view(ntr, *spatial)              # (L*C, *S)
                y_b = y[b].view(ntr, K)                     # (L*C, K)

            # acquire/alloc grid scratch (ntr, *G)
            grid = None
            if scratch is not None: grid = scratch.get("grid", None)
            if grid is None and arena is not None:
                numel = ntr
                for s in G: numel *= s
                grid = arena.request(numel, dtype=x.dtype, anchor=y).reshape((ntr,)+tuple(G))  # type: ignore[attr-defined]
            if not (isinstance(grid, torch.Tensor) and tuple(grid.shape) == ((ntr,)+tuple(G)) and grid.device == x.device and grid.dtype == x.dtype):
                grid = torch.empty((ntr,)+tuple(G), device=x.device, dtype=x.dtype)

            # 1) center‑pad into grid and FFT in place
            grid.zero_()
            sl_big = self._center_slices(G, tuple(spatial))
            grid[(slice(None),)+sl_big] = x_b
            grid = torch.fft.fftn(grid, dim=tuple(range(1, 1+self.ndim)), norm=self.fft_norm)

            # 2) degrid: gather neighbors in K‑chunks to cap memory
            kc = int(self.chunk_k) if self.chunk_k is not None else K
            grid_flat = grid.reshape(ntr, -1)                # (ntr, Gtot)
            for s in range(0, K, max(1, kc)):
                sl = slice(s, min(K, s + max(1, kc)))
                idx_s = idx[sl]                              # (kc, M) long
                w_s   = w[sl]                                # (kc, M) float32
                # gather: (ntr, kc, M) complex
                g_nm  = grid_flat[:, idx_s]                  # advanced indexing
                # weighted sum over M → (ntr, kc)
                y_blk = (g_nm * w_s.to(g_nm.dtype)).sum(dim=-1)
                # 3) DCF (balanced/standard) in place on y
                if self._dcf_BK is not None:
                    if self.dcf_mode == "standard" and self.apply_dcf_in_fwd:
                        y_blk *= self._dcf_BK[b, sl].to(y_blk.dtype).unsqueeze(0)
                    elif self.dcf_mode == "balanced":
                        y_blk *= torch.sqrt(self._dcf_BK[b, sl].clamp_min(0)).to(y_blk.dtype).unsqueeze(0)
                y_b[:, sl] = y_blk

        return y

    # ------------------------------ adjoint (AH) -------------------------------
    @torch.no_grad()
    def AH(self,
           y: torch.Tensor,
           out: Optional[torch.Tensor] = None,
           *,
           scratch: Optional[Dict[str, torch.Tensor]] = None,
           arena: Optional[Any] = None) -> torch.Tensor:
        """
        Adjoint NUFFT (Type‑1): k-space -> image, coil‑combined inside.
        Writes directly into `out`; minimal scratch: 'grid' (n_trans,*G).
        """
        assert self._im_shape is not None and self._grid_shape is not None and self._maps is not None
        C, *spatial = self._im_shape
        G = self._grid_shape
        B = int(self._traj_BndK.shape[0]); K = self._K

        # Parse shapes & get/alloc OUT
        if y.ndim == 3:
            By, Cy = int(y.shape[0]), int(y.shape[1]); L = 1
            if Cy != C: raise ValueError("y C mismatches maps")
            need = (B, 1, *spatial)
        elif y.ndim == 4:
            By, L, Cy = int(y.shape[0]), int(y.shape[1]), int(y.shape[2])
            if Cy != C: raise ValueError("y C mismatches maps")
            need = (B, L, 1, *spatial)
        else:
            raise ValueError("y must be (B,C,K) or (B,L,C,K)")
        if By != B: raise ValueError("y batch B must match traj batch B")

        if out is None:
            x_out = torch.empty(need, device=y.device, dtype=y.dtype)
        else:
            if tuple(out.shape) != need: raise ValueError(f"out has shape {tuple(out.shape)} but expected {need}")
            x_out = out

        # deapod per-dim cache (triangle placeholder -> ones)
        deapo = self._deapo_cache

        for b in range(B):
            idx = self._idx_B[b]; w = self._w_B[b]; M = int(w.shape[1])

            if L == 1:
                ntr = C
                y_b = y[b].view(ntr, K)             # (C,K)
                x_b = x_out[b].view(1, *spatial)    # (1, *S) after combine
            else:
                ntr = L*C
                y_b = y[b].view(ntr, K)             # (L*C,K)
                x_b = x_out[b].view(L, 1, *spatial) # (L,1,*S)

            # acquire/alloc grid scratch
            grid = None
            if scratch is not None: grid = scratch.get("grid", None)
            if grid is None and arena is not None:
                numel = ntr
                for s in G: numel *= s
                grid = arena.request(numel, dtype=y.dtype, anchor=x_out).reshape((ntr,)+tuple(G))  # type: ignore[attr-defined]
            if not (isinstance(grid, torch.Tensor) and tuple(grid.shape) == ((ntr,)+tuple(G)) and grid.device == y.device and grid.dtype == y.dtype):
                grid = torch.empty((ntr,)+tuple(G), device=y.device, dtype=y.dtype)
            grid.zero_()

            # spread in K-chunks: grid_flat.scatter_add_(dim=1, idx, val)
            grid_flat = grid.reshape(ntr, -1)       # (ntr, Gtot)
            kc = int(self.chunk_k) if self.chunk_k is not None else K
            for s in range(0, K, max(1, kc)):
                sl = slice(s, min(K, s + max(1, kc)))
                idx_s = idx[sl]                     # (kc,M)
                w_s   = w[sl]                       # (kc,M)
                # weights & potential DCF on the fly → (ntr,kc)
                vals = y_b[:, sl]                   # (ntr,kc)
                if self._dcf_BK is not None:
                    if self.dcf_mode == "standard" and self.apply_dcf_in_adj:
                        vals = vals * self._dcf_BK[b, sl].to(vals.dtype).unsqueeze(0)
                    elif self.dcf_mode == "balanced":
                        vals = vals * torch.sqrt(self._dcf_BK[b, sl].clamp_min(0)).to(vals.dtype).unsqueeze(0)
                # scatter-add over neighbors m
                # For each m, add (ntr,kc) * (kc) → (ntr,kc)
                for m in range(M):
                    idx_m = idx_s[:, m]                                    # (kc,)
                    w_m   = w_s[:, m].to(vals.dtype)                        # (kc,)
                    grid_flat.scatter_add_(1,
                        idx_m.unsqueeze(0).expand(ntr, -1),                 # (ntr,kc)
                        vals * w_m.unsqueeze(0))                            # (ntr,kc)

            # IFFT, center-crop & coil combine (sum over C)
            grid = torch.fft.ifftn(grid, dim=tuple(range(1, 1+self.ndim)), norm=self.fft_norm)
            sl_big = self._center_slices(G, tuple(spatial))
            im_like_coils = grid[(slice(None),)+sl_big].contiguous()  # (ntr,*S)
            # deapodization (separable) in-place
            self._apply_separable_inplace(im_like_coils.unsqueeze(0), self._deapo_1d)
            im_like_coils = im_like_coils.view(-1, C, *spatial)       # (L,C,*S) or (1,C,*S)
            maps = self._maps
            x_like = (im_like_coils * torch.conj(maps).view(1, C, *spatial)).sum(dim=1, keepdim=True)

            x_b.copy_(x_like)

        return x_out

    def _compute_deapo_1d(self, S: int, G: int) -> torch.Tensor:
        """
        Compute 1-D roll-off correction (deapodization) for Kaiser–Bessel on an oversampled grid.
        Steps:
        1) Build 1-D kernel samples φ[k] on length-G k-grid (centered support at DC).
        2) rolloff = IFFT(φ) on length G (real, positive).
        3) Center-crop to image length S, then deapo = 1 / rolloff_crop (eps clamp).
        Returns: (S,) float32 on device self._dev
        """
        device = self._dev
        dtype_r = torch.float32
        r = self.kwidth // 2
        phi = torch.zeros((G,), device=device, dtype=dtype_r)
        # center index
        c = G // 2
        offs = torch.arange(-r, -r + self.kwidth, device=device, dtype=torch.float32)
        # distance to cell centers (… + 0.5)
        dx = offs + 0.5
        # KB weight at those offsets
        w = self._kb1d_weight(dx)  # (kw,)
        # place symmetrically around DC
        idx = (c + offs.to(torch.long)) % G
        phi[idx] = w.to(dtype_r)
        # periodic IFFT; rolloff is real & positive
        phi_shift = torch.fft.ifftshift(phi)
        roll = torch.fft.ifft(phi_shift, n=G, norm=self.fft_norm).real
        # center-crop to S
        sl = self._center_slices((G,), (S,))
        roll_crop = roll[sl]
        # avoid zeros; produce deapodization
        eps = 1e-8
        deapo = 1.0 / roll_crop.clamp_min(eps)
        return deapo.to(dtype_r)

    def _apply_separable_inplace(self, x: torch.Tensor, vecs: Tuple[torch.Tensor, ...]) -> None:
        """
        Multiply x (ntr, *spatial) in-place by separable real 1-D vectors per dim.
        vecs[d] is shape (Sd,) on same device/dtype-real; x is complex.
        """
        # x dims: (ntr, S0, S1[, S2])
        assert x.ndim == (1 + self.ndim)
        ntr = x.shape[0]
        # broadcast each 1-D vector across the corresponding axis
        for d, v in enumerate(vecs):
            # reshape v -> (1, 1,..., Sd, 1,...) to align with x (ntr, S0, S1[, S2])
            shape = [1] * (1 + self.ndim)
            shape[1 + d] = int(v.shape[0])
            x.mul_(v.view(*shape).to(device=x.device, dtype=x.real.dtype))
