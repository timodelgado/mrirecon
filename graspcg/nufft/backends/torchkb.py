from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Literal
import math
import torch


@dataclass
class TorchKbNUFFTAdapter:
    """
    torchkbnufft backend adapter (North‑Star surface + minimal scratch).

    North‑Star Shapes (adapter surface):
      x (image)   : (B_eff, 1, X, Y[, Z]) or (B_eff, C, X, Y[, Z])   complex
      y (k-space) : (B_eff, C, K)                                     complex
      maps        : (C, X, Y[, Z])                                    complex
      traj        : (B, nd, K) | (B, K, nd) | (nd, K)   (units via traj_units)

    DCF modes
    ---------
    dcf_mode='balanced' (default):
        A : y  = NUFFT(x * S) ; y *= sqrt(dcf)   (after execute)
        AH: yw = y * sqrt(dcf);  x = Σ_c conj(S_c) * NUFFT^H(yw)
    dcf_mode='standard':
        A : multiply by dcf IFF apply_dcf_in_fwd=True
        AH: multiply by dcf IFF apply_dcf_in_adj=True
    dcf_mode='none':
        No internal density weighting.

    Notes
    -----
    • Vectorization across *like* dims happens in the front‑end as B_eff = B×L_other;
      this adapter expands ω/DCF from B → B_eff as needed. :contentReference[oaicite:2]{index=2}
    • Map application happens inside A/AH; AH coil‑combines to a single‑coil image
      (image side uses coil=1) while k‑space preserves the coil axis. :contentReference[oaicite:3]{index=3}
    """

    # --- configuration
    ndim: int
    backend_name: str = "torchkb"
    traj_units: Literal["rad", "norm"] = "rad"
    dcf_mode: Literal["balanced", "standard", "none"] = "balanced"
    apply_dcf_in_fwd: bool = False   # used only when dcf_mode == 'standard'
    apply_dcf_in_adj: bool = True    # used only when dcf_mode == 'standard'
    chunk_beff: Optional[int] = None # optional micro-batch on B_eff to cap memory

    # --- prepared state
    _im_shape: Optional[Tuple[int, ...]] = None           # (C, X, Y[,Z])
    _maps: Optional[torch.Tensor] = None                  # (C, X, Y[,Z]) complex on device
    _traj_BndK: Optional[torch.Tensor] = None             # (B, nd, K) float32 (radians)
    _dcf_BK: Optional[torch.Tensor] = None                # (B, K) float32 or None
    _K: int = 0
    _fwd: Optional[torch.nn.Module] = None
    _adj: Optional[torch.nn.Module] = None
    _dev: Optional[torch.device] = None
    _dtype: torch.dtype = torch.complex64

    # cached calibration
    _alpha_profile: Optional[torch.Tensor] = None
    _alpha_scalar: Optional[float] = None

    # ----------------------------- prepare & scaling -----------------------------

    def _to_BndK(self, traj: torch.Tensor) -> torch.Tensor:
        """
        Normalize to (B, nd, K) float32. Accepts (B,nd,K), (B,K,nd), or (nd,K).
        """
        if traj.ndim == 2:
            t = traj
            # (nd,K) or (K,nd)
            if int(t.shape[0]) != self.ndim and int(t.shape[1]) == self.ndim:
                t = t.transpose(0, 1).contiguous()
            if int(t.shape[0]) != self.ndim:
                raise ValueError(f"traj has wrong nd; expected {self.ndim}")
            return t.unsqueeze(0).to(dtype=torch.float32)
        elif traj.ndim == 3:
            t = traj
            if int(t.shape[1]) == self.ndim:
                return t.to(dtype=torch.float32)
            if int(t.shape[2]) == self.ndim:
                return t.transpose(1, 2).contiguous().to(dtype=torch.float32)
            raise ValueError("traj must be (B,nd,K) or (B,K,nd)")
        else:
            raise ValueError("traj must have 2 or 3 dims")

    def scale_traj(self, traj: torch.Tensor, im_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Convert trajectory to radians if needed. Returns (B, nd, K) float32.
        """
        t = self._to_BndK(traj)
        if self.traj_units == "rad":
            return t
        if self.traj_units == "norm":
            return (2.0 * math.pi) * t
        raise ValueError("traj_units must be 'rad' or 'norm'")

    def prepare(
        self,
        im_shape: Tuple[int, ...],
        traj: torch.Tensor,
        dcf: Optional[torch.Tensor],
        maps: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
        like_prod: int = 1,  # accepted for interface parity; TorchKb folds like→batch in front‑end
    ) -> None:
        """
        Build modules and cache ω/DCFs. 'like_prod' is ignored (front‑end already folds like dims). 
        """
        if self.ndim not in (2, 3):
            raise ValueError("TorchKbNUFFTAdapter supports ndim ∈ {2,3}")
        self._im_shape = tuple(int(s) for s in im_shape)   # (C, X, Y[,Z])
        self._dev = device
        self._dtype = dtype

        # Cache maps & traj/DCFs on device
        self._maps = maps.to(device=device, dtype=dtype, non_blocking=True).contiguous()
        self._traj_BndK = self.scale_traj(traj, self._im_shape).to(device=device, non_blocking=True).contiguous()
        B, _, K = self._traj_BndK.shape
        self._K = int(K)

        if dcf is None:
            self._dcf_BK = None
        else:
            d = dcf
            if d.ndim == 1: d = d.view(1, -1)
            if d.ndim != 2:
                raise ValueError("dcf must be (B,K) or (K)")
            d = d.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
            if int(d.shape[0]) == 1 and int(d.shape[1]) == self._K:
                d = d.expand(B, -1)
            self._dcf_BK = d

        # Build torchkbnufft modules
        from torchkbnufft import KbNufft, KbNufftAdjoint
        spatial = tuple(int(s) for s in self._im_shape[1:])
        dtype_r = torch.float32 if dtype == torch.complex64 else torch.float64
        self._fwd = KbNufft(im_size=spatial, grid_size=None, numpoints=6).to(device=device, dtype=dtype_r)
        self._adj = KbNufftAdjoint(im_size=spatial, grid_size=None, numpoints=6).to(device=device, dtype=dtype_r)

        # Invalidate calibration caches
        self._alpha_profile = None
        self._alpha_scalar = None

    def k_per_frame(self) -> int:
        return int(self._K)

    # ----------------------------- expand helpers -----------------------------

    def _expand_omega(self, Bx: int) -> torch.Tensor:
        """Expand ω to effective batch Bx."""
        assert self._traj_BndK is not None
        om = self._traj_BndK
        Bom = int(om.shape[0])
        if Bom == Bx:
            return om
        if Bom == 1 and Bx > 1:
            return om.expand(Bx, -1, -1)
        if Bx % Bom == 0:
            r = Bx // Bom
            return om.repeat_interleave(r, dim=0)
        raise ValueError(
            f"omega batch {Bom} incompatible with requested {Bx}. "
            "Prepare with matching B or fold like→batch only when Bx is a multiple of B."
        )

    def _expand_dcf(self, Bx: int, K: int) -> Optional[torch.Tensor]:
        """Expand DCF to (Bx,K) if needed; None if not set."""
        dw = self._dcf_BK
        if dw is None:
            return None
        if int(dw.shape[-1]) != K:
            return None
        Bd = int(dw.shape[0])
        if Bd == Bx:
            return dw
        if Bd == 1 and Bx > 1:
            return dw.expand(Bx, -1)
        if Bx % Bd == 0:
            r = Bx // Bd
            return dw.repeat_interleave(r, dim=0)
        raise ValueError(f"dcf batch {Bd} incompatible with requested {Bx} for K={K}.")

    def _weights(self, Bx: int, K: int, direction: Literal['A','AH']) -> Optional[torch.Tensor]:
        """Return (Bx,K) weight or None based on dcf_mode and flags."""
        d = self._expand_dcf(Bx, K)
        if d is None:
            return None
        if self.dcf_mode == 'balanced':
            return torch.sqrt(d.clamp_min(0))
        if self.dcf_mode == 'standard':
            if direction == 'A' and self.apply_dcf_in_fwd:
                return d
            if direction == 'AH' and self.apply_dcf_in_adj:
                return d
            return None
        # 'none'
        return None

    # --------------------------------- operators ---------------------------------
    @torch.no_grad()
    def A(self, x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward NUFFT: image → k-space (per coil).
        Accepts x with Cin ∈ {1, Cmaps}; multiplies maps internally if Cin==1.
        Vectorizes over coils by folding C into batch for torchkbnufft.
        y shape always keeps the coil axis Cmaps.
        """
        assert self._fwd is not None and self._maps is not None and self._traj_BndK is not None and self._im_shape is not None
        Bx = int(x.shape[0]); K = self._K; Cmaps = int(self._im_shape[0]); spatial = tuple(int(s) for s in self._im_shape[1:])
        om_all = self._expand_omega(Bx)
        w_all = self._weights(Bx, K, direction='A')

        # Parse shapes & OUT
        if x.ndim == (2 + self.ndim):      # (B,Cin,*S)
            L = 1
            Cin = int(x.shape[1])
            need = (Bx, Cmaps, K)
        elif x.ndim == (3 + self.ndim):    # (B,L,Cin,*S)
            L = int(x.shape[1])
            Cin = int(x.shape[2])
            need = (Bx, L, Cmaps, K)
        else:
            raise ValueError(f"x must be (B,C,*S) or (B,L,C,*S); got shape {tuple(x.shape)} for ndim={self.ndim}")

        if Cin not in (1, Cmaps):
            raise ValueError(f"image coil slot must be 1 or {Cmaps}; got {Cin}")

        if out is None:
            y_out = torch.empty(need, dtype=x.dtype, device=x.device)
        else:
            if tuple(out.shape) != need: raise ValueError(f"out has shape {tuple(out.shape)} but expected {need}")
            y_out = out

        # Chunk over B_eff
        step = int(self.chunk_beff) if self.chunk_beff is not None else Bx
        for s in range(0, Bx, max(1, step)):
            sl = slice(s, min(Bx, s + max(1, step)))
            b = sl.stop - sl.start

            # Build (b*Cmaps,1,*S) and (b*Cmaps,nd,K)
            if L == 1:
                if Cin == 1:
                    x_coils = (x[sl, 0].unsqueeze(1) * self._maps.unsqueeze(0)).reshape(b * Cmaps, 1, *spatial)
                else:
                    x_coils = x[sl].reshape(b * Cmaps, 1, *spatial)
                om_rep = om_all[sl].repeat_interleave(Cmaps, dim=0)
                y_flat = self._fwd(x_coils, om_rep)
                if y_flat.ndim == 3: y_flat = y_flat.squeeze(1)
                if w_all is not None:
                    w_rep = w_all[sl].unsqueeze(1).expand(b, Cmaps, K).reshape(b * Cmaps, K)
                    y_flat.mul_(w_rep)
                y_out[sl].copy_(y_flat.reshape(b, Cmaps, K).to(dtype=x.dtype))
            else:
                if Cin == 1:
                    # (b,L,1,*S) * (1,1,C,*S) → (b,L,C,*S)
                    x_lc = x[sl].view(b, L, 1, *spatial) * self._maps.view(1, 1, Cmaps, *spatial)
                else:
                    x_lc = x[sl]
                x_flat = x_lc.reshape(b * L * Cmaps, 1, *spatial)
                om_rep = om_all[sl].repeat_interleave(L * Cmaps, dim=0)
                y_flat = self._fwd(x_flat, om_rep)
                if y_flat.ndim == 3: y_flat = y_flat.squeeze(1)
                if w_all is not None:
                    w_rep = w_all[sl].unsqueeze(1).unsqueeze(1).expand(b, L, Cmaps, K).reshape(b * L * Cmaps, K)
                    y_flat.mul_(w_rep)
                y_out[sl].copy_(y_flat.reshape(b, L, Cmaps, K).to(dtype=x.dtype))

        return y_out
    
    @torch.no_grad()
    def AH(self, y: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Adjoint NUFFT: k-space (per coil) → coil‑combined image.
        Folds C into batch, runs adjoint, then combines with conj(maps).
        """
        assert self._adj is not None and self._maps is not None and self._traj_BndK is not None and self._im_shape is not None
        Bx = int(y.shape[0]); K = self._K; Cmaps = int(self._im_shape[0]); spatial = tuple(int(s) for s in self._im_shape[1:])
        om_all = self._expand_omega(Bx)
        w_all = self._weights(Bx, K, direction='AH')

        # Parse shapes & OUT
        if y.ndim == 3:            # (B,C,K)
            L = 1; C = int(y.shape[1]); need = (Bx, 1, *spatial)
        elif y.ndim == 4:          # (B,L,C,K)
            L = int(y.shape[1]); C = int(y.shape[2]); need = (Bx, L, 1, *spatial)
        else:
            raise ValueError("y must be (B,C,K) or (B,L,C,K)")
        if C != Cmaps:
            raise ValueError(f"y coil dim {C} must equal maps C {Cmaps}")

        if out is None:
            x_out = torch.empty(need, dtype=y.dtype, device=y.device)
        else:
            if tuple(out.shape) != need: raise ValueError(f"out has shape {tuple(out.shape)} but expected {need}")
            x_out = out

        # Chunk over B_eff
        step = int(self.chunk_beff) if self.chunk_beff is not None else Bx
        for s in range(0, Bx, max(1, step)):
            sl = slice(s, min(Bx, s + max(1, step)))
            b = sl.stop - sl.start

            if L == 1:
                yw = y[sl] if w_all is None else (y[sl] * w_all[sl].unsqueeze(1))
                y_flat = yw.reshape(b * Cmaps, 1, K)
                om_rep = om_all[sl].repeat_interleave(Cmaps, dim=0)
                x_flat = self._adj(y_flat, om_rep)                 # (b*C,1,*S)
                if x_flat.ndim == 5: x_flat = x_flat  # (b*C,1,*S)
                x_c = x_flat.reshape(b, Cmaps, *spatial)
                x_sum = (x_c * torch.conj(self._maps).unsqueeze(0)).sum(dim=1, keepdim=True)
                x_out[sl].copy_(x_sum.to(dtype=y.dtype))
            else:
                yw = y[sl] if w_all is None else (y[sl] * w_all[sl].unsqueeze(1).unsqueeze(1))
                y_flat = yw.reshape(b * L * Cmaps, 1, K)
                om_rep = om_all[sl].repeat_interleave(L * Cmaps, dim=0)
                x_flat = self._adj(y_flat, om_rep)                 # (b*L*C,1,*S)
                x_lc = x_flat.reshape(b, L, Cmaps, *spatial)
                x_like = (x_lc * torch.conj(self._maps).view(1, 1, Cmaps, *spatial)).sum(dim=2, keepdim=True)
                x_out[sl].copy_(x_like.to(dtype=y.dtype))

        return x_out


    # -------------------------------- calibration ----------------------------------

    @torch.no_grad()
    def diag_AHA_profile(self) -> torch.Tensor:
        """
        Per‑frame proxy α_b ≈ sum_k dcf[b,k] (or K if no DCF). (Matches AᴴA diag scaling.)
        """
        assert self._traj_BndK is not None
        B = int(self._traj_BndK.shape[0])
        if self._dcf_BK is None:
            return torch.full((B,), float(self._K), device=self._traj_BndK.device, dtype=torch.float32)
        if int(self._dcf_BK.shape[0]) == 1:
            return self._dcf_BK.sum(dim=1).expand(B).to(torch.float32)
        return self._dcf_BK.sum(dim=1).to(torch.float32)

    @torch.no_grad()
    def diag_AHA_scalar(self) -> float:
        """
        Calibrate α by α = AᴴA(δ_center)/Σ_c |S_c|²(center). Cached after first call.
        """
        if self._alpha_scalar is not None:
            return float(self._alpha_scalar)
        assert self._im_shape is not None and self._dev is not None and self._maps is not None

        spatial = tuple(int(s) for s in self._im_shape[1:])
        ctr = tuple(s // 2 for s in spatial)
        B = int(self._traj_BndK.shape[0]) if self._traj_BndK is not None else 1
        C = int(self._im_shape[0])

        x0 = torch.zeros((B, C) + spatial, device=self._dev, dtype=self._dtype)
        x0[0, 0][ctr] = 1.0 + 0.0j
        y = self.A(x0)       # (B,C,K)
        z = self.AH(y)       # (B,1,spatial)

        zc = float(z[0, 0][ctr].real.item())
        sos = float((self._maps.abs() ** 2).sum(dim=0)[ctr].clamp_min(1e-20).item())
        alpha = zc / sos
        self._alpha_scalar = alpha
        return alpha
