from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Literal
import math
import torch


@dataclass
class TorchKbNUFFTAdapter:
    """
    TorchKb NUFFT backend adapter (streamlined).

    Shapes (North‑Star surface):
      x (image)   : (B_eff, C, X, Y[, Z])      complex
      y (k-space) : (B_eff, C, K)              complex
      maps        : (C, X, Y[, Z])             complex
      traj        : (B, nd, K) | (B, K, nd) | (nd, K)  (units controlled by traj_units)

    Notes
    -----
    • This adapter multiplies coil maps inside the operator:
        A: y = NUFFT( x * S )      (per-coil)
       AH: x = sum_c conj(S_c) * NUFFT^H( y_c * dcf )
      This matches the previous implementation and keeps tests/contracts intact. :contentReference[oaicite:3]{index=3}
    • Vectorization across *like* dims: fold them into batch (B_eff = B×L_other) at the
      call site; this adapter will expand ω/DCFs from B→B_eff automatically.
    • Trajectory units are explicit: 'rad' (pass-through) or 'norm' (cycles/pixel → radians).
    """

    ndim: int
    backend_name: str = "torchkb"
    traj_units: Literal["rad", "norm"] = "rad"
    dcf_mode: Literal["standard", "balanced", "none"] = "balanced"

    # Prepared state
    _im_shape: Optional[Tuple[int, ...]] = None           # (C, X, Y[,Z])
    _maps: Optional[torch.Tensor] = None                  # (C, X, Y[,Z]) complex on device
    _traj_BndK: Optional[torch.Tensor] = None             # (B, nd, K) float32
    _dcf_BK: Optional[torch.Tensor] = None                # (B, K) float32 or None
    _K: int = 0
    _fwd: Optional[torch.nn.Module] = None
    _adj: Optional[torch.nn.Module] = None
    _dev: Optional[torch.device] = None
    _dtype: torch.dtype = torch.complex64
    
    # Flags
    apply_dcf_in_fwd: bool = False
    apply_dcf_in_adj: bool = True

    # Optional cached calibration
    _alpha_profile: Optional[torch.Tensor] = None   # (B,)
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
        like_prod: int = 1,  # accepted for interface parity; TorchKb ignores
    ) -> None:
        """
        Initialize operators and cache ω/DCFs. 'like_prod' is ignored (we fold like→batch).
        """
        if self.ndim not in (2, 3):
            raise ValueError("TorchKbNUFFTAdapter supports ndim ∈ {2,3}")
        self._im_shape = tuple(int(s) for s in im_shape)   # (C, X, Y[,Z])
        self._dev = device
        self._dtype = dtype

        # Cache maps (complex) on device
        self._maps = maps.to(device=device, dtype=dtype, non_blocking=True).contiguous()

        # Cache trajectory as (B, nd, K) float32 on device
        self._traj_BndK = self.scale_traj(traj, self._im_shape).to(device=device, non_blocking=True).contiguous()
        B, _, K = self._traj_BndK.shape
        self._K = int(K)

        # Cache DCF as (B, K) float32 on device (expand from (K) if provided)
        if dcf is None:
            self._dcf_BK = None
        else:
            d = dcf
            if d.ndim == 1:
                d = d.view(1, -1)
            if d.ndim != 2:
                raise ValueError("dcf must be (B,K) or (K)")
            d = d.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
            if int(d.shape[0]) == 1 and int(d.shape[1]) == self._K:
                d = d.expand(B, -1)
            self._dcf_BK = d

        # Build TorchKb modules
        from torchkbnufft import KbNufft, KbNufftAdjoint

        spatial = tuple(int(s) for s in self._im_shape[1:])
        dtype_r = torch.float32 if dtype == torch.complex64 else torch.float64

        # Keep defaults lean & robust; grid_size=None is fine; numpoints=6 is a good tradeoff.
        self._fwd = KbNufft(im_size=spatial, grid_size=None, numpoints=6).to(device=device, dtype=dtype_r)
        self._adj = KbNufftAdjoint(im_size=spatial, grid_size=None, numpoints=6).to(device=device, dtype=dtype_r)

        # Invalidate cached calibration on re-prepare
        self._alpha_profile = None
        self._alpha_scalar = None

    def k_per_frame(self) -> int:
        return int(self._K)

    # ----------------------------- batch expand helpers -----------------------------

    
    def _expand_omega(self, Bx: int) -> torch.Tensor:
        """
        Expand ω to effective batch Bx.
        If prepared B == Bx: return as-is.
        If prepared B == 1:  broadcast to Bx.
        If Bx is an integer multiple of prepared B: repeat_interleave.
        Else: raise.
        """
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
            return None  # defensive: only apply when K matches
        Bd = int(dw.shape[0])
        if Bd == Bx:
            return dw
        if Bd == 1 and Bx > 1:
            return dw.expand(Bx, -1)
        if Bx % Bd == 0:
            r = Bx // Bd
            return dw.repeat_interleave(r, dim=0)
        raise ValueError(
            f"dcf batch {Bd} incompatible with requested {Bx} for K={K}."
        )
    # --------------------------------- operators -----------------------------------

    @torch.no_grad()
    def A(self, x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward NUFFT: image → k-space per coil.

        x: (B_eff, C, X, Y[,Z]) complex
        y: (B_eff, C, K)        complex
        """
        assert self._fwd is not None and self._maps is not None
        Bx, C = int(x.shape[0]), int(x.shape[1])
        K = self._K

        # Apply coil maps inside the op (contract legacy behavior)
        x_coils = x * self._maps.unsqueeze(0)

        # Expand ω/DCFs to the effective batch when folding like→batch
        om = self._expand_omega(Bx)

        # Execute batched NUFFT
        y = self._fwd(x_coils, om)  # (Bx, C, K), real dtype inside module
        y = y.to(dtype=x.dtype)

        # DCF handling
        if self.dcf_mode == "standard":
            if self.apply_dcf_in_fwd:
                dcf = self._expand_dcf(Bx, K)
                if dcf is not None: y = y * dcf.unsqueeze(1)
        elif self.dcf_mode == "balanced":
            dcf = self._expand_dcf(Bx, K)
            if dcf is not None: y = y * torch.sqrt(dcf.clamp_min(0)).unsqueeze(1)
        # 'none' -> no weighting

        if out is not None:
            if out.shape != y.shape:
                raise ValueError(f"out has shape {tuple(out.shape)} but expected {tuple(y.shape)}")
            out.copy_(y)
            return out
        return y

    @torch.no_grad()
    def AH(self, y: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Adjoint NUFFT: k-space per coil → coil-combined image.

        y: (Bx, C, K)        complex
        x: (Bx, 1, X, Y[,Z]) complex
        """
        assert self._adj is not None and self._maps is not None
        Bx, C = int(y.shape[0]), int(y.shape[1])
        spatial = tuple(int(s) for s in self._im_shape[1:])
        K = self._K

        om = self._expand_omega(Bx)

        # DCF handling in adjoint
        if self.dcf_mode == "standard":
            dw = self._expand_dcf(Bx, K) if self.apply_dcf_in_adj else None
            yw = y if dw is None else (y * dw.unsqueeze(1))
        elif self.dcf_mode == "balanced":
            dw = self._expand_dcf(Bx, K)
            yw = y if dw is None else (y * torch.sqrt(dw.clamp_min(0)).unsqueeze(1))
        else:
            yw = y  # 'none'

        x_c = self._adj(yw, om)  # (Bx, C, X,Y[,Z]) real dtype inside module
        x_sum = (x_c * torch.conj(self._maps).unsqueeze(0)).sum(dim=1, keepdim=True).to(dtype=y.dtype)

        if out is not None:
            if out.shape != (Bx, 1, *spatial):
                raise ValueError(f"out has shape {tuple(out.shape)} but expected {(Bx,1,*spatial)}")
            out.copy_(x_sum)
            return out
        return x_sum


    # -------------------------------- calibration ----------------------------------

    @torch.no_grad()
    def diag_AHA_profile(self) -> torch.Tensor:
        """
        Per-frame proxy α_b ≈ sum_k dcf[b,k] (or K if dcf is None).
        Constant across frames when frames share sampling/DCF. (B matches prepared ω.)
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
        Calibrate α by applying AH(A(δ)) at the spatial center and normalizing by
        Σ_c |S_c(center)|^2. Cached after the first call.
        """
        if self._alpha_scalar is not None:
            return float(self._alpha_scalar)
        assert self._im_shape is not None and self._dev is not None and self._maps is not None

        spatial = tuple(int(s) for s in self._im_shape[1:])
        ctr = tuple(s // 2 for s in spatial)

        # Use prepared B for ω (expand rules in A/AH handle larger B_eff at call-time)
        B = int(self._traj_BndK.shape[0]) if self._traj_BndK is not None else 1
        C = int(self._im_shape[0])

        x0 = torch.zeros((B, C) + spatial, device=self._dev, dtype=self._dtype)
        x0[0, 0][ctr] = 1.0 + 0.0j
        y = self.A(x0)     # (B,C,K)
        z = self.AH(y)     # (B,1,spatial)

        zc = float(z[0, 0][ctr].real.item())
        sos = float((self._maps.abs() ** 2).sum(dim=0)[ctr].clamp_min(1e-20).item())
        alpha = zc / sos
        self._alpha_scalar = alpha
        return alpha
