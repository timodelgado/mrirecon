# -----------------------------------------------------------------------------
# graspcg/nufft/api.py
# NUFFT front-end with non-canonical layouts + backend-aware vectorization
# -----------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Literal
import torch

# Layout helpers
from .layout import (
    AxisSpec,
    plan_image_layout, plan_kspace_layout,
    plan_image_layout_from_sizes, plan_kspace_layout_from_sizes,
    image_pack_for_torchkb, image_pack_for_cufi,
    kspace_pack_for_torchkb, kspace_pack_for_cufi,
    image_unpack_from_torchkb, kspace_unpack_from_torchkb,
    image_unpack_from_cufi,  kspace_unpack_from_cufi,
)

# Existing backends (unchanged)
from .backends.torchkb import TorchKbNUFFTAdapter   # (B,C,sp) <-> (B,C,K)  :contentReference[oaicite:4]{index=4}
from .backends.cufi    import CuFiNUFFTAdapter     # (B,C,sp) <-> (B,C,K)  :contentReference[oaicite:5]{index=5}


@dataclass(frozen=True)
class NUFFTConfig:
    # Spatial FFT dimensionality (adapters validate support; ≤3 per North Star)
    ndim: int
    # Backend
    backend: Literal['torchkb', 'cufi'] = 'torchkb'
    # Trajectory units
    traj_units: Literal['norm', 'rad'] = 'norm'
    # Density compensation mode:
    #   'standard'  -> apply DCF on adjoint (legacy default), optional in forward;
    #   'balanced'  -> multiply by sqrt(DCF) in both A and AH;
    #   'none'      -> never apply DCF internally.
    dcf_mode: Literal['standard', 'balanced', 'none'] = 'standard'
    # Legacy flags remain respected by adapters (ignored if dcf_mode != 'standard')
    apply_dcf_in_fwd: bool = False
    apply_dcf_in_adj: bool = True
    # CUFI tuning
    vectorize_coils: bool = True
    eps: float = 1e-6
    isign: int = -1
    # Tiny LRU cache size for CUFI plans keyed by n_trans=C×L
    max_cufi_plan_cache: int = 2


class NUFFT:
    """
    NUFFT operator honoring user axis labeling and exploiting backend vectorization.

    Inputs at construction:
      maps: (C, X, Y[, Z]) complex
      traj: (B, nd, K) or (B, K, nd) or (nd, K)
      dcf : (B, K) or (K) or None
      axis: AxisSpec
      config: NUFFTConfig

    Forward  A(x_user): image → k-space (per coil)
    Adjoint AH(y_user): k-space (per coil) → coil-combined image

    Vectorization:
      • TorchKb: folds like dims into batch (B_eff = B×L_other); batched over coils C.
      • CUFI   : loops like dims in front-end; vectorizes coils C inside CUFI plans.
    """
    def __init__(
        self,
        maps: torch.Tensor,
        traj: torch.Tensor,
        dcf: Optional[torch.Tensor],
        *,
        axis: AxisSpec,
        config: NUFFTConfig,
        dtype: torch.dtype = torch.complex64,
        device: Optional[torch.device] = None,
    ):
        if maps.ndim not in (3, 4):
            raise ValueError("maps must be (C,X,Y) or (C,X,Y,Z)")
        ndim = 2 if maps.ndim == 3 else 3
        if config.ndim != ndim:
            raise ValueError(f"config.ndim={config.ndim} must match maps dimensionality {ndim}")

        self.maps = maps
        self.traj0 = traj
        self.dcf0 = dcf
        self.axis = axis
        self.cfg = config
        self.dtype = dtype
        self.device = device or maps.device

        # Normalize traj to (B, nd, K) on device; adapters do units scaling.
        self._traj_BndK = self._to_BndK(self.traj0, ndim).to(self.device, dtype=torch.float32, non_blocking=True).contiguous()

        if self.dcf0 is None:
            self._dcf_BK = None
        else:
            d = self.dcf0
            if d.ndim == 1:
                d = d.view(1, -1)
            if d.ndim != 2:
                raise ValueError("dcf must be (B,K) or (K)")
            self._dcf_BK = d.to(self.device, dtype=torch.float32, non_blocking=True).contiguous()

        # Backend selection + policy propagation
        if self.cfg.backend == 'torchkb':
            be = TorchKbNUFFTAdapter(ndim=ndim)
            be.traj_units = self.cfg.traj_units
            be.dcf_mode   = self.cfg.dcf_mode
            be.apply_dcf_in_fwd = self.cfg.apply_dcf_in_fwd
            be.apply_dcf_in_adj = self.cfg.apply_dcf_in_adj
            self._backend = be
            like_prod = int(maps.shape[0])   # coils only; TorchKb folds like→batch up front
        elif self.cfg.backend == 'cufi':
            be = CuFiNUFFTAdapter(
                ndim=ndim, vectorize_coils=self.cfg.vectorize_coils,
                eps=self.cfg.eps, isign=self.cfg.isign
            )
            be.traj_units = self.cfg.traj_units
            be.dcf_mode   = self.cfg.dcf_mode
            be.apply_dcf_in_fwd = self.cfg.apply_dcf_in_fwd
            be.apply_dcf_in_adj = self.cfg.apply_dcf_in_adj
            be.max_cache  = int(self.cfg.max_cufi_plan_cache)
            self._backend = be
            like_prod = int(maps.shape[0])   # start with n_trans=C; may grow to C×L on demand
        else:
            raise ValueError("backend must be 'torchkb' or 'cufi'")


        im_shape = (int(maps.shape[0]),) + tuple(int(s) for s in maps.shape[1:])
        self._backend.prepare(
            im_shape,
            self._traj_BndK,
            self._dcf_BK,
            maps.to(self.device, dtype=self.dtype, non_blocking=True).contiguous(),
            dtype=self.dtype,
            device=self.device,
            like_prod=like_prod,
        )
        
    def __call__(self, x_user: torch.Tensor, out: Optional[torch.Tensor] = None, *, scratch: Optional[dict] = None) -> torch.Tensor:
        return self.A(x_user, out=out, scratch=scratch)

    def adjoint(self, y_user: torch.Tensor, out: Optional[torch.Tensor] = None, *, scratch: Optional[dict] = None) -> torch.Tensor:
        return self.AH(y_user, out=out, scratch=scratch)

    @torch.no_grad()
    def A(self, x_user: torch.Tensor, out: Optional[torch.Tensor] = None, *, scratch: Optional[dict] = None) -> torch.Tensor:
        """
        Forward NUFFT honoring (Batch, Like, FFT) → (Batch, Like, Coil, K).

        If 'out' is provided, the final result is written in-place with no new output allocation.
        Large intermediates (e.g., (B*L,C,K) or (B,L,C,K)) may be taken from 'scratch' if provided.
        """
        C = int(self.maps.shape[0])
        K = int(self.k_per_frame())

        if self.cfg.backend == 'torchkb':
            # Fold Like→Batch and execute once; unpack to user order (no allocation if out is given)
            x_BL1sp, lay_img = image_pack_for_torchkb(x_user, self.axis)
            B_eff = int(lay_img.B * lay_img.L_other)
            # Pick a BLK staging target (prefer Arena scratch)
            y_BLK = None
            if scratch is not None:
                buf = scratch.get("y_BLK", None)
                if isinstance(buf, torch.Tensor) and tuple(buf.shape) == (B_eff, C, K) and buf.dtype == x_user.dtype and buf.device == self.device:
                    y_BLK = buf
            # Execute; adapter honors out= for BLK
            y_BLK = self._backend.A(x_BL1sp, out=y_BLK)
            lay_k = plan_kspace_layout_from_sizes(lay_img.B, (lay_img.L_other,), C, K, self.axis)
            return kspace_unpack_from_torchkb(y_BLK, lay_k, out)

        # CUFI path (vectorized across Like → single execute per frame with n_trans=C×L)
        x_BL1sp, lay_img = image_pack_for_cufi(x_user, self.axis)  # (B,L,1,spatial)
        B, L = int(x_BL1sp.shape[0]), int(x_BL1sp.shape[1])
        # Ensure plan for n_trans=C×L (adapter uses tiny LRU cache)
        self._backend.ensure_like_prod(int(C * max(1, L)))
        # Broadcast object across coils as a view; adapter multiplies by maps internally
        x_BLCHW = x_BL1sp.expand(B, max(1, L), 1, *x_BL1sp.shape[3:]).expand(B, max(1, L), C, *x_BL1sp.shape[3:])

        # Choose BLCK staging (prefer Arena scratch); adapter honors out=
        y_BLCK = None
        if scratch is not None:
            buf = scratch.get("y_BLCK", None)
            if isinstance(buf, torch.Tensor) and tuple(buf.shape) == (B, max(1, L), C, K) and buf.dtype == x_user.dtype and buf.device == self.device:
                y_BLCK = buf
        y_BLCK = self._backend.A(x_BLCHW, out=y_BLCK)
        lay_k  = plan_kspace_layout_from_sizes(B, (max(1, L),), C, K, self.axis)
        return kspace_unpack_from_cufi(y_BLCK, lay_k, out)

    @torch.no_grad()
    def AH(self, y_user: torch.Tensor, out: Optional[torch.Tensor] = None, *, scratch: Optional[dict] = None) -> torch.Tensor:
        """
        Adjoint NUFFT honoring (Batch, Like, Coil, K) → (Batch, Like, 1, FFT…).

        If 'out' is provided, the final result is written in-place with no new output allocation.
        Large intermediates (e.g., (B,L,1,spatial)) may be taken from 'scratch' if provided.
        """
        spatial = tuple(int(s) for s in self.maps.shape[1:])

        if self.cfg.backend == 'torchkb':
            y_BLK, lay_k = kspace_pack_for_torchkb(y_user, self.axis)  # (B*L, C, K)
            x_B1sp = self._backend.AH(y_BLK)                           # (B*L, 1, spatial)
            lay_img = plan_image_layout_from_sizes(lay_k.B, (lay_k.L_other,), spatial, self.axis)
            return image_unpack_from_torchkb(x_B1sp, lay_img, out)

        # CUFI path (vectorized across Like)
        y_BLCK, lay_k = kspace_pack_for_cufi(y_user, self.axis)  # (B,L,C,K)
        B, L = int(lay_k.B), int(lay_k.L_other)
        self._backend.ensure_like_prod(int(y_BLCK.shape[1] * y_BLCK.shape[2]))  # L*C

        # Choose BL1sp staging (prefer Arena scratch)
        x_BL1sp = None
        if scratch is not None:
            buf = scratch.get("x_BL1sp", None)
            if isinstance(buf, torch.Tensor) and tuple(buf.shape) == (B, max(1, L), 1, *spatial) and buf.dtype == y_user.dtype and buf.device == self.device:
                x_BL1sp = buf
        x_BL1sp = self._backend.AH(y_BLCK, out=x_BL1sp)  # (B,L,1,spatial)
        lay_img = plan_image_layout_from_sizes(B, (max(1, L),), spatial, self.axis)
        return image_unpack_from_cufi(x_BL1sp, lay_img, out)


    # ------------------------------------------------------------------ shapes
    def k_per_frame(self) -> int:
        return int(self._backend.k_per_frame())

    def image_shape(self) -> Tuple[int, ...]:
        """Expected AH output shape given current maps/traj (coils already combined)."""
        B = int(self._traj_BndK.shape[0])
        spatial = tuple(int(s) for s in self.maps.shape[1:])
        return (B, 1, *spatial)


    def domain_shape(self) -> Tuple[int, ...]:
        spatial = tuple(int(s) for s in self.maps.shape[1:])
        return (1, 1, *spatial)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _to_BndK(traj: torch.Tensor, ndim: int) -> torch.Tensor:
        if traj.ndim == 2:
            t = traj
            if int(t.shape[0]) != ndim and int(t.shape[1]) == ndim:
                t = t.transpose(0, 1).contiguous()
            if int(t.shape[0]) != ndim:
                raise ValueError(f"traj has wrong nd; expected {ndim}")
            return t.unsqueeze(0)
        if traj.ndim == 3:
            t = traj
            if int(t.shape[1]) == ndim:
                return t
            if int(t.shape[2]) == ndim:
                return t.transpose(1, 2).contiguous()
            raise ValueError("traj must be (B,nd,K) or (B,K,nd)")
        raise ValueError("traj must have 2 or 3 dims")


# Backwards-compat alias (keeps existing imports functional)
NSNUFFT = NUFFT

__all__ = ["NUFFT", "NSNUFFT", "NUFFTConfig", "AxisSpec"]
