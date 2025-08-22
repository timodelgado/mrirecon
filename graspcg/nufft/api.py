# -----------------------------------------------------------------------------
# graspcg/nufft/api.py
# Streamlined North‑Star NUFFT front‑end (no big temporaries; out-first I/O)
# -----------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Sequence, Dict
import torch

# Axis + layout helpers (no heavy copies here; we only use planners)
from .layout import (
    AxisSpec,
    plan_image_layout, plan_kspace_layout,
    plan_image_layout_from_sizes, plan_kspace_layout_from_sizes,
)

# Adapters (unchanged in this step)
from .backends.torchkb import TorchKbNUFFTAdapter
from .backends.cufi    import CuFiNUFFTAdapter
from .backends.ptnufft import PTNUFFTAdapter


# Roles for Workspace/solvers
from ..core.roles import Roles


# --------------------------- Configuration ------------------------------------

@dataclass(frozen=True)
class NUFFTConfig:
    """
    Front-end configuration. Backends may ignore some fields (e.g., like_prod).
    """
    ndim: Literal[2, 3]
    backend: Literal['torchkb', 'cufi'] = 'torchkb'
    traj_units: Literal['norm', 'rad'] = 'norm'

    # DCF policy (North‑Star default: 'balanced'; backends honor in Step 2)
    dcf_mode: Literal['standard', 'balanced', 'none'] = 'balanced'
    apply_dcf_in_fwd: bool = False   # used only when dcf_mode == 'standard'
    apply_dcf_in_adj: bool = True    # used only when dcf_mode == 'standard'

    # CUFI tuning
    vectorize_coils: bool = True
    eps: float = 1e-6
    isign: int = -1


# ------------------------------ Front‑end -------------------------------------

class NUFFT:
    """
    North‑Star NUFFT front‑end with user-controlled axis labeling.

    Construction
    ------------
      maps : (C, X, Y[, Z]) complex
      traj : (B, nd, K) | (B, K, nd) | (nd, K)   [units via cfg.traj_units]
      dcf  : (B, K) | (K) | None
      axis : AxisSpec (Image: (B, Like..., FFT...), K‑space: (B, Like..., C, K))
      cfg  : NUFFTConfig

    Contracts
    ---------
    • Forward  A(x_user):  image  (B, Like..., 1, FFT...) → k-space (B, Like..., C, K)
    • Adjoint AH(y_user):  k-space (B, Like..., C, K)     → image  (B, Like..., 1, FFT...)
    • Coil combination happens **inside** the operator (AH sums conj(maps)·grid).  # adapters
    • When `out` is provided, the **final result** is written in-place (no extra result alloc).
      This front‑end produces a canonical **view** of `out` in backend order so adapters can
      write directly; no large intermediate staging buffers are allocated here.
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

        self.maps   = maps
        self.traj0  = traj
        self.dcf0   = dcf
        self.axis   = axis
        self.cfg    = config
        self.dtype  = dtype
        self.device = device or maps.device

        # Normalize trajectory to (B, nd, K) on device; adapters do units scaling again if needed.
        self._traj_BndK = self._to_BndK(self.traj0, ndim).to(self.device, dtype=torch.float32, non_blocking=True).contiguous()

        # Normalize DCF to (B,K) float32 on device
        if self.dcf0 is None:
            self._dcf_BK = None
        else:
            d = self.dcf0
            if d.ndim == 1: d = d.view(1, -1)
            if d.ndim != 2: raise ValueError("dcf must be (B,K) or (K)")
            self._dcf_BK = d.to(self.device, dtype=torch.float32, non_blocking=True).contiguous()

        # --------------------------- Backend selection -------------------------
        if self.cfg.backend == 'torchkb':
            be = TorchKbNUFFTAdapter(ndim=ndim)
            be.traj_units = self.cfg.traj_units
            be.apply_dcf_in_fwd = self.cfg.apply_dcf_in_fwd
            be.apply_dcf_in_adj = self.cfg.apply_dcf_in_adj
            try: be.dcf_mode = self.cfg.dcf_mode
            except Exception: pass
            self._backend = be
            like_prod = int(maps.shape[0])

        elif self.cfg.backend == 'cufi':
            be = CuFiNUFFTAdapter(ndim=ndim, eps=self.cfg.eps, isign=self.cfg.isign)
            be.traj_units = self.cfg.traj_units
            be.apply_dcf_in_fwd = self.cfg.apply_dcf_in_fwd
            be.apply_dcf_in_adj = self.cfg.apply_dcf_in_adj
            try: be.dcf_mode = self.cfg.dcf_mode
            except Exception: pass
            self._backend = be
            like_prod = int(maps.shape[0])

        elif self.cfg.backend == 'ptnufft':
            be = PTNUFFTAdapter(ndim=ndim)
            be.traj_units = self.cfg.traj_units
            be.apply_dcf_in_fwd = self.cfg.apply_dcf_in_fwd
            be.apply_dcf_in_adj = self.cfg.apply_dcf_in_adj
            try: be.dcf_mode = self.cfg.dcf_mode
            except Exception: pass
            self._backend = be
            like_prod = int(maps.shape[0])

        else:
            raise ValueError("backend must be 'torchkb', 'cufi', or 'ptnufft'")

        # Prepare backend once (we may re-prepare when Like>1 on CUFI)
        im_shape = (int(maps.shape[0]),) + tuple(int(s) for s in maps.shape[1:])
        self._backend.prepare(
            im_shape,
            self._traj_BndK,
            self._dcf_BK,
            maps.to(self.device, dtype=self.dtype, non_blocking=True).contiguous(),
            dtype=self.dtype,
            device=self.device,
            like_prod=int(like_prod),
        )

        # --------------------------- Roles exposure ----------------------------
        # Image roles: (B, Like..., FFT...), no coil axis.
        like_img = tuple(lbl for lbl in axis.image if (lbl not in axis.image_fft) and (lbl not in ('B', axis.coil)))
        like_ks  = tuple(lbl for lbl in axis.kspace if (lbl not in axis.kspace_fft) and (lbl not in ('B', axis.coil)))
        self.roles_image  = Roles(unlike=1, like=len(like_img), nufft=len(axis.image_fft))
        self.roles_kspace = Roles(unlike=1, like=len(like_ks),  nufft=1)

    # ------------------------------------------------------------------ core ops

    @torch.no_grad()
    def A(self, x_user: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward NUFFT:
            image (B, Like..., 1, FFT...) → k-space (B, Like..., C, K)   [user axis order]
        If `out` is provided, the result is written in-place (no extra result allocation).
        """
        # Infer B, Like sizes & spatial from the *input* (no copies)
        img_plan = plan_image_layout(x_user, self.axis)     # B, like_labels, like_sizes, spatial
        B        = int(img_plan.B)
        L_sizes  = tuple(img_plan.like_sizes)
        L_other  = int(img_plan.L_other)
        spatial  = tuple(int(s) for s in img_plan.spatial)
        C        = int(self.maps.shape[0])
        K        = int(self.k_per_frame())

        # Ensure we have an output tensor in *user order*
        if out is None:
            # Build the user-layout shape for k-space (B, Like..., C, K)
            y_shape = self._user_kspace_shape(B, L_sizes, C, K)
            y_user  = torch.empty(y_shape, device=self.device, dtype=x_user.dtype)
        else:
            # Validate shape against expected user-layout shape
            exp_shape = self._user_kspace_shape(B, L_sizes, C, K)
            if tuple(out.shape) != tuple(exp_shape):
                raise ValueError(f"out has shape {tuple(out.shape)} but expected {tuple(exp_shape)}")
            y_user = out

        # Build a canonical *view* of y_user in backend order and call the adapter
        if self.cfg.backend == 'torchkb':
            # TorchKb canonical out view: (B_eff, C, K) with B_eff = B * L_other
            y_view = self._kspace_view_for_backend(y_user, B, L_sizes, C, K, fold_like_to_batch=True)
            # Image canonical view: (B_eff, 1, spatial)   (no big temp)
            x_view = self._image_view_for_backend(x_user, img_plan, fold_like_to_batch=True)  # (B_eff,1,...)
            self._backend.A(x_view, out=y_view)  # backend copies into y_view
            return y_user

        # CUFI: vectorize across Like using broadcasted view (no copies)
        # Canonical out view: (B, L, C, K)
        y_view = self._kspace_view_for_backend(y_user, B, L_sizes, C, K, fold_like_to_batch=False)

        # Canonical image view: (B, L, 1, spatial) then broadcast to (B,L,C,spatial)
        x_view_BL1 = self._image_view_for_backend(x_user, img_plan, fold_like_to_batch=False)  # (B,L,1,...)
        x_view_BLCHW = x_view_BL1.expand(B, max(1, L_other), 1, *spatial).expand(B, max(1, L_other), C, *spatial)

        # Re-prepare CUFI plans if Like product changed (LRU arrives in Step 2)
        ntr = getattr(self._backend, "_n_trans", C)
        if int(ntr) != int(C * max(1, L_other)):
            self._backend.prepare(
                (C,) + spatial, self._traj_BndK, self._dcf_BK, self.maps.to(self.device, dtype=self.dtype).contiguous(),
                dtype=self.dtype, device=self.device, like_prod=int(C * max(1, L_other))
            )

        self._backend.A(x_view_BLCHW, out=y_view)
        return y_user

    @torch.no_grad()
    def AH(self, y_user: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Adjoint NUFFT:
            k-space (B, Like..., C, K) → image (B, Like..., 1, FFT...)   [user axis order]
        If `out` is provided, the result is written in-place.
        """
        # Infer B, Like sizes & K from the *input* (no copies)
        ks_plan = plan_kspace_layout(y_user, self.axis)   # B, like_labels, like_sizes, C, K
        B        = int(ks_plan.B)
        L_sizes  = tuple(ks_plan.like_sizes)
        L_other  = int(ks_plan.L_other)
        spatial  = tuple(int(s) for s in self.maps.shape[1:])
        C        = int(self.maps.shape[0])
        K        = int(ks_plan.K)

        # Ensure we have an output tensor in *user order*
        if out is None:
            x_shape = self._user_image_shape(B, L_sizes, spatial)
            x_user  = torch.empty(x_shape, device=self.device, dtype=y_user.dtype)
        else:
            exp_shape = self._user_image_shape(B, L_sizes, spatial)
            if tuple(out.shape) != tuple(exp_shape):
                raise ValueError(f"out has shape {tuple(out.shape)} but expected {tuple(exp_shape)}")
            x_user = out

        if self.cfg.backend == 'torchkb':
            # Canonical input view:  (B_eff, C, K)
            y_view = self._kspace_view_for_backend(y_user, B, L_sizes, C, K, fold_like_to_batch=True)
            # Canonical out view:    (B_eff, 1, spatial)
            x_view = self._image_view_for_backend(x_user, None, fold_like_to_batch=True, B=B, L_sizes=L_sizes, spatial=spatial)
            self._backend.AH(y_view, out=x_view)
            return x_user

        # CUFI: Canonical input view: (B, L, C, K)
        y_view = self._kspace_view_for_backend(y_user, B, L_sizes, C, K, fold_like_to_batch=False)
        # Canonical out view: (B, L, 1, spatial)
        x_view = self._image_view_for_backend(x_user, None, fold_like_to_batch=False, B=B, L_sizes=L_sizes, spatial=spatial)

        # Re-prepare CUFI plans if Like product changed
        ntr = getattr(self._backend, "_n_trans", C)
        if int(ntr) != int(C * max(1, L_other)):
            self._backend.prepare(
                (C,) + spatial, self._traj_BndK, self._dcf_BK, self.maps.to(self.device, dtype=self.dtype).contiguous(),
                dtype=self.dtype, device=self.device, like_prod=int(C * max(1, L_other))
            )

        self._backend.AH(y_view, out=x_view)
        return x_user

    # ------------------------------------------------------------------ shapes

    def k_per_frame(self) -> int:
        return int(self._backend.k_per_frame())

    def image_shape(self, y_user: torch.Tensor) -> Tuple[int, ...]:
        # Convenience: uses maps for spatial dims; batch from traj
        B = int(self._to_BndK(self.traj0, self.cfg.ndim).shape[0])
        spatial = tuple(int(s) for s in self.maps.shape[1:])
        return (B, 1, *spatial)

    def domain_shape(self) -> Tuple[int, ...]:
        spatial = tuple(int(s) for s in self.maps.shape[1:])
        return (1, 1, *spatial)

    # ------------------------------------------------------------ view helpers

    def _user_kspace_shape(self, B: int, L_sizes: Sequence[int], C: int, K: int) -> Tuple[int, ...]:
        """Materialize user-order k-space shape from sizes."""
        plan = plan_kspace_layout_from_sizes(B, L_sizes, C, K, self.axis)
        size: Dict[str, int] = {'B': plan.B, self.axis.coil: plan.C, self.axis.kspace_fft[0]: plan.K}
        for lbl, s in zip(plan.like_labels, plan.like_sizes): size[lbl] = int(s)
        return tuple(int(size[lbl]) for lbl in self.axis.kspace)

    def _user_image_shape(self, B: int, L_sizes: Sequence[int], spatial: Sequence[int]) -> Tuple[int, ...]:
        """Materialize user-order image shape from sizes (coil-combined; coil dim=1 if present)."""
        plan = plan_image_layout_from_sizes(B, L_sizes, spatial, self.axis)
        size: Dict[str, int] = {'B': plan.B, self.axis.coil: 1}
        for lbl, s in zip(plan.like_labels, plan.like_sizes): size[lbl] = int(s)
        for i, lbl in enumerate(self.axis.image_fft): size[lbl] = int(spatial[i])
        return tuple(int(size[lbl]) for lbl in self.axis.image)

    def _kspace_view_for_backend(self,
                                 y_user: torch.Tensor,
                                 B: int, L_sizes: Sequence[int], C: int, K: int,
                                 *, fold_like_to_batch: bool) -> torch.Tensor:
        """
        Return a *view* of y_user arranged for the backend:
          • TorchKb: (B_eff, C, K) with B_eff=B×∏L
          • CUFI   : (B, L, C, K)
        """
        plan = plan_kspace_layout_from_sizes(B, L_sizes, C, K, self.axis)
        # target dims in canonical order
        dims_target = ('B',) + tuple(plan.like_labels) + (self.axis.coil, self.axis.kspace_fft[0])
        # current user order
        mp_user = {lbl: i for i, lbl in enumerate(self.axis.kspace)}
        perm = [mp_user[lbl] for lbl in dims_target]
        y_perm = y_user.permute(*perm)  # (B, *like, C, K) view
        if fold_like_to_batch:
            L_other = int(plan.L_other)
            return y_perm.reshape(B * max(1, L_other), C, K)
        return y_perm.reshape(B, max(1, int(plan.L_other)), C, K)

    def _image_view_for_backend(self,
                                x_user: torch.Tensor,
                                img_plan,  # ImageLayoutPlan or None
                                *,
                                fold_like_to_batch: bool,
                                B: Optional[int] = None,
                                L_sizes: Optional[Sequence[int]] = None,
                                spatial: Optional[Sequence[int]] = None) -> torch.Tensor:
        """
        Return a *view* of x_user arranged for the backend:
          • TorchKb: (B_eff, 1, spatial)
          • CUFI   : (B, L, 1, spatial)
        """
        if img_plan is None:
            assert B is not None and L_sizes is not None and spatial is not None
            img_plan = plan_image_layout_from_sizes(int(B), L_sizes, spatial, self.axis)
        # target dims: (B, *like, 1, spatial...)
        dims_target = ['B'] + list(img_plan.like_labels) + [self.axis.coil] + list(self.axis.image_fft)
        mp_user = {lbl: i for i, lbl in enumerate(self.axis.image)}
        perm = [mp_user[lbl] for lbl in dims_target]
        x_perm = x_user.permute(*perm)  # (B, *like, 1, spatial) view
        if fold_like_to_batch:
            B_eff = int(img_plan.B) * max(1, int(img_plan.L_other))
            return x_perm.reshape(B_eff, 1, *img_plan.spatial)
        return x_perm.reshape(int(img_plan.B), max(1, int(img_plan.L_other)), 1, *img_plan.spatial)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _to_BndK(traj: torch.Tensor, ndim: int) -> torch.Tensor:
        """Normalize trajectory to (B, nd, K)."""
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


# Backwards-compat alias
NSNUFFT = NUFFT

__all__ = ["NUFFT", "NSNUFFT", "NUFFTConfig", "AxisSpec"]
