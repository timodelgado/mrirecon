# graspcg/nufft/nufftoperator.py
from __future__ import annotations
import math, torch
from typing import Tuple, Dict, Optional
from torchkbnufft import KbNufft, AdjKbNufft  # pip install torchkbnufft
from graspcg.workspace.layout import LayoutSpec
from graspcg.workspace.unified_arena import UnifiedArena

def _as_complex32(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.complex64) if x.dtype != torch.complex64 else x

def _as_float32(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32) if x.dtype != torch.float32 else x

class TorchKbSenseNUFFT:
    """
    General NUFFT (2D/3D) with SENSE coil maps, trajectory axes, and batch axes.

    Shapes (compute order):
      x       : (B, *traj_shape, *im_size)                    complex64
      csm     : (Nc, *traj_shape, *im_size)   or broadcastable complex64
      ktraj   : (ndim, nK) | (Ttraj, ndim, nK) | (B, Ttraj, ndim, nK) float32
      dcomp   : optional, same broadcast semantics as ktraj but last dim nK

    Forward  A(x): (B, Nc, *traj_shape, nK)
    Adjoint AH(k): (B, *traj_shape, *im_size)

    Notes
    -----
    • `layout.nufft_axes` defines `im_size` (2D or 3D).
    • `layout.traj_axes` defines how many “per-trajectory” groups (e.g. z-slices).
    • All batch axes are flattened externally by the workspace into B.
    • Per-device modules are cached; attach_arena is supported.
    """
    def __init__(self,
                 layout: LayoutSpec,
                 *,
                 im_size: Tuple[int, ...] | None = None,
                 grid_size: Tuple[int, ...] | None = None,
                 table_ov: float = 2.0,
                 n_spatial_dims: int | None = None,
                 dcomp_mode: str = "sqrt",  # 'none'|'sqrt'|'full'
                 arena: Optional[UnifiedArena] = None):

        layout.validate()
        self.layout = layout
        self.ndim = n_spatial_dims or len(layout.nufft_axes)
        assert self.ndim in (2,3), "Only 2D/3D NUFFT supported by torchkbnufft."

        # infer im_size from layout if not provided
        if im_size is None:
            im_size = tuple(layout.image_shape[a] for a in layout.nufft_axes)
        self.im_size = tuple(int(s) for s in im_size)
        self.grid_size = tuple(int(math.ceil(s * 1.5)) for s in self.im_size) if grid_size is None else grid_size

        self.table_ov = float(table_ov)
        assert dcomp_mode in ("none","sqrt","full")
        self.dcomp_mode = dcomp_mode

        self.arena = arena  # set later via attach_arena if None
        # device → {'fwd':KbNufft, 'adj':AdjKbNufft}
        self._per_device: Dict[torch.device, Dict[str, torch.nn.Module]] = {}

    # .............................................................
    def attach_arena(self, arena: UnifiedArena):
        self.arena = arena

    # .............................................................
    def _ensure_modules(self, dev: torch.device):
        m = self._per_device.get(dev)
        if m is not None:
            return m
        # Construct modules on device
        fwd = KbNufft(
            im_size=self.im_size, grid_size=self.grid_size, num_spokes=None
        ).to(dev)
        adj = AdjKbNufft(
            im_size=self.im_size, grid_size=self.grid_size, num_spokes=None
        ).to(dev)
        self._per_device[dev] = {"fwd": fwd, "adj": adj}
        return self._per_device[dev]

    # .............................................................
    def _canon_traj(self, ktraj: torch.Tensor, B: int, Ttraj: int, dev: torch.device) -> torch.Tensor:
        """
        Return ktraj as shape (B, Ttraj, ndim, nK) on device.
        Accepts (ndim,nK), (Ttraj,ndim,nK), or (B,Ttraj,ndim,nK).
        """
        kt = _as_float32(ktraj).to(dev, non_blocking=True)
        if kt.ndim == 2:               # (ndim,nK) -> (1,1,ndim,nK) -> expand
            kt = kt.unsqueeze(0).unsqueeze(0).expand(B, Ttraj, *kt.shape)
        elif kt.ndim == 3:             # (Ttraj,ndim,nK) -> (1,Ttraj,ndim,nK)->expand
            kt = kt.unsqueeze(0).expand(B, *kt.shape)
        elif kt.ndim == 4:
            assert kt.shape[0] == B and kt.shape[1] == Ttraj, "ktraj batch/ traj dims mismatch"
        else:
            raise ValueError("ktraj must have shape (ndim,nK) or (Ttraj,ndim,nK) or (B,Ttraj,ndim,nK)")
        return kt

    def _canon_dcomp(self, dcomp: torch.Tensor | None, B: int, Ttraj: int, nK: int, dev: torch.device) -> Optional[torch.Tensor]:
        """
        Return dcomp as shape (B, Ttraj, nK) on device, or None.
        Accepts (nK), (Ttraj,nK), or (B,Ttraj,nK).
        """
        if dcomp is None:
            return None
        dc = _as_float32(dcomp).to(dev, non_blocking=True)
        if dc.ndim == 1:
            dc = dc.view(1,1,nK).expand(B,Ttraj,nK)
        elif dc.ndim == 2:
            assert dc.shape[0] == Ttraj and dc.shape[1] == nK
            dc = dc.view(1,Ttraj,nK).expand(B,Ttraj,nK)
        elif dc.ndim == 3:
            assert dc.shape[:2] == (B,Ttraj) and dc.shape[2] == nK
        else:
            raise ValueError("dcomp must have shape (nK) or (Ttraj,nK) or (B,Ttraj,nK)")
        return dc

    # .............................................................
    @torch.no_grad()
    def A(self,
          x: torch.Tensor,                  # (B, *traj_shape, *im_size)
          csm: torch.Tensor,                # (Nc, *traj_shape, *im_size)   (broadcastable)
          ktraj: torch.Tensor,              # any canonical shape
          dcomp: torch.Tensor | None = None,
          *,
          out: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward NUFFT with SENSE:
            x -> expand coils -> NUFFT -> (optional) density compensation.
        Returns (B, Nc, *traj_shape, nK) (or copies into `out` if provided).
        """
        dev = x.device
        mods = self._ensure_modules(dev)
        fwd = mods["fwd"]

        x = _as_complex32(x)
        B = x.shape[0]
        traj_shape = x.shape[1:-self.ndim]     # e.g., (Z,)
        Ttraj = int(math.prod(traj_shape)) if traj_shape else 1
        im_size = x.shape[-self.ndim:]
        assert tuple(im_size) == self.im_size, f"image size {im_size} vs op {self.im_size}"

        # canonicalize traj & dcomp: (B, Ttraj, ndim, nK) and (B,Ttraj,nK)
        # We need nK; infer from ktraj
        kt_c = self._canon_traj(ktraj, B, Ttraj, dev)
        nK   = int(kt_c.shape[-1])
        dc_c = self._canon_dcomp(dcomp, B, Ttraj, nK, dev)

        # reshape x and csm to (B,Ttraj,1,*im_size) then to (B*Ttraj, 1, *im_size)
        x_ = x.view(B, Ttraj, *im_size)
        # coil maps: (Nc,*traj_shape,*im_size) -> (1,Ttraj,Nc,*im_size) -> broadcast to (B,Ttraj,Nc,*im_size)
        Nc = int(csm.shape[0])
        csm_dev = _as_complex32(csm).to(dev, non_blocking=True)
        if csm_dev.ndim == self.ndim + 1:     # (Nc,*im_size) shared for all Ttraj
            csm_dev = csm_dev.unsqueeze(0).unsqueeze(0).expand(B, Ttraj, *csm_dev.shape)
        else:
            # assume (Nc,*traj_shape,*im_size)
            # bring traj dims right after batch for broadcast: (1,*traj_shape,Nc,*im_size)
            while csm_dev.ndim < 2 + self.ndim:  # pad if missing traj dims
                csm_dev = csm_dev.unsqueeze(1)    # add a size-1 traj dim
            csm_dev = csm_dev.unsqueeze(0)        # add batch axis
            csm_dev = csm_dev.expand(B, *csm_dev.shape[1:])

        x_bt  = x_.unsqueeze(2)                  # (B,Ttraj,1,*im_size)
        coil = x_bt * csm_dev                    # (B,Ttraj,Nc,*im_size)
        coil = coil.view(B*Ttraj, Nc, *im_size)

        # flatten ktraj to (B*Ttraj, ndim, nK)
        kt_bt = kt_c.view(B*Ttraj, self.ndim, nK)

        # NUFFT forward
        kdata = fwd(coil, kt_bt)                 # (B*Ttraj, Nc, nK)

        # density compensation
        if dc_c is not None:
            dc_bt = dc_c.view(B*Ttraj, nK)
            if self.dcomp_mode == "sqrt":
                kdata.mul_(dc_bt.sqrt().unsqueeze(1))
            elif self.dcomp_mode == "full":
                kdata.mul_(dc_bt.unsqueeze(1))

        # reshape to (B,Nc,*traj_shape,nK)
        out_shape = (B, Nc) + tuple(traj_shape) + (nK,)
        if out is not None:
            assert out.shape == out_shape and out.device == dev and out.dtype == kdata.dtype
            out.copy_(kdata.view(out_shape))
            return out
        return kdata.view(out_shape)

    # .............................................................
    @torch.no_grad()
    def AH(self,
           kdata: torch.Tensor,              # (B, Nc, *traj_shape, nK)
           csm: torch.Tensor,                # (Nc, *traj_shape, *im_size)
           ktraj: torch.Tensor,
           dcomp: torch.Tensor | None = None,
           *,
           out: torch.Tensor | None = None) -> torch.Tensor:
        """
        Adjoint NUFFT with SENSE:
            (optional) density weighting -> adjoint NUFFT -> SENSE combine.
        Returns (B, *traj_shape, *im_size) (or copies into `out`).
        """
        dev = kdata.device
        mods = self._ensure_modules(dev)
        adj = mods["adj"]

        kdata = _as_complex32(kdata)
        B, Nc = kdata.shape[:2]
        traj_shape = kdata.shape[2:-1]
        Ttraj = int(math.prod(traj_shape)) if traj_shape else 1
        nK = kdata.shape[-1]

        # canonicalize traj & dcomp
        kt_c = self._canon_traj(ktraj, B, Ttraj, dev)
        assert int(kt_c.shape[-1]) == nK, "ktraj last dim must match kdata last dim"
        dc_c = self._canon_dcomp(dcomp, B, Ttraj, nK, dev)

        # apply density weighting first (to k-space)
        k_bt = kdata.view(B*Ttraj, Nc, nK)
        if dc_c is not None:
            dc_bt = dc_c.view(B*Ttraj, nK)
            if self.dcomp_mode == "sqrt":
                k_bt.mul_(dc_bt.sqrt().unsqueeze(1))
            elif self.dcomp_mode == "full":
                k_bt.mul_(dc_bt.unsqueeze(1))

        # Adjoint NUFFT -> (B*Ttraj,Nc,*im_size)
        kt_bt = kt_c.view(B*Ttraj, self.ndim, nK)
        imc  = adj(k_bt, kt_bt)

        # SENSE combine with conj(csm)
        im_size = self.im_size
        imc = imc.view(B, Ttraj, Nc, *im_size)

        csm_dev = _as_complex32(csm).to(dev, non_blocking=True)
        if csm_dev.ndim == self.ndim + 1:  # (Nc,*im_size)
            csm_dev = csm_dev.unsqueeze(0).unsqueeze(0).expand(B, Ttraj, *csm_dev.shape)
        else:
            while csm_dev.ndim < 2 + self.ndim:
                csm_dev = csm_dev.unsqueeze(1)
            csm_dev = csm_dev.unsqueeze(0).expand(B, *csm_dev.shape[1:])

        # elementwise multiply conj(csm) and sum over coils
        im  = (imc * csm_dev.conj()).sum(dim=2)   # (B,Ttraj,*im_size)

        out_shape = (B,) + tuple(traj_shape) + tuple(im_size)
        if out is not None:
            assert out.shape == out_shape and out.device == dev and out.dtype == im.dtype
            out.copy_(im.view(out_shape))
            return out
        return im.view(out_shape)
