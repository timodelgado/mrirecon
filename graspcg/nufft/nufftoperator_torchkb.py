# Append to: /data2/timod/mrirecon/graspcg/nufft/nufftoperator_torchkb.py
from typing import Optional, Literal
import torch
import math
from torchkbnufft import KbNufft, KbNufftAdjoint

_DcompMode = Literal["none", "sqrt", "full"]

def _assert_same_device_tb(*tensors: torch.Tensor):
    devs = {t.device for t in tensors if t is not None}
    if len(devs) > 1:
        raise ValueError(f"[TORCHKB] All tensors must be on the same device, got {devs}.")

class SenseNUFFT_TORCHKB:
    """
    Minimal SENSE NUFFT using torchkbnufft with the same public API as CUFI:
      A(image, ktraj, dcomp, out=None)  -> (Nch,Ndz,Ndth,Ndr)
      AH(kdata,  ktraj, dcomp, out=None) -> (Ndz,Ndx,Ndy)
    """
    def __init__(self, *, Ndx: int, Ndy: int, Ndth: int, Ndr: int,
                 Ndz: int, Nch: int, coil_sensitivity_maps: torch.Tensor,
                 dcomp_mode: _DcompMode = "sqrt"):
        self.Ndx, self.Ndy = int(Ndx), int(Ndy)
        self.Ndth, self.Ndr = int(Ndth), int(Ndr)
        self.Ndz, self.Nch = int(Ndz), int(Nch)
        if dcomp_mode not in ("none", "sqrt", "full"):
            raise ValueError("dcomp_mode must be one of {'none','sqrt','full'}")
        self.dcomp_mode: _DcompMode = dcomp_mode

        # coil maps
        sos = torch.sqrt(torch.sum(torch.abs(coil_sensitivity_maps)**2,
                                   dim=0, keepdim=True)) + 1e-8
        self.csm = coil_sensitivity_maps / sos
        self.csm_conj = self.csm.conj()
        del sos

        self.device = self.csm.device

        # lazily-constructed modules (on csm.device)
        self._fwd = KbNufft(im_size=(self.Ndx, self.Ndy), grid_size=None).to(self.device)
        self._adj = KbNufftAdjoint(im_size=(self.Ndx, self.Ndy), grid_size=None).to(self.device)

        # arena hook (not used here but kept for symmetry)
        self._arena = None

        # keep ad-hoc scale
        c = 1.0 / math.sqrt(2.0 * math.pi)
        self._adhoc_scale = (
            c
            * (self.Ndth ** (-1.0 / (8.0 * math.sqrt(3.0))))
            * (self.Ndr  ** (-1.0 / 12.0))
            * ((self.Ndx * self.Ndy) ** (-1.0 / 3.0))
        )

    def attach_arena(self, arena) -> None:
        self._arena = arena

    @torch.no_grad()
    def A(self, image: torch.Tensor, ktraj: torch.Tensor, dcomp: Optional[torch.Tensor],
          *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.F(image, ktraj, dcomp, out=out)

    @torch.no_grad()
    def AH(self, kdata: torch.Tensor, ktraj: torch.Tensor, dcomp: Optional[torch.Tensor],
           *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.G(kdata, ktraj, dcomp, out=out)

    @torch.no_grad()
    def F(self, image: torch.Tensor, ktraj: torch.Tensor, dcomp: Optional[torch.Tensor],
          *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        if image.shape != (self.Ndz, self.Ndx, self.Ndy):
            raise ValueError(f"[TORCHKB.F] image shape {(self.Ndz,self.Ndx,self.Ndy)} expected, got {tuple(image.shape)}")
        if ktraj.shape != (2, self.Ndth, self.Ndr):
            raise ValueError(f"[TORCHKB.F] ktraj shape (2,Ndth,Ndr) expected, got {tuple(ktraj.shape)}")
        _assert_same_device_tb(image, ktraj, *( () if dcomp is None else (dcomp,) ))
        if image.device != self.device:
            raise ValueError(f"[TORCHKB.F] inputs on {image.device}, but modules on {self.device}.")

        # coil expansion, per-z batch: (Ndz,Nch,Ndx,Ndy)
        csm_b = self.csm.permute(1, 0, 2, 3)  # (Ndz,Nch,Ndx,Ndy)
        coil = image.unsqueeze(1) * csm_b

        # traj per batch: (Ndz, 2, Ndth*Ndr)
        kt = ktraj.reshape(2, -1).unsqueeze(0).expand(self.Ndz, 2, self.Ndth * self.Ndr)

        # NUFFT
        k_bt = self._fwd(coil, kt)  # (Ndz, Nch, Ndth*Ndr)
        k_bt = k_bt.view(self.Ndz, self.Nch, self.Ndth, self.Ndr)
        k_out = k_bt.permute(1, 0, 2, 3).contiguous()  # (Nch,Ndz,Ndth,Ndr)

        # dcomp
        if dcomp is not None and self.dcomp_mode != "none":
            w = dcomp.sqrt() if self.dcomp_mode == "sqrt" else dcomp
            k_out = k_out * w.view(1, 1, self.Ndth, self.Ndr)

        # ad-hoc scale (kept)
        k_out = k_out * self._adhoc_scale

        if out is not None:
            if out.shape != k_out.shape or out.device != image.device or out.dtype != k_out.dtype:
                raise ValueError("[TORCHKB.F] `out` has wrong shape/dtype/device.")
            out.copy_(k_out)
            return out
        return k_out

    @torch.no_grad()
    def G(self, kdata: torch.Tensor, ktraj: torch.Tensor, dcomp: Optional[torch.Tensor],
          *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        if kdata.shape != (self.Nch, self.Ndz, self.Ndth, self.Ndr):
            raise ValueError(f"[TORCHKB.G] kdata shape {(self.Nch,self.Ndz,self.Ndth,self.Ndr)} expected, got {tuple(kdata.shape)}")
        if ktraj.shape != (2, self.Ndth, self.Ndr):
            raise ValueError(f"[TORCHKB.G] ktraj shape (2,Ndth,Ndr) expected, got {tuple(ktraj.shape)}")
        _assert_same_device_tb(kdata, ktraj, *( () if dcomp is None else (dcomp,) ))
        if kdata.device != self.device:
            raise ValueError(f"[TORCHKB.G] inputs on {kdata.device}, but modules on {self.device}.")

        # weight k-space into a new tensor (caller input read-only)
        if dcomp is not None and self.dcomp_mode != "none":
            w = dcomp.sqrt() if self.dcomp_mode == "sqrt" else dcomp
            k_w = kdata * w.view(1, 1, self.Ndth, self.Ndr)
        else:
            k_w = kdata

        # arrange to (Ndz,Nch,Ndth*Ndr)
        k_bt = k_w.permute(1, 0, 2, 3).contiguous().view(self.Ndz, self.Nch, self.Ndth * self.Ndr)
        kt = ktraj.reshape(2, -1).unsqueeze(0).expand(self.Ndz, 2, self.Ndth * self.Ndr)

        # adjoint NUFFT -> (Ndz,Nch,Ndx,Ndy)
        imc = self._adj(k_bt, kt)

        # SENSE combine
        csm_b = self.csm.permute(1, 0, 2, 3)  # (Ndz,Nch,Ndx,Ndy)
        im = (imc * csm_b.conj()).sum(dim=1)  # (Ndz,Ndx,Ndy)

        # ad-hoc scale (kept)
        im = im * self._adhoc_scale

        if out is not None:
            if out.shape != im.shape or out.device != kdata.device or out.dtype != im.dtype:
                raise ValueError("[TORCHKB.G] `out` has wrong shape/dtype/device.")
            out.copy_(im)
            return out
        return im