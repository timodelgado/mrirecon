import torch
from graspcg.workspace.unified_arena import UnifiedArena
from graspcg.nufft.nufftoperator import get_cufi_opSENSE2
# ──────────────────────────────────────────────────────────────────────────────
# stateframe_nufft3_md.py
# ──────────────────────────────────────────────────────────────────────────────
class StateFrameNUFFT3MD:
    """
    Frame‑wise NUFFT on *multiple* GPUs.

    Frames are assigned round‑robin to the list of ``devices`` passed in
    the constructor.  All public methods keep exactly the same signature
    as the original `StateFrameNUFFT3`.
    """
    # ------------------------------------------------------------------ init
    def __init__(
            self,
            coil_sensitivity_maps : torch.Tensor,
            organized_ktraj       : torch.Tensor,
            organized_dcomp       : torch.Tensor,
            *,
            devices: list[int]
        ):
        self.devices          = devices
        self.organized_ktraj  = organized_ktraj
        self.organized_dcomp  = organized_dcomp

        # ---------- infer sizes ---------------------------------------------
        self.Nt,  self.Ndth, self.Ndr = organized_dcomp.shape
        self.Nch, self.Ndz, self.Ndx, self.Ndy = coil_sensitivity_maps.shape
        self.imshape = [self.Nt, self.Ndz, self.Ndx, self.Ndy]
        self.kshape = [self.Nt, self.Nch, self.Ndth, self.Ndr]
        # ---------- 1 NUFFT op per device -----------------------------------
        self.nufft_ops: dict[int, get_cufi_opSENSE2] = {}
        for dev in devices:
            with torch.cuda.device(dev):
                maps_dev = coil_sensitivity_maps.to(dev, non_blocking=True)
                op_pars  = dict(
                    Ndx=self.Ndx, Ndy=self.Ndy, Ndz=self.Ndz,
                    Nch=self.Nch, Ndth=self.Ndth, Ndr=self.Ndr,
                    device_id=dev)
                self.nufft_ops[dev] = get_cufi_opSENSE2(op_pars, maps_dev)

        # (optional) filled later by solver
        self.frame_scale = None
        self.scale_emp   = None

    # ---------------------------------------------------------- arena wiring
    def attach_arena(self, arena: UnifiedArena):
        for op in self.nufft_ops.values():
            op.attach_arena(arena)

    # ---------------------------------------------------------- helpers ----
    def _dev_for_frame(self, f: int) -> int:
        """Round‑robin mapping frame → device."""
        return self.devices[f % len(self.devices)]

    # ---------------------------------------------------------- AH operator
    def AH(self, organized_k_space, out: torch.Tensor | None = None):
        if out is not None:
            image_space = out.zero_()
        else:
            image_space = organized_k_space.new_zeros(
                (self.Nt, self.Ndz, self.Ndx, self.Ndy))

        # ----------- per‑frame dispatch -------------------------------------
        for f in range(self.Nt):
            dev   = self._dev_for_frame(f)
            op    = self.nufft_ops[dev]

            kdata =  organized_k_space[f].to(dev, non_blocking=True)
            ktraj = self.organized_ktraj[f].to(dev, non_blocking=True)
            dcomp = self.organized_dcomp[f].to(dev, non_blocking=True)

            im    = op.G(kdata, ktraj, dcomp)            # adjoint NUFFT
            image_space[f].copy_(im.to(image_space.device, non_blocking=True))

        return image_space

    # ---------------------------------------------------------- A operator
    def A(self, image_space, out: torch.Tensor | None = None):
        if out is not None:
            k_space = out.zero_()
        else:
            k_space = image_space.new_zeros(
                (self.Nt, self.Nch, self.Ndz, self.Ndth, self.Ndr))
            
        for f in range(self.Nt):
            dev   = self._dev_for_frame(f)
            op    = self.nufft_ops[dev]

            img   = image_space[f].to(dev, non_blocking=True)
            ktraj = self.organized_ktraj[f].to(dev, non_blocking=True)
            dcomp = self.organized_dcomp[f].to(dev, non_blocking=True)

            kd    = op.F(img, ktraj, dcomp)              # forward NUFFT
            k_space[f].copy_(kd.to(k_space.device, non_blocking=True))

        return k_space
