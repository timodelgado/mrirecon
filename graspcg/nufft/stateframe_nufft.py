import torch
from typing import Optional
from graspcg.workspace.unified_arena import UnifiedArena
from .nufftoperator_cufi import get_cufi_opSENSE2
# ──────────────────────────────────────────────────────────────────────────────
# stateframe_nufft3_md.py
# ──────────────────────────────────────────────────────────────────────────────
class StateFrameNUFFT3MD:
    """
    Frame‑wise NUFFT on *multiple* GPUs.

    Frames are assigned round‑robin to the list of ``devices`` passed in
    the constructor.  All public methods keep exactly the same signature
    as the original `StateFrameNUFFT3`.

    Batched execution for torchkbnufft can be limited by `frames_per_batch` to control peak memory.
    """
    # ------------------------------------------------------------------ init
    def __init__(
            self,
            coil_sensitivity_maps : torch.Tensor,
            organized_ktraj       : torch.Tensor,
            organized_dcomp       : torch.Tensor,
            *,
            devices: list[int],
            prestage_constants: bool = True,   # default on
        ):
        self.devices          = devices
        self.organized_ktraj  = organized_ktraj
        self.organized_dcomp  = organized_dcomp
        self.prestage_constants = prestage_constants

        # infer sizes
        self.Nt,  self.Ndth, self.Ndr = organized_dcomp.shape
        self.Nch, self.Ndz, self.Ndx, self.Ndy = coil_sensitivity_maps.shape
        self.imshape = [self.Nt, self.Ndz, self.Ndx, self.Ndy]
        self.kshape = [self.Nt, self.Nch, self.Ndth, self.Ndr]

        # 1 NUFFT op per device
        self.nufft_ops: dict[int, get_cufi_opSENSE2] = {}
        for dev in devices:
            with torch.cuda.device(dev):
                maps_dev = coil_sensitivity_maps.to(dev, non_blocking=True)
                op_pars  = dict(
                    Ndx=self.Ndx, Ndy=self.Ndy, Ndz=self.Ndz,
                    Nch=self.Nch, Ndth=self.Ndth, Ndr=self.Ndr,
                    device_id=dev)
                self.nufft_ops[dev] = get_cufi_opSENSE2(op_pars, maps_dev)

        # Frame → device mapping (round-robin) and local index per device
        self.frames_per_dev: dict[int, list[int]] = {dev: [] for dev in devices}
        for f in range(self.Nt):
            self.frames_per_dev[self.devices[f % len(self.devices)]].append(f)

        # Pre-stage immutable per-frame constants (ktraj, dcomp) once per device
        self.ktraj_dev: dict[int, torch.Tensor] = {}
        self.dcomp_dev: dict[int, torch.Tensor] = {}
        self.local_index: dict[int, int] = {}  # frame f -> index within its device block

        if self.prestage_constants:
            for dev in devices:
                f_list = self.frames_per_dev[dev]
                if not f_list:
                    continue
                ktraj_buf = torch.empty((len(f_list), 2, self.Ndth, self.Ndr),
                                        dtype=self.organized_ktraj.dtype,
                                        device=torch.device(f"cuda:{dev}"))
                dcomp_buf = torch.empty((len(f_list), self.Ndth, self.Ndr),
                                        dtype=self.organized_dcomp.dtype,
                                        device=torch.device(f"cuda:{dev}"))

                # One-time copies (P2P if source already on a GPU)
                for j, f in enumerate(f_list):
                    ktraj_buf[j].copy_(self.organized_ktraj[f].to(ktraj_buf.device, non_blocking=False))
                    dcomp_buf[j].copy_(self.organized_dcomp[f].to(dcomp_buf.device, non_blocking=False))
                    self.local_index[f] = j

                self.ktraj_dev[dev] = ktraj_buf
                self.dcomp_dev[dev] = dcomp_buf

        # (optional) filled later by solver
        self.frame_scale = None
        # optional batching knob for torchkbnufft (None = all frames in one batch)
        self.frames_per_batch: Optional[int] = None
        self.scale_emp   = None

    # ---------------------------------------------------------- arena wiring
    def attach_arena(self, arena: UnifiedArena):
        for op in self.nufft_ops.values():
            op.attach_arena(arena)

    # ---------------------------------------------------------- helpers ----
    def _guard_on_device(self, t: torch.Tensor, dev: int, name: str):
        """Ensure tensor `t` is resident on cuda:`dev`."""
        if t.device.type != "cuda" or t.device.index != dev:
            raise ValueError(
                f"[StateFrameNUFFT3MD] {name} is on {t.device}, expected cuda:{dev}. "
                f"Move it explicitly before calling to avoid implicit copies."
            )

    def _dev_for_frame(self, f: int) -> int:
        """Round‑robin mapping frame → device."""
        return self.devices[f % len(self.devices)]

    def AH(self, organized_k_space, out: torch.Tensor | None = None, frames_per_batch: Optional[int] = None):
        """
        Adjoint NUFFT (k-space → image space).

        Parameters
        ----------
        organized_k_space : torch.Tensor
            Input k-space data.
        out : torch.Tensor, optional
            Output tensor to write results into.
        frames_per_batch : Optional[int], optional
            Number of frames per batch for processing (only affects torchkbnufft backend).
            CUFI backend still runs per-frame.

        Returns
        -------
        torch.Tensor
            Image space data.
        """
        if out is not None:
            image_space = out.zero_()
        else:
            image_space = organized_k_space.new_zeros((self.Nt, self.Ndz, self.Ndx, self.Ndy))

        for f in range(self.Nt):
            dev = self.devices[f % len(self.devices)]
            op  = self.nufft_ops[dev]

            # Operator device guard (if the backend exposes device_id)
            op_dev = getattr(op, "device_id", dev)
            if op_dev != dev:
                raise ValueError(f"[StateFrameNUFFT3MD.AH] Operator bound to cuda:{op_dev}, but frame assigned to cuda:{dev}.")

            # Data moves per frame (k-space often lives on CPU or a single GPU)
            kdata = organized_k_space[f].to(dev, non_blocking=True)

            # Constants: reuse pre-staged slices if available
            if self.prestage_constants:
                j = self.local_index[f]
                ktraj = self.ktraj_dev[dev][j]
                dcomp = self.dcomp_dev[dev][j]
            else:
                ktraj = self.organized_ktraj[f].to(dev, non_blocking=True)
                dcomp = self.organized_dcomp[f].to(dev, non_blocking=True)

            # Device guards to prevent accidental cross-device use
            self._guard_on_device(kdata, dev, "kdata")
            self._guard_on_device(ktraj, dev, "ktraj")
            self._guard_on_device(dcomp, dev, "dcomp")

            im = op.G(kdata, ktraj, dcomp)
            image_space[f].copy_(im.to(image_space.device, non_blocking=True))
        return image_space

    def A(self, image_space, out: torch.Tensor | None = None, frames_per_batch: Optional[int] = None):
        """
        Forward NUFFT (image space → k-space).

        Parameters
        ----------
        image_space : torch.Tensor
            Input image space data.
        out : torch.Tensor, optional
            Output tensor to write results into.
        frames_per_batch : Optional[int], optional
            Number of frames per batch for processing (only affects torchkbnufft backend).
            CUFI backend still runs per-frame.

        Returns
        -------
        torch.Tensor
            k-space data.
        """
        if out is not None:
            k_space = out.zero_()
        else:
            k_space = image_space.new_zeros((self.Nt, self.Nch, self.Ndz, self.Ndth, self.Ndr))

        for f in range(self.Nt):
            dev = self.devices[f % len(self.devices)]
            op  = self.nufft_ops[dev]

            # Operator device guard (if the backend exposes device_id)
            op_dev = getattr(op, "device_id", dev)
            if op_dev != dev:
                raise ValueError(f"[StateFrameNUFFT3MD.A] Operator bound to cuda:{op_dev}, but frame assigned to cuda:{dev}.")

            img = image_space[f].to(dev, non_blocking=True)

            if self.prestage_constants:
                j = self.local_index[f]
                ktraj = self.ktraj_dev[dev][j]
                dcomp = self.dcomp_dev[dev][j]
            else:
                ktraj = self.organized_ktraj[f].to(dev, non_blocking=True)
                dcomp = self.organized_dcomp[f].to(dev, non_blocking=True)

            # Device guards to prevent accidental cross-device use
            self._guard_on_device(img,   dev, "image")
            self._guard_on_device(ktraj, dev, "ktraj")
            self._guard_on_device(dcomp, dev, "dcomp")

            kd = op.F(img, ktraj, dcomp)
            k_space[f].copy_(kd.to(k_space.device, non_blocking=True))
        return k_space
