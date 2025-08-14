import os
import math
import torch
from typing import Literal, Optional, Tuple
from .utils import get_ktraj2D_cufi, get_dcomp_2d, frame_scaling

# optional backends
from .nufftoperator_cufi import get_cufi_opSENSE2 as CufiOp
from .nufftoperator_torchkb import SenseNUFFT_TORCHKB as TorchKbOp  # new simple class

Backend = Literal["cufi", "torchkb"]
DcompPolicy = Literal["op_sqrt", "none", "preweight_cache"]

def sliding_window_view_kspace(k_space: torch.Tensor, temp_window: int, temp_slide: int) -> torch.Tensor:
    # k_space: (Nch, Ndz, Ndth_total, Ndr)
    Nch, Ndz, Ndth, Ndr = k_space.shape
    s_ch, s_dz, s_dth, s_dr = k_space.stride()
    Nframes = (Ndth - temp_window) // temp_slide + 1
    return torch.as_strided(
        k_space,
        size=(Nframes, Nch, Ndz, temp_window, Ndr),
        stride=(temp_slide * s_dth, s_ch, s_dz, s_dth, s_dr),
    )

def _sliding_window_view_traj(full_ktraj: torch.Tensor, temp_window: int, temp_slide: int) -> torch.Tensor:
    # full_ktraj: (2, Ndth_total, Ndr) -> (Nframes, 2, temp_window, Ndr) view
    _, Ndth, Ndr = full_ktraj.shape
    s0, s1, s2 = full_ktraj.stride()
    Nframes = (Ndth - temp_window) // temp_slide + 1
    return torch.as_strided(
        full_ktraj,
        size=(Nframes, 2, temp_window, Ndr),
        stride=(temp_slide * s1, s0, s1, s2),
    )
# --- Helpers for CUFI plan caching (module scope) -----------------------------

class _BoundCufiOp:
    """
    Wraps a CUFI operator "bound" to a frame's (ktraj, dcomp).
    Does a small warm-up to make sure the plan/points are primed once.
    """
    def __init__(self, base_op, *, ktraj: torch.Tensor, dcomp: Optional[torch.Tensor],
                 Ndz: int, Ndx: int, Ndy: int, Nch: int, Ndth: int, Ndr: int,
                 dtype, device):
        self.op = base_op
        self.ktraj = ktraj
        self.dcomp = dcomp
        # Warm-up (forward + adjoint) to ensure the plan is ready
        zero_img = torch.zeros((Ndz, Ndx, Ndy), dtype=dtype, device=device)
        tmp_k = torch.empty((Nch, Ndz, Ndth, Ndr), dtype=dtype, device=device)
        self.op.F(zero_img, self.ktraj, self.dcomp, out=tmp_k)
        tmp_im = torch.empty((Ndz, Ndx, Ndy), dtype=dtype, device=device)
        self.op.G(tmp_k, self.ktraj, self.dcomp, out=tmp_im)

    @torch.no_grad()
    def F(self, image: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return self.op.F(image, self.ktraj, self.dcomp, out=out)

    @torch.no_grad()
    def G(self, kdata: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return self.op.G(kdata, self.ktraj, self.dcomp, out=out)


class _CufiFramePlanFactory:
    """
    Lazily creates and caches CUFI operators with per-frame bound plans.
    One entry per frame index; subsequent requests reuse the same object.
    """
    def __init__(self, op_params: dict, csm: torch.Tensor, dcomp_mode: str,
                 Ndz: int, Ndx: int, Ndy: int, Nch: int, Ndth: int, Ndr: int,
                 dtype, device):
        self._op_params = dict(op_params)  # copy to avoid external mutation
        self._csm = csm
        self._dcomp_mode = dcomp_mode
        self._cache: dict[int, _BoundCufiOp] = {}
        self._shape = (Ndz, Ndx, Ndy, Nch, Ndth, Ndr)
        self._dtype = dtype
        self._device = device

    def get(self, f: int, *, ktraj: torch.Tensor, dcomp: Optional[torch.Tensor]) -> _BoundCufiOp:
        if f in self._cache:
            return self._cache[f]
        op = CufiOp(self._op_params, self._csm, dcomp_mode=self._dcomp_mode)
        bound = _BoundCufiOp(
            op, ktraj=ktraj, dcomp=dcomp,
            Ndz=self._shape[0], Ndx=self._shape[1], Ndy=self._shape[2],
            Nch=self._shape[3], Ndth=self._shape[4], Ndr=self._shape[5],
            dtype=self._dtype, device=self._device
        )
        self._cache[f] = bound
        return bound


# --- Replacement class --------------------------------------------------------

class FrameSenseNUFFT:
    """
    Frame-wise SENSE NUFFT wrapper with A/AH that iterates frames and calls
    a per-frame backend operator (CUFI or TORCHKB). Exposes:
      • A(frames)  -> (Nframes, Nch, Ndz, Ndth, Ndr)
      • AH(kspace) -> (Nframes, Ndz, Ndx, Ndy)
      • attach_arena(arena)
      • .scale_emp, .frame_scale, .dcomp_mode

    Supports: torchkbnufft batching (with optional frames_per_batch chunking),
    optional torch.compile on the torchkb path, and CUFI per-frame plan caching.
    """
    def __init__(self,
                 sensitivity_maps: torch.Tensor,
                 org_trj: torch.Tensor,          # (Nf, 2, W, Ndr)
                 org_cmp: torch.Tensor,          # (Nf, W, Ndr)
                 *,
                 backend: Backend,
                 dcomp_mode: Literal["none", "sqrt", "full"],
                 frames_per_batch: Optional[int] = None,
                 compile_mode: Literal["off", "static", "dynamic"] = "off",
                 use_cufi_frame_plan_cache: bool = True,
                 device_id: Optional[int] = None):
        self.csm = sensitivity_maps
        self.org_trj = org_trj
        self.org_cmp = org_cmp
        self.backend: Backend = backend
        self.dcomp_mode = dcomp_mode
        self.scale_emp = None   # filled after construction
        self.frame_scale = None

        # batching & compile controls
        self.frames_per_batch: Optional[int] = frames_per_batch
        self.compile_mode: Literal["off","static","dynamic"] = compile_mode
        self.use_cufi_frame_plan_cache: bool = bool(use_cufi_frame_plan_cache)

        # dims
        self.Nframes = int(org_trj.shape[0])
        self.Ndz, self.Ndx, self.Ndy = map(int, sensitivity_maps.shape[1:])
        self.Nch = int(sensitivity_maps.shape[0])
        self.Ndth = int(org_trj.shape[2])
        self.Ndr  = int(org_trj.shape[3])

        dev = sensitivity_maps.device
        self._cufi_op_params = None  # stored for plan cache

        # backend op
        if backend == "cufi":
            op_params = {
                "Ndth": self.Ndth, "Ndr": self.Ndr,
                "Ndx": self.Ndx, "Ndy": self.Ndy, "Ndz": self.Ndz,
                "Nch": self.Nch, "device_id": dev.index if device_id is None else device_id,
            }
            self._cufi_op_params = dict(op_params)
            self._op = CufiOp(op_params, sensitivity_maps, dcomp_mode=self.dcomp_mode)
        elif backend == "torchkb":
            self._op = TorchKbOp(Ndx=self.Ndx, Ndy=self.Ndy, Ndth=self.Ndth, Ndr=self.Ndr,
                                 Ndz=self.Ndz, Nch=self.Nch,
                                 coil_sensitivity_maps=sensitivity_maps,
                                 dcomp_mode=self.dcomp_mode)
        else:
            raise ValueError("backend must be 'cufi' or 'torchkb'.")

        # Optional compile for torchkbnufft
        if self.backend == "torchkb" and self.compile_mode != "off" and hasattr(torch, "compile"):
            dynamic = (self.compile_mode == "dynamic")
            try:
                self._op._fwd = torch.compile(self._op._fwd, dynamic=dynamic)
                self._op._adj = torch.compile(self._op._adj, dynamic=dynamic)
            except Exception:
                # fail open: keep eager if compile fails in this environment
                self.compile_mode = "off"

        # Optional CUFI per-frame plan cache
        self._cufi_factory = None
        if self.backend == "cufi" and self.use_cufi_frame_plan_cache:
            self._cufi_factory = _CufiFramePlanFactory(
                self._cufi_op_params, self.csm, self.dcomp_mode,
                self.Ndz, self.Ndx, self.Ndy, self.Nch, self.Ndth, self.Ndr,
                dtype=self.csm.dtype, device=self.csm.device
            )

    def attach_arena(self, arena) -> None:
        if hasattr(self._op, "attach_arena"):
            self._op.attach_arena(arena)

    def enable_compile(self, mode: Literal["static","dynamic"]="static") -> bool:
        """
        Enable torch.compile for the torchkbnufft backend after construction.
        Returns True if enabled; False if not available or failed.
        """
        if self.backend != "torchkb" or not hasattr(torch, "compile"):
            return False
        dynamic = (mode == "dynamic")
        try:
            self._op._fwd = torch.compile(self._op._fwd, dynamic=dynamic)
            self._op._adj = torch.compile(self._op._adj, dynamic=dynamic)
            self.compile_mode = "dynamic" if dynamic else "static"
            return True
        except Exception:
            return False

    @torch.no_grad()
    def A(self, x: torch.Tensor, *, out: Optional[torch.Tensor] = None,
          frames_per_batch: Optional[int] = None) -> torch.Tensor:
        """
        Forward NUFFT (SENSE):
          x: (Nframes, Ndz, Ndx, Ndy)  ->  (Nframes, Nch, Ndz, Ndth, Ndr)

        `frames_per_batch` (torchkbnufft only) chunks frames to cap peak memory.
        """
        if x.shape != (self.Nframes, self.Ndz, self.Ndx, self.Ndy):
            raise ValueError(f"[FrameNUFFT.A] x shape must be {(self.Nframes,self.Ndz,self.Ndx,self.Ndy)}")

        if out is None:
            out = torch.empty((self.Nframes, self.Nch, self.Ndz, self.Ndth, self.Ndr),
                              dtype=x.dtype, device=x.device)

        # --- Fast batched path for torchkbnufft
        if self.backend == "torchkb":
            if x.device != self.csm.device:
                raise ValueError(f"[FrameNUFFT.A torchkb] inputs on {x.device}, but operator on {self.csm.device}.")
            B = self.Nframes
            chunk = frames_per_batch if frames_per_batch is not None else self.frames_per_batch
            if chunk is None or chunk <= 0:
                chunk = B

            csm_b = self._op.csm.permute(1, 0, 2, 3).unsqueeze(0)  # (1,Ndz,Nch,Nx,Ny)
            adhoc = float(getattr(self._op, "_adhoc_scale", 1.0))

            for start in range(0, B, chunk):
                end = min(start + chunk, B)
                bsz = end - start

                # Coil expansion -> (bsz*Ndz, Nch, Nx, Ny)
                coil = (x[start:end].unsqueeze(2) * csm_b).reshape(
                    bsz * self.Ndz, self.Nch, self.Ndx, self.Ndy
                )
                # Per-batch trajectory -> (bsz*Ndz, 2, Ndth*Ndr)
                kt = self.org_trj[start:end].reshape(bsz, 2, self.Ndth * self.Ndr).repeat_interleave(self.Ndz, dim=0)

                # NUFFT forward
                k_bn = self._op._fwd(coil, kt).view(bsz, self.Ndz, self.Nch, self.Ndth, self.Ndr)
                k_out = k_bn.permute(0, 2, 1, 3, 4).contiguous()  # (bsz,Nch,Ndz,Ndth,Ndr)

                # dcomp weighting if configured
                if self.dcomp_mode != "none":
                    w = self.org_cmp[start:end]
                    w = w.sqrt() if self.dcomp_mode == "sqrt" else w
                    k_out = k_out * w.view(bsz, 1, 1, self.Ndth, self.Ndr)

                # ad-hoc scale
                if adhoc != 1.0:
                    k_out = k_out * adhoc

                out[start:end].copy_(k_out)
            return out

        # --- CUFINUFFT (per-frame loop; optional bound plans)
        for f in range(self.Nframes):
            if self._cufi_factory is not None:
                bound = self._cufi_factory.get(
                    f,
                    ktraj=self.org_trj[f],
                    dcomp=self.org_cmp[f] if self.dcomp_mode != "none" else None,
                )
                bound.F(image=x[f], out=out[f])
            else:
                self._op.F(
                    image=x[f],
                    ktraj=self.org_trj[f],
                    dcomp=self.org_cmp[f] if self.dcomp_mode != "none" else None,
                    out=out[f],
                )
        return out

    @torch.no_grad()
    def AH(self, k: torch.Tensor, *, out: Optional[torch.Tensor] = None,
           frames_per_batch: Optional[int] = None) -> torch.Tensor:
        """
        Adjoint NUFFT (SENSE):
          k: (Nframes, Nch, Ndz, Ndth, Ndr)  ->  (Nframes, Ndz, Ndx, Ndy)

        `frames_per_batch` (torchkbnufft only) chunks frames to cap peak memory.
        """
        if k.shape != (self.Nframes, self.Nch, self.Ndz, self.Ndth, self.Ndr):
            raise ValueError(f"[FrameNUFFT.AH] k shape must be {(self.Nframes,self.Nch,self.Ndz,self.Ndth,self.Ndr)}")

        if out is None:
            out = torch.empty((self.Nframes, self.Ndz, self.Ndx, self.Ndy),
                              dtype=k.dtype, device=k.device)

        # --- Fast batched path for torchkbnufft
        if self.backend == "torchkb":
            if k.device != self.csm.device:
                raise ValueError(f"[FrameNUFFT.AH torchkb] inputs on {k.device}, but operator on {self.csm.device}.")
            B = self.Nframes
            chunk = frames_per_batch if frames_per_batch is not None else self.frames_per_batch
            if chunk is None or chunk <= 0:
                chunk = B

            csm_b = self._op.csm.permute(1, 0, 2, 3)  # (Ndz,Nch,Nx,Ny)
            adhoc = float(getattr(self._op, "_adhoc_scale", 1.0))

            for start in range(0, B, chunk):
                end = min(start + chunk, B)
                bsz = end - start

                # dcomp weighting if configured
                if self.dcomp_mode != "none":
                    w = self.org_cmp[start:end]
                    w = w.sqrt() if self.dcomp_mode == "sqrt" else w
                    k_w = k[start:end] * w.view(bsz, 1, 1, self.Ndth, self.Ndr)
                else:
                    k_w = k[start:end]

                # Arrange for adjoint: (bsz*Ndz, Nch, Ndth*Ndr)
                k_bt = k_w.permute(0, 2, 1, 3, 4).contiguous().view(
                    bsz * self.Ndz, self.Nch, self.Ndth * self.Ndr
                )
                kt = self.org_trj[start:end].reshape(bsz, 2, self.Ndth * self.Ndr).repeat_interleave(self.Ndz, dim=0)

                # Adjoint NUFFT -> (bsz*Ndz, Nch, Nx, Ny)
                imc = self._op._adj(k_bt, kt).view(bsz, self.Ndz, self.Nch, self.Ndx, self.Ndy)

                # SENSE combine over coils: (bsz,Ndz,Nx,Ny)
                im = (imc * csm_b.unsqueeze(0).conj()).sum(dim=2)

                # ad-hoc scale
                if adhoc != 1.0:
                    im = im * adhoc

                out[start:end].copy_(im)
            return out

        # --- CUFINUFFT (per-frame loop; optional bound plans)
        for f in range(self.Nframes):
            if self._cufi_factory is not None:
                bound = self._cufi_factory.get(
                    f,
                    ktraj=self.org_trj[f],
                    dcomp=self.org_cmp[f] if self.dcomp_mode != "none" else None,
                )
                bound.G(kdata=k[f], out=out[f])
            else:
                self._op.G(
                    kdata=k[f],
                    ktraj=self.org_trj[f],
                    dcomp=self.org_cmp[f] if self.dcomp_mode != "none" else None,
                    out=out[f],
                )
        return out

@torch.no_grad()
def organize_k_space_slidingwindow(
    k_space: torch.Tensor,
    sensitivity_maps: torch.Tensor,
    temp_window: int,
    temp_slide: int,
    *,
    backend: Backend = "cufi",
    dcomp_policy: DcompPolicy = "op_sqrt",
    devices: Optional[list[int]] = None,   # kept for signature compatibility
):
    """
    Slice the long radial acquisition into overlapping time-frames and return:
      (organized_k_space, FrameSenseNUFFT)

    Args
    ----
    backend: "cufi" | "torchkb"
    dcomp_policy:
      - "op_sqrt"        : operator applies √dcomp in A and AH (recommended)
      - "none"           : operator applies no dcomp; org_k is raw
      - "preweight_cache": organizer multiplies √dcomp into org_k (uses memory)

    Returns
    -------
    org_k : (Nframes, Nch, Ndz, temp_window, Ndr)  (view unless preweighted)
    op    : FrameSenseNUFFT with A/AH and attributes .scale_emp, .frame_scale
    """
    # ----------------------------- dims
    Nch, Ndz, Ndth_total, Ndr = k_space.shape
    Nframes = (Ndth_total - temp_window) // temp_slide + 1
    _, _, Ndx, Ndy = sensitivity_maps.shape
    dev = k_space.device

    # ----------------------------- cache dirs
    base_dir = os.path.dirname(__file__)
    traj_dir = os.path.join(base_dir, "ktraj")
    dcmp_dir = os.path.join(base_dir, "dcomp")
    os.makedirs(traj_dir, exist_ok=True)
    os.makedirs(dcmp_dir,  exist_ok=True)

    traj_path = os.path.join(traj_dir, f"ktraj_{Ndth_total}_{Ndr}.pt")
    # dcmp_path is defined after selecting backend units (radians vs cycles)

   # ----------------------------- load/build trajectory (on dev)
    if os.path.exists(traj_path):
        full_ktraj = torch.load(traj_path, map_location=dev)
        if full_ktraj.shape != (2, Ndth_total, Ndr):
            full_ktraj = get_ktraj2D_cufi(Ndth_total, Ndr).to(dev)
            torch.save(full_ktraj.cpu(), traj_path)
    else:
        full_ktraj = get_ktraj2D_cufi(Ndth_total, Ndr).to(dev)
        torch.save(full_ktraj.cpu(), traj_path)

    # ----------------------------- select units per backend
    # CUFINUFFT expects radians [-pi, pi); torchkbnufft expects cycles [-0.5, 0.5).
    if backend == "torchkb":
        full_ktraj_backend = full_ktraj / (2.0 * math.pi)
        units_tag = "cycles"
    else:
        full_ktraj_backend = full_ktraj
        units_tag = "rad"

    # density-comp cache name must encode units to avoid cross-backend reuse
    dcmp_path = os.path.join(
        dcmp_dir,
        f"dcomp_{Ndth_total}_{Ndr}_w{temp_window}_s{temp_slide}_{units_tag}.pt"
    )

    # ----------------------------- sliding-window views
    org_k_view   = sliding_window_view_kspace(k_space, temp_window, temp_slide)                   # (Nf,Nch,Ndz,W,Ndr)
    org_trj_view = _sliding_window_view_traj(full_ktraj_backend, temp_window, temp_slide)         # (Nf,2,W,Ndr)

    # ----------------------------- per-frame dcomp (cached with w/s and units)
    if os.path.exists(dcmp_path):
        org_cmp = torch.load(dcmp_path, map_location=dev).to(k_space.dtype)
        if org_cmp.shape != (Nframes, temp_window, Ndr):
            org_cmp = None
    else:
        org_cmp = None

    if org_cmp is None:
        org_cmp = torch.empty((Nframes, temp_window, Ndr),
                              dtype=k_space.dtype, device=dev)
        for f in range(Nframes):
            s = slice(f * temp_slide, f * temp_slide + temp_window)
            org_cmp[f] = get_dcomp_2d(full_ktraj_backend[:, s, :], dtype=k_space.dtype)
        torch.save(org_cmp.cpu(), dcmp_path)

    # ----------------------------- build operator wrapper
    dcomp_mode = {"op_sqrt": "sqrt", "none": "none", "preweight_cache": "none"}[dcomp_policy]
    nufft_op = FrameSenseNUFFT(
        sensitivity_maps, org_trj_view, org_cmp,
        backend=backend, dcomp_mode=dcomp_mode,
        device_id=(dev.index if dev.type == "cuda" else None),
    )

    # ----------------------------- empirical scaling (unit impulse)
    one = torch.zeros((Nframes, Ndz, Ndx, Ndy), dtype=torch.complex64, device=dev)
    one[Nframes // 2, :, Ndx // 2, Ndy // 2] = 1
    # use operator A (forward), sum coil energies, average over z
    nufft_op.scale_emp = nufft_op.A(one).abs().pow(2).sum() / Ndz

    # ----------------------------- frame scaling (uses whatever org_k we return)
    # If we preweight, compute it on the preweighted copy; otherwise on the raw view.
    if dcomp_policy == "preweight_cache":
        org_k = org_k_view * torch.sqrt(org_cmp).unsqueeze(1).unsqueeze(1)
        nufft_op.frame_scale = frame_scaling(org_k)
    else:
        org_k = org_k_view
        nufft_op.frame_scale = frame_scaling(org_k)

    return org_k, nufft_op