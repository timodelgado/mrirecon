# graspcg/nufft/organize_k_space_nomotion.py
# -----------------------------------------------------------
import os, math, torch, cupy as cp
from .stateframe_nufft import StateFrameNUFFT3MD
from .utils import get_ktraj2D_cufi, get_dcomp_2d, _safe_smooth_1d, frame_scaling
# -----------------------------------------------------------

@torch.no_grad()
def organize_k_space_slidingwindow(k_space: torch.Tensor,
                              sensitivity_maps: torch.Tensor,
                              temp_window: int,
                              temp_slide : int,
                              *,
                              devices: list[int] | None = None):
    """
    Slice the long radial acquisition into overlapping time‑frames,
    apply √dcomp pre‑conditioning and return (organized_k_space, NUFFT_op).

    Cached tensors
    --------------
    If ``graspcg/nufft/ktraj/ktraj_<Ndth>_<Ndr>.pt`` or
    ``graspcg/nufft/dcomp/dcomp_<Ndth>_<Ndr>.pt`` exist and have the expected
    shapes they are used directly; otherwise they are generated and saved
    for next run.
    """
    # ------------------------------------------------------------------ dims
    Nch, Ndz, Ndth_total, Ndr = k_space.shape
    Nframes = (Ndth_total - temp_window) // temp_slide + 1
    _, _, Ndx, Ndy = sensitivity_maps.shape
    dev   = k_space.device

    # ------------------------------------------------------------------ cache folders
    base_dir  = os.path.dirname(__file__)
    traj_dir  = os.path.join(base_dir, "ktraj")
    dcmp_dir  = os.path.join(base_dir, "dcomp")
    os.makedirs(traj_dir, exist_ok=True)
    os.makedirs(dcmp_dir,  exist_ok=True)

    traj_path = os.path.join(traj_dir,  f"ktraj_{Ndth_total}_{Ndr}.pt")
    dcmp_path = os.path.join(dcmp_dir,  f"dcomp_{Ndth_total}_{Ndr}.pt")

    # ------------------------------------------------------------------ load / build trajectory
    if os.path.exists(traj_path):
        full_ktraj = torch.load(traj_path, map_location=dev)
        if full_ktraj.shape != (2, Ndth_total, Ndr):
            print("[organize] ktraj cache shape mismatch – regenerating.")
            full_ktraj = get_ktraj2D_cufi(Ndth_total, Ndr).to(dev)
            torch.save(full_ktraj.cpu(), traj_path)
    else:
        full_ktraj = get_ktraj2D_cufi(Ndth_total, Ndr).to(dev)
        torch.save(full_ktraj.cpu(), traj_path)

    # ------------------------------------------------------------------ load / build density comp
    if os.path.exists(dcmp_path):
        full_dcomp = torch.load(dcmp_path, map_location=dev).to(k_space.dtype)
        if full_dcomp.shape != (Ndth_total, Ndr):
            print("[organize] dcomp cache shape mismatch – recomputing.")
            full_dcomp = None
    else:
        full_dcomp = None

    if full_dcomp is None:                       # compute once, save later
        full_dcomp = torch.empty(Ndth_total, Ndr,
                                 dtype=k_space.dtype, device=dev)
        for f in range(Nframes):
            s = slice(f * temp_slide, f * temp_slide + temp_window)
            full_dcomp[s] = get_dcomp_2d(full_ktraj[:, s, :],
                                          dtype=k_space.dtype)
        torch.save(full_dcomp.cpu(), dcmp_path)

    # ------------------------------------------------------------------ output tensors
    org_k   = torch.zeros((Nframes, Nch, Ndz, temp_window, Ndr),
                          dtype=k_space.dtype, device=dev)
    org_trj = torch.zeros((Nframes, 2,  temp_window, Ndr),
                          dtype=torch.float32, device=dev)
    org_cmp = torch.zeros((Nframes,      temp_window, Ndr),
                          dtype=k_space.dtype, device=dev)

    # ------------------------------------------------------------------ slice per‑frame
    for f in range(Nframes):
        s = slice(f * temp_slide, f * temp_slide + temp_window)
        org_k[f]   = k_space[:, :, s, :]
        org_trj[f] = full_ktraj[:, s, :]
        org_cmp[f] = full_dcomp[s]

    # ------------------------------------------------------------------ pre-conditioning
    org_k.div_(org_k.abs().max())


    # ------------------------------------------------------------------ NUFFT operator
    nufft_op = StateFrameNUFFT3MD(sensitivity_maps, org_trj, org_cmp,
                                  devices=devices or [dev.index])

    # empirical scaling & frame scaling
    one = torch.zeros((Nframes, Ndz, Ndx, Ndy), dtype=torch.complex64,
                      device=dev)
    one[Nframes//2, :, Ndx//2, Ndy//2] = 1
    nufft_op.scale_emp  = nufft_op.A(one).abs().pow(2).sum() / Ndz
    nufft_op.frame_scale = frame_scaling(org_k)
    org_k.mul_(torch.sqrt(org_cmp).unsqueeze(1).unsqueeze(1))
    cp._default_memory_pool.free_all_blocks()
    torch.cuda.empty_cache()
    return org_k, nufft_op