# /data2/timod/mrirecon/graspcg/nufft/nufftoperator_cufi.py
import math
import torch
import cufinufft
from typing import Optional, Literal

_DcompMode = Literal["none", "sqrt", "full"]

def _assert_same_device(*tensors: torch.Tensor):
    devs = {t.device for t in tensors if t is not None}
    if len(devs) > 1:
        raise ValueError(f"[CUFI] All tensors must be on the same device, got {devs}.")

def _as_float32(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == torch.float32 else x.float()

class get_cufi_opSENSE2:
    """
    CUFINUFFT SENSE operator for 2D NUFFT applied per-z slice with Nc coils.

    Shapes:
      image    : (Ndz, Ndx, Ndy)          complex64
      ktraj    : (2, Ndth, Ndr)           float32
      dcomp    : (Ndth, Ndr)              float32 (or real-compatible)
      kdata    : (Nch, Ndz, Ndth, Ndr)    complex64

    Conventions:
      • Balanced weighting by default: dcomp_mode="sqrt" multiplies √dcomp
        in both A (F) and AH (G).
      • Inputs are treated as read-only; no in-place mutation of caller buffers.
      • No `.to(device)` – the caller controls device placement.
      • Keeps the ad-hoc scale factor you previously used.
    """
    def __init__(self, op_params, coil_sensitivity_maps: torch.Tensor,
                 dcomp_mode: _DcompMode = "sqrt"):
        # --- Dimensions/params
        self.Ndth = int(op_params['Ndth'])
        self.Ndr  = int(op_params['Ndr'])
        self.Ndx  = int(op_params['Ndx'])
        self.Ndy  = int(op_params['Ndy'])
        self.Ndz  = int(op_params['Ndz'])
        self.Nch  = int(op_params['Nch'])
        self.device_id = int(op_params['device_id'])
        if dcomp_mode not in ("none", "sqrt", "full"):
            raise ValueError("dcomp_mode must be one of {'none','sqrt','full'}")
        self.dcomp_mode: _DcompMode = dcomp_mode

        # --- Coil maps (normalize; cache conj)
        sos = torch.sqrt(torch.sum(torch.abs(coil_sensitivity_maps)**2,
                                   dim=0, keepdim=True)) + 1e-8
        self.coil_sensitivity_maps = coil_sensitivity_maps / sos
        self.coil_sensitivity_maps_conj = self.coil_sensitivity_maps.conj()
        del sos

        # --- CUFINUFFT plans (type-1 = adjoint G, type-2 = forward F)
        self.Gp = self._make_plan_G()
        self.Fp = self._make_plan_F()

        # --- Optional arena (allocator) will be injected by a workspace
        self._arena = None

        # --- Ad-hoc legacy scale (kept exactly)
        self.c = 1.0 / math.sqrt(2.0 * math.pi)
        self._adhoc_scale = (
            self.c
            * (self.Ndth ** (-1.0 / (8.0 * math.sqrt(3.0))))
            * (self.Ndr  ** (-1.0 / 12.0))
            * ((self.Ndx * self.Ndy) ** (-1.0 / 3.0))
        )

    # ------------------------ Arena wiring ------------------------
    def attach_arena(self, arena) -> None:
        self._arena = arena

    def _local_buf(self, name: str, *, shape, dtype, device) -> torch.Tensor:
        """
        Return a cached scratch tensor keyed by (device, dtype, shape).
        Uses a dedicated attribute '<name>__cache' to avoid clobbering methods.
        """
        cache_attr = f"{name}__cache"
        buf_dict = getattr(self, cache_attr, None)
        if buf_dict is None or not isinstance(buf_dict, dict):
            buf_dict = {}
            setattr(self, cache_attr, buf_dict)

        key = (device.index if device.type == "cuda" else -1, dtype, tuple(shape))
        buf = buf_dict.get(key)
        if (
            buf is None
            or buf.dtype != dtype
            or tuple(buf.shape) != tuple(shape)
            or buf.device != device
        ):
            buf = torch.empty(shape, dtype=dtype, device=device)
            buf_dict[key] = buf
        return buf

    # ------------------------ Plans -------------------------------
    def _make_plan_G(self):
        return cufinufft.Plan(
            nufft_type=1,
            n_modes=(self.Ndx, self.Ndy),
            n_trans=self.Ndz * self.Nch,
            dtype="complex64",
            gpu_device_id=self.device_id,
        )

    def _make_plan_F(self):
        return cufinufft.Plan(
            nufft_type=2,
            n_modes=(self.Ndx, self.Ndy),
            n_trans=self.Ndz * self.Nch,
            dtype="complex64",
            gpu_device_id=self.device_id,
        )

    # ------------------------ Public API --------------------------
    # Forward: image -> k-space
    @torch.no_grad()
    def A(self, image: torch.Tensor, ktraj: torch.Tensor, dcomp: Optional[torch.Tensor],
          *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.F(image, ktraj, dcomp, out=out)

    # Adjoint: k-space -> image
    @torch.no_grad()
    def AH(self, kdata: torch.Tensor, ktraj: torch.Tensor, dcomp: Optional[torch.Tensor],
           *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.G(kdata, ktraj, dcomp, out=out)

    # ---------- adjoint NUFFT  (k‑space ➜ image) ------------------
    @torch.no_grad()
    def G(self, kdata: torch.Tensor, ktraj: torch.Tensor, dcomp: Optional[torch.Tensor],
          *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        AH: (apply density weighting if configured) -> NUFFT^H -> SENSE combine.
        kdata: (Nch, Ndz, Ndth, Ndr)  on the same device as ktraj / dcomp.
        Returns (Ndz, Ndx, Ndy) complex64.
        """
        # --- basic checks
        if kdata.ndim != 4 or kdata.shape != (self.Nch, self.Ndz, self.Ndth, self.Ndr):
            raise ValueError(f"[CUFI.G] kdata shape must be (Nch,Ndz,Ndth,Ndr) = "
                             f"({self.Nch},{self.Ndz},{self.Ndth},{self.Ndr}), got {tuple(kdata.shape)}")
        if ktraj.shape != (2, self.Ndth, self.Ndr):
            raise ValueError(f"[CUFI.G] ktraj shape must be (2,Ndth,Ndr) = (2,{self.Ndth},{self.Ndr}), "
                             f"got {tuple(ktraj.shape)}")
        _assert_same_device(kdata, ktraj, *( () if dcomp is None else (dcomp,) ))
        kt = _as_float32(ktraj).reshape(2, -1)

        # --- flatten batches
        src = kdata.flatten(0, 1).flatten(1, 2)  # (Nch*Ndz, Ndth*Ndr)
        # --- apply density weighting into a scratch buffer (inputs read-only)
        k_flat = self._scratch_kflat(device=kdata.device, dtype=kdata.dtype)
        if dcomp is None or self.dcomp_mode == "none":
            k_flat.copy_(src)
        else:
            w = dcomp
            if self.dcomp_mode == "sqrt":
                w = torch.sqrt(w)
            torch.mul(src, w.reshape(1, -1), out=k_flat)  # (Nch*Ndz, Ndth*Ndr)

        # --- scratch for output image stack
        im_stack = self._scratch_im(device=kdata.device, dtype=kdata.dtype)
        im_stack.zero_()

        # --- set points & execute
        self.Gp.setpts(kt[0], kt[1])
        self.Gp.execute(k_flat, out=im_stack)  # (Nch*Ndz, Ndx, Ndy)

        # --- reshape, scale, SENSE combine
        im = im_stack.view(self.Nch, self.Ndz, self.Ndx, self.Ndy)
        # keep the ad-hoc scale
        im = im * self._adhoc_scale
        imcc = torch.sum(im * self.coil_sensitivity_maps_conj, dim=0)  # (Ndz,Ndx,Ndy)

        if out is not None:
            if out.shape != (self.Ndz, self.Ndx, self.Ndy) or out.device != kdata.device or out.dtype != imcc.dtype:
                raise ValueError("[CUFI.G] `out` has wrong shape/dtype/device.")
            out.copy_(imcc)
            return out
        return imcc

    # ---------- forward NUFFT  (image ➜ k‑space) ------------------
    @torch.no_grad()
    def F(self, image: torch.Tensor, ktraj: torch.Tensor, dcomp: Optional[torch.Tensor],
          *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        A: image -> coil expand -> NUFFT -> (apply density weighting if configured).
        image: (Ndz, Ndx, Ndy), ktraj: (2, Ndth, Ndr), dcomp: (Ndth, Ndr) or None
        Returns (Nch, Ndz, Ndth, Ndr) complex64.
        """
        if image.shape != (self.Ndz, self.Ndx, self.Ndy):
            raise ValueError(f"[CUFI.F] image shape must be (Ndz,Ndx,Ndy)=({self.Ndz},{self.Ndx},{self.Ndy}), "
                             f"got {tuple(image.shape)}")
        if ktraj.shape != (2, self.Ndth, self.Ndr):
            raise ValueError(f"[CUFI.F] ktraj shape must be (2,Ndth,Ndr)=(2,{self.Ndth},{self.Ndr}), "
                             f"got {tuple(ktraj.shape)}")
        _assert_same_device(image, ktraj, *( () if dcomp is None else (dcomp,) ))
        kt = _as_float32(ktraj).reshape(2, -1)

        # --- coil expansion (does not modify caller tensor)
        # im_coils: (Nch, Ndz, Ndx, Ndy)
        im_coils = (image.unsqueeze(0) * self.coil_sensitivity_maps).contiguous()
        im_flat = im_coils.flatten(0, 1)  # (Nch*Ndz, Ndx, Ndy)

        # --- scratch for k-space
        k_flat = self._scratch_kflat(device=image.device, dtype=image.dtype)

        # --- set points & execute
        self.Fp.setpts(kt[0], kt[1])
        self.Fp.execute(im_flat, out=k_flat)  # (Nch*Ndz, Ndth*Ndr)

        # --- reshape, apply density comp & ad-hoc scale
        k_unw = k_flat.view(self.Nch, self.Ndz, self.Ndth, self.Ndr)
        if out is None:
            out = self._scratch_kout(device=image.device, dtype=image.dtype)

        if dcomp is not None and self.dcomp_mode != "none":
            w = dcomp
            if self.dcomp_mode == "sqrt":
                w = torch.sqrt(w)
            torch.mul(k_unw, w.view(1, 1, self.Ndth, self.Ndr), out=out)
        else:
            out.copy_(k_unw)

        # keep the ad-hoc scale on forward as you had
        out.mul_(self._adhoc_scale)
        return out

    # -------------------- scratch helpers -------------------------
    def _scratch_im(self, device, dtype) -> torch.Tensor:
        if self._arena is not None:
            elems = self.Nch * self.Ndz * self.Ndx * self.Ndy
            return self._arena.request(elems, dtype).view(self.Nch * self.Ndz, self.Ndx, self.Ndy)
        return self._local_buf("_scratch_im", shape=(self.Nch * self.Ndz, self.Ndx, self.Ndy),
                               dtype=dtype, device=device)

    def _scratch_kflat(self, device, dtype) -> torch.Tensor:
        if self._arena is not None:
            elems = self.Nch * self.Ndz * self.Ndth * self.Ndr
            return self._arena.request(elems, dtype).view(self.Nch * self.Ndz, self.Ndth * self.Ndr)
        return self._local_buf("_scratch_kflat",
                               shape=(self.Nch * self.Ndz, self.Ndth * self.Ndr),
                               dtype=dtype, device=device)

    def _scratch_kout(self, device, dtype) -> torch.Tensor:
        return self._local_buf("_scratch_kout",
                               shape=(self.Nch, self.Ndz, self.Ndth, self.Ndr),
                               dtype=dtype, device=device)