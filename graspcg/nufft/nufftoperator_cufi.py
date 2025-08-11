
import torch
import math
import cufinufft

class get_cufi_opSENSE2:
    # NUFFT Operator for [multi-channel k-space <-> single channel image-space]
    def __init__(self, op_params, coil_sensitivity_maps):
        self.Ndth = op_params['Ndth']
        self.Ndr = op_params['Ndr']
        self.Ndx = op_params['Ndx']
        self.Ndy = op_params['Ndy']
        self.Ndz = op_params['Ndz']
        self.Nch = op_params['Nch']
        self.device_id = op_params['device_id']
        self.Gp = self.Gplan()  # CUFI Plan Object
        self.Fp = self.Fplan()  # CUFI Plan Object
        # Normalize coil sensitivity maps
        sos = torch.sqrt(torch.sum(torch.abs(coil_sensitivity_maps)**2, axis=0, keepdims=True))+ 1e-8
        self.coil_sensitivity_maps = coil_sensitivity_maps / sos
        del sos
        # arena will be injected later by a workspace/solver
        self._arena = None
        self.c =  1 / math.sqrt(2 * math.pi)
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Arena wiring
    # ------------------------------------------------------------------
    def attach_arena(self, arena):
        """
        Receive a reference to the workspace arena so NUFFT scratch
        slices are allocated by the common allocator.
        """
        self._arena = arena

    # ------------------------------------------------------------------
    # Local fallback scratch when no arena is attached
    # ------------------------------------------------------------------
    def _local_buf(self, name: str, *, shape, dtype, device):
        buf_dict = getattr(self, name, None)
        if buf_dict is None:
            buf_dict = {}
            setattr(self, name, buf_dict)

        buf = buf_dict.get(device.index)
        if buf is None or buf.shape != shape or buf.dtype != dtype:
            buf = torch.empty(shape, dtype=dtype, device=device)
            buf_dict[device.index] = buf
        return buf
        
        
    def Gplan(self):
        # Specify Type 1 NUFFT Parameters
        plan_G = cufinufft.Plan(
            nufft_type=1,
            n_modes=(self.Ndx, self.Ndy),
            n_trans=self.Ndz * self.Nch,
            dtype="complex64",
            gpu_device_id = self.device_id
        )
        return plan_G

    def Gsetpoints(self, ktraj):
        self.Gp.setpts(ktraj[0].flatten(), ktraj[1].flatten())

    # ---------- adjoint NUFFT  (k‑space ➜ image) -----------------------
    @torch.no_grad()
    def G(self, kdata, ktraj, dcomp, *, out: torch.Tensor = None):
        # density compensation
        kdata = kdata.mul_(dcomp.sqrt().unsqueeze(0).unsqueeze(0))

        # flatten batch: (Nch,Ndz,Ndth,Ndr) -> (Nch*Ndz, Ndth*Ndr)
        k_flat = kdata.flatten(0, 1).flatten(1, 2)

        # plan expects 1‑D arrays for setpts
        kt = ktraj.to(kdata.device, non_blocking=True)
        self.Gp.setpts(kt[0].reshape(-1), kt[1].reshape(-1))

        # ---------------- choose scratch tensor -------------------------------
        elems = self.Nch * self.Ndz * self.Ndx * self.Ndy
        if self._arena is not None:
            scratch = self._arena.request(elems, kdata.dtype).view(
                self.Nch * self.Ndz, self.Ndx, self.Ndy)
        else:
            scratch = self._local_buf(
                "_scratch_G",
                shape=(self.Nch * self.Ndz, self.Ndx, self.Ndy),
                dtype=kdata.dtype,
                device=kdata.device,
            )
        scratch.zero_() 
        # ---------- NUFFT directly into scratch -------------------------------
        self.Gp.execute(k_flat, out=scratch)
        im = scratch.view(self.Nch, self.Ndz, self.Ndx, self.Ndy)
        
        im.mul_(self.c * ((self.Ndth) ** (-1/(8*math.sqrt(3))))* ((self.Ndr) ** (-1/12))* ((self.Ndx*self.Ndy)**(-1/3)))

        # combine coils
        imcc = torch.sum(im * self.coil_sensitivity_maps.conj(), dim=0)
        if out is not None:
            assert out.shape == (self.Ndz, self.Ndx, self.Ndy), \
                "out tensor has wrong shape"
            out.copy_(imcc)
            del kdata, im
            torch.cuda.empty_cache()
            return out
        del kdata, im

        return imcc

    
    def Fplan(self):
        # Specify Type 2 NUFFT Parameters
        plan_F = cufinufft.Plan(
            nufft_type=2,
            n_modes=(self.Ndx, self.Ndy),
            n_trans=self.Ndz * self.Nch,
            dtype="complex64",
            gpu_device_id = self.device_id
        )
        return plan_F
    
    def Fsetpoints(self, ktraj):
        self.Fp.setpts(ktraj[0].flatten(), ktraj[1].flatten())


    @torch.no_grad()
    def F(self, image, ktraj, dcomp, *, out: torch.Tensor = None):
        # expand over coils & apply maps
        im = (image.unsqueeze(0) * self.coil_sensitivity_maps).contiguous()

        # batch flatten
        im_flat = im.flatten(0, 1)

        kt = ktraj.to(im.device, non_blocking=True)
        self.Fp.setpts(kt[0].reshape(-1), kt[1].reshape(-1))

        elems = (self.Nch * self.Ndz * self.Ndth * self.Ndr)
        if self._arena is not None:
            scratch = self._arena.request(elems, im.dtype).view(
                          self.Nch * self.Ndz, self.Ndth * self.Ndr)
        else:
            scratch = self._local_buf("_scratch_F",
                          shape=(self.Nch * self.Ndz, self.Ndth * self.Ndr),
                          dtype=im.dtype, device=im.device)

        self.Fp.execute(im_flat, out=scratch)

        kdata = scratch.view(self.Nch, self.Ndz, self.Ndth, self.Ndr)
        kdata.mul_(dcomp.sqrt().unsqueeze(0).unsqueeze(0))
        # ad-hoc scaling
        c = 1 / (math.sqrt(2 * math.pi))
        kdata.mul_(c * ((self.Ndth) ** (-1/(8*math.sqrt(3)))* ((self.Ndr) ** (-1/12)))* ((self.Ndx*self.Ndy)**(-1/3)))
        if out is not None:
            assert out.shape == (self.Nch, self.Ndz, self.Ndth, self.Ndr), \
                "out tensor has wrong shape"
            out.copy_(kdata)
            del im_flat, im
            torch.cuda.empty_cache()
            return out
        del im_flat, im

        return kdata