# graspcg/numerics/utils.py
from __future__ import annotations
from typing import Mapping, Iterable, Optional, Union, Callable
import torch

DeviceLike = Union[torch.device, Callable[[], torch.device]]
ValsLike = Union[
    Mapping[torch.device, torch.Tensor],
    Iterable[torch.Tensor],
    torch.Tensor,
    float,
    int,
]

@torch.no_grad()
def reduce0d(
    values: ValsLike,
    *,
    prefer_device: Optional[DeviceLike] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Reduce a collection of 0‑D REAL scalars to a single 0‑D tensor.

    Accepts:
      • dict[device -> 0‑D tensor]
      • iterable of 0‑D tensors
      • a single 0‑D tensor
      • a Python number

    The result is placed on `prefer_device` if provided; otherwise it uses the
    first tensor's device (or CPU if none). If `dtype` is provided, the result
    is cast to that dtype.
    """
    # Normalize prefer_device (handle callable property vs attribute)
    if callable(prefer_device):
        try:
            prefer_device = prefer_device()
        except TypeError:
            # Was already a device-like attribute
            pass

    # Normalize inputs into a list of tensors
    if isinstance(values, torch.Tensor):
        ts = [values]
    elif isinstance(values, Mapping):
        ts = list(values.values())
    elif isinstance(values, Iterable):
        ts = list(values)
    else:
        ts = [torch.as_tensor(values)]

    # Choose target device/dtype
    if ts:
        first = ts[0]
        if isinstance(first, torch.Tensor):
            target_dev = prefer_device or first.device
            target_dtype = dtype or first.dtype
        else:
            target_dev = prefer_device or torch.device("cpu")
            target_dtype = dtype or torch.float32
    else:
        target_dev = prefer_device or torch.device("cpu")
        target_dtype = dtype or torch.float32

    out = torch.zeros((), device=target_dev, dtype=target_dtype)

    # Accumulate (sum) onto target device/dtype
    for t in ts:
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t, device=target_dev, dtype=target_dtype)
        else:
            t = t.to(device=target_dev, dtype=target_dtype, non_blocking=True)
        out.add_(t)

    return out

# Optional legacy alias if older code imports _reduce0d
_reduce0d = reduce0d



def real_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype is torch.complex128 else (torch.float32 if dtype is torch.complex64 else dtype)


@torch.no_grad()
def norm0(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(dot_real0(x, x).clamp_min_(0))



@torch.no_grad()
def dot_real0(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Memory‑aware dot: uses torch.vdot to avoid materializing a*b.
    Works for real & complex, any shape (flattened).
    Returns a 0‑D REAL tensor on a.device.
    """
    # vdot returns complex if inputs are complex; take real part
    return torch.vdot(a.reshape(-1), b.reshape(-1)).real


@torch.no_grad()
def dot_precond0(
    a: torch.Tensor,
    b: torch.Tensor,
    diag: torch.Tensor,
    *,
    arena=None,
    chunk_elems: int | None = None,
    eps: float | None = None,  # optional: stabilize small diag
) -> torch.Tensor:
    """
    Memory-aware ⟨a, b/diag⟩ with broadcasting support.

    Semantics match: (a.conj() * (b / diag)).real.sum()
    Returns a 0-D REAL tensor on a.device.
    """
    # Devices
    dev = a.device
    if not (b.device == dev and diag.device == dev):
        raise AssertionError("Devices must match")

    # Common dtypes
    dtype_c = torch.promote_types(a.dtype, b.dtype)
    if diag.dtype.is_complex:
        dtype_c = torch.promote_types(dtype_c, diag.dtype)
    dtype_r = torch.empty((), dtype=dtype_c).real.dtype

    # Broadcast shape (no allocation)
    out_shape = torch.broadcast_shapes(a.shape, b.shape, diag.shape)
    if 0 in out_shape or a.numel() == 0 or b.numel() == 0 or diag.numel() == 0:
        return torch.zeros((), device=dev, dtype=dtype_r)

    # Make broadcasted *views* (expand creates stride-0 views; still O(1) memory)
    aB = a.to(dtype_c, copy=False).expand(out_shape)
    bB = b.to(dtype_c, copy=False).expand(out_shape)
    dB = diag.to(dtype_c, copy=False).expand(out_shape)

    n = int(torch.tensor(out_shape, device=dev).prod().item())

    # Choose chunk size
    if chunk_elems is None:
        if arena is not None:
            free = int(arena.free_elems(dtype_c, device=dev))
            chunk_elems = max(int(free * 0.90), 1 << 16)
        else:
            elem_sz = torch.empty((), dtype=dtype_c, device=dev).element_size()
            chunk_elems = max((8 * (1 << 20)) // elem_sz, 1 << 16)

    # Scratch
    if arena is not None:
        scratch_base = arena.request(chunk_elems, dtype_c, anchor=aB)
        scratch = scratch_base[:chunk_elems]
        will_release = True
    else:
        scratch_base = None
        scratch = torch.empty(chunk_elems, dtype=dtype_c, device=dev)
        will_release = False

    # Accumulator
    out = torch.zeros((), device=dev, dtype=dtype_r)

    # Fast path: if all three are contiguous AND already same shape, we can slice views
    same_shape = a.shape == b.shape == diag.shape
    all_contig  = a.is_contiguous() and b.is_contiguous() and diag.is_contiguous()
    fast_flat   = same_shape and all_contig

    if fast_flat:
        af = aB.view(-1)
        bf = bB.view(-1)
        df = dB.view(-1)

    i = 0
    while i < n:
        j = min(i + chunk_elems, n)
        span = j - i

        if fast_flat:
            a_chunk = af.narrow(0, i, span)
            b_chunk = bf.narrow(0, i, span)
            d_chunk = df.narrow(0, i, span)
        else:
            # Gather only this slice using flattened indices; works with stride-0 expands
            idx = torch.arange(i, j, device=dev)
            a_chunk = torch.take(aB, idx)
            b_chunk = torch.take(bB, idx)
            d_chunk = torch.take(dB, idx)

        # tmp = b_chunk / d_chunk  (into scratch)
        tmp = scratch[:span]
        tmp.copy_(b_chunk)
        if eps is None:
            tmp.div_(d_chunk)
        else:
            tmp.div_(d_chunk + eps)

        # accumulate Re{ conj(a)·tmp }
        out.add_(torch.vdot(a_chunk, tmp).real)

        i = j

    if will_release:
        arena.release(scratch_base)

    return out