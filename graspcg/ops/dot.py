import torch
from .utils import like

@torch.no_grad()
def dot_chunked(a: torch.Tensor,
                b: torch.Tensor,
                *,
                diag: torch.Tensor | None = None,
                chunk: int | None = None,
                arena=None) -> float:
    """Chunked complex dot product with optional diagonal pre‑conditioning.

    If `arena` (UnifiedArena) is supplied **and** `chunk is None`,
    the routine queries `arena.free_elems()` on the anchor device to choose
    the largest scratch it can safely allocate, then requests that buffer
    once and releases it on exit.

    Args:
        a, b : tensors with identical shape/dtype/device.
        diag : if given, computes Σ conj(a)·(b/diag)  (Jacobi pre‑conditioning)
        chunk: explicit chunk size in *elements* (overrides arena logic).
        arena: UnifiedArena instance (optional but recommended).

    Returns:
        scalar Python float with the real part of the dot product.
    """
    assert a.shape == b.shape and a.dtype == b.dtype and a.device == b.device

    # ------------------------------------------------------------------ pick chunk
    if chunk is None:
        if arena is not None:
            avail = arena.free_elems(a.dtype, device=a.device)
            # leave a 5 % safety head‑room and clamp
            chunk = max(int(avail * 0.95), 1 << 16)
        else:
            # aim for ~128 MB scratch if no arena given
            elem = torch.tensor([], dtype=a.dtype,
                                device=a.device).element_size()
            chunk = (128 * 2**20) // elem
            chunk = max(chunk, 1 << 16)

    af, bf = a.flatten(), b.flatten()
    df     = diag.flatten() if diag is not None else None
    numel  = af.numel()

    # ------------------------------------------------------------------ scratch buf
    if arena is not None:
        prod_buf = arena.request(chunk, af.dtype, anchor=af).view(chunk)
        release  = True
    else:
        prod_buf = like(af, chunk)
        release  = False

    tot = 0.0
    for i in range(0, numel, chunk):
        end   = min(i + chunk, numel)
        span  = end - i
        s     = slice(i, end)

        if df is None:
            torch.mul(af[s].conj(), bf[s], out=prod_buf[:span])
        else:
            torch.mul(af[s].conj(), bf[s].div(df[s]), out=prod_buf[:span])

        tot += torch.real(prod_buf[:span].sum())

    if release:
        arena.release(prod_buf)

    del prod_buf
    return float(tot)
