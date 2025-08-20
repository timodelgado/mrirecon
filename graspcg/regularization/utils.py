# regularization/utils.py
import torch
from typing import Sequence, Tuple

@torch.no_grad()
def add_degreewise_diag_(
    D: torch.Tensor,                     # interior view of ws.diag (REAL), shape (B_loc, C, H, W)
    axes: Tuple[int, ...],               # axes to operate on, in D's coordinates
    axis_weights: Sequence[float],       # e.g. spacing weights (1/dx, 1/dy, 1/dz, 1/dt)
    base_scale: torch.Tensor             # broadcastable factor (e.g. (1/s_t)^2 with shape (B_loc,1,1,1))
) -> None:
    """
    Adds, for each axis a in `axes`, the degree pattern: +w^2 everywhere and an *extra* +w^2 on the interior
    along that axis. This produces the [1,2,...,2,1] profile without allocating big temporaries.
    """
    assert D.is_floating_point()
    dev, dt = D.device, D.dtype
    # Broadcastable scalar/tensor factor
    S = base_scale.to(device=dev, dtype=dt)

    for a, w in zip(axes, axis_weights):
        w2 = torch.as_tensor(float(w) ** 2, device=dev, dtype=dt)
        c = w2  # The global lambda factor can be multiplied outside.
        # +1 everywhere along axis 'a'
        D.add_(S * c)

        # +1 extra on interior along axis 'a'
        n = D.shape[a]
        if n > 2:
            sl = [slice(None)] * D.ndim
            sl[a] = slice(1, n - 1)
            # For time axis (a == 0), S may be (B_loc,1,1,1) → slice it too
            if S.ndim == D.ndim and S.shape[a] == n:
                S_int = S[(slice(None),)*a + (slice(1, n-1),)]
                D[tuple(sl)].add_(S_int * c)
            else:
                # S broadcasts along axis a
                D[tuple(sl)].add_(S * c)