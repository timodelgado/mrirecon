# graspcg/nufft/adapter.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any, Protocol

import torch
from ..core.roles import Roles  # single source of truth for roles :contentReference[oaicite:4]{index=4}


class _Backend(Protocol):
    """
    Concrete backend bound to a *single device*.
    Must implement device-local A / AH using OUT buffers (no allocations).
    The adapter handles cross-device by copying inputs as needed.
    """
    def A (self, x: torch.Tensor, out: torch.Tensor) -> None: ...
    def AH(self, y: torch.Tensor, out: torch.Tensor) -> None: ...


@dataclass(frozen=True)
class NUFFTMeta:
    roles_image: Roles
    roles_kspace: Roles
    # If known at construction; else adapter will probe with AH(y[:1]).
    image_ndim: Optional[int] = None  # number of non-batch image axes (e.g., 2 for 2D)


class NUFFT:
    """
    Unified NUFFT adapter used by Workspace/Objective.

    Contract (mirrors usage in Objective & Workspace):
      • A(x,  out=Y) : write Ax into Y (shape==y-slice), *on out.device*
      • AH(y, out=X) : write A^H y into X (shape==x-slice), *on out.device*
      • roles_image / roles_kspace describe (unlike, like, nufft) axis counts
      • image_shape(y) -> Tuple[int,...] for x given a y sample (probe if needed)

    Cross-device guarantee:
      If input.device != out.device, the adapter copies the input to out.device
      and executes the backend there (backend is per-device).
      (Objective uses this in its global scratch path.) :contentReference[oaicite:5]{index=5}
    """

    def __init__(self,
                 make_backend: Callable[[torch.device], _Backend],
                 meta: NUFFTMeta,
                 *,
                 scale_emp: Optional[float] = None):
        self._make_backend = make_backend
        self._meta = meta
        self._scale_emp = float(scale_emp) if scale_emp is not None else None
        self._per_dev: Dict[torch.device, _Backend] = {}

    # ---------- public attributes expected by other modules ----------
    @property
    def roles_image(self) -> Roles:
        return self._meta.roles_image

    @property
    def roles_kspace(self) -> Roles:
        return self._meta.roles_kspace

    # Optional, used by preconditioner builders in some code paths
    @property
    def scale_emp(self) -> Optional[float]:
        return self._scale_emp

    # ---------- main ops ----------
    @torch.no_grad()
    def A(self, x: torch.Tensor, *, out: torch.Tensor) -> None:
        """Compute y = A x into out (device == out.device)."""
        b = self._impl(out.device)
        xin = x if x.device == out.device else x.to(out.device, non_blocking=True)
        b.A(xin, out=out)

    @torch.no_grad()
    def AH(self, y: torch.Tensor, *, out: torch.Tensor) -> None:
        """Compute x = A^H y into out (device == out.device)."""
        b = self._impl(out.device)
        yin = y if y.device == out.device else y.to(out.device, non_blocking=True)
        b.AH(yin, out=out)

    # ---------- shape helpers ----------
    @torch.no_grad()
    def image_shape(self, y_like: torch.Tensor) -> Tuple[int, ...]:
        """
        Best-effort image shape inference from a k-space example.
        If image_ndim is known, keep batch as y_like.shape[0] and reuse that many inner dims.
        Otherwise, probe with AH on a 1-frame tensor.
        """
        B = int(y_like.shape[0])
        if self._meta.image_ndim is not None:
            # Assume square inner dims if not known; caller can override elsewhere
            # This path is mainly for backends that already know their im_size.
            # It is safe to fall back to the probe below.
            pass  # let probe happen for correctness

        # Probe with AH on y[:1] to avoid allocations elsewhere, as Workspace may do. 
        y0 = y_like[:1]
        x0 = torch.empty(1, *([1] * (y_like.ndim - 1)),
                         dtype=y_like.dtype, device=y_like.device)
        # allocate a temp output on y.device to receive AH(y0)
        tmp = torch.empty(1, *([1] * (y_like.ndim - 1)),
                          dtype=y_like.dtype, device=y_like.device)
        # We need the true shape; make a minimal dry run
        # Allocate a correctly shaped out by calling AH into a dummy and reading its shape.
        # Backends are expected to write exactly one frame back.
        out0 = None
        # Build an output by actually running AH; we need a buffer, not its shape beforehand.
        # Use a scratch attempt with zeros; then read its resulting shape.
        z = torch.zeros_like(y0)
        # optimistic guess: image and kspace frames match batch dimension
        # create a 1-frame output on y.device; backend must resize it correctly
        x_guess = torch.zeros((1, *y_like.shape[1:]), dtype=y_like.dtype, device=y_like.device)
        try:
            self.AH(z, out=x_guess)
            shape = (B, *x_guess.shape[1:])
            return tuple(int(s) for s in shape)
        except Exception:
            # Fallback: ask backend via a 1-frame “AH then measure”
            x_probe = torch.empty_like(x_guess)
            self.AH(z, out=x_probe)
            shape = (B, *x_probe.shape[1:])
            return tuple(int(s) for s in shape)

    # ---------- internal ----------
    def _impl(self, dev: torch.device) -> _Backend:
        if dev not in self._per_dev:
            self._per_dev[dev] = self._make_backend(dev)
        return self._per_dev[dev]


# ------------------------------
# Simple reference backend (FFT) for tests and CPU fallback
# ------------------------------
class _FFTBackend:
    """Uniform FFT backend (orthonormal), useful for tests; x/y share the same inner shape."""
    def __init__(self, device: torch.device, norm: str = "ortho"):
        self.device = device
        self.norm = norm

    @torch.no_grad()
    def A(self, x: torch.Tensor, *, out: torch.Tensor) -> None:
        # y = FFT(x)
        out.copy_(torch.fft.fftn(x, dim=tuple(range(1, x.ndim)), norm=self.norm))

    @torch.no_grad()
    def AH(self, y: torch.Tensor, *, out: torch.Tensor) -> None:
        # x = IFFT(y)
        out.copy_(torch.fft.ifftn(y, dim=tuple(range(1, y.ndim)), norm=self.norm))


def make_fft_adapter(roles_image: Roles, roles_kspace: Roles) -> NUFFT:
    def factory(dev: torch.device) -> _FFTBackend:
        return _FFTBackend(dev)
    return NUFFT(factory, NUFFTMeta(roles_image=roles_image, roles_kspace=roles_kspace))
