# graspcg/regularization/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple, Union
import torch

from ..core.roles import Roles  # (unlike, like, nufft) axis counts

AxesSpec = Union[str, Sequence[int], Tuple[int, ...]]

@dataclass(frozen=True)
class RegParams:
    """
    Minimal parameters common to all regularizers.

    • weight: λ
    • eps   : smoothing (TV/Huber-style)
    • axes  : which axes to operate on; accepts "temporal"/"unlike", "like", "spatial"/"nufft"/"image",
              or explicit axis indices (deduplicated; negative allowed).
    • isotropic: TV flavor (ignored by regs that don't use it).
    """
    weight: float = 0.0
    eps: float = 0.0
    axes: AxesSpec = "spatial"
    isotropic: bool = True
@dataclass(frozen=True)
class RegContext:
    """
    Per-shard call context. All tensors live on the shard device; shapes are stable.
    """
    # Core shard tensors
    x: torch.Tensor                 # image shard (complex)
    g: torch.Tensor                 # gradient shard to accumulate into (complex)
    diag: Optional[torch.Tensor]    # preconditioner diagonal (real) or None

    # Axis semantics for 'x' (tensors ordered as (unlike, like, nufft))
    roles_image: Roles

    # Dtypes/devices
    device: torch.device
    dtype_c: torch.dtype
    dtype_r: torch.dtype
    # Helpers (injected by the manager)
    axes_resolver: Optional[callable] = None      # (AxesSpec) -> Tuple[int,...]
    arena: Optional[Any] = None                   # DeviceArena or scratch provider
    write_interior_slice: Optional[Tuple[slice, ...]] = None

    # Workspace/shard/halo (used by mapped regularizers)
    ws: Optional[object] = None
    shard_index: Optional[int] = None
    halo_map: Optional[dict] = None


class Regularizer(Protocol):
    """
    Minimal interface a regularizer must implement.
    Kernels should be compile-friendly: no .item()/float() in the hot path.
    """
    name: str
    params: RegParams

    def energy_grad(self, ctx: RegContext) -> torch.Tensor:
        """
        Accumulate ∂E/∂x into ctx.g **in-place** and return a 0‑D REAL tensor (shard energy).
        """
        ...

    def energy_grad_fixed_axes(self, ctx: RegContext, axes: Tuple[int, ...]) -> torch.Tensor:
        """
        Same as energy_grad, but with axes pre-resolved and closed over by the caller (manager).
        Preferred for torch.compile & caching.
        """
        ...

    # Optional hooks
    def add_diag(self, ctx: RegContext) -> None: ...
    def continuation_update(self, stats: Mapping[str, Any]) -> bool: ...
    def scaling_policy(self, ctx: RegContext): ...
    def prox_inplace(self, ctx: RegContext, step: float) -> None: ...
    def majorizer_diag(self, ctx: RegContext) -> Optional[torch.Tensor]: ...
    def halo(self, roles: Roles): ...
