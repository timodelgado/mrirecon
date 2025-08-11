# graspcg/workspace/layout.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class LayoutSpec:
    """
    Describe an image layout and how operators act on it.

    image_shape : full shape of x (e.g., (T, E, Z, X, Y))
    nufft_axes  : axes (indices into image_shape) the NUFFT acts on (e.g., (-2, -1) for X,Y)
    traj_axes   : axes that parameterize trajectory grouping but are not transformed
                  by the NUFFT (e.g., Z slices for 2D multi‑slice)
    batch_axes  : axes that are independent batch dimensions (e.g., time, echo)
    names       : optional names for pretty printing (len == ndim)
    """
    image_shape: Tuple[int, ...]
    nufft_axes : Tuple[int, ...]
    traj_axes  : Tuple[int, ...] = ()
    batch_axes : Tuple[int, ...] = ()
    names      : Tuple[str, ...] | None = None

    def validate(self):
        nd = len(self.image_shape)
        all_axes = tuple(sorted(set(self.nufft_axes + self.traj_axes + self.batch_axes)))
        assert all(0 <= a < nd for a in all_axes), "axis out of range"
        assert len(all_axes) == nd, "axes must be a partition of image axes"
        assert len(set(self.nufft_axes)) == len(self.nufft_axes)
        assert len(set(self.traj_axes))  == len(self.traj_axes)
        assert len(set(self.batch_axes)) == len(self.batch_axes)

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        return tuple(self.image_shape[a] for a in self.batch_axes)

    @property
    def inner_shape(self) -> Tuple[int, ...]:
        # compute‑order inner dims (traj first, then nufft)
        return tuple(self.image_shape[a] for a in self.traj_axes + self.nufft_axes)

    @property
    def compute_perm(self) -> Tuple[int, ...]:
        """
        Permutation that brings (batch_axes, traj_axes, nufft_axes) to the front in this order.
        Useful to allocate x as (B, *inner_shape) contiguous for operators.
        """
        return self.batch_axes + self.traj_axes + self.nufft_axes
