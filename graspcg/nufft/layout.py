# graspcg/nufft/layout.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Sequence
from functools import reduce
import operator as op
import torch

# ------------------------------ axis grammar ----------------------------------

@dataclass(init=False)
class AxisSpec:
    """
    Minimal axis grammar for non-canonical layouts.

    Accepts **either** new names (used in tests):
      image_labels, kspace_labels, image_fft_labels, kspace_fft_labels, coil_label
    **or** legacy names:
      image, kspace, image_fft, kspace_fft, coil
    """
    # stored fields
    image_labels: Tuple[str, ...]
    kspace_labels: Tuple[str, ...]
    image_fft_labels: Tuple[str, ...]
    kspace_fft_labels: Tuple[str, ...]
    coil_label: str

    def __init__(
        self,
        *,
        # new names (preferred)
        image_labels: Tuple[str, ...] | None = None,
        kspace_labels: Tuple[str, ...] | None = None,
        image_fft_labels: Tuple[str, ...] | None = None,
        kspace_fft_labels: Tuple[str, ...] | None = None,
        coil_label: str | None = None,
        # legacy names (back‑compat)
        image: Tuple[str, ...] | None = None,
        kspace: Tuple[str, ...] | None = None,
        image_fft: Tuple[str, ...] | None = None,
        kspace_fft: Tuple[str, ...] | None = None,
        coil: str | None = None,
    ):
        # prefer new names when provided; otherwise fall back to legacy
        image_labels  = image_labels  if image_labels  is not None else image
        kspace_labels = kspace_labels if kspace_labels is not None else kspace
        image_fft_labels  = image_fft_labels  if image_fft_labels  is not None else image_fft
        kspace_fft_labels = kspace_fft_labels if kspace_fft_labels is not None else kspace_fft
        coil_label = coil_label if coil_label is not None else (coil if coil is not None else 'coil')

        if image_labels is None or kspace_labels is None or image_fft_labels is None:
            raise TypeError("AxisSpec requires image(_labels)=..., kspace(_labels)=..., image_fft(_labels)=...")
        if kspace_fft_labels is None:
            kspace_fft_labels = ('K',)

        object.__setattr__(self, "image_labels",  tuple(image_labels))
        object.__setattr__(self, "kspace_labels", tuple(kspace_labels))
        object.__setattr__(self, "image_fft_labels", tuple(image_fft_labels))
        object.__setattr__(self, "kspace_fft_labels", tuple(kspace_fft_labels))
        object.__setattr__(self, "coil_label", str(coil_label))

    # ---- Read‑only aliases used by existing code
    @property
    def image(self) -> Tuple[str, ...]: return self.image_labels
    @property
    def kspace(self) -> Tuple[str, ...]: return self.kspace_labels
    @property
    def image_fft(self) -> Tuple[str, ...]: return self.image_fft_labels
    @property
    def kspace_fft(self) -> Tuple[str, ...]: return self.kspace_fft_labels
    @property
    def coil(self) -> str: return self.coil_label

__all__ = ["AxisSpec",  # explicit export helps avoid stale imports
           "ImageLayoutPlan", "KspaceLayoutPlan",
           "plan_image_layout", "plan_kspace_layout",
           "plan_image_layout_from_sizes", "plan_kspace_layout_from_sizes",
           "image_pack_for_torchkb", "image_pack_for_cufi",
           "kspace_pack_for_torchkb", "kspace_pack_for_cufi",
           "image_unpack_from_torchkb", "kspace_unpack_from_torchkb",
           "image_unpack_from_cufi",  "kspace_unpack_from_cufi",
           "ensure_flat_kspace", "infer_nd_from_axes"]

# --------------------------- internal small helpers ---------------------------

def _prod(seq: Sequence[int]) -> int:
    return int(reduce(op.mul, seq, 1))

def _like_labels_image(axis: AxisSpec) -> Tuple[str, ...]:
    return tuple(lbl for lbl in axis.image
                 if (lbl not in axis.image_fft) and (lbl != 'B') and (lbl != axis.coil))

def _like_labels_kspace(axis: AxisSpec) -> Tuple[str, ...]:
    return tuple(lbl for lbl in axis.kspace
                 if (lbl not in axis.kspace_fft) and (lbl not in ('B', axis.coil)))

def _index_map(labels: Tuple[str, ...]) -> dict[str, int]:
    return {lbl: i for i, lbl in enumerate(labels)}

# ------------------------------- layout plans ---------------------------------

@dataclass(frozen=True)
class ImageLayoutPlan:
    axis: AxisSpec
    B: int
    like_labels: Tuple[str, ...]
    like_sizes: Tuple[int, ...]
    L_other: int
    spatial: Tuple[int, ...]            # ordered as axis.image_fft

@dataclass(frozen=True)
class KspaceLayoutPlan:
    axis: AxisSpec
    B: int
    like_labels: Tuple[str, ...]
    like_sizes: Tuple[int, ...]
    L_other: int
    C: int
    K: int                               # label axis.kspace_fft[0] (‘K’)

# --------------------------- planners from tensors ----------------------------

def plan_image_layout(x_user: torch.Tensor, axis: AxisSpec) -> ImageLayoutPlan:
    im = axis.image
    mp = _index_map(im)
    # sizes by label
    sizes = {lbl: int(x_user.shape[mp[lbl]]) for lbl in im}
    # coil rules
    if axis.coil in im and sizes[axis.coil] != 1:
        raise ValueError("Image is expected coil-combined; axis.coil dim must be size 1 if present.")
    B = sizes.get('B', 1)
    like_labels = _like_labels_image(axis)
    like_sizes = tuple(sizes[lbl] for lbl in like_labels) if like_labels else tuple()
    L_other = _prod(like_sizes) if like_labels else 1
    spatial = tuple(sizes[lbl] for lbl in axis.image_fft)
    return ImageLayoutPlan(axis=axis, B=B, like_labels=like_labels,
                           like_sizes=like_sizes, L_other=L_other, spatial=spatial)

def plan_kspace_layout(y_user: torch.Tensor, axis: AxisSpec) -> KspaceLayoutPlan:
    ks = axis.kspace
    mp = _index_map(ks)
    sizes = {lbl: int(y_user.shape[mp[lbl]]) for lbl in ks}
    B = sizes.get('B', 1)
    C = sizes[axis.coil]
    like_labels = _like_labels_kspace(axis)
    like_sizes = tuple(sizes[lbl] for lbl in like_labels) if like_labels else tuple()
    L_other = _prod(like_sizes) if like_labels else 1
    K = sizes[axis.kspace_fft[0]]
    return KspaceLayoutPlan(axis=axis, B=B, like_labels=like_labels,
                            like_sizes=like_sizes, L_other=L_other, C=C, K=K)

# ------------------------ planners from bare sizes ----------------------------

def plan_image_layout_from_sizes(B: int,
                                 L_sizes: Sequence[int],
                                 spatial: Sequence[int],
                                 axis: AxisSpec) -> ImageLayoutPlan:
    like_labels = _like_labels_image(axis)
    if len(like_labels) == 0:
        like_sizes, L_other = tuple(), 1
    elif len(L_sizes) == len(like_labels):
        like_sizes = tuple(int(v) for v in L_sizes)
        L_other = _prod(like_sizes)
    elif len(L_sizes) == 1:
        # Assign the product to the first like label; set others to 1
        L_other = int(L_sizes[0])
        pad = [1] * len(like_labels); pad[0] = L_other
        like_sizes = tuple(pad)
    else:
        raise ValueError("L_sizes length must be 0, 1, or match number of like labels.")
    return ImageLayoutPlan(axis=axis, B=int(B),
                           like_labels=like_labels, like_sizes=like_sizes,
                           L_other=int(L_other), spatial=tuple(int(s) for s in spatial))

def plan_kspace_layout_from_sizes(B: int,
                                  L_sizes: Sequence[int],
                                  C: int, K: int,
                                  axis: AxisSpec) -> KspaceLayoutPlan:
    like_labels = _like_labels_kspace(axis)
    if len(like_labels) == 0:
        like_sizes, L_other = tuple(), 1
    elif len(L_sizes) == len(like_labels):
        like_sizes = tuple(int(v) for v in L_sizes)
        L_other = _prod(like_sizes)
    elif len(L_sizes) == 1:
        L_other = int(L_sizes[0])
        pad = [1] * len(like_labels); pad[0] = L_other
        like_sizes = tuple(pad)
    else:
        raise ValueError("L_sizes length must be 0, 1, or match number of like labels.")
    return KspaceLayoutPlan(axis=axis, B=int(B),
                            like_labels=like_labels, like_sizes=like_sizes,
                            L_other=int(L_other), C=int(C), K=int(K))

# ------------------------- image pack/unpack (TorchKb) ------------------------

def image_pack_for_torchkb(x_user: torch.Tensor, axis: AxisSpec) -> tuple[torch.Tensor, ImageLayoutPlan]:
    plan = plan_image_layout(x_user, axis)
    im = axis.image; mp = _index_map(im)
    # order: B, like..., (drop coil if present), spatial...
    keep = []
    if 'B' in mp: keep.append(mp['B'])
    keep += [mp[lbl] for lbl in plan.like_labels]
    keep += [mp[lbl] for lbl in axis.image_fft]
    x_perm = x_user.permute(*keep).contiguous()
    B_eff = plan.B * plan.L_other
    x_BL1sp = x_perm.reshape(B_eff, 1, *plan.spatial)
    return x_BL1sp, plan

def image_unpack_from_torchkb(x_BL1sp: torch.Tensor,
                              plan: ImageLayoutPlan,
                              out: Optional[torch.Tensor] = None) -> torch.Tensor:
    # current order is [B*L, 1, spatial] → [B, L, 1, spatial] → expand L across like labels
    tmp = x_BL1sp.reshape(plan.B, plan.L_other, 1, *plan.spatial)
    if len(plan.like_labels) > 0:
        tmp = tmp.reshape(plan.B, *plan.like_sizes, 1, *plan.spatial)
        dims_labels = ('B',) + tuple(plan.like_labels) + (plan.axis.coil,) + tuple(plan.axis.image_fft)
    else:
        tmp = tmp.reshape(plan.B, 1, *plan.spatial)
        dims_labels = ('B', plan.axis.coil) + tuple(plan.axis.image_fft)
    # map to user order
    target_labels = plan.axis.image
    label_to_idx = _index_map(dims_labels)
    order = [label_to_idx[lbl] for lbl in target_labels]
    out_view = tmp.permute(*order)
    if out is not None:
        if tuple(out.shape) != tuple(out_view.shape):
            raise ValueError(f"out has shape {tuple(out.shape)} but expected {tuple(out_view.shape)}")
        out.copy_(out_view)
        return out
    return out_view

# -------------------------- image pack/unpack (CUFI) --------------------------

def image_pack_for_cufi(x_user: torch.Tensor, axis: AxisSpec) -> tuple[torch.Tensor, ImageLayoutPlan]:
    plan = plan_image_layout(x_user, axis)
    im = axis.image; mp = _index_map(im)
    keep = []
    if 'B' in mp: keep.append(mp['B'])
    keep += [mp[lbl] for lbl in plan.like_labels]
    keep += [mp[lbl] for lbl in axis.image_fft]
    x_perm = x_user.permute(*keep).contiguous()  # (B, L1, L2, ..., X, Y[,Z])
    # make (B, L, 1, spatial) where L = product of like sizes
    L_other = plan.L_other
    x_BL1sp = x_perm.reshape(plan.B, L_other, 1, *plan.spatial)
    return x_BL1sp, plan

def image_unpack_from_cufi(x_BL1sp: torch.Tensor,
                           plan: ImageLayoutPlan,
                           out: Optional[torch.Tensor] = None) -> torch.Tensor:
    # [B, L, 1, spatial] → [B, *like, 1, spatial] → permute to user order
    if len(plan.like_labels) > 0:
        tmp = x_BL1sp.reshape(plan.B, *plan.like_sizes, 1, *plan.spatial)
        dims_labels = ('B',) + tuple(plan.like_labels) + (plan.axis.coil,) + tuple(plan.axis.image_fft)
    else:
        tmp = x_BL1sp.reshape(plan.B, 1, *plan.spatial)
        dims_labels = ('B', plan.axis.coil) + tuple(plan.axis.image_fft)
    target_labels = plan.axis.image
    label_to_idx = _index_map(dims_labels)
    order = [label_to_idx[lbl] for lbl in target_labels]
    out_view = tmp.permute(*order)
    if out is not None:
        if tuple(out.shape) != tuple(out_view.shape):
            raise ValueError(f"out has shape {tuple(out.shape)} but expected {tuple(out_view.shape)}")
        out.copy_(out_view)
        return out
    return out_view

# ------------------------ k-space pack/unpack (TorchKb) -----------------------

def kspace_pack_for_torchkb(y_user: torch.Tensor, axis: AxisSpec) -> tuple[torch.Tensor, KspaceLayoutPlan]:
    plan = plan_kspace_layout(y_user, axis)
    ks = axis.kspace; mp = _index_map(ks)
    # order: B, like..., C, K
    keep = []
    if 'B' in mp: keep.append(mp['B'])
    keep += [mp[lbl] for lbl in plan.like_labels]
    keep += [mp[axis.coil], mp[axis.kspace_fft[0]]]
    y_perm = y_user.permute(*keep).contiguous()
    y_BLK = y_perm.reshape(plan.B * plan.L_other, plan.C, plan.K)
    return y_BLK, plan

def kspace_unpack_from_torchkb(y_BLK: torch.Tensor,
                               plan: KspaceLayoutPlan,
                               out: Optional[torch.Tensor] = None) -> torch.Tensor:
    # (B*L, C, K) → (B, L, C, K) → (B, *like, C, K) → permute to user order
    tmp = y_BLK.reshape(plan.B, plan.L_other, plan.C, plan.K)
    if len(plan.like_labels) > 0:
        tmp = tmp.reshape(plan.B, *plan.like_sizes, plan.C, plan.K)
        dims_labels = ('B',) + tuple(plan.like_labels) + (plan.axis.coil, plan.axis.kspace_fft[0])
    else:
        tmp = tmp.reshape(plan.B, plan.C, plan.K)
        dims_labels = ('B', plan.axis.coil, plan.axis.kspace_fft[0])
    target_labels = plan.axis.kspace
    label_to_idx = _index_map(dims_labels)
    order = [label_to_idx[lbl] for lbl in target_labels]
    out_view = tmp.permute(*order)
    if out is not None:
        if tuple(out.shape) != tuple(out_view.shape):
            raise ValueError(f"out has shape {tuple(out.shape)} but expected {tuple(out_view.shape)}")
        out.copy_(out_view)
        return out
    return out_view

# ------------------------- k-space pack/unpack (CUFI) -------------------------

def kspace_pack_for_cufi(y_user: torch.Tensor, axis: AxisSpec) -> tuple[torch.Tensor, KspaceLayoutPlan]:
    plan = plan_kspace_layout(y_user, axis)
    ks = axis.kspace; mp = _index_map(ks)
    keep = []
    if 'B' in mp: keep.append(mp['B'])
    keep += [mp[lbl] for lbl in plan.like_labels]
    keep += [mp[axis.coil], mp[axis.kspace_fft[0]]]
    y_perm = y_user.permute(*keep).contiguous()
    y_BLCK = y_perm.reshape(plan.B, plan.L_other, plan.C, plan.K)
    return y_BLCK, plan

def kspace_unpack_from_cufi(y_BLCK: torch.Tensor,
                            plan: KspaceLayoutPlan,
                            out: Optional[torch.Tensor] = None) -> torch.Tensor:
    # (B, L, C, K) → (B, *like, C, K) → permute to user order
    if len(plan.like_labels) > 0:
        tmp = y_BLCK.reshape(plan.B, *plan.like_sizes, plan.C, plan.K)
        dims_labels = ('B',) + tuple(plan.like_labels) + (plan.axis.coil, plan.axis.kspace_fft[0])
    else:
        tmp = y_BLCK.reshape(plan.B, plan.C, plan.K)
        dims_labels = ('B', plan.axis.coil, plan.axis.kspace_fft[0])
    target_labels = plan.axis.kspace
    label_to_idx = _index_map(dims_labels)
    order = [label_to_idx[lbl] for lbl in target_labels]
    out_view = tmp.permute(*order)
    if out is not None:
        if tuple(out.shape) != tuple(out_view.shape):
            raise ValueError(f"out has shape {tuple(out.shape)} but expected {tuple(out_view.shape)}")
        out.copy_(out_view)
        return out
    return out_view

# --------------------------- legacy helpers retained --------------------------

def ensure_flat_kspace(
    y: torch.Tensor,
    om: torch.Tensor,
    dcf: Optional[torch.Tensor],
    kspace_labels: tuple[str, ...],
    like_labels: tuple[str, ...] = (),
):
    """
    Flatten only sampling axes into one trailing K; keep B, C and all like_labels.
    Returns y_flat:(...,C,K), om_flat:(B,K,nd), dcf_flat:(B,K)|None, new_labels:(..., 'C','K').
    """
    keep_idx = [i for i, lbl in enumerate(kspace_labels) if lbl in ('B', 'C') or (lbl in like_labels)]
    flat_idx = [i for i, _ in enumerate(kspace_labels) if i not in keep_idx]
    perm = keep_idx + flat_idx
    yp = y.permute(*perm).contiguous()
    K = 1
    for ax in flat_idx:
        K *= y.shape[ax]
    y_flat = yp.reshape(*yp.shape[:len(keep_idx)], K)

    # trajectory to (B,K,nd)
    if om.ndim == 3 and om.shape[1] in (2, 3) and om.shape[2] != om.shape[1]:
        om = om.transpose(1, 2).contiguous()
    elif om.ndim >= 4:  # e.g., (B, P, Sp, Ro, nd) -> (B, P, K, nd)
        *lead, nd = om.shape
        B = lead[0]
        K = int(torch.tensor(lead[1:]).prod())
        om = om.reshape(B, K, nd).contiguous()

    if dcf is not None:
        if dcf.ndim == 1:
            dcf = dcf.view(1, -1)
        elif dcf.ndim >= 3:
            B = dcf.shape[0]
            dcf = dcf.reshape(B, int(torch.tensor(dcf.shape[1:]).prod())).contiguous()

    new_labels = tuple([kspace_labels[i] for i in keep_idx] + ['K'])
    return y_flat, om, dcf, new_labels


def infer_nd_from_axes(axis: AxisSpec) -> int:
    """3D only if a third image FFT axis is present. Otherwise 2D multi-slice."""
    return 3 if len(axis.image_fft) == 3 else 2
