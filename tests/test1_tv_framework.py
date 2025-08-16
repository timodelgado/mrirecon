import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import pytest
import torch

from graspcg.regularization.base import Roles, RegParams
from graspcg.regularization.tv_nd import TVND, TVParams
from graspcg.regularization.manager import RegManager


# -----------------------
# Helpers / Fake workspace
# -----------------------

class _FakeShard:
    def __init__(self, device: torch.device):
        self.device = device


class _FakeArena:
    def stream_for(self, device: torch.device):
        # No explicit streams on CPU tests; CUDA tests can add real streams later.
        return None


@dataclass
class _FakePlan:
    roles_image: Roles
    num_shards: int


class _WS:
    """
    Minimal workspace stub to exercise RegManager:
    - Tensors follow image ordering: (unlike, like, nufft).
    - Sharding occurs along the first (unlike) axis, dim=0.
    """
    def __init__(
        self,
        x_full: torch.Tensor,
        roles_image: Roles,
        num_shards: int = 1,
        with_diag: bool = False,
        scale_field: Optional[object] = None,
    ):
        assert x_full.ndim == roles_image.unlike + roles_image.like + roles_image.nufft
        self.plan = _FakePlan(roles_image=roles_image, num_shards=num_shards)
        self.arena = _FakeArena()
        self.scale_field = scale_field

        # Split x across shards along dim=0
        B = x_full.shape[0]
        if num_shards == 1:
            splits = [B]
        else:
            # Nearly equal splits
            base = B // num_shards
            rem = B % num_shards
            splits = [base + (1 if i < rem else 0) for i in range(num_shards)]
        self._x: List[torch.Tensor] = list(x_full.split(splits, dim=0))
        self._g: List[torch.Tensor] = [torch.zeros_like(t) for t in self._x]
        self._diag: Optional[List[torch.Tensor]] = None
        if with_diag:
            self._diag = [torch.zeros(t.shape, device=t.device, dtype=t.real.dtype) for t in self._x]

    @property
    def num_shards(self) -> int:
        return len(self._x)

    def iter_shards(self) -> Iterable[Tuple[_FakeShard, int]]:
        for i, t in enumerate(self._x):
            yield _FakeShard(t.device), i

    def get(self, name: str, shard_idx: int) -> torch.Tensor:
        if name == "x":
            return self._x[shard_idx]
        if name == "g":
            return self._g[shard_idx]
        if name == "diag":
            assert self._diag is not None
            return self._diag[shard_idx]
        raise KeyError(name)

    def has(self, name: str) -> bool:
        if name == "diag":
            return self._diag is not None
        return name in ("x", "g")

    def concat(self, name: str) -> torch.Tensor:
        if name == "x":
            return torch.cat(self._x, dim=0)
        if name == "g":
            return torch.cat(self._g, dim=0)
        if name == "diag":
            assert self._diag is not None
            return torch.cat(self._diag, dim=0)
        raise KeyError(name)

    def zero_grad(self) -> None:
        for g in self._g:
            g.zero_()


# -----------------------
# Fixtures
# -----------------------

@pytest.fixture(autouse=True)
def _set_seed():
    torch.manual_seed(0)


def _ramp_temporal(shape: Tuple[int, ...], dtype=torch.complex64) -> torch.Tensor:
    """
    Build x[t, ...] = t (complex allowed) so TV along temporal axis is predictable.
    """
    B = shape[0]
    real = torch.arange(B, dtype=torch.float32).view(B, *([1] * (len(shape) - 1))).expand(shape)
    if dtype.is_complex:
        return (real + 0j).to(dtype)
    return real.to(dtype)


# -----------------------
# Tests: TVND core kernels
# -----------------------

def test_tv1d_energy_grad_matches_reference_iso():
    """
    1D TV along temporal axis (B), isotropic with eps=0 on a linear ramp.
    Energy should be (B-1) * prod(other dims), gradient should be [-1, 0,..,0, +1] broadcast.
    """
    B, H, W = 6, 3, 2
    roles = Roles(unlike=1, like=0, nufft=2)  # dims: (B, H, W)
    x = _ramp_temporal((B, H, W), dtype=torch.complex64)

    tv = TVND("tv_t", TVParams(weight=1.0, eps=0.0, axes="temporal", isotropic=True))
    mgr = RegManager([tv], compile_kernels=False)

    ws = _WS(x, roles, num_shards=1)
    E = mgr.energy_and_grad(ws)  # 0-d tensor

    # Energy check: forward diff => ones for t=0..B-2, zeros at last
    expected_energy = float((B - 1) * H * W)
    assert isinstance(E, torch.Tensor) and E.dim() == 0
    assert abs(E.item() - expected_energy) < 1e-5

    # Gradient pattern: -bwd_diff(sign(fwd_diff(x)))
    g = ws.concat("g")
    # Expected per-voxel temporal pattern: [-1, 0, ..., 0, +1]
    per_t = torch.zeros(B, dtype=torch.float32)
    per_t[0] = -1.0
    per_t[-1] = +1.0
    expected_g_real = per_t.view(B, 1, 1).expand(B, H, W)
    # Gradient is complex with zero imaginary part for real ramp
    assert torch.allclose(g.real, expected_g_real, atol=1e-6)
    assert torch.allclose(g.imag, torch.zeros_like(g.imag), atol=1e-6)


def test_tv2d_aniso_ge_iso_energy():
    """
    For a nontrivial 2D field, anisotropic TV ≥ isotropic TV (with eps≈0).
    """
    B, H, W = 1, 5, 6
    roles = Roles(unlike=1, like=0, nufft=2)
    x = (torch.randn((B, H, W)) + 1j * torch.randn((B, H, W))).to(torch.complex64)

    ws_iso = _WS(x.clone(), roles)
    ws_aniso = _WS(x.clone(), roles)

    tv_iso = TVND("tv_sp", TVParams(weight=1.0, eps=1e-8, axes="spatial", isotropic=True))
    tv_aniso = TVND("tv_sp", TVParams(weight=1.0, eps=1e-8, axes="spatial", isotropic=False))
    mgr_iso = RegManager([tv_iso], compile_kernels=False)
    mgr_aniso = RegManager([tv_aniso], compile_kernels=False)

    E_iso = mgr_iso.energy_and_grad(ws_iso).item()
    E_aniso = mgr_aniso.energy_and_grad(ws_aniso).item()
    assert E_aniso >= E_iso - 1e-5  # allow tiny num diff


def test_tv_add_diag_adds_expected_amount():
    """
    add_diag should add λ * 2 * (#axes) to the diagonal (broadcast over all voxels).
    """
    B, H, W = 2, 4, 3
    roles = Roles(unlike=1, like=0, nufft=2)
    x = torch.zeros((B, H, W), dtype=torch.complex64)
    weight = 0.5
    axes = "spatial"  # two axes: H and W
    expected_add = 2.0 * 2 * weight  # 2 * (#axes) * λ

    ws = _WS(x, roles, with_diag=True)
    tv = TVND("tv_sp", TVParams(weight=weight, eps=0.0, axes=axes, isotropic=True))
    mgr = RegManager([tv], compile_kernels=False)

    mgr.add_diag(ws)
    D = ws.concat("diag")
    assert torch.allclose(D, torch.full_like(D, expected_add), atol=1e-6)


def test_tv_energy_grad_returns_tensor_scalar_and_complex_grad():
    """
    The manager must return a 0-d REAL tensor; gradient buffer must be complex dtype.
    """
    B, H, W = 3, 3, 3
    roles = Roles(unlike=1, like=0, nufft=2)
    x = (torch.randn((B, H, W)) + 1j * torch.randn((B, H, W))).to(torch.complex64)

    ws = _WS(x, roles)
    tv = TVND("tv_sp", TVParams(weight=1.0, eps=1e-3, axes="spatial", isotropic=True))
    mgr = RegManager([tv], compile_kernels=False)
    E = mgr.energy_and_grad(ws)

    assert isinstance(E, torch.Tensor) and E.dim() == 0 and torch.is_floating_point(E)
    g = ws.concat("g")
    assert g.dtype.is_complex


def test_axes_resolution_tokens_local_and_manager():
    """
    Ensure token mapping follows (unlike, like, nufft) policy.
    roles: (1, 1, 2) => dims: [t, z, x, y] => 'temporal'->(0), 'like'->(1), 'spatial'->(2,3)
    """
    roles = Roles(unlike=1, like=1, nufft=2)
    tv = TVND("tv_any", TVParams(weight=0.0, axes="spatial"))
    local_sp = TVND._resolve_axes_local("spatial", roles)
    local_t = TVND._resolve_axes_local("temporal", roles)
    local_l = TVND._resolve_axes_local("like", roles)
    assert local_sp == (2, 3)
    assert local_t == (0,)
    assert local_l == (1,)

    # Manager resolver path
    B, Z, X, Y = 2, 1, 4, 5
    x = torch.zeros((B, Z, X, Y), dtype=torch.complex64)
    ws = _WS(x, roles)
    # Build a tiny manager just to access its resolver through context in a run
    mgr = RegManager([tv], compile_kernels=False)
    # Running once will internally resolve axes; this is a smoke check (should not throw)
    mgr.energy_and_grad(ws)


def test_temporal_halo_sharded_matches_unsharded():
    """
    With TV along temporal axis, sharded + halo calculation should match an unsharded reference.
    """
    B, H, W = 6, 2, 2
    roles = Roles(unlike=1, like=0, nufft=2)
    x = (torch.randn((B, H, W)) + 1j * torch.randn((B, H, W))).to(torch.complex64)

    tv = TVND("tv_t", TVParams(weight=1.0, eps=1e-8, axes="temporal", isotropic=True))
    mgr = RegManager([tv], compile_kernels=False)

    # Unsharded reference
    ws_ref = _WS(x.clone(), roles, num_shards=1)
    E_ref = mgr.energy_and_grad(ws_ref)
    g_ref = ws_ref.concat("g").clone()

    # Sharded (2 shards)
    ws_sh = _WS(x.clone(), roles, num_shards=2)
    E_sh = mgr.energy_and_grad(ws_sh)
    g_sh = ws_sh.concat("g").clone()

    # Energies must match closely; grads must match elementwise
    assert torch.allclose(E_sh, E_ref, atol=1e-6)
    assert torch.allclose(g_sh, g_ref, atol=1e-6)


def test_manager_uses_fixed_axes_kernel_when_available():
    """
    If a regularizer implements energy_grad_fixed_axes but NOT energy_grad (or raises),
    the manager should call the fixed-axes variant (tested by toggling a flag).
    """
    class FixedOnlyReg:
        def __init__(self):
            self.name = "fixed_only"
            self.params = RegParams(weight=0.0, axes="spatial")
            self.called_fixed = False

        def energy_grad(self, ctx):
            raise RuntimeError("should_not_call_generic_energy_grad")

        def energy_grad_fixed_axes(self, ctx, axes):
            self.called_fixed = True
            return torch.zeros((), device=ctx.device, dtype=ctx.dtype_r)

        def add_diag(self, ctx): ...
        def continuation_update(self, stats): return False
        def scaling_policy(self, ctx): return None
        def prox_inplace(self, ctx, step): ...
        def majorizer_diag(self, ctx): return None
        def halo(self, roles): return {}

    B, H, W = 2, 3, 4
    roles = Roles(unlike=1, like=0, nufft=2)
    x = torch.zeros((B, H, W), dtype=torch.complex64)

    reg = FixedOnlyReg()
    mgr = RegManager([reg], compile_kernels=False)
    ws = _WS(x, roles)
    _ = mgr.energy_and_grad(ws)  # should not raise
    assert reg.called_fixed is True


def test_scale_field_passthrough_noop():
    """
    Manager should tolerate a scale_field object even if the reg ignores it.
    """
    class ScaleField:
        def inv_s2_for_shard(self, sh, anchor):
            # return (B_loc, 1, 1) ones on the shard's device
            B_loc = anchor.shape[0]
            return torch.ones((B_loc, 1, 1), device=anchor.device, dtype=anchor.real.dtype)

    B, H, W = 3, 2, 2
    roles = Roles(unlike=1, like=0, nufft=2)
    x = torch.randn((B, H, W), dtype=torch.complex64)

    ws = _WS(x, roles, scale_field=ScaleField())
    tv = TVND("tv_sp", TVParams(weight=1.0, eps=1e-3, axes="spatial", isotropic=True))
    mgr = RegManager([tv], compile_kernels=False)

    # Should run without error; return a tensor scalar
    E = mgr.energy_and_grad(ws)
    assert isinstance(E, torch.Tensor) and E.dim() == 0