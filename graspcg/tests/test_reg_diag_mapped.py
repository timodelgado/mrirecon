import torch
import pytest

from graspcg.regularization.tv_nd import TVND, TVParams
from graspcg.regularization.manager import RegManager
from graspcg.regularization.wrappers import Transformed
from graspcg.regularization.mapping import ScaleOp, TemporalBasisOp

# ---- minimal roles/plan/ws stubs ----

class DummyRoles:
    def resolve_axes(self, spec):
        if isinstance(spec, (list, tuple)):
            return tuple(int(a) for a in spec)
        if spec in ("spatial", "image", "nufft"):
            return (2, 3)  # CHW -> H,W
        if spec in ("like",):
            return (1,)
        if spec in ("unlike", "temporal"):
            return (0,)
        raise ValueError(f"Unknown axes spec {spec!r}")

class DummyPlan:
    def __init__(self):
        self.roles_image = DummyRoles()
        self.num_shards = 1

class DummyShard:
    def __init__(self, device, b_start=0):
        self.device = device
        self.b_start = b_start

class DummyScale:
    def __init__(self, inv_scalar: float = 1.0):
        self.inv_scalar = float(inv_scalar)
    def inv_for_shard(self, shard, anchor):
        shape = (anchor.shape[0],) + (1,) * (anchor.ndim - 1)
        return torch.full(shape, self.inv_scalar, device=anchor.device, dtype=anchor.dtype)

class DummyWS:
    def __init__(self, x, diag=None, V=None, diag_V=None, scale=None, device=None):
        # Canonical names
        self._bufs = {"var": [x]}
        if diag is not None:
            self._bufs["diag"] = [diag]
        if V is not None:
            self._bufs["V"] = [V]
        if diag_V is not None:
            self._bufs["diag_V"] = [diag_V]
        self.device = device or x.device
        self.plan = DummyPlan()
        self._shard = DummyShard(self.device, b_start=0)
        self.scale = scale
        self.arena = None
        self.num_shards = 1
    def get(self, key, i):
        return self._bufs[key][i]
    def has(self, key):
        return key in self._bufs
    def shard_for_index(self, i):
        return self._shard
    def iter_shards(self):
        yield self._shard, 0
    def _safe_get(self, name, i, default=None):
        try:
            return self.get(name, i)
        except Exception:
            return default

# ---- tests ----

@pytest.mark.parametrize("device", [torch.device("cpu")])
def test_diag_push_scaleop(device):
    B, C, H, W = 2, 1, 4, 4
    x = torch.zeros((B, C, H, W), device=device, dtype=torch.complex64)
    diag = torch.zeros_like(x.real, dtype=torch.float32)

    reg = TVND(name="tv", params=TVParams(weight=1e-3, eps=1e-3, axes="spatial", isotropic=True))
    regm = RegManager([Transformed(base=reg, op=ScaleOp(), name="tv_scaled", params=reg.params)],
                      compile_kernels=False)

    inv = 2.5  # u = (1/s) ⊙ x with (1/s)=2.5 ⇒ diag scales by (2.5)^2
    ws = DummyWS(x=x, diag=diag, scale=DummyScale(inv_scalar=inv), device=device)

    regm.add_diag(ws)

    # Expected constant from TV majorizer: k = 2 * λ * (#axes)
    k = 2.0 * 1e-3 * 2.0
    expected = k * (inv**2)
    assert torch.allclose(ws.get("diag", 0), torch.full_like(diag, expected), atol=1e-7, rtol=0)

@pytest.mark.parametrize("device", [torch.device("cpu")])
def test_diag_push_temporal_basis(device):
    # Field: BxCxHxW, Params: (K, C, H, W)
    B, C, H, W = 2, 1, 3, 2
    K = 3
    x = torch.zeros((B, C, H, W), device=device, dtype=torch.complex64)
    V = torch.zeros((K, C, H, W), device=device, dtype=torch.complex64)
    diag_V = torch.zeros_like(V.real, dtype=torch.float32)

    # Simple U s.t. |U|^2 = [[1,0,0],[0,4,0]]  (B=2, K=3)
    U = torch.tensor([[1.0, 0.0, 0.0],
                      [0.0, 2.0, 0.0]], device=device)

    reg = TVND(name="tv", params=TVParams(weight=1e-3, eps=1e-3, axes="spatial", isotropic=True))
    op  = TemporalBasisOp(U=U, param_key="V", grad_key="g_V", diag_key="diag_V")
    regm = RegManager([Transformed(base=reg, op=op, name="tv_on_UV", params=reg.params)],
                      compile_kernels=False)

    ws = DummyWS(x=x, V=V, diag_V=diag_V, device=device)

    regm.add_diag(ws)

    # k from TV majorizer
    k = 2.0 * 1e-3 * 2.0  # 2*λ*(#spatial axes)
    # Expected:
    #   diag_V[0,:,:,:] += sum_t k * |U[t,0]|^2 = k * (1 + 0) = k
    #   diag_V[1,:,:,:] += sum_t k * |U[t,1]|^2 = k * (0 + 4) = 4k
    #   diag_V[2,:,:,:] += 0
    assert torch.allclose(ws.get("diag_V", 0)[0], torch.full((C, H, W), k,  device=device), atol=1e-7, rtol=0)
    assert torch.allclose(ws.get("diag_V", 0)[1], torch.full((C, H, W), 4*k, device=device), atol=1e-7, rtol=0)
    assert torch.all(ws.get("diag_V", 0)[2] == 0)

def test_diag_profile_temporal_only_identity(device=torch.device("cpu")):
    B, C, H, W = 5, 1, 2, 2
    x = torch.zeros((B, C, H, W), device=device, dtype=torch.complex64)
    diag = torch.zeros_like(x.real, dtype=torch.float32)
    reg = TVND("tv", TVParams(weight=1e-3, eps=1e-3, axes="temporal", isotropic=True))
    regm = RegManager([reg], compile_kernels=False)
    ws = DummyWS(x=x, diag=diag, device=device)
    regm.add_diag(ws)
    expected = torch.tensor([1,2,2,2,1], device=device, dtype=torch.float32) * (2*1e-3*1.0)
    assert torch.allclose(ws.get("diag",0).mean(dim=(1,2,3)), expected, atol=1e-7, rtol=0)
