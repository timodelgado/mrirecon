# graspcg/tests/test_regularization_ns.py
import math
import torch
import pytest


from ..regularization.tv_nd import TVND, TVParams
from ..regularization.manager import RegManager, RegContext
from ..regularization.stats_board import StatsBoard
from ..regularization.mapping import MappedRegularizer, IdentityOp, TemporalBasisOp, ComposeOp
from ..core.roles import Roles

# -------------------------- helpers / fakes --------------------------
def device_list():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))
        devices.append(torch.device("cuda:1"))
    return devices
class _Arena:
    def stream_for(self, device):
        return None

class _Shard:
    def __init__(self, b_start, device):
        self.b_start = int(b_start)
        self.device = device

class _Plan:
    def __init__(self, roles_image, num_shards):
        self.roles_image = roles_image
        self.num_shards = num_shards

class FakeWS:
    """
    Minimal workspace covering what RegManager/mapping need:
      • plan.roles_image, plan.num_shards
      • iter_shards(), shard_for_index(i)
      • get(name, i), has(name)
      • arena, stats (optional)
      • optional: shared parameter buffers ('V', 'g_V')
    """
    def __init__(self, x_shards, *, stats=None, extras=None, device=None):
        self.x_shards = list(x_shards)
        self.num_shards = len(self.x_shards)
        self._bstarts = []
        b = 0
        for xi in self.x_shards:
            self._bstarts.append(b)
            b += int(xi.shape[0])

        self.buffers = {
            "x": self.x_shards,
            "g": [torch.zeros_like(xi) for xi in self.x_shards],
            "diag": [torch.zeros_like(xi.real) for xi in self.x_shards],
        }
        if extras:
            for k, v in extras.items():
                # allow shared tensors across shards
                if isinstance(v, (list, tuple)):
                    self.buffers[k] = list(v)
                else:
                    self.buffers[k] = [v for _ in self.x_shards]

        dev = device or self.x_shards[0].device
        self.arena = _Arena()
        self.plan = _Plan(roles_image=Roles(unlike=1, like=1, nufft=2), num_shards=self.num_shards)
        self._shards = [_Shard(b0, xi.device) for b0, xi in zip(self._bstarts, self.x_shards)]
        self.stats = stats

    def iter_shards(self):
        for i, sh in enumerate(self._shards):
            yield sh, i

    def shard_for_index(self, i):
        return self._shards[i]

    def get(self, name, i):
        return self.buffers[name][i]

    def has(self, name):
        return name in self.buffers
    def _safe_get(self, name, i, default=None):
        try:
            return self.get(name, i)
        except Exception:
            return default
# -------------------------- fixtures --------------------------

def device_list():
    devs = [torch.device("cpu")]
    if torch.cuda.is_available():
        devs.append(torch.device("cuda"))
    return devs

# -------------------------- tests --------------------------

@pytest.mark.parametrize("device", device_list())
def test_tvnd_energy_grad_constant_zero(device):
    B, C, H, W = 4, 1, 8, 8
    x = torch.ones((B, C, H, W), device=device, dtype=torch.complex64)
    ws = FakeWS([x])
    tv = TVND("tv_sp", TVParams(weight=1.0, eps=1e-3, axes="spatial", isotropic=True))
    regm = RegManager([tv], compile_kernels=False)

    E = regm.energy_and_grad(ws)

    assert E.dtype == ws.get("g", 0).real.dtype
    assert float(E.detach().cpu()) == pytest.approx(0.0, abs=1e-6)
    assert torch.allclose(ws.get("g", 0), torch.zeros_like(ws.get("g", 0)))

@pytest.mark.parametrize("device", device_list())
def test_manager_sharded_vs_whole_energy_temporal_halo(device):
    # Make a signal that varies over the sharded axis (B) so halos are needed.
    B, C, H, W = 8, 1, 4, 4
    x_full = torch.linspace(0, 1, B, device=device).reshape(B, 1, 1, 1).expand(B, C, H, W).to(torch.complex64)

    # Whole (single shard)
    ws_whole = FakeWS([x_full])
    tv_t = TVND("tv_t", TVParams(weight=1.0, eps=1e-3, axes="temporal", isotropic=False))
    regm = RegManager([tv_t], compile_kernels=False)
    E_whole = regm.energy_and_grad(ws_whole)

    # Sharded (two shards) — should match
    ws_sh = FakeWS([x_full[:4], x_full[4:]])
    E_sh = regm.energy_and_grad(ws_sh)

    assert float(E_sh.detach().cpu()) == pytest.approx(float(E_whole.detach().cpu()), rel=1e-5, abs=1e-6)

@pytest.mark.parametrize("device", device_list())
def test_mapped_temporal_basis_updates_gV_not_x(device):
    # Build u = U @ V then TV on spatial dims; gradient should land in g_V, not g(x)
    T, K = 6, 3
    C, H, W = 1, 4, 4
    U = torch.randn(T, K)
    V = torch.randn(K, C, H, W, dtype=torch.complex64, device=device) * 0.1

    # two shards of length 3 each
    x1 = torch.zeros((3, C, H, W), device=device, dtype=torch.complex64)
    x2 = torch.zeros((3, C, H, W), device=device, dtype=torch.complex64)
    gV = torch.zeros_like(V)

    ws = FakeWS([x1, x2], extras={"V": V, "g_V": gV})
    tv = TVND("tv_sp", TVParams(weight=0.5, eps=1e-3, axes="spatial", isotropic=True))
    op = TemporalBasisOp(U=U, param_key="V", grad_key="g_V")
    reg = MappedRegularizer(name="tv_UV", inner=tv, op=op)
    regm = RegManager([reg], compile_kernels=False)

    E = regm.energy_and_grad(ws)
    # x-grad should be zero (param-only mapping)
    assert torch.allclose(ws.get("g", 0), torch.zeros_like(ws.get("g", 0)))
    assert torch.allclose(ws.get("g", 1), torch.zeros_like(ws.get("g", 1)))
    # g_V should receive updates on both shards (shared tensor)
    assert torch.linalg.norm(ws.get("g_V", 0)).item() > 0.0
    # energy positive
    assert float(E.detach().cpu()) >= 0.0

@pytest.mark.parametrize("device", device_list())
def test_stats_board_tv_quantile_collection(device):
    B, C, H, W = 8, 1, 8, 8
    x = (torch.rand((B, C, H, W), device=device) - 0.5).to(torch.complex64)
    sb = StatsBoard()
    sb.enable("tv_quantile", True)
    sb.tv_percentile = 0.9
    sb.tv_sample_K = 128

    ws = FakeWS([x], stats=sb)
    tv = TVND("tv_sp", TVParams(weight=1.0, eps=1e-3, axes="spatial", isotropic=True))
    regm = RegManager([tv], compile_kernels=False)

    _ = regm.energy_and_grad(ws)
    q = sb.read_scalar(f"tv_q/{tv.name}")
    assert q >= 0.0

@pytest.mark.parametrize("device", device_list())
def test_add_diag_behavior_identity_vs_mapped(device):
    B, C, H, W = 4, 1, 8, 8
    x = torch.randn((B, C, H, W), device=device, dtype=torch.complex64) * 0.01

    # Identity TVND: diag must increase
    ws1 = FakeWS([x.clone()])
    tv1 = TVND("tv_id", TVParams(weight=1.0, eps=1e-3, axes="spatial"))
    regm1 = RegManager([tv1], compile_kernels=False)
    d_before = ws1.get("diag", 0).clone()
    regm1.add_diag(ws1)
    d_after = ws1.get("diag", 0)
    assert torch.all(d_after > d_before)

    # Mapped (TemporalBasis): diag should not change (no-op add_diag)
    T, K = B, 2
    U = torch.randn(T, K)
    V = torch.randn(K, C, H, W, dtype=torch.complex64, device=device) * 0.01
    ws2 = FakeWS([x.clone()], extras={"V": V, "g_V": torch.zeros_like(V)})
    tv2 = TVND("tv_map", TVParams(weight=1.0, eps=1e-3, axes="spatial"))
    reg = MappedRegularizer("tv_UV", tv2, TemporalBasisOp(U=U))
    regm2 = RegManager([reg], compile_kernels=False)
    d2_before = ws2.get("diag", 0).clone()
    regm2.add_diag(ws2)
    d2_after = ws2.get("diag", 0)
    assert torch.allclose(d2_after, d2_before)

@pytest.mark.parametrize("device", device_list())
def test_graph_diag_degree(device):
    B, C, H, W = 5, 1, 4, 3
    x = torch.zeros((B,C,H,W), device=device, dtype=torch.complex64)
    g = torch.zeros_like(x)
    D = torch.zeros_like(x.real)

    # Simple chain: neighbors weight=1 -> deg=[1,2,2,2,1]
    W = torch.zeros(B, B)
    for t in range(B-1):
        W[t, t+1] = 1.0; W[t+1, t] = 1.0

    from graspcg.core.roles import Roles
    from types import SimpleNamespace
    from graspcg.regularization.graph_laplacian import GraphLaplacian, GraphLapParams

    reg = GraphLaplacian("gl", W, GraphLapParams(weight=2.0, normalize="none"))

    class WS:
        def __init__(self): self._bufs={"x":[x], "g":[g], "diag":[D]}
        def iter_shards(self): yield (SimpleNamespace(b_start=0,b_stop=B), 0)
        def get(self, k, i): return self._bufs[k][i]
        def shard_for_index(self, i): return SimpleNamespace(b_start=0, b_stop=B)
        @property
        def plan(self): return SimpleNamespace(roles_image=Roles(unlike=1, like=1, nufft=2))
        @property
        def arena(self): return None
        def has(self, k): return k in self._bufs

    ws = WS()
    # Minimal context per manager.add_diag impl
    from graspcg.regularization.manager import RegManager
    RegManager([reg]).add_diag(ws)

    deg = torch.tensor([1,2,2,2,1], dtype=D.dtype, device=device) * 2.0
    target = deg.view(B,1,1,1).expand_as(D)
    assert torch.allclose(D, target, atol=1e-6, rtol=0)

@pytest.mark.parametrize("device", device_list())
def test_graph_energy_grad_matches_fd(device):
    B, C, H, W = 6, 1, 4, 4
    x = (torch.randn((B,C,H,W), device=device, dtype=torch.complex64) * 0.05).requires_grad_(False)
    g = torch.zeros_like(x)
    D = torch.ones_like(x.real)

    W = torch.rand(B, B); W = 0.5*(W+W.t()); W.fill_diagonal_(0)
    from graspcg.regularization.graph_laplacian import GraphLaplacian, GraphLapParams
    reg = GraphLaplacian("gl", W, GraphLapParams(weight=1e-3, normalize="sym"))

    from graspcg.core.roles import Roles
    ctx = RegContext(
        x=x, g=g, diag=D, roles_image=Roles(unlike=1, like=1, nufft=2),
        device=x.device, dtype_c=x.dtype, dtype_r=x.real.dtype,
        axes_resolver=lambda spec:(1,2,3), arena=None,
        write_interior_slice=(slice(None),)*x.ndim,
        ws=type("W", (), {
            "iter_shards": lambda self: [ (type("S", (), {"b_start":0,"b_stop":B})(), 0) ].__iter__(),
            "get": lambda self,k,i: {"x":x,"g":g,"diag":D}[k],
            "shard_for_index": lambda self,i: type("S", (), {"b_start":0,"b_stop":B},)()
        })(), shard_index=0
    )

    # Analytical E, g
    E0 = reg.energy_grad(ctx)
    g0 = g.clone()

    # Finite-difference slope along random d
    d = torch.randn_like(x) * 0.1
    eps = torch.tensor(1e-4, device=device, dtype=x.real.dtype)

    def E_at(t):
        xt = x + (t * d)
        gtmp = torch.zeros_like(gtmp := g)
        # re-evaluate through a fresh context
        ctx2 = RegContext(**{**ctx.__dict__, "x": xt, "g": gtmp})
        return reg.energy_grad(ctx2)

    Ep = E_at(eps); Em = E_at(-eps)
    gdot_fd = ((Ep - Em) / (2*eps)).item()

    gdot_an = (g0.conj() * d).real.sum().item()
    assert abs(gdot_an - gdot_fd) < 5e-4
