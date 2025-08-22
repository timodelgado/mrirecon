import pytest
import torch

from graspcg.ops.dot import dot_chunked


def _rand_cplx(n, device, dtype=torch.complex64, seed=0):
    if isinstance(device, torch.device) and device.type == "cuda":
        gen = torch.Generator(device=device)
    else:
        gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    re = torch.randn(n, generator=gen, device=device, dtype=torch.float32)
    im = torch.randn(n, generator=gen, device=device, dtype=torch.float32)
    return (re + 1j * im).to(dtype)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_dot_chunked_matches_naive(dtype):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    a = _rand_cplx(100, device=device, dtype=dtype, seed=1)
    b = _rand_cplx(100, device=device, dtype=dtype, seed=2)
    res = dot_chunked(a, b, chunk=17)
    expected = torch.real((a.conj() * b).sum()).item()
    tol = 1e-5 if dtype == torch.complex64 else 1e-12
    assert abs(res - expected) <= tol * (1.0 + abs(expected))


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_dot_chunked_with_diag(dtype):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    a = _rand_cplx(80, device=device, dtype=dtype, seed=3)
    b = _rand_cplx(80, device=device, dtype=dtype, seed=4)
    diag = torch.rand(80, device=device, dtype=torch.float32) + 0.5
    diag = diag.to(dtype)
    res = dot_chunked(a, b, diag=diag, chunk=13)
    expected = torch.real((a.conj() * (b / diag)).sum()).item()
    tol = 1e-5 if dtype == torch.complex64 else 1e-12
    assert abs(res - expected) <= tol * (1.0 + abs(expected))


class DummyArena:
    def __init__(self, chunk):
        self.chunk = chunk
        self.requested = False
    def free_elems(self, dtype, device):
        return self.chunk
    def request(self, numel, dtype, anchor=None):
        self.requested = True
        return torch.empty(numel, dtype=dtype, device=anchor.device)
    def release(self, tensor):
        pass


def test_dot_chunked_uses_arena_for_scratch():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    a = _rand_cplx(120, device=device, seed=5)
    b = _rand_cplx(120, device=device, seed=6)
    arena = DummyArena(chunk=32)
    res = dot_chunked(a, b, arena=arena)
    expected = torch.real((a.conj() * b).sum()).item()
    assert arena.requested
    assert abs(res - expected) <= 1e-5 * (1.0 + abs(expected))
