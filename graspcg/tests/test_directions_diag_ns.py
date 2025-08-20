import pytest
import torch

from graspcg.numerics.directions import _diag_like


def test_diag_like_batch_expand():
    a = torch.zeros(2, 3, 4, 5)
    D = torch.ones(3, 4, 5)
    out = _diag_like(a, D)
    assert out.shape == a.shape
    assert torch.all(out == 1.0)
    assert out.device == a.device


def test_diag_like_returns_same_object():
    a = torch.randn(3, 4)
    D = torch.randn(3, 4)
    out = _diag_like(a, D)
    assert out is D


def test_diag_like_mismatch_raises():
    a = torch.randn(2, 3, 4)
    D = torch.randn(3, 5)
    with pytest.raises(ValueError):
        _diag_like(a, D)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_diag_like_moves_to_device():
    a = torch.zeros(2, 3, 4, device=torch.device('cuda:0'))
    D = torch.ones(3, 4, device=torch.device('cpu'))
    out = _diag_like(a, D)
    assert out.device == a.device
