import torch
import pytest
from graspcg.utils.operations import dot_chunked


def test_dot_chunked_matches_direct():
    a = torch.randn(6, dtype=torch.cfloat)
    b = torch.randn(6, dtype=torch.cfloat)

    expected = torch.real((a.conj() * b).sum()).item()
    result = dot_chunked(a, b)

    assert result == pytest.approx(expected)


def test_dot_chunked_with_diagonal():
    a = torch.randn(5, dtype=torch.cfloat)
    b = torch.randn(5, dtype=torch.cfloat)
    diag = torch.rand(5, dtype=torch.cfloat) + 0.1

    expected = torch.real((a.conj() * (b / diag)).sum()).item()
    result = dot_chunked(a, b, diag=diag)

    assert result == pytest.approx(expected)


def test_dot_chunked_respects_chunk_size():
    a = torch.randn(10, dtype=torch.cfloat)
    b = torch.randn(10, dtype=torch.cfloat)
    chunk = 3  # smaller than tensor length to force chunking

    expected = torch.real((a.conj() * b).sum()).item()
    result = dot_chunked(a, b, chunk=chunk)

    assert result == pytest.approx(expected)
