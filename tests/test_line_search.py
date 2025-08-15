"""Tests for the line search zoom routine.

These tests avoid requiring a full PyTorch installation by stubbing just the
`torch.no_grad` context manager used by the module under test.
"""

import sys
import types
import contextlib
from pathlib import Path

# ---------------------------------------------------------------------------
# create a minimal torch stub with the no_grad decorator used in the module
torch_stub = types.ModuleType("torch")

@contextlib.contextmanager
def _no_grad():
    yield

torch_stub.no_grad = _no_grad
sys.modules.setdefault("torch", torch_stub)

# make package importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

from graspcg.numerics import line_search


class DummyObj:
    def f_g(self, ws, t):
        f = 0.5 * t ** 2 + t
        gdot = t + 1.0
        return f, gdot


class DummySolver:
    def __init__(self):
        self.ws = None
        self.obj = DummyObj()
        self.c1 = 0.5
        self.c2 = 0.1
        self.ls_max_iter = 1
        self.ls_name = "wolfe"
        self.ls_zoom = True


def test_zoom_returns_correct_gradient():
    solver = DummySolver()
    ok, t, f, gdot = line_search._zoom(
        solver,
        t_low=1.0,
        f_low=1.5,
        t_high=2.0,
        f0=0.0,
        g0d=1.0,
    )
    assert ok is True
    # gradient at t_low=1.0 is gdot=2.0 (since derivative of f is t+1)
    assert abs(gdot - 2.0) < 1e-6
