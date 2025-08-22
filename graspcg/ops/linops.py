from __future__ import annotations
from typing import Optional, Protocol, Callable, Sequence, Tuple, Union
import contextlib
import inspect
import torch

# ----------------------------- Protocol -----------------------------

class LinearOp(Protocol):
    """
    Abstract linear acquisition operator with an optional diagonal A^H A.
      A : image  -> kspace   (B, C, H, W[,D]) -> (B, C, K)
      AH: kspace -> image    (B, C, K)        -> (B, 1, H, W[,D])
    All methods must be device/dtype safe and (ideally) honor out=.
    """
    def A(self, x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor: ...
    def AH(self, y: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor: ...
    def diag_AHA(self, ref: torch.Tensor) -> Optional[torch.Tensor]: ...

# ----------------------------- Helpers ------------------------------

def _call_maybe_out(fn: Callable, arg: torch.Tensor, out: Optional[torch.Tensor]):
    """Call fn(arg, out=...) if supported; else fn(arg)."""
    try:
        sig = inspect.signature(fn)
        if "out" in sig.parameters and out is not None:
            return fn(arg, out=out)
    except Exception:
        pass
    return fn(arg) if out is None else out.copy_(fn(arg))

# ------------------------ Concrete wrappers -------------------------

class _ObjectLinearOp:
    def __init__(self, obj):
        self.obj = obj
        import inspect
        try:
            self._A_has_out  = "out" in inspect.signature(obj.A).parameters
            self._AH_has_out = "out" in inspect.signature(obj.AH).parameters
        except Exception:
            self._A_has_out = self._AH_has_out = False

    def A(self, x, out=None):
        if out is not None and self._A_has_out:
            return self.obj.A(x, out=out)
        r = self.obj.A(x)
        return r if out is None else out.copy_(r)

    def AH(self, y, out=None):
        if out is not None and self._AH_has_out:
            return self.obj.AH(y, out=out)
        r = self.obj.AH(y)
        return r if out is None else out.copy_(r)


class CallableLinearOp:
    """Wrap (A_fn, AH_fn [, diag_fn]). Each fn may or may not accept out=."""
    def __init__(self, A_fn: Callable, AH_fn: Callable, diag_fn: Optional[Callable] = None) -> None:
        self.A_fn = A_fn; self.AH_fn = AH_fn; self.diag_fn = diag_fn
    def A(self, x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        return _call_maybe_out(self.A_fn, x, out)
    def AH(self, y: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        return _call_maybe_out(self.AH_fn, y, out)
    def diag_AHA(self, ref: torch.Tensor) -> Optional[torch.Tensor]:
        if self.diag_fn is None:
            return None
        try:
            return self.diag_fn(ref)
        except Exception:
            return None

class ComposeLinearOp:
    """
    Chain of LinearOps: A = A_n ◦ ... ◦ A_1, AH = AH_1 ◦ ... ◦ AH_n.
    We intentionally return None for diag_AHA (safe default). Specific chains
    that admit a cheap diagonal can implement their own subclass if needed.
    """
    def __init__(self, ops: Sequence[LinearOp]) -> None:
        self.ops = list(ops)
        if len(self.ops) == 0:
            raise ValueError("ComposeLinearOp requires at least one operator.")
    def A(self, x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = x
        for i, op in enumerate(self.ops):
            last = (i == len(self.ops) - 1)
            z = op.A(z, out=out if last else None)
        return z if out is None else out
    def AH(self, y: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = y
        for i, op in enumerate(reversed(self.ops)):
            last = (i == len(self.ops) - 1)
            z = op.AH(z, out=out if last else None)
        return z if out is None else out
    def diag_AHA(self, ref: torch.Tensor) -> Optional[torch.Tensor]:
        return None  # safe default; override in a specialized composition if needed

# --------------------------- Converters -----------------------------

def as_linear_op(obj: Union[LinearOp, Tuple[Callable, Callable, Optional[Callable]], object]) -> LinearOp:
    """
    Accepts:
      • any object with .A/.AH (and optional .diag_AHA) -> wrapped view
      • a tuple (A_fn, AH_fn [, diag_fn])               -> CallableLinearOp
      • a LinearOp                                      -> returned as-is
    """
    # Structural duck-typing: if it "looks like" a LinearOp, keep it.
    if hasattr(obj, "A") and hasattr(obj, "AH"):
        return _ObjectLinearOp(obj)
    if isinstance(obj, tuple):
        if len(obj) < 2:
            raise ValueError("Need at least (A_fn, AH_fn).")
        return CallableLinearOp(obj[0], obj[1], obj[2] if len(obj) > 2 else None)
    # If user passed a subclass implementing the Protocol fully, return it
    if isinstance(obj, LinearOp.__constraints__ if hasattr(LinearOp, "__constraints__") else object):
        return obj  # type: ignore
    raise TypeError("Unsupported linear operator; provide .A/.AH, a (A_fn, AH_fn) tuple, or a LinearOp.")

def chain(*ops: LinearOp) -> LinearOp:
    """Convenience for ComposeLinearOp(*ops)."""
    return ComposeLinearOp(ops)
