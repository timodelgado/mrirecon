# graspcg/workspace/shardops.py
from __future__ import annotations
from typing import Callable, Iterable, Optional, Sequence, Tuple, Dict
import contextlib
import torch

from .unified_arena import DeviceArena  # streams & scratch
from .workspace import Workspace        # for type hints only

class ShardOps:
    """
    Tiny facade over Workspace sharding that keeps user functions tensor-only.
    The user supplies `fn(*tensors)`; we compile it per (shape,dtype,device).
    """
    def __init__(self, ws: Workspace, *, compile_kernels: bool = True, use_streams: bool = True):
        self.ws = ws
        self.compile_kernels = bool(compile_kernels)
        self.use_streams = bool(use_streams)
        self._cache: Dict[Tuple[int, Tuple[torch.Size, ...], Tuple[torch.dtype, ...], torch.device], Callable] = {}

    # -------------- public API --------------

    @torch.no_grad()
    def foreach(self, names: Sequence[str], fn: Callable[..., None]) -> None:
        """
        Apply an in-place function on per-shard buffers:
            fn(*bufs_on_this_shard) -> None
        """
        for sh, i in self.ws.iter_shards():
            bufs = self.ws.bind(i, *names)
            f = self._compiled(fn, bufs)
            self._call_on_shard(f, bufs, sh.device)

    @torch.no_grad()
    def map_reduce(self,
                   names: Sequence[str],
                   fn: Callable[..., torch.Tensor],
                   *,
                   reducer: str = "sum",
                   out_device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Call fn(*bufs) on each shard; fn must return a **0-D tensor** (REAL or COMPLEX).
        Reduce across devices to a single 0-D tensor (on `out_device` or a primary one).
        """
        assert reducer in ("sum",), "Only 'sum' reduction is provided for now."

        dev_accum: Dict[torch.device, torch.Tensor] = {}
        dev_order: list[torch.device] = []

        for sh, i in self.ws.iter_shards():
            bufs = self.ws.bind(i, *names)
            f = self._compiled(fn, bufs)
            val = self._call_on_shard(f, bufs, sh.device)
            if val.dim() != 0:
                raise RuntimeError("map_reduce expects fn to return a 0-D tensor")
            if sh.device not in dev_accum:
                dev_accum[sh.device] = val
                dev_order.append(sh.device)
            else:
                dev_accum[sh.device].add_(val)

        if not dev_accum:
            # Degenerate case: no shards (return a CPU float32 zero)
            return torch.zeros((), dtype=torch.float32)

        primary = out_device or dev_order[0]
        dtype = next(iter(dev_accum.values())).dtype
        total = torch.zeros((), device=primary, dtype=dtype)
        for d in dev_order:
            v = dev_accum[d]
            total.add_(v if d == primary else v.to(primary, non_blocking=True))
        return total

    # Convenience: complex dot or preconditioned dot on a sharded buffer
    @torch.no_grad()
    def dot(self, a: str, b: str, diag: Optional[str] = None) -> torch.Tensor:
        def _kernel(x: torch.Tensor, y: torch.Tensor, *rest) -> torch.Tensor:
            if rest:
                D = rest[0]
                z = x.conj() * (y / D)
            else:
                z = x.conj() * y
            return z.real.sum()  # 0-D real tensor
        names = (a, b) if diag is None else (a, b, diag)
        return self.map_reduce(names, _kernel)

    # -------------- internals --------------

    def _compiled(self, fn: Callable, bufs: Tuple[torch.Tensor, ...]) -> Callable:
        """
        Compile/cache per (id(fn), shapes, dtypes, device) signature.
        Shapes can differ across shards ⇒ expect a small number of specializations.
        """
        sig = (id(fn),
               tuple(t.shape for t in bufs),
               tuple(t.dtype for t in bufs),
               bufs[0].device)
        f = self._cache.get(sig)
        if f is not None or not self.compile_kernels:
            return f or fn
        f = torch.compile(fn)
        self._cache[sig] = f
        return f

    def _call_on_shard(self, f: Callable, bufs: Tuple[torch.Tensor, ...], dev: torch.device):
        if self.use_streams and dev.type == "cuda":
            stream = self.ws.arena.stream_for(dev)
            with torch.cuda.stream(stream):
                return f(*bufs)
        else:
            return f(*bufs)
