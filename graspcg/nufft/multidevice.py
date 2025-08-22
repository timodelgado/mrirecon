# -----------------------------------------------------------------------------
# graspcg/nufft/multidevice.py
# -----------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import contextlib

import torch

from .api import NUFFT, NUFFTConfig
from .layout import AxisSpec

# ------------------------------ helpers ---------------------------------------

def _is_cuda(dev: torch.device) -> bool:
    return getattr(dev, "type", None) == "cuda"

def _sync(dev: torch.device) -> None:
    if _is_cuda(dev):
        torch.cuda.synchronize(device=dev)

def _b_slice(x: torch.Tensor, axis_labels: Sequence[str], b_start: int, b_stop: int) -> torch.Tensor:
    """Return a view slicing the B dimension in user order."""
    try:
        b_pos = axis_labels.index('B')
    except ValueError:
        raise ValueError("AxisSpec.image / .kspace must include 'B' as the batch axis.")
    sl = [slice(None)] * x.ndim
    sl[b_pos] = slice(b_start, b_stop)
    return x[tuple(sl)]

def _maybe_list_devices(ws: Any) -> List[torch.device]:
    # Preferred: ws.plan.shards[*].device
    devs: List[torch.device] = []
    plan = getattr(ws, "plan", None)
    if plan is not None and hasattr(plan, "shards"):
        for sh in plan.shards:
            devs.append(sh.device)
        if devs:
            return devs
    # Fallback: ws.devices
    dlist = getattr(ws, "devices", None)
    if isinstance(dlist, (list, tuple)) and all(isinstance(d, torch.device) for d in dlist):
        return list(dlist)
    # Last resort: all available CUDA devices or CPU
    if torch.cuda.is_available():
        return [torch.device("cuda", i) for i in range(torch.cuda.device_count())]
    return [torch.device("cpu")]

def _maybe_shards(ws: Any, B_total: int, devs: Sequence[torch.device]) -> List[Tuple[torch.device, int, int]]:
    """
    Return [(device, b_start, b_stop)].
    Prefer ws.plan.shards; else split B evenly across devs.
    """
    plan = getattr(ws, "plan", None)
    out: List[Tuple[torch.device, int, int]] = []
    if plan is not None and hasattr(plan, "shards"):
        for sh in plan.shards:
            out.append((sh.device, int(sh.b_start), int(sh.b_stop)))
        if out:
            return out
    # Even split
    ndev = max(1, len(devs))
    per = (B_total + ndev - 1) // ndev
    s = 0
    for d in devs:
        e = min(B_total, s + per)
        if s < e:
            out.append((d, s, e))
        s = e
    return out

def _resolve_out_list(out: Any, nshards: int) -> Optional[List[torch.Tensor]]:
    """
    Normalize out= into a per-shard list if possible.

    Accepted forms:
      - list[Tensor] of length nshards
      - dict[int|device] -> Tensor (order will match self.shards)
      - None -> None
    """
    if out is None:
        return None
    if isinstance(out, list):
        if len(out) != nshards:
            raise ValueError(f"out list must have length {nshards}")
        return out
    if isinstance(out, dict):
        # Map devices or indices
        return None  # handled by caller with access to self.shards
    # Anything else → unsupported (single global tensor cannot span devices)
    raise ValueError("For multi-device, 'out' must be a list of per-shard tensors or a dict keyed by device.")

# ------------------------------ public API ------------------------------------

@dataclass
class ShardHandle:
    device: torch.device
    b_start: int
    b_stop: int
    op: NUFFT

try:
    from torch._dynamo import disable as _dynamo_disable
except Exception:
    def _dynamo_disable(fn):
        return fn

@dataclass
class MultiDeviceNUFFT:
    """
    Streamlined, North‑Star multi-device NUFFT.

    • Shards over the batch axis 'B' across devices. Like dims remain local to each device.
    • Builds one single-device NUFFT per shard using the same AxisSpec & NUFFTConfig.
    • Accepts a Workspace-like object `ws` for buffer discovery; duck-typed:
        - ws.plan.shards : list of objects with (device, b_start, b_stop)
        - ws.get(name, shard_idx) -> Tensor  (per-shard buffers, e.g., 'x', 'y')
        - ws.arena : optional device arena; if present, passed through to adapters via scratch
        - ws.maps, ws.omB (or ws.traj), ws.dcfB : optional global data; else pass via ctor

      Fallbacks:
        - If ws.plan/shards is missing, split B evenly across visible devices.
        - If maps/omB/dcfB are missing on ws, they must be passed to the constructor.

    North‑Star shapes:
      image  (B, Like..., 1, FFT...)  →  k-space (B, Like..., C, K)
      k-space (B, Like..., C, K)      →  image  (B, Like..., 1, FFT...)
    """

    axis: AxisSpec
    config: NUFFTConfig
    ws: Any
    maps: Optional[torch.Tensor] = None
    traj: Optional[torch.Tensor] = None
    dcf: Optional[torch.Tensor] = None
    compile_per_shard: bool = False
    shard_like: bool = False  # NEW: when True, split the Like extent across devices (B stays local)


    # built
    _shards: List[ShardHandle] = None

    def __post_init__(self):
        # Resolve maps/traj/dcf
        maps = self.maps if self.maps is not None else getattr(self.ws, "maps", None)
        traj = self.traj if self.traj is not None else (getattr(self.ws, "omB", None) or getattr(self.ws, "traj", None))
        dcf  = self.dcf  if self.dcf  is not None else getattr(self.ws, "dcfB", None)

        if maps is None or traj is None:
            raise ValueError("MultiDeviceNUFFT requires maps and trajectory: pass (maps,traj,dcf) or expose ws.maps and ws.omB/ws.traj")

        # Normalize to device list and shards
        # B dimension from traj: accept (B,nd,K) or (B,K,nd) or (nd,K)
        t = traj
        if t.ndim == 2:
            B_total = 1
        elif t.ndim == 3:
            B_total = int(t.shape[0])
        else:
            raise ValueError("traj must have 2 or 3 dims")

        devs = _maybe_list_devices(self.ws)
        shard_triplets = _maybe_shards(self.ws, B_total, devs)

        self._shards = []

        # Build one NUFFT per shard (move/crop data lazily per device)
        for (dev, b_start, b_stop) in shard_triplets:
            # Slice trajectory/DCF for this shard. Let single-device NUFFT normalize (B,nd,K).
            if t.ndim == 2:
                traj_i = t
                dcf_i  = dcf
            else:
                traj_i = t[b_start:b_stop]
                dcf_i  = None if dcf is None else dcf[b_start:b_stop]

            maps_i = maps.to(dev, non_blocking=True)
            traj_i = traj_i.to(dev, non_blocking=True)
            dcf_i  = None if dcf_i is None else dcf_i.to(dev, non_blocking=True)

            op = NUFFT(maps=maps_i, traj=traj_i, dcf=dcf_i,
                       axis=self.axis, config=self.config,
                       dtype=maps.dtype, device=dev)
            if self.compile_per_shard:
                try:
                    op = torch.compile(op)  # __call__ dispatches to A
                except Exception:
                    pass
            self._shards.append(ShardHandle(device=dev, b_start=b_start, b_stop=b_stop, op=op))

    # ------------------------------- execution ---------------------------------

    @_dynamo_disable
    def A(self,
          x: Optional[Union[torch.Tensor, List[torch.Tensor], Dict[Union[int, torch.device], torch.Tensor]]] = None,
          out: Optional[Union[List[torch.Tensor], Dict[Union[int, torch.device], torch.Tensor]]] = None,
          *,
          scratch: Optional[Union[List[Dict[str, torch.Tensor]], Dict[Union[int, torch.device], Dict[str, torch.Tensor]]]] = None,
          async_streams: bool = False) -> Union[List[torch.Tensor], None]:
        """
        Multi-device forward:
          - If `x` is None, try ws.get('x', i) per shard; else slice from the provided tensor by B.
          - `out` must be per-shard (list or dict keyed by device/index). If None, try ws.get('y', i).
          - `scratch` may be per-shard dict(s); passed to adapters.
          - Returns the list of per-shard outputs if `out` is None and ws has no 'y'; otherwise None.
        """
        y_list: List[torch.Tensor] = []
        launch: List[Tuple[ShardHandle, torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]] = []

        # Normalize user-provided dicts into shard order
        out_map = out if isinstance(out, dict) else None
        scr_map = scratch if isinstance(scratch, dict) else None

        for i, sh in enumerate(self._shards):
            dev, b0, b1, op = sh.device, sh.b_start, sh.b_stop, sh.op

            # Resolve input image for this shard
            x_i: Optional[torch.Tensor] = None
            if x is None:
                # Try workspace
                get = getattr(self.ws, "get", None)
                if callable(get):
                    try:
                        x_i = get("x", i)
                    except Exception:
                        x_i = None
            elif isinstance(x, list):
                x_i = x[i]
            elif isinstance(x, dict):
                key = i if i in x else (dev if dev in x else None)
                if key is not None:
                    x_i = x[key]
            elif isinstance(x, torch.Tensor):
                # Slice along B
                x_i = _b_slice(x, self.axis.image, b0, b1).to(dev, non_blocking=True)
            else:
                raise ValueError("Unsupported x type for multi-device")

            if x_i is None:
                raise ValueError(f"Missing image input for shard {i} (device={dev}). Provide x[...] or ws.get('x', {i}).")

            # Resolve output buffer
            y_i: Optional[torch.Tensor] = None
            if out_map is not None:
                key = i if i in out_map else (dev if dev in out_map else None)
                if key is not None:
                    y_i = out_map[key]
            elif isinstance(out, list):
                y_i = out[i]
            elif out is None:
                get = getattr(self.ws, "get", None)
                if callable(get):
                    try:
                        y_i = get("y", i)
                    except Exception:
                        y_i = None
            if y_i is None:
                # We'll allocate one-time for return path (caller can collect/gather)
                # NOTE: This does NOT allocate inside the single-device op; it's a user-visible tensor.
                #       For zero-alloc end-to-end, pass per-shard 'out' or use ws buffers.
                # Use op.expected_shapes if present
                K = op.k_per_frame()
                C = int(op.maps.shape[0])
                # We cannot infer Like sizes from x_i robustly without planners; trust user x_i
                like_dims = tuple(int(s) for s in x_i.shape[1:-1-self.axis.image.count('coil')])  # rough fallback
                # Safer: simply allocate same (B_shard, Like..., C, K) as x_i switching last dims
                B_shard = int(x_i.shape[self.axis.image.index('B')]) if 'B' in self.axis.image else (b1-b0)
                # We can't re-order from here robustly; delegate to single-device by not passing out
                y_i = None

            # Resolve scratch
            scr_i: Optional[Dict[str, torch.Tensor]] = None
            if scr_map is not None:
                k = i if i in scr_map else (dev if dev in scr_map else None)
                scr_i = scr_map.get(k, None) if k is not None else None
            elif isinstance(scratch, list):
                if len(scratch) > i:
                    scr_i = scratch[i]

            launch.append((sh, x_i, y_i, scr_i))

        # Execute
        if async_streams:
            # optional per-device streams via ws.arena if available
            arena = getattr(self.ws, "arena", None)
            streams: Dict[torch.device, torch.cuda.Stream] = {}
            if arena is not None:
                for sh, _, _, _ in launch:
                    s = arena.stream_for(sh.device)  # may create on demand
                    streams[sh.device] = s
            for sh, x_i, y_i, scr_i in launch:
                dev = sh.device
                s = streams.get(dev, None)
                ctx = torch.cuda.stream(s) if (s is not None and _is_cuda(dev)) else contextlib.nullcontext()
                with ctx:
                    sh.op.A(x_i, out=y_i) if scr_i is None else sh.op.A(x_i, out=y_i, scratch=scr_i)
            # Sync before returning
            for sh, _, _, _ in launch:
                _sync(sh.device)
        else:
            for sh, x_i, y_i, scr_i in launch:
                y_out = sh.op.A(x_i, out=y_i) if scr_i is None else sh.op.A(x_i, out=y_i, scratch=scr_i)
                if y_i is None:
                    y_list.append(y_out)

        return None if out is not None or hasattr(self.ws, "get") else y_list

    @_dynamo_disable
    def AH(self,
           y: Optional[Union[torch.Tensor, List[torch.Tensor], Dict[Union[int, torch.device], torch.Tensor]]] = None,
           out: Optional[Union[List[torch.Tensor], Dict[Union[int, torch.device], torch.Tensor]]] = None,
           *,
           scratch: Optional[Union[List[Dict[str, torch.Tensor]], Dict[Union[int, torch.device], Dict[str, torch.Tensor]]]] = None,
           async_streams: bool = False) -> Union[List[torch.Tensor], None]:
        """
        Multi-device adjoint:
          - If `y` is None, try ws.get('y', i); else slice by B from provided tensor.
          - `out` must be per-shard (list/dict) or omitted (then try ws.get('x', i)).
          - Returns list of per-shard images if `out` is None and ws has no 'x'.
        """
        x_list: List[torch.Tensor] = []
        out_map = out if isinstance(out, dict) else None
        scr_map = scratch if isinstance(scratch, dict) else None
        launch: List[Tuple[ShardHandle, torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]] = []

        for i, sh in enumerate(self._shards):
            dev, b0, b1, op = sh.device, sh.b_start, sh.b_stop, sh.op

            # Resolve k-space input
            y_i: Optional[torch.Tensor] = None
            if y is None:
                get = getattr(self.ws, "get", None)
                if callable(get):
                    try:
                        y_i = get("y", i)
                    except Exception:
                        y_i = None
            elif isinstance(y, list):
                y_i = y[i]
            elif isinstance(y, dict):
                key = i if i in y else (dev if dev in y else None)
                if key is not None:
                    y_i = y[key]
            elif isinstance(y, torch.Tensor):
                y_i = _b_slice(y, self.axis.kspace, b0, b1).to(dev, non_blocking=True)
            else:
                raise ValueError("Unsupported y type for multi-device")
            if y_i is None:
                raise ValueError(f"Missing k-space input for shard {i}; provide y[...] or ws.get('y', {i}).")

            # Resolve out image
            x_i: Optional[torch.Tensor] = None
            if out_map is not None:
                key = i if i in out_map else (dev if dev in out_map else None)
                if key is not None:
                    x_i = out_map[key]
            elif isinstance(out, list):
                x_i = out[i]
            elif out is None:
                get = getattr(self.ws, "get", None)
                if callable(get):
                    try:
                        x_i = get("x", i)
                    except Exception:
                        x_i = None

            # Resolve scratch
            scr_i: Optional[Dict[str, torch.Tensor]] = None
            if scr_map is not None:
                k = i if i in scr_map else (dev if dev in scr_map else None)
                scr_i = scr_map.get(k, None) if k is not None else None
            elif isinstance(scratch, list):
                if len(scratch) > i:
                    scr_i = scratch[i]

            launch.append((sh, y_i, x_i, scr_i))

        # Execute
        if async_streams:
            arena = getattr(self.ws, "arena", None)
            streams: Dict[torch.device, torch.cuda.Stream] = {}
            if arena is not None:
                for sh, _, _, _ in launch:
                    s = arena.stream_for(sh.device)
                    streams[sh.device] = s
            for sh, y_i, x_i, scr_i in launch:
                dev = sh.device
                s = streams.get(dev, None)
                ctx = torch.cuda.stream(s) if (s is not None and _is_cuda(dev)) else contextlib.nullcontext()
                with ctx:
                    sh.op.AH(y_i, out=x_i) if scr_i is None else sh.op.AH(y_i, out=x_i, scratch=scr_i)
            for sh, _, _, _ in launch:
                _sync(sh.device)
        else:
            for sh, y_i, x_i, scr_i in launch:
                x_out = sh.op.AH(y_i, out=x_i) if scr_i is None else sh.op.AH(y_i, out=x_i, scratch=scr_i)
                if x_i is None:
                    x_list.append(x_out)

        return None if out is not None or hasattr(self.ws, "get") else x_list

    # ------------------------------- utilities ---------------------------------

    def per_shard_ops(self) -> List[NUFFT]:
        return [sh.op for sh in self._shards]

    def shards(self) -> List[Tuple[torch.device, int, int]]:
        return [(sh.device, sh.b_start, sh.b_stop) for sh in self._shards]
