# -----------------------------------------------------------------------------
# graspcg/nufft/distributed.py
# Build and apply per-device NUFFT operators; shard over frames (B).
# Adds a high-level workspace function to frame, shard, and build the operator.
# -----------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import torch

# Framing helper (spokes -> frames)
# Returns (yB, omB, dcfB) with shapes:
#   yB : (B, C, K),   omB : (B, K, nd),   dcfB : (B, K) or None
from ..sliding_window import SlidingWindowSpec, sliding_window_sort  # :contentReference[oaicite:4]{index=4}

from .api import NUFFT, NUFFTConfig
from .layout import AxisSpec

# ---- Internal sharding policy (frame-wise, even partition) -------------------
def _even_slices(B: int, n: int) -> List[slice]:
    if n <= 0:
        raise ValueError("Need at least one device to shard.")
    q, r = divmod(B, n)
    out: List[slice] = []
    s = 0
    for i in range(n):
        bi = q + (1 if i < r else 0)
        out.append(slice(s, s + bi))
        s += bi
    return out


def _default_backend_for_device(dev: torch.device) -> str:
    # Prefer CUFI on CUDA, TorchKb on CPU
    return "cufi" if dev.type == "cuda" else "torchkb"


# ---- Workspace (now with high-level function) --------------------------------
@dataclass
class Workspace:
    """
    Device workspace used by DistNUFFT.
      - devices: list of torch.devices to use
      - streams: optional map device->torch.cuda.Stream for async work
    """
    devices: Sequence[torch.device]
    streams: Optional[Dict[torch.device, torch.cuda.Stream]] = None

    def get_stream(self, dev: torch.device) -> Optional[torch.cuda.Stream]:
        if self.streams is None: return None
        return self.streams.get(dev, None)

    # -------------------------------------------------------------------------
    # HIGH-LEVEL ENTRY POINT
    # -------------------------------------------------------------------------
    def frame_and_build_nufft(
        self,
        *,
        maps: torch.Tensor,
        ktraj: torch.Tensor,
        kspace: Optional[torch.Tensor] = None,
        dcf: Optional[torch.Tensor] = None,
        win: Optional[SlidingWindowSpec] = None,
        backend_overrides: Optional[Dict[torch.device, str]] = None,
        dtype: torch.dtype = torch.complex64,
        pin_host: bool = False,
        preload_device: bool = False,
    ) -> Tuple["DistNUFFT", torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        End-to-end builder:
          - If 'win' is provided, treat (kspace, ktraj, dcf) as SPOKES and frame them into (B,*,K).
          - Else, treat inputs as already FRAMED: kspace=(B,C,K) or None; ktraj=(B,K,nd) or (B,nd,K); dcf=(B,K) or None.
          - Shard frames across devices; build a distributed NUFFT operator on those shards.
          - Optionally pin host buffers and/or preload framed data to device shards.

        Returns:
          dist  : DistNUFFT distributed operator
          yB    : framed kspace  (B, C, K)        (if kspace=None, this is an empty tensor with the correct shape)
          omB   : framed ktraj   (B, K, nd)
          dcfB  : framed dcf     (B, K) or None
        """
        # ---- 1) Frame or pass-through
        # Trajectory is required; maps are required
        if win is not None:
            if kspace is None:
                raise ValueError("When 'win' is provided, 'kspace' (spokes) must be given.")
            # spokes → frames
            yB, omB, dcfB = sliding_window_sort(kspace, ktraj, dcf, win)  # :contentReference[oaicite:6]{index=6}
        else:
            # already framed; normalize om layout to (B,K,nd)
            if ktraj.ndim != 3:
                raise ValueError("ktraj must be (B,K,nd) or (B,nd,K) or (S, spp, nd) with 'win'.")
            if ktraj.shape[1] in (2, 3) and ktraj.shape[2] != ktraj.shape[1]:
                # assume (B,nd,K) -> transpose to (B,K,nd)
                omB = ktraj.transpose(1, 2).contiguous()
            else:
                omB = ktraj.contiguous()
            if (kspace is None):
                # fabricate an empty yB so shapes are discoverable
                B = int(omB.shape[0]); K = int(omB.shape[1]); C = int(maps.shape[0])
                yB = torch.empty((B, C, K), dtype=dtype, device=maps.device)
                yB.zero_()
            else:
                yB = kspace.contiguous()
            dcfB = None if (dcf is None) else dcf.contiguous()

        # ---- 2) Optional host pinning (for faster H2D)
        if pin_host and (any(d.type == "cuda" for d in self.devices)):
            if yB.is_cuda:  # already on GPU
                pass
            else:
                yB = yB.pin_memory()
            if not omB.is_cuda:
                omB = omB.pin_memory()
            if (dcfB is not None) and (not dcfB.is_cuda):
                dcfB = dcfB.pin_memory()

        # ---- 3) Build distributed operator (this shards traj/dcf internally)
        dist = DistNUFFT(maps, omB, dcfB, workspace=self,
                         backend_overrides=backend_overrides, dtype=dtype)

        # ---- 4) Optional preloading of framed data to shard devices
        if preload_device:
            for w in dist._workers:
                sl = w.frames
                dev = w.device
                # move only the slices this worker will consume
                _ = yB[sl].to(dev, non_blocking=True)
                _ = omB[sl].to(dev, non_blocking=True)
                if dcfB is not None:
                    _ = dcfB[sl].to(dev, non_blocking=True)
                # we don't keep references here on purpose; DistNUFFT will still
                # stage inputs per-call (keeps API simple and memory predictable)

        return dist, yB, omB, dcfB


# ---- Per-device worker handle -------------------------------------------------
@dataclass
class _Worker:
    device: torch.device
    backend: str
    frames: slice               # frames [start:stop] from the global B-axis
    op: NUFFT                   # per-device NUFFT prepared on traj[frames]

    @property
    def B_local(self) -> int:
        return int(self.frames.stop - self.frames.start)

    def A(self, xB: torch.Tensor, out: Optional[torch.Tensor]=None) -> torch.Tensor:
        return self.op.A(xB, out=out)

    def AH(self, yB: torch.Tensor, out: Optional[torch.Tensor]=None) -> torch.Tensor:
        return self.op.AH(yB, out=out)

    def k_per_frame(self) -> int:
        return int(self.op.k_per_frame())


# =============================================================================
# Distributed NUFFT over multiple devices (frame-sharded)
# =============================================================================
class DistNUFFT:
    """
    Distributed NUFFT wrapper that shards frames (B) across devices, builds
    per-device NSNUFFT operators once, and applies them in parallel.

    Forward A(x): x  (B, C, H, W[,D]) → y (B, C, K)
    Adjoint AH(y): y (B, C, K)        → x (B, 1, H, W[,D])
    """
    def __init__(
        self,
        maps: torch.Tensor,
        traj: torch.Tensor,
        dcf: Optional[torch.Tensor],
        *,
        workspace: Workspace,
        backend_overrides: Optional[Dict[torch.device, str]] = None,
        dtype: torch.dtype = torch.complex64,
    ):
        if maps.ndim not in (3, 4):
            raise ValueError("maps must be (C,H,W) or (C,H,W,D)")
        if traj.ndim != 3:
            raise ValueError("traj must be (B,nd,K) or (B,K,nd)")

        self.maps = maps
        self.traj = traj
        self.dcf = dcf
        self.dtype = dtype

        self.workspace = workspace
        self._devices = list(workspace.devices)
        self._workers: List[_Worker] = []

        # Normalize traj to (B, nd, K) for NSNUFFT; NSNUFFT/adapters accept both and normalize internally. :contentReference[oaicite:7]{index=7}
        if traj.shape[1] in (2, 3) and traj.shape[2] != traj.shape[1]:
            traj_BndK = traj
        else:
            traj_BndK = traj.transpose(1, 2).contiguous()

        # Shard B across devices
        B = int(traj_BndK.shape[0])
        slices = _even_slices(B, len(self._devices))

        # Build a per-device NSNUFFT on each shard
        for dev, sl in zip(self._devices, slices):
            backend = (backend_overrides or {}).get(dev, _default_backend_for_device(dev))
            traj_i = traj_BndK[sl].contiguous()
            dcf_i  = None if (dcf is None) else (dcf[sl] if dcf.ndim == 2 else dcf)

            maps_i = maps.to(device=dev, dtype=dtype, non_blocking=True).contiguous()
            traj_i = traj_i.to(device=dev, dtype=torch.float32, non_blocking=True).contiguous()
            dcf_i  = None if (dcf_i is None) else dcf_i.to(device=dev, dtype=torch.float32, non_blocking=True).contiguous()

            op = NSNUFFT(
                maps=maps_i, traj=traj_i, dcf=dcf_i,
                backend=backend, dtype_c=dtype, device=dev
            )  # builds TorchKb or CUFI backend plans under the hood  :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}
            self._workers.append(_Worker(device=dev, backend=backend, frames=sl, op=op))

        self._K = int(self._workers[0].k_per_frame()) if self._workers else 0
        self._spatial = tuple(int(s) for s in maps.shape[1:])

    # ------------------------------- properties --------------------------------
    def k_per_frame(self) -> int:
        return int(self._K)

    def devices(self) -> Sequence[torch.device]:
        return tuple(w.device for w in self._workers)

    def spatial(self) -> Tuple[int, ...]:
        return self._spatial

    # ---------------------------- distributed apply -----------------------------
    @torch.no_grad()
    def A(self, x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:   (B, C, H, W[,D]) on any device
        out: (B, C, K)
        """
        B = int(x.shape[0]); C = int(x.shape[1]); K = self.k_per_frame()
        # Allocate/validate out
        if out is None:
            out = torch.empty((B, C, K), dtype=x.dtype, device=x.device)
        else:
            if tuple(out.shape) != (B, C, K):
                raise ValueError(f"out has shape {tuple(out.shape)} but expected {(B,C,K)}")

        # Launch per device
        pending: List[Tuple[_Worker, torch.Tensor, Optional[torch.cuda.Stream]]] = []
        for w in self._workers:
            sl = w.frames
            xB = x[sl]
            stream = self.workspace.get_stream(w.device)
            if w.device.type == "cuda" and stream is not None:
                with torch.cuda.stream(stream):
                    xB_dev = xB.to(w.device, non_blocking=True)
                    yB = w.A(xB_dev)
                pending.append((w, yB, stream))
            else:
                xB_dev = xB.to(w.device, non_blocking=True)
                yB = w.A(xB_dev)
                out[sl].copy_(yB.to(out.device, non_blocking=True))
        for w, yB, stream in pending:
            stream.synchronize()
            out[w.frames].copy_(yB.to(out.device, non_blocking=True))
        return out

    @torch.no_grad()
    def AH(self, y: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        y:   (B, C, K)
        out: (B, 1, H, W[,D])
        """
        B = int(y.shape[0]); HWD = self.spatial()
        if out is None:
            out = torch.empty((B, 1, *HWD), dtype=y.dtype, device=y.device)
        else:
            if tuple(out.shape) != (B, 1, *HWD):
                raise ValueError(f"out has shape {tuple(out.shape)} but expected {(B,1,*HWD)}")

        pending: List[Tuple[_Worker, torch.Tensor, Optional[torch.cuda.Stream]]] = []
        for w in self._workers:
            sl = w.frames
            yB = y[sl]
            stream = self.workspace.get_stream(w.device)
            if w.device.type == "cuda" and stream is not None:
                with torch.cuda.stream(stream):
                    yB_dev = yB.to(w.device, non_blocking=True)
                    xB = w.AH(yB_dev)
                pending.append((w, xB, stream))
            else:
                yB_dev = yB.to(w.device, non_blocking=True)
                xB = w.AH(yB_dev)
                out[sl].copy_(xB.to(out.device, non_blocking=True))
        for w, xB, stream in pending:
            stream.synchronize()
            out[w.frames].copy_(xB.to(out.device, non_blocking=True))
        return out

    # ------------------------------ calibration ---------------------------------
    @torch.no_grad()
    def diag_AHA_profile(self) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        for w in self._workers:
            if hasattr(w.op._impl, "diag_AHA_profile"):
                parts.append(w.op._impl.diag_AHA_profile().to("cpu"))
            else:
                Bi = w.B_local
                parts.append(torch.full((Bi,), float(self._K), dtype=torch.float32))
        return torch.cat(parts, dim=0)

    @torch.no_grad()
    def diag_AHA_scalar(self) -> float:
        vals: List[float] = []
        for w in self._workers:
            if hasattr(w.op._impl, "diag_AHA_scalar"):
                vals.append(float(w.op._impl.diag_AHA_scalar()))
        if not vals:
            return float(self._K)
        vals_t = torch.tensor(vals, dtype=torch.float32)
        return float(vals_t.median().item())
