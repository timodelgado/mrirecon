from __future__ import annotations

import math
import time
import contextlib
from dataclasses import dataclass, field
from typing import Optional, Iterator, Tuple, List, Dict, Literal, Sequence, Any
from collections import defaultdict

import torch
try:
    import psutil  # optional; used to estimate CPU RAM if needed
except Exception:
    psutil = None  # type: ignore

from .unified_arena import DeviceArena


# ==============================
# Declarative buffer manifest
# ==============================
Role       = Literal["image", "kspace", "scalar"]
Layout     = Literal["per_shard", "global"]
ShapeRef   = Literal["image", "kspace", "spatial", "scalar"]
Lifetime   = Literal["iter", "ls", "accept", "precond"]  # rough phases for peak memory modeling


@dataclass(frozen=True)
class BufSpec:
    """
    Solver-declared buffer spec.

    name     : unique name for lookup, e.g. "x", "g", "dx", "diag", "Ax_sh", "Ad_sh", "r_sh"
    role     : informational ("image" | "kspace" | "scalar")
    layout   : "per_shard" or "global"
    shape    : "image" (B, ...), "kspace" (B, ... like y), "spatial" (...), "scalar" (), or explicit tuple
    dtype    : torch.dtype
    persist  : keep across iterations (True) vs ephemeral (False) – still allocated here
    init     : "zeros" | "ones" | None
    co_shard : label; per_shard buffers with the same label are sharded together (same frames & device)
               (default: "frame" for per_shard, None for global)
    co_locate: for globals – place on the same device as this *thing* ("y" or a device string like "cuda:0")
    lifetime : rough phase when the buffer is live ("iter","ls","accept","precond")
    """
    name: str
    role: Role
    layout: Layout
    shape: ShapeRef | Tuple[int, ...]
    dtype: torch.dtype
    persist: bool = True
    init: Optional[str] = None
    co_shard: Optional[str] = None
    co_locate: Optional[str] = None
    lifetime: Lifetime = "iter"

    def normalized_co_shard(self) -> Optional[str]:
        if self.layout == "per_shard":
            return self.co_shard or "frame"
        return None


# ==============================
# Domain role descriptors
# ==============================
@dataclass(frozen=True)
class Roles:
    """
    Number of axes in each category for a given domain.
    Tensors must be ordered (unlike, like, nufft).
    Example (2D multi-slice, k-space): unlike=1 (frames), like=2 (coils,z), nufft=2 (views,readout)
             (2D multi-slice, image): unlike=1 (frames), like=1 (z),       nufft=2 (x,y)
    """
    unlike: int
    like:   int
    nufft:  int

    @property
    def ndim(self) -> int:
        return int(self.unlike + self.like + self.nufft)


# ==============================
# Shards & planning
# ==============================
@dataclass
class Shard:
    device: torch.device
    b_start: int
    b_stop: int
    image_shape: Tuple[int, ...]  # full image shape (B_total, ...)

    @property
    def span(self) -> int:
        return self.b_stop - self.b_start

    @property
    def shape(self) -> Tuple[int, ...]:
        return (self.span,) + self.image_shape[1:]


@dataclass
class WorkspacePlan:
    image_shape: Tuple[int, ...]
    kspace_shape: Tuple[int, ...]
    shards: List[Shard]
    dtype_c: torch.dtype
    dtype_r: torch.dtype
    kspace_mode: Literal["sharded", "global"] = "sharded"
    # telemetry
    tpf_by_device: Dict[int, float] = field(default_factory=dict)  # seconds per frame per device
    b_by_device  : Dict[int, int]   = field(default_factory=dict)  # frames assigned per device
    # roles (axis counts; tensors must be ordered (unlike, like, nufft))
    roles_image: Roles = field(default_factory=lambda: Roles(unlike=1, like=0, nufft=2))
    roles_kspace: Roles = field(default_factory=lambda: Roles(unlike=1, like=0, nufft=2))


# ==============================
# Workspace core
# ==============================
class Workspace:
    """
    Solver-agnostic workspace that:
      1) infers image/k-space shapes,
      2) plans shards across available devices (throughput & capacity aware),
      3) allocates buffers from a *mandatory* manifest,
      4) exposes shard iteration and buffer lookups.

    No implicit per-shard attributes exist; solvers and objectives
    must use ws.get("name", shard_idx) to access their buffers.
    """

    # ---------- construction ----------
    def __init__(self,
                 y: torch.Tensor,
                 nufft_op: Any,
                 arena: DeviceArena,
                 *,
                 buf_specs: Sequence[BufSpec],
                 image_shape: Optional[Tuple[int, ...]] = None,
                 dtype_c: torch.dtype = torch.complex64,
                 dtype_r: torch.dtype = torch.float32,
                 kspace_mode: Literal["sharded", "global"] = "sharded",
                 headroom: float = 0.20,
                 benchmark: bool = True,
                 bench_frames: int = 2,
                 comm_bandwidth_GBps: float = 12.0,   # for global k-space
                 comm_overhead_us: float = 30.0):
        assert len(buf_specs) > 0, "Workspace requires a non-empty manifest (buf_specs)."

        self.y         = y
        self.nufft_op  = nufft_op
        self.arena     = arena
        self.dtype_c   = dtype_c
        self.dtype_r   = dtype_r
        self.kspace_mode = kspace_mode

        # Manifest defaults
        self.specs = list(buf_specs)
        self._normalize_specs_inplace()

        # Build plan
        self.plan = self.infer_plan(
            nufft_op, y, arena, self.specs,
            image_shape=image_shape,
            dtype_c=dtype_c, dtype_r=dtype_r,
            kspace_mode=kspace_mode,
            headroom=headroom,
            benchmark=benchmark,
            bench_frames=bench_frames,
            comm_bandwidth_GBps=comm_bandwidth_GBps,
            comm_overhead_us=comm_overhead_us,
        )

        # registry: name -> tensor (global) OR list[tensor] (per_shard)
        self._bufs: Dict[str, object] = {}

        # allocate per the manifest
        self._allocate_from(self.specs)

    # ---------- manifest helpers ----------
    def _normalize_specs_inplace(self) -> None:
        """Set defaults for co_shard; basic validation."""
        names = set()
        for i, s in enumerate(self.specs):
            if s.name in names:
                raise ValueError(f"Duplicate buffer name in manifest: {s.name}")
            names.add(s.name)
            if s.layout == "per_shard" and s.co_shard is None:
                self.specs[i] = BufSpec(**{**s.__dict__, "co_shard": "frame"})  # type: ignore[arg-type]

    # ---------- planning ----------
    @staticmethod
    @torch.no_grad()
    def _infer_image_shape(nufft_op, y: torch.Tensor,
                           *, dtype_c=torch.complex64) -> Tuple[int, ...]:
        # Prefer operator hints
        if hasattr(nufft_op, "image_shape") and callable(getattr(nufft_op, "image_shape")):
            shp = nufft_op.image_shape(y)
            if isinstance(shp, (tuple, list)):
                return tuple(int(s) for s in shp)
        if hasattr(nufft_op, "domain_shape") and callable(getattr(nufft_op, "domain_shape")):
            shp = nufft_op.domain_shape()
            if isinstance(shp, (tuple, list)):
                shp = tuple(int(s) for s in shp)
                return shp if len(shp) == len(y.shape) else (int(y.shape[0]),) + shp

        # Fallback: probe one frame with AH
        y0 = y[:1]
        x0 = nufft_op.AH(y0)
        return (int(y.shape[0]),) + tuple(int(s) for s in x0.shape[1:])

    @staticmethod
    def _elem_size(dt: torch.dtype) -> int:
        return torch.empty((), dtype=dt).element_size()

    @staticmethod
    def _prod(shape: Sequence[int]) -> int:
        out = 1
        for s in shape:
            out *= int(s)
        return out

    @classmethod
    def _elements_per_frame(cls, spec: BufSpec, image_shape: Tuple[int, ...], kspace_shape: Tuple[int, ...]) -> int:
        """Elements that scale linearly with frames (B)."""
        if spec.shape == "image":
            return cls._prod(image_shape[1:])  # one frame slice
        if spec.shape == "kspace":
            return cls._prod(kspace_shape[1:])
        return 0

    @classmethod
    def _elements_const(cls, spec: BufSpec, image_shape: Tuple[int, ...], kspace_shape: Tuple[int, ...]) -> int:
        """Elements that are constant per shard (do not scale with B)."""
        if spec.shape == "spatial":
            return cls._prod(image_shape[1:])
        if isinstance(spec.shape, tuple):
            return cls._prod(spec.shape)
        return 0

    @classmethod
    def _bytes_per_frame_and_const(
        cls, specs: Sequence[BufSpec],
        image_shape: Tuple[int, ...],
        kspace_shape: Tuple[int, ...],
        lifetimes: Sequence[Lifetime] = ("iter","ls","accept","precond"),
    ) -> Tuple[int, int, int]:
        """
        Estimate peak memory: returns (per_frame_bytes_peak, const_bytes_peak, kspace_bytes_per_frame).
        """
        pf_by_life: Dict[Lifetime, int] = defaultdict(int)
        c_by_life : Dict[Lifetime, int] = defaultdict(int)
        kpf = 0  # kspace bytes per frame (used for comm pricing)

        for s in specs:
            es = cls._elem_size(s.dtype)
            pf = cls._elements_per_frame(s, image_shape, kspace_shape) * es
            c  = cls._elements_const(s,  image_shape, kspace_shape) * es

            pf_by_life[s.lifetime] += pf if s.layout == "per_shard" else 0
            c_by_life[s.lifetime]  += c if s.layout == "per_shard" else 0

            if s.role == "kspace":
                kpf = max(kpf, cls._prod(kspace_shape[1:]) * es)  # one frame worth

        per_frame_peak = max((pf_by_life.get(l, 0) for l in lifetimes), default=0)
        const_peak     = max((c_by_life.get(l, 0) for l in lifetimes), default=0)
        return int(per_frame_peak), int(const_peak), int(kpf)

    @classmethod
    @torch.no_grad()
    def _bench_tpf(
        cls, nufft_op, image_shape: Tuple[int, ...], kspace_shape: Tuple[int, ...],
        device: torch.device, dtype_c: torch.dtype, dtype_k: torch.dtype,
        frames: int = 2
    ) -> float:
        """
        Measure seconds-per-frame for A and AH on `device` with a tiny slice.
        """
        B_probe = max(1, min(frames, int(kspace_shape[0])))
        x = torch.zeros((B_probe, *image_shape[1:]), dtype=dtype_c, device=device)
        y = torch.zeros((B_probe, *kspace_shape[1:]), dtype=dtype_k, device=device)

        y_out = torch.empty_like(y)
        x_out = torch.empty_like(x)

        def sync():
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        for _ in range(2):
            nufft_op.A(x, out=y_out)
            nufft_op.AH(y, out=x_out)
            sync()

        iters = 4
        if device.type == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                nufft_op.A(x, out=y_out); nufft_op.AH(y, out=x_out)
            end.record(); end.synchronize()
            ms = start.elapsed_time(end)
            total_s = (ms / 1e3)
        else:
            t0 = time.perf_counter()
            for _ in range(iters):
                nufft_op.A(x, out=y_out); nufft_op.AH(y, out=x_out)
            total_s = time.perf_counter() - t0

        per_iter_s = total_s / iters
        tpf = per_iter_s / float(B_probe)
        return float(max(tpf, 1e-6))

    @classmethod
    def _solve_assignment(
        cls,
        B_total: int,
        free_bytes_by_dev: Dict[int, int],
        bytes_per_frame: int,
        const_bytes: int,
        tpf_by_dev: Dict[int, float],
        kspace_bytes_per_frame: int,
        kspace_mode: Literal["sharded", "global"],
        comm_bandwidth_GBps: float,
        comm_overhead_us: float,
    ) -> Dict[int, int]:
        """
        Return frames per device (device index -> frames), minimizing max time under capacities.
        """
        dev_ids = sorted(free_bytes_by_dev.keys())

        # capacities
        cap: Dict[int, int] = {}
        for i in dev_ids:
            usable = max(0, free_bytes_by_dev[i] - const_bytes)
            if bytes_per_frame <= 0:
                cap[i] = B_total
            else:
                cap[i] = max(0, usable // max(1, bytes_per_frame))

        # effective tpf (add comm per frame for 'global')
        tpf_eff: Dict[int, float] = {}
        comm_pf = 0.0
        if kspace_mode == "global" and kspace_bytes_per_frame > 0:
            BW = max(1e-6, comm_bandwidth_GBps) * (1024 ** 3)  # bytes/sec
            comm_pf = kspace_bytes_per_frame / BW + (comm_overhead_us * 1e-6)
        for i in dev_ids:
            tpf_eff[i] = float(tpf_by_dev[i] + comm_pf)

        if len(dev_ids) == 0:
            return {}
        t_min = 0.0
        t_max = max(tpf_eff.values()) * max(1, B_total) * 2.0
        for _ in range(40):
            τ = 0.5 * (t_min + t_max)
            total = 0
            for i in dev_ids:
                if tpf_eff[i] <= 0:
                    total += cap[i]
                else:
                    total += min(cap[i], int(τ / tpf_eff[i]))
            if total >= B_total:
                t_max = τ
            else:
                t_min = τ

        b_by_dev: Dict[int, int] = {}
        total = 0
        for i in dev_ids:
            frames = min(cap[i], int(t_max / max(tpf_eff[i], 1e-9)))
            b_by_dev[i] = frames
            total += frames

        while total > B_total:
            i = max((i for i in dev_ids if b_by_dev[i] > 0), key=lambda j: tpf_eff[j], default=None)
            if i is None:
                break
            b_by_dev[i] -= 1
            total -= 1
        while total < B_total:
            i = min((i for i in dev_ids if b_by_dev[i] < cap[i]), key=lambda j: tpf_eff[j], default=None)
            if i is None:
                break
            b_by_dev[i] += 1
            total += 1

        return b_by_dev

    @classmethod
    def infer_plan(
        cls,
        nufft_op: Any,
        y: torch.Tensor,
        arena: DeviceArena,
        specs: Sequence[BufSpec],
        *,
        image_shape: Optional[Tuple[int, ...]] = None,
        dtype_c: torch.dtype = torch.complex64,
        dtype_r: torch.dtype = torch.float32,
        kspace_mode: Literal["sharded", "global"] = "sharded",
        headroom: float = 0.20,
        benchmark: bool = True,
        bench_frames: int = 2,
        comm_bandwidth_GBps: float = 12.0,
        comm_overhead_us: float = 30.0,
    ) -> WorkspacePlan:
        assert y.dim() >= 1, "Expected leading dimension to be frames/batch"
        B_total = int(y.shape[0])
        img_shape = image_shape or cls._infer_image_shape(nufft_op, y, dtype_c=dtype_c)
        k_shape   = tuple(int(s) for s in y.shape)

        # ---- Fetch roles from NUFFT and validate canonical order -----------
        def _coerce_roles(obj, default: Roles) -> Roles:
            if isinstance(obj, Roles): return obj
            if isinstance(obj, (tuple, list)) and len(obj) == 3:
                return Roles(int(obj[0]), int(obj[1]), int(obj[2]))
            if isinstance(obj, dict):
                return Roles(int(obj.get("unlike", default.unlike)),
                             int(obj.get("like",   default.like)),
                             int(obj.get("nufft",  default.nufft)))
            return default

        def _roles_from_nufft(op) -> Tuple[Roles, Roles]:
            img, ksp = getattr(op, "roles_image", None), getattr(op, "roles_kspace", None)
            if img is None or ksp is None:
                roles_fn = getattr(op, "roles", None)
                if callable(roles_fn):
                    out = roles_fn()
                    if isinstance(out, dict):
                        img = out.get("image", img); ksp = out.get("kspace", ksp)
                    elif isinstance(out, (tuple, list)) and len(out) == 2:
                        img, ksp = out
            default_img = Roles(unlike=1, like=0, nufft=2)
            default_ksp = Roles(unlike=1, like=0, nufft=2)
            return _coerce_roles(img, default_img), _coerce_roles(ksp, default_ksp)

        roles_image, roles_kspace = _roles_from_nufft(nufft_op)

        def _check_roles(shape: Tuple[int, ...], roles: Roles, domain: str):
            if len(shape) != roles.ndim:
                raise ValueError(
                    f"{domain} shape {shape} has {len(shape)} dims but roles specify {roles.ndim} "
                    f"(unlike={roles.unlike}, like={roles.like}, nufft={roles.nufft}). "
                    "Expected tensors ordered as (unlike, like, nufft)."
                )
        _check_roles(img_shape, roles_image,  "image")
        _check_roles(k_shape,   roles_kspace, "k-space")

        # Memory model (peak) derived from manifest
        bytes_per_frame, const_bytes, kspace_bytes_pf = cls._bytes_per_frame_and_const(specs, img_shape, k_shape)

        # Devices
        devs = arena.cuda_devices()
        if not devs:
            devs = [y.device]

        # Free bytes per device (apply headroom)
        free_bytes_by_dev: Dict[int, int] = {}
        for d in devs:
            if d.type == "cuda":
                free_b, _ = torch.cuda.mem_get_info(d)
                free_bytes_by_dev[int(d.index)] = int(float(free_b) * (1.0 - headroom))
            else:
                if psutil is not None:
                    free_bytes_by_dev[-1] = int(psutil.virtual_memory().available * (1.0 - headroom))
                else:
                    free_bytes_by_dev[-1] = int(2e9)

        # Time-per-frame per device
        tpf_by_dev: Dict[int, float] = {}
        for d in devs:
            idx = int(d.index) if d.type == "cuda" else -1
            if benchmark:
                with (torch.cuda.device(d) if d.type == "cuda" else contextlib.nullcontext()):
                    tpf = cls._bench_tpf(nufft_op, img_shape, k_shape, d, dtype_c, y.dtype, frames=bench_frames)
            else:
                tpf = 1.0
            tpf_by_dev[idx] = float(max(1e-6, tpf))

        # Assignment
        b_by_dev = cls._solve_assignment(
            B_total=B_total,
            free_bytes_by_dev=free_bytes_by_dev,
            bytes_per_frame=bytes_per_frame,
            const_bytes=const_bytes,
            tpf_by_dev=tpf_by_dev,
            kspace_bytes_per_frame=kspace_bytes_pf,
            kspace_mode=kspace_mode,
            comm_bandwidth_GBps=comm_bandwidth_GBps,
            comm_overhead_us=comm_overhead_us,
        )

        # Build contiguous shard ranges per device
        shards: List[Shard] = []
        b_cursor = 0
        for d in devs:
            idx = int(d.index) if d.type == "cuda" else -1
            frames = int(b_by_dev.get(idx, 0))
            if frames <= 0:
                continue
            b_start = b_cursor
            b_stop  = min(B_total, b_start + frames)
            if b_stop > b_start:
                shards.append(Shard(device=d, b_start=b_start, b_stop=b_stop, image_shape=img_shape))
                b_cursor = b_stop

        if b_cursor < B_total:
            if not shards:
                shards.append(Shard(device=devs[0], b_start=0, b_stop=B_total, image_shape=img_shape))
            else:
                shards[-1].b_stop = B_total

        plan = WorkspacePlan(
            image_shape=img_shape,
            kspace_shape=k_shape,
            shards=shards,
            dtype_c=dtype_c,
            dtype_r=dtype_r,
            kspace_mode=kspace_mode,
            tpf_by_device=tpf_by_dev,
            b_by_device=b_by_dev,
            roles_image=roles_image,
            roles_kspace=roles_kspace,
        )
        return plan

    # ---------- allocation ----------
    @torch.no_grad()
    def _resolve_shape(self, spec: BufSpec, sh: Optional[Shard]) -> Tuple[int, ...]:
        if spec.layout == "per_shard":
            assert sh is not None, "per_shard shape requires a shard"
        if spec.shape == "image":
            if sh is not None:
                return (sh.span,) + self.plan.image_shape[1:]
            return self.plan.image_shape
        if spec.shape == "kspace":
            if sh is not None:
                return (sh.span,) + self.plan.kspace_shape[1:]
            return self.plan.kspace_shape
        if spec.shape == "spatial":
            return self.plan.image_shape[1:]
        if spec.shape == "scalar":
            return ()
        if isinstance(spec.shape, tuple):
            return tuple(int(s) for s in spec.shape)
        raise ValueError(f"Unknown shape ref: {spec.shape}")

    @torch.no_grad()
    def _allocate_from(self, specs: Sequence[BufSpec]) -> None:
        # Per-shard
        for spec in specs:
            if spec.layout == "per_shard":
                bufs: List[torch.Tensor] = []
                for sh in self.plan.shards:
                    shape = self._resolve_shape(spec, sh)
                    n = int(torch.tensor(shape).prod().item()) if len(shape) else 1
                    t = self.arena.request(n, spec.dtype, device=sh.device).reshape(shape)
                    if spec.init == "zeros": t.zero_()
                    elif spec.init == "ones": t.fill_(1)
                    bufs.append(t)
                self._bufs[spec.name] = bufs

        # Globals
        for spec in specs:
            if spec.layout != "global":
                continue
            shape = self._resolve_shape(spec, None)
            n = int(torch.tensor(shape).prod().item()) if len(shape) else 1

            dev = self.plan.shards[0].device
            if spec.co_locate:
                cl = spec.co_locate.lower()
                if cl == "y":
                    dev = self.y.device
                elif cl.startswith("cuda"):
                    dev = torch.device(cl)
                elif cl == "cpu":
                    dev = torch.device("cpu")

            t = self.arena.request(n, spec.dtype, device=dev).reshape(shape)
            if spec.init == "zeros": t.zero_()
            elif spec.init == "ones": t.fill_(1)
            self._bufs[spec.name] = t

    # ---------- iteration ----------
    def iter_shards(self) -> Iterator[Tuple[Shard, int]]:
        for i, sh in enumerate(self.plan.shards):
            yield sh, i

    # ---------- lookups ----------
    def get(self, name: str, shard_idx: Optional[int] = None) -> torch.Tensor:
        obj = self._bufs[name]
        if shard_idx is None:
            if isinstance(obj, list):
                raise ValueError(f"Buffer '{name}' is per_shard; provide shard_idx")
            return obj
        else:
            if not isinstance(obj, list):
                raise ValueError(f"Buffer '{name}' is global; omit shard_idx")
            return obj[shard_idx]

    def list_bufs(self) -> List[str]:
        return list(self._bufs.keys())

    def bind(self, shard_idx: int, *names: str):
        """Return a tuple of named per-shard buffers for this shard index."""
        return tuple(self.get(n, shard_idx) for n in names)

    def concat(self, name: str) -> torch.Tensor:
        obj = self._bufs[name]
        if not isinstance(obj, list):
            return obj
        return torch.cat(obj, dim=0)