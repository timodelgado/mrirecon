# -----------------------------------------------------------------------------
# graspcg/nufft/factory.py
# -----------------------------------------------------------------------------
from __future__ import annotations
from typing import Optional, Tuple
import torch
from .specs import ProblemSpec, DistributionPlan
from .api import NUFFT, NUFFTConfig           # new layout-aware front-end
from .layout import AxisSpec
from .distributed import DistNUFFT

def build(spec: ProblemSpec, plan: DistributionPlan, ws) -> object:
    """
    Returns a NUFFT operator built from the workspace data & plan:
      - DistNUFFT if >1 shard, else single-device NUFFT
    """
    maps, omB, dcfB = (ws.maps, ws.omB, ws.dcfB)
    assert maps is not None and omB is not None

    # AxisSpec is required by the new front-end; use the provided one.
    if spec.axis is None:
        raise ValueError("ProblemSpec.axis must be provided for the new NUFFT front-end.")

    # Build a base config; per-shard backend can override 'backend'.
    base_cfg = NUFFTConfig(
        ndim=spec.ndim,
        backend='torchkb' if spec.default_backend == 'auto' else spec.default_backend,  # device overrides happen below
        traj_units=spec.traj_units,
    )

    # Single device short-circuit
    if len(plan.shards) == 1:
        sh = plan.shards[0]
        dev = sh.device
        maps_i = maps.to(dev)
        traj_i = _to_BndK(omB[sh.frames])            # accept (B,K,nd) or (B,nd,K)
        dcf_i  = None if dcfB is None else dcfB[sh.frames].to(dev)
        cfg_i = NUFFTConfig(
            ndim=base_cfg.ndim,
            backend=sh.backend,
            traj_units=base_cfg.traj_units,
        )
        return NUFFT(maps=maps_i, traj=traj_i, dcf=dcf_i,
                     axis=spec.axis, config=cfg_i, dtype=maps.dtype, device=dev)

    # Multi-device composition: pass axis + base config; per-device backend overrides handled inside DistNUFFT
    return DistNUFFT(maps, omB, dcfB,
                     workspace=ws,
                     backend_overrides={s.device: s.backend for s in plan.shards},
                     axis=spec.axis,
                     config=base_cfg,
                     dtype=maps.dtype)

def _to_BndK(omB: torch.Tensor) -> torch.Tensor:
    # Keep (B,K,nd) if already; otherwise (B,nd,K) -> transpose
    if omB.ndim != 3:
        raise ValueError("trajectory must be (B,K,nd) or (B,nd,K)")
    if omB.shape[1] in (2,3) and omB.shape[2] != omB.shape[1]:
        return omB.transpose(1,2).contiguous()
    return omB

def infer_problem_semantics(axis, maps, omB):
    # Always flat K
    assert axis.kspace_fft == ('K',), "kspace_fft must be ('K',)"
    # Determine nd from image FFT labels
    if any(lbl in axis.image_fft for lbl in ('Z','z','Pz')):   # user may alias Z
        nd = 3
        like_L = 1
        like_label = None
    else:
        nd = 2
        # If a partitions-like label exists in image but not in image_fft → 2D multi-slice
        like_label = next((lbl for lbl in axis.image if lbl.upper() in ('P','SLICE','SLICES') and lbl not in axis.image_fft), None)
        like_L = maps.shape[axis.image.index(like_label)] if like_label else 1
    return nd, like_label, like_L
