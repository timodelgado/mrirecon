from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type
import contextlib
import torch

from .base import Regularizer, RegContext, AxesSpec
from ..core.roles import Roles
from .mapping import MappedRegularizer, IdentityOp

@dataclass
class RegSpec:
    kind: str
    name: str
    params: Mapping[str, Any]


class RegManager:
    """
    Regularizer orchestration (agnostic to reg kinds and mappings).

    • Builds RegContext per shard (halo-aware, compile-friendly).
    • Accumulates energies; collects optional probes (quantile/MAD) via reg hooks.
    • Adds diagonal majorizers with a robust 3-tier path:
        (1) axis-profile, (2) image-dependent add_diag, (3) scalar/1-D fallback.
    • Buffer names: primary 'var/grad/diag' with legacy fallback 'x/g/diag'.
    • Stats: writes both generic keys and legacy equivalents.
    """
    def __init__(self, regs: Sequence[Regularizer], *, compile_kernels: bool = True):
        self._regs: List[Regularizer] = list(regs)
        self._compile_kernels = bool(compile_kernels)
        self._kernel_cache: Dict[Tuple[Any, ...], Callable[[RegContext], torch.Tensor]] = {}

    # ---------------- Main API ----------------

    @torch.no_grad()
    def energy_and_grad(self, ws) -> torch.Tensor:
        roles = ws.plan.roles_image
        halo  = self._aggregate_halo(roles)
        n_shards = self._num_shards(ws)
        sb = getattr(ws, "stats", None)

        # Resolve axes once per reg
        reg_axes: List[Tuple[int, ...]] = []
        for reg in self._regs:
            axes_spec = getattr(getattr(reg, "params", None), "axes", "spatial")
            reg_axes.append(self._resolve_axes(axes_spec, roles))

        sb = getattr(ws, "stats", None)

        # Unified toggles (new) + tolerate legacy ones
        collect_regE  = bool(sb and (sb.enabled.get("term_reg_energy", False) or sb.enabled.get("reg_energy", False)))
        collect_q     = bool(sb and (sb.enabled.get("probe_quantile",   False) or sb.enabled.get("tv_quantile", False)))
        collect_mad   = bool(sb and sb.enabled.get("probe_mad", False))

        q_p = float(getattr(sb, "probe_percentile", 0.90)) if sb is not None else 0.90
        K_s = max(1, int(getattr(sb, "probe_sample_K", 4096))) if sb is not None else 4096


        dev_accum: Dict[torch.device, torch.Tensor] = {}
        dev_order: List[torch.device] = []
        samples_by_reg: Dict[str, List[torch.Tensor]] = {}

        for sh, i in ws.iter_shards():
            # buffers with fallback
            x = self._get(ws, i, "var", "x")
            g = self._get(ws, i, "grad", "g")
            D = self._get(ws, i, "diag", "diag") if self._has(ws, "diag") else None

            dev = x.device
            if dev not in dev_accum:
                dev_accum[dev] = torch.zeros((), device=dev, dtype=x.real.dtype)
                dev_order.append(dev)

            stream = ws.arena.stream_for(dev) if dev.type == "cuda" else None
            ctx_mgr = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
            with ctx_mgr:
                # halo-extended read (x_ext), interior write slice for g/diag
                x_ext, interior = self._with_halo_x(ws, i, n_shards, halo, anchor=x)

                for reg, axes in zip(self._regs, reg_axes):
                    ctx = RegContext(
                        x=x_ext, g=g, diag=D, roles_image=roles, device=dev,
                        dtype_c=x.dtype, dtype_r=g.real.dtype,
                        axes_resolver=lambda spec: self._resolve_axes(spec, roles),
                        arena=ws.arena, write_interior_slice=interior,
                        ws=ws, shard_index=i, halo_map=halo,
                    )

                    fn = self._compiled_fixed_axes(reg, axes, self._kernel_signature_axes(reg, axes, ctx))
                    e_shard = fn(ctx)
                    dev_accum[dev].add_(e_shard)

                    if collect_regE and sb is not None:
                        sb.scalar_slot(f"term/reg/{reg.name}/energy", dev, e_shard.dtype).add_(e_shard)

                    if (collect_q or collect_mad):
                        sample_fn = getattr(reg, "quantile_sample_shard", None)
                        if callable(sample_fn):
                            K_shard = max(1, (K_s + n_shards - 1) // max(1, n_shards))
                            smp = sample_fn(ctx, axes, K_shard, q_p)
                            if (smp is not None) and int(smp.numel()) > 0:
                                samples_by_reg.setdefault(reg.name, []).append(smp)


        # ---- total (across devices) ----
        if not dev_order:
            return torch.zeros((), device="cpu", dtype=torch.float32)
        primary = dev_order[0]
        dtype_r = next(iter(dev_accum.values())).dtype
        total = torch.zeros((), device=primary, dtype=dtype_r)
        for dev in dev_order:
            v = dev_accum[dev]
            total.add_(v if dev == primary else v.to(primary, non_blocking=True))

        if sb is not None:
            # total reg energy (new + legacy for tolerance)
            sb.scalar_slot("term/reg/total/energy", primary, dtype_r).add_(total)
            sb.scalar_slot("E_reg_total",           primary, dtype_r).add_(total)

            # quantile/MAD writes (if enabled)
            if (collect_q or collect_mad) and samples_by_reg:
                for name, chunks in samples_by_reg.items():
                    if not chunks:
                        continue
                    S = chunks[0] if (len(chunks) == 1 and chunks[0].device == primary) \
                        else torch.cat([c if c.device == primary else c.to(primary, non_blocking=True) for c in chunks], dim=0)
                    if int(S.numel()) == 0:
                        continue
                    if collect_q:
                        qv = torch.quantile(S, q_p)
                        sb.scalar_slot(f"probe/q/{name}", primary, qv.dtype).add_(qv)
                        # tolerate legacy readers
                        sb.scalar_slot(f"tv_q/{name}",    primary, qv.dtype).add_(qv)
                    if collect_mad:
                        med = torch.quantile(S, 0.5)
                        mad = torch.quantile((S - med).abs(), 0.5)
                        sb.scalar_slot(f"probe/mad/{name}", primary, mad.dtype).add_(mad)


        return total

    @torch.no_grad()
    def add_diag(self, ws) -> None:
        """
        Add diagonal preconditioner contributions (image/param).
        Preference per reg:
          (1) axis-profile majorizer (fast, deterministic)  → image diag
          (2) image-dependent add_diag (lagged-diffusivity) → image diag
          (3) scalar/1-D majorizer via majorizer_diag       → image/param via mapping
        """
        roles = ws.plan.roles_image
        halo  = self._aggregate_halo(roles)
        n_shards = self._num_shards(ws)

        has_img_diag = self._has(ws, "diag")

        for sh, i in ws.iter_shards():
            # required field; if missing, skip
            try:
                x = self._get(ws, i, "var", "x")
            except Exception:
                continue

            D = self._get(ws, i, "diag", "diag") if has_img_diag else None
            g_placeholder = torch.empty(0, dtype=x.dtype, device=x.device)

            x_ext, interior = self._with_halo_x(ws, i, n_shards, halo, anchor=x)
            dev = x.device

            for reg in self._regs:
                ctx = RegContext(
                    x=x_ext, g=g_placeholder, diag=D, roles_image=roles, device=dev,
                    dtype_c=x.dtype, dtype_r=x.real.dtype,
                    axes_resolver=lambda spec: self._resolve_axes(spec, roles),
                    arena=ws.arena, write_interior_slice=interior,
                    ws=ws, shard_index=i, halo_map=halo,
                )

                # (1) axis-profile if image diag is present
                if D is not None:
                    prof = getattr(reg, "majorizer_profile", None)
                    if callable(prof):
                        try:
                            prof_list = prof(ctx)
                        except Exception:
                            prof_list = None
                        if prof_list:
                            Dint = D[interior]
                            for axis, v1d in prof_list:
                                v = v1d.to(Dint.device, dtype=Dint.dtype)
                                shape = [1] * Dint.ndim
                                shape[int(axis)] = -1
                                Dint.add_(v.reshape(shape))
                            continue  # profile handled this reg

                # (2) image-dependent in the mapped space (Transformed will push into params if needed)
                try:
                    reg.add_diag(ctx)
                    continue
                except Exception:
                    pass

                # (3) scalar/1-D fallback via majorizer_diag (image or mapped)
                try:
                    kval = reg.majorizer_diag(ctx)
                except Exception:
                    kval = None
                if (kval is None) or (D is None):
                    continue
                Dint = D[interior]
                if kval.ndim == 0:
                    Dint.add_(kval.to(Dint.device, dtype=Dint.dtype))
                elif kval.ndim == 1 and int(kval.shape[0]) == int(Dint.shape[0]):
                    v = kval.to(Dint.device, dtype=Dint.dtype).view(-1, *([1] * (Dint.ndim - 1)))
                    Dint.add_(v)
                # else: shapes unknown → skip

    # ---------------- Continuation passthrough ----------------
    def maybe_update(self, stats: Mapping[str, Any]) -> bool:
        changed = False
        for reg in self._regs:
            try:
                changed |= bool(reg.continuation_update(stats))
            except AttributeError:
                pass
        return changed

    # ---------------- Utilities ----------------
    def _has(self, ws, name: str) -> bool:
        try:
            return bool(ws.has(name))
        except Exception:
            return False

    def _get(self, ws, i: int, primary: str, legacy: str):
        if self._has(ws, primary):
            return ws.get(primary, i)
        return ws.get(legacy, i)

    def _num_shards(self, ws) -> int:
        n = getattr(ws, "num_shards", None)
        if n is not None:
            return int(n)
        plan = getattr(ws, "plan", None)
        if plan is not None and hasattr(plan, "num_shards"):
            return int(plan.num_shards)  # type: ignore[attr-defined]
        return sum(1 for _ in ws.iter_shards())

    def _resolve_axes(self, spec: AxesSpec, roles: Roles) -> Tuple[int, ...]:
        return roles.resolve_axes(spec)

    def _aggregate_halo(self, roles: Roles) -> Dict[int, int]:
        merged: Dict[int, int] = {}
        for reg in self._regs:
            try:
                h = reg.halo(roles) or {}
            except AttributeError:
                h = {}
            for ax, r in h.items():
                r = int(r)
                if r < 0:
                    continue
                merged[ax] = max(merged.get(ax, 0), r)
        return merged

    def _with_halo_x(self,
                     ws,
                     shard_idx: int,
                     num_shards: int,
                     halo_map: Dict[int, int],
                     *,
                     anchor: torch.Tensor) -> Tuple[torch.Tensor, Tuple[slice, ...]]:
        """
        Return (x_ext, interior_slice). We support halo along absolute axis 0
        (the sharded batch/time axis).
        """
        t_halo = int(halo_map.get(0, 0))
        x_i = self._get(ws, shard_idx, "var", "x")
        if t_halo <= 0:
            return x_i, (slice(None),) * x_i.ndim

        dev = x_i.device
        prev_tail = None
        next_head = None

        if shard_idx > 0:
            x_prev = self._get(ws, shard_idx - 1, "var", "x")
            prev_tail = x_prev[-t_halo:].to(dev, non_blocking=True)
        if shard_idx + 1 < num_shards:
            x_next = self._get(ws, shard_idx + 1, "var", "x")
            next_head = x_next[:t_halo].to(dev, non_blocking=True)

        parts = [p for p in (prev_tail, x_i, next_head) if p is not None]
        x_ext = torch.cat(parts, dim=0) if len(parts) > 1 else x_i
        interior = [slice(None)] * x_ext.ndim
        start = t_halo if prev_tail is not None else 0
        stop  = start + x_i.shape[0]
        interior[0] = slice(start, stop)
        return x_ext, tuple(interior)

    def _kernel_signature_axes(self, reg: Regularizer, axes: Tuple[int, ...], ctx: RegContext) -> Tuple[Any, ...]:
        return (type(reg), tuple(int(a) for a in axes), ctx.device, ctx.dtype_c, ctx.dtype_r)

    def _compiled_fixed_axes(self, reg: Regularizer, axes: Tuple[int, ...], sig: Tuple[Any, ...]) -> Callable[[RegContext], torch.Tensor]:
        fn = self._kernel_cache.get(sig)
        if fn is not None:
            return fn
        # prefer fixed-axes kernel if available
        if hasattr(reg, "energy_grad_fixed_axes"):
            def inner(ctx: RegContext) -> torch.Tensor:
                return reg.energy_grad_fixed_axes(ctx, axes)
        else:
            def inner(ctx: RegContext) -> torch.Tensor:
                return reg.energy_grad(ctx)

        if self._compile_kernels and hasattr(torch, "compile"):
            try:
                inner = torch.compile(inner, fullgraph=False, dynamic=False)  # type: ignore[arg-type]
            except Exception:
                pass
        self._kernel_cache[sig] = inner
        return inner
