# graspcg/regularization/manager.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type

import contextlib
import torch

from .base import Regularizer, RegParams, RegContext, AxesSpec
from ..core.roles import Roles
from .mapping import MappedRegularizer, LinOpAdapter, Op, IdentityOp


@dataclass
class RegSpec:
    """
    Construction-time specification for a single regularizer.
    - kind: registry key (e.g., "tv")
    - name: instance name (e.g., "tv_spatial"); must be unique in a manager
    - params: dict of parameter overrides (weight, eps, axes, use_scale, ...)
    """
    kind: str
    name: str
    params: Mapping[str, Any]


class RegManager:
    """
    Orchestrates a set of regularizers, one pass per shard:
      • Builds RegContext (on the shard device)
      • Applies per-term scale policy (optional) as 1/s (not 1/s^2)
      • Dispatches energy+grad kernels (compile-friendly)
      • Accumulates per-reg and total energies into StatsBoard
      • (Optional) Emits TV percentiles via uniform subsample (scalar-only)

    Notes on scaling:
      - If reg.params.use_scale is True and a per-frame scale exists,
        ctx.scale_field_shard is set to (B_loc,1,...,1) containing 1/s.
      - Regularizers decide how to apply 1/s (e.g., exact TV(x/s) + chain rule).
    """
    def __init__(self,
                 regs: Sequence[Regularizer],
                 *,
                 compile_kernels: bool = True):
        self._regs: List[Regularizer] = list(regs)
        self._compile_kernels = bool(compile_kernels)
        # registry of regularizer classes (kind -> class)
        self._local_registry: Dict[str, Type[Regularizer]] = {}
        # cache of compiled/eager kernels keyed by (type, axes, device, dtypes)
        self._kernel_cache: Dict[Tuple[Any, ...], Callable[[RegContext], torch.Tensor]] = {}
        

    # ---------------- Main API (used by Objective/Precond/Solver) ----------------

    @torch.no_grad()
    def energy_and_grad(self, ws) -> torch.Tensor:
        """
        Total regularization energy; accumulates ∂E/∂x into ws.g.
        Mapped‑friendly:
          • No in‑kernel scaling; transforms handled by wrappers/maps.
          • Quantiles: prefer reg.quantile_sample_shard(ctx, axes, K_shard, q)
            if available; else fallback to TVND on x_ext.
        """
        import contextlib
        from .tv_nd import TVND
        from ..ops.ndops import fwd_diff, tv_iso_energy, tv_aniso_energy
        from .base import RegContext

        roles = ws.plan.roles_image
        halo  = self._aggregate_halo(roles)
        n_shards = self._num_shards(ws)

        # Resolve axes ONCE per regularizer (compile-static)
        reg_axes = []
        for reg in self._regs:
            axes_spec = getattr(getattr(reg, "params", None), "axes", "spatial")
            reg_axes.append(self._resolve_axes(axes_spec, roles))

        # Stats toggles
        sb = getattr(ws, "stats", None)
        collect_regE = bool(sb and sb.enabled.get("reg_energy", False))
        collect_tvq  = bool(sb and sb.enabled.get("tv_quantile", False))
        tvq_q = float(getattr(sb, "tv_percentile", 0.90)) if sb is not None else 0.90
        tvq_K = int(getattr(sb, "tv_sample_K", 4096)) if sb is not None else 4096
        tvq_K = max(1, tvq_K)

        dev_order = []
        dev_accum = {}
        samples_by_reg = {}

        for sh, i in ws.iter_shards():
            dev = sh.device
            if dev not in dev_accum:
                # Anchor dtype on x.real (FakeWS may not expose 'g')
                x_dtype_r = ws.get("x", i).real.dtype
                dev_accum[dev] = torch.zeros((), device=dev, dtype=x_dtype_r)
                dev_order.append(dev)

            stream = ws.arena.stream_for(dev) if getattr(dev, "type", "cpu") == "cuda" else None
            ctx_mgr = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
            with ctx_mgr:                # Ensure per-device work stream waits for writes to x on this and neighbor devices

                x = ws.get("x", i)
                g = ws.get("g", i)
                D = ws.get("diag", i) if getattr(ws, "has", lambda _: False)("diag") else None

                # halo-extended read (x_ext), interior write slice for g/diag
                x_ext, interior = self._with_halo_x(ws, i, n_shards, halo, anchor=x)

                for reg, axes in zip(self._regs, reg_axes):
                    # Build context (now carries ws/shard_index/halo_map for mapped regs)
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

                    if collect_regE:
                        sb.scalar_slot(f"E_reg/{reg.name}", dev, e_shard.dtype).add_(e_shard)

                    if collect_tvq:
                        # Preferred: mapped regularizer provides a sample on its field
                        sample_fn = getattr(reg, "quantile_sample_shard", None)
                        if callable(sample_fn):
                            K_shard = max(1, (tvq_K + n_shards - 1) // max(1, n_shards))
                            smp = sample_fn(ctx, axes, K_shard, tvq_q)
                            if (smp is not None) and int(smp.numel()) > 0:
                                samples_by_reg.setdefault(reg.name, []).append(smp)
                        elif isinstance(reg, TVND):
                            # Fallback: TV magnitude on x_ext (no scale)
                            grads = [fwd_diff(x_ext, ax) for ax in axes]
                            eps = torch.as_tensor(getattr(reg.params, "eps", 0.0), device=dev, dtype=g.real.dtype)
                            e_den_full = tv_iso_energy(grads, eps) if getattr(reg.params, "isotropic", True) else tv_aniso_energy(grads, eps)
                            e_den = e_den_full[interior] if interior is not None else e_den_full
                            flat = e_den.reshape(-1)
                            nvox = int(flat.numel())
                            if nvox > 0:
                                K_shard = max(1, (tvq_K + n_shards - 1) // max(1, n_shards))
                                stride  = max(1, nvox // K_shard)
                                sample  = flat[::stride]
                                if int(sample.numel()) > K_shard:
                                    sample = sample[:K_shard]
                                samples_by_reg.setdefault(reg.name, []).append(sample)

        if not dev_order:
            return torch.zeros((), device="cpu", dtype=torch.float32)
        primary = dev_order[0]
        dtype_r = next(iter(dev_accum.values())).dtype
        total = torch.zeros((), device=primary, dtype=dtype_r)
        for dev in dev_order:
            v = dev_accum[dev]
            total.add_(v if dev == primary else v.to(primary, non_blocking=True))

        if collect_tvq and samples_by_reg:
            for name, chunks in samples_by_reg.items():
                if not chunks:
                    continue
                if len(chunks) == 1 and chunks[0].device == primary:
                    S = chunks[0]
                else:
                    S = torch.cat([c if c.device == primary else c.to(primary, non_blocking=True) for c in chunks], dim=0)
                if int(S.numel()) > 0:
                    qv = torch.quantile(S, tvq_q)
                    sb.scalar_slot(f"tv_q/{name}", primary, qv.dtype).add_(qv)

        if sb is not None:
            sb.scalar_slot("E_reg_total", primary, dtype_r).add_(total)
    
        return total

    @torch.no_grad()
    def add_diag(self, ws) -> None:
        """
        Invoke each regularizer's diagonal contribution per shard.

        Order of preference per reg:
        1) axis-profile majorizer (e.g., temporal degree) → cheap & deterministic
        2) the reg's own image-dependent add_diag (lagged-diffusivity)
        3) mapped regs may still terminalize into parameter diags via op.diag_push
            even when image diag is absent.
        """
        roles = ws.plan.roles_image
        halo  = self._aggregate_halo(roles)
        n_shards = self._num_shards(ws)

        has_img_diag = getattr(ws, "has", lambda _: False)("diag")

        # Detect whether any mapped op wants to write a parameter-space diagonal
        def _has_param_diag() -> bool:
            for reg in self._regs:
                op = getattr(getattr(reg, "_mr", None), "op", None)  # Transformed(...)._mr.op
                diag_key = getattr(op, "diag_key", None)
                if isinstance(diag_key, str) and getattr(ws, "has", lambda _: False)(diag_key):
                    return True
            return False

        if not (has_img_diag or _has_param_diag()):
            return  # nothing to do anywhere

        for sh, i in ws.iter_shards():
            # Required x; continue if missing
            try:
                x = ws.get("x", i)
            except Exception:
                continue

            # Optional image diag for this shard
            D = ws.get("diag", i) if has_img_diag else None

            # NO large gradient allocation here
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

                # --- (1) axis-profile path if we do have an image diag ---
                if D is not None:
                    prof = getattr(reg, "majorizer_profile", None)
                    if callable(prof):
                        try:
                            prof_list = prof(ctx)
                            if prof_list:
                                Dint = D[interior]
                                for axis, v1d in prof_list:
                                    v = v1d.to(Dint.device, dtype=Dint.dtype)
                                    shape = [1] * Dint.ndim
                                    shape[int(axis)] = -1
                                    Dint.add_(v.reshape(shape))
                                continue  # profile handled this reg
                        except Exception:
                            # profile is an optimization; fall through to reg.add_diag
                            pass

                # --- (2) image-dependent path OR (3) parameter-diag push path ---
                try:
                    reg.add_diag(ctx)
                    # Note: Transformed(...).add_diag will push to parameter diag via op.diag_push
                    # even when ctx.diag is None (e.g., TemporalBasisOp→diag_V).
                except Exception:
                    # diagonal is an enhancement; never be fatal
                    pass

    # ---------------- Continuation broadcast hook ----------------
    def maybe_update(self, stats: Mapping[str, Any]) -> bool:
        changed = False
        for reg in self._regs:
            try:
                changed |= bool(reg.continuation_update(stats))
            except AttributeError:
                pass
        return changed

    # ---------------- Context & utilities ----------------
    def build_context(self, *,
                      x: torch.Tensor,
                      g: torch.Tensor,
                      diag: Optional[torch.Tensor],
                      roles_image: Roles,
                      device: torch.device,
                      dtype_c: torch.dtype,
                      dtype_r: torch.dtype,
                      arena: Optional[Any],
                      write_interior_slice: Optional[Tuple[slice, ...]] = None) -> RegContext:
        return RegContext(
            x=x, g=g, diag=diag, roles_image=roles_image, device=device,
            dtype_c=dtype_c, dtype_r=dtype_r,
            axes_resolver=lambda spec: self._resolve_axes(spec, roles_image),
            arena=arena, write_interior_slice=write_interior_slice,
        )
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
        Return (x_ext, interior_slice). We support halo only along the first 'unlike'
        axis (absolute axis 0) because sharding happens there.
        """
        t_halo = int(halo_map.get(0, 0))
        x_i = ws.get("x", shard_idx)
        if t_halo <= 0:
            return x_i, (slice(None),) * x_i.ndim

        dev = x_i.device
        prev_tail = None
        next_head = None

        if shard_idx > 0:
            x_prev = ws.get("x", shard_idx - 1)
            prev_tail = x_prev[-t_halo:].to(dev, non_blocking=True)
        if shard_idx + 1 < num_shards:
            x_next = ws.get("x", shard_idx + 1)
            next_head = x_next[:t_halo].to(dev, non_blocking=True)

        parts = [p for p in (prev_tail, x_i, next_head) if p is not None]
        x_ext = torch.cat(parts, dim=0) if len(parts) > 1 else x_i
        # interior slice (drop halo rows on writes)
        interior = [slice(None)] * x_ext.ndim
        start = t_halo if prev_tail is not None else 0
        stop  = start + x_i.shape[0]
        interior[0] = slice(start, stop)
        return x_ext, tuple(interior)


    def _kernel_signature_axes(self, reg: Regularizer, axes: Tuple[int, ...], ctx: RegContext) -> Tuple[Any, ...]:
        # signature must not capture large tensors; only shapes/dtypes/devices
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

        # Optional: hand off to torch.compile when requested by the caller
        if self._compile_kernels and hasattr(torch, "compile"):
            try:
                inner = torch.compile(inner, fullgraph=False, dynamic=False)  # type: ignore[arg-type]
            except Exception:
                # still cache the eager version
                pass

        self._kernel_cache[sig] = inner
        return inner
    
    

    @torch.no_grad()
    def refresh_preconditioner(
        self,
        ws,
        *,
        mode: str = "auto",
        ema_alpha: float = 1.0,
        floor: float = 1e-12,
        mix_data: float = 0.0,
        verbose: bool = False,
    ) -> None:
        """
        Rebuild Jacobi preconditioner in-place:
          • recompute diag field(s) from regs (mapped-aware),
          • optionally mix a data-term diagonal,
          • optionally EMA-smooth,
          • clamp to floor for stability.
        """
        # 1) Recompute (delegates to reg.add_diag / majorizer push)
        #    Clear any param diagonals you maintain (e.g., diag_V) before accumulation
        for sh, i in ws.iter_shards():
            if ws.has("diag"):
                D = ws.get("diag", i)
                D.zero_()

            # clear known param diags if present
            for name in ("diag_V",):
                if ws.has(name):
                    ws.get(name, i).zero_()

        # Use existing add_diag logic (profile-aware + scalar fallback)
        self.add_diag(ws)

        # 2) Optional mixing with data-term diag(A^H A)
        if mix_data > 0.0 and hasattr(ws.nufft_op, "diag_AHA"):
            for sh, i in ws.iter_shards():
                try:
                    D = ws.get("diag", i)
                    # ws.nufft_op.diag_AHA should return a broadcastable REAL tensor
                    D.add_(float(mix_data) * ws.nufft_op.diag_AHA(D))
                except Exception:
                    pass  # non-fatal; silently skip if not available

        # 3) Optional EMA smoothing to avoid oscillations
        if ema_alpha < 1.0:
            a = float(max(0.0, min(1.0, ema_alpha)))
            one_minus = 1.0 - a
            # Keep a shadow diag_prev on workspace if you want persistent EMA.
            # If none exists, we do a one-shot blend vs current (no-op).
            if not ws.has("_diag_prev"):
                # allocate on-demand, one per shard
                for sh, i in ws.iter_shards():
                    if ws.has("diag"):
                        D = ws.get("diag", i)
                        ws._bufs.setdefault("_diag_prev", [None]*ws.num_shards)  # crude but effective
                        ws._bufs["_diag_prev"][i] = D.clone()
            try:
                for sh, i in ws.iter_shards():
                    if ws.has("diag"):
                        D  = ws.get("diag", i)
                        Dp = ws.get("_diag_prev", i)
                        if Dp is None:
                            # initialize prev with current
                            ws._bufs["_diag_prev"][i] = D.clone()
                        else:
                            D.mul_(a).add_(Dp, alpha=one_minus)
                            Dp.copy_(D)
                    # Example for param diag: diag_V EMA
                    if ws.has("diag_V"):
                        DV  = ws.get("diag_V", i)
                        key = "_diag_V_prev"
                        if not ws.has(key):
                            ws._bufs[key] = [None]*ws.num_shards
                        DVp = ws.get(key, i)
                        if DVp is None:
                            ws._bufs[key][i] = DV.clone()
                        else:
                            DV.mul_(a).add_(DVp, alpha=one_minus)
                            DVp.copy_(DV)
            except Exception:
                pass

        # 4) Clamp for stability
        for sh, i in ws.iter_shards():
            if ws.has("diag"):
                D = ws.get("diag", i)
                D.clamp_(min=float(floor))

            if ws.has("diag_V"):
                ws.get("diag_V", i).clamp_(min=float(floor))

        if verbose:
            try:
                devs = {sh.device.index for sh, _ in ws.iter_shards() if sh.device.type == "cuda"}
                print(f"[regm] preconditioner refresh (mode={mode}, ema={ema_alpha}, mix={mix_data}) on devices {sorted(devs)}")
            except Exception:
                print(f"[regm] preconditioner refresh (mode={mode}, ema={ema_alpha}, mix={mix_data})")