# graspcg/regularization/wrappers.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import torch
from .base import Regularizer, RegContext
from .mapping import MappedRegularizer, LinOpAdapter, Op, IdentityOp

@dataclass
class Transformed(Regularizer):
    """
    Wrap a Regularizer with a mapping:
      • If 'op' implements the unified Op API (forward_apply/adjoint_apply), we use it directly.
      • Else, if 'op' provides .fwd(x)->y and .adj(y)->x, we adapt it via LinOpAdapter.
    """
    base: Regularizer
    op: Any                    # Op or object with fwd/adj
    name: str
    params: Any                # optional passthrough of base.params for policies
    _mr: MappedRegularizer = field(init=False)

    def __post_init__(self):
        # Unified mapping first
        if hasattr(self.op, "forward_apply") and hasattr(self.op, "adjoint_apply"):
            op_u = self.op  # already a unified Op
        else:
            # Legacy fwd/adj -> wrap
            op_u = LinOpAdapter(self.op)
        self._mr = MappedRegularizer(name=self.name, inner=self.base, op=op_u)

    def halo(self, roles):
        try:
            return self._mr.halo(roles)
        except Exception:
            return {}

    def energy_grad(self, ctx: RegContext) -> torch.Tensor:
        return self._mr.energy_grad(ctx)

    def add_diag(self, ctx: RegContext) -> None:
        """
        Operator‑aware diagonal: ask inner for a majorizer on the mapped field
        and push it back via the op. If the op is identity and the inner
        doesn't expose a majorizer, fall back to inner.add_diag(ctx).
        """
        try:
            interior = ctx.write_interior_slice or (slice(None),) * ctx.x.ndim
            roles_u  = self._mr.op.roles_for_output(ctx.roles_image) if hasattr(self._mr.op, "roles_for_output") else ctx.roles_image

            # Build the mapped field
            u_ext = self._mr.op.forward_apply(ctx.x, ctx, interior)
            g_u   = torch.zeros_like(u_ext, device=u_ext.device, dtype=ctx.dtype_c)

            sub = RegContext(
                x=u_ext, g=g_u, diag=None,
                roles_image=roles_u, device=u_ext.device,
                dtype_c=ctx.dtype_c, dtype_r=ctx.dtype_r,
                axes_resolver=ctx.axes_resolver,
                arena=ctx.arena, write_interior_slice=interior,
                ws=ctx.ws, shard_index=ctx.shard_index, halo_map=ctx.halo_map,
            )
            # (A) Try profile majorizer first (push through op), if caller provided image diag.
            if ctx.diag is not None:
                prof_fn = getattr(self.base, "majorizer_profile", None)
                if callable(prof_fn):
                    prof_list = prof_fn(sub)
                    if prof_list:
                        axis, v1d = prof_list[0]  # TVND returns only time-axis
                        if hasattr(self._mr.op, "diag_push_profile"):
                            pushed = self._mr.op.diag_push_profile(axis, v1d, ctx, interior)
                            if pushed is not None:
                                axis2, v1d2 = pushed
                                Dint = ctx.diag[interior]
                                v = v1d2.to(Dint.device, dtype=Dint.dtype)
                                shape = [1] * Dint.ndim
                                shape[int(axis2)] = -1
                                Dint.add_(v.reshape(shape))
                                return
            k = None
            try:
                k = self.base.majorizer_diag(sub)
            except Exception:
                k = None

            if k is None:
                # Fallback: identity‑only legacy path
                if isinstance(self._mr.op, IdentityOp):
                    self.base.add_diag(ctx)
                return

            # Materialize interior weights (broadcast‑safe)
            u_int = u_ext[interior] if interior is not None else u_ext
            if k.ndim == 0:
                k_int = torch.ones_like(u_int.real, device=u_int.device, dtype=sub.dtype_r) * k.to(device=u_int.device, dtype=sub.dtype_r)
            else:
                k_int = (k.to(device=u_int.device, dtype=sub.dtype_r) * torch.ones_like(u_int.real))

            pushed = None
            if hasattr(self._mr.op, "diag_push"):
                pushed = self._mr.op.diag_push(k_int, ctx, interior)

            if pushed is not None and ctx.diag is not None:
                ctx.diag[interior].add_(pushed)
            # else: param‑space op handled its own diag (e.g., TemporalBasisOp)

        except Exception:
            # Keep this very safe—diag is a preconditioner enhancement.
            # If anything goes wrong, we silently skip the addition.
            print('Warning: Failed to add diag for transformed regularizer', self.name)
            return None



    def continuation_update(self, stats):
        try:
            return bool(self.base.continuation_update(stats))
        except Exception:
            return False

    def scaling_policy(self, ctx):  # deprecated in unified mapping
        return None

    def prox_inplace(self, ctx, step: float):
        pass

    def majorizer_diag(self, ctx):
        try:
            return self.base.majorizer_diag(ctx)
        except Exception:
            return None
