# graspcg/regularization/wrappers.py
from __future__ import annotations
from dataclasses import dataclass
import torch
from .base import Regularizer, RegContext
from ..ops.opchain import LinOp

@dataclass
class Transformed(Regularizer):
    """Wrap any Regularizer to act on u = op.fwd(z) and pull back grad with op.adj."""
    base: Regularizer
    op: LinOp
    name: str  # expose a unique name for manager/policy
    params: any  # forward base.params for policy/printing if desired

    def halo(self, roles):
        # combine base halo with op halo
        h1 = self.base.halo(roles)
        h2 = self.op.halo(roles)
        return {ax: max(h1.get(ax, 0), h2.get(ax, 0))}

    def energy_grad(self, ctx: RegContext) -> torch.Tensor:
        # u = Phi(z), build a temp grad buffer shaped like u
        u = self.op.fwd(ctx.x)
        g_u = torch.zeros_like(u)  # temp grad target

        # Make a shallow context for the base reg: writes to g_u, reads u
        sub = RegContext(
            x=u, g=g_u, diag=None,
            roles_image=ctx.roles_image, device=ctx.device,
            dtype_c=ctx.dtype_c, dtype_r=ctx.dtype_r,
            scale_field_shard=None, axes_resolver=ctx.axes_resolver,
            arena=ctx.arena, write_interior_slice=ctx.write_interior_slice
        )
        E = self.base.energy_grad(sub)

        # Pull back: g_z += (D Phi)^* g_u
        ctx.g.add_(self.op.adj(g_u))
        return E

    # Pass-throughs (optional implementations)
    def add_diag(self, ctx: RegContext) -> None:
        # conservative: skip, or approximate via op energy; identity by default
        return
    def continuation_update(self, stats): return False
    def scaling_policy(self, ctx): return None
    def prox_inplace(self, ctx, step: float): pass
    def majorizer_diag(self, ctx): return None