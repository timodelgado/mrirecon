from __future__ import annotations
import torch
from typing import Literal

@torch.no_grad()
def build_precond_diag(
    ws,
    regm,
    *,
    mode: Literal["full", "update"] = "full",
    nufft_diag_val: float = 1.0,
    use_nufft_norm: bool = True,
) -> None:
    """
    Fill sh.diag on each shard with data + regulariser contributions.
    - Data term: constant diagonal scale (NUFFT norm or provided value)
    - Regularisers: delegated to RegManager
    """
    nuf   = ws.nufft_op
    n_val = float(getattr(nuf, "scale_emp", nufft_diag_val)) if use_nufft_norm else float(nufft_diag_val)

    for sh, _ in ws.iter_shards():
        if mode == "full" or not hasattr(sh, "diag"):
            # Full per‑variable diagonal; same shape as x (complex → real)
            sh.diag = torch.full_like(sh.x.real, n_val, dtype=torch.float32, device=sh.x.device)
        else:
            sh.diag.fill_(n_val)

        # Ask the manager to add regulariser contributions for THIS shard
        if hasattr(regm, "add_diag_shard"):
            regm.add_diag_shard(ws, sh, sh.diag)
        else:
            # Older managers: fall back to one-call interface
            regm.add_diag(ws, sh.diag)