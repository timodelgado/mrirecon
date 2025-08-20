# graspcg/regularization/graph_laplacian.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Sequence, Union
import torch

from .base import Regularizer, RegParams, RegContext

TensorLike = Union[torch.Tensor, torch.nn.Parameter]

@dataclass(frozen=True)
class GraphLapParams(RegParams):
    weight: float = 0.0
    # 'none' -> L = D - W
    # 'rw'   -> L = I - D^{-1} W
    # 'sym'  -> L = I - D^{-1/2} W D^{-1/2}
    normalize: str = "none"
    # Matmul along spatial columns is chunked to limit peak memory.
    # Each chunk computes (B x chunk_cols); tune based on VRAM.
    mm_chunk_cols: int = 65536

class GraphLaplacian(Regularizer):
    """
    Graph Laplacian regularizer along absolute axis 0 (time/batch) with a general neighborhood W.

    Energy:
        E(x) = (λ/2) * sum_p < x[:,p], L x[:,p] >
    Gradient:
        ∂E/∂x = λ * (L x)     (L depends on 'normalize')

    Supports dense or sparse W on CPU/GPU. For multi-device sharding, we gather x across
    shards onto the current shard device, compute Y = L @ X in (B x N) form (chunked over N),
    write gradient only to the local interior frames, and accumulate local energy.
    """

    Params = GraphLapParams

    def __init__(self,
                 name: str,
                 W: TensorLike,                  # (B, B) real, adjacency weights
                 params: GraphLapParams):
        self.name = name
        self.params = params

        # Store a CPU master copy of W (float32), symmetrize, zero diag
        Wc = W.detach()
        if Wc.is_sparse:
            # coalesce; symmetrize: (W + W^T)/2
            Wc = Wc.coalesce()
            WT = torch.sparse_coo_tensor(Wc.indices().flip(0), Wc.values(),
                                         Wc.shape, dtype=Wc.dtype, device=Wc.device).coalesce()
            Wavg = 0.5 * (Wc + WT)
            # zero diagonal (remove diag entries)
            idx = Wavg.indices()
            val = Wavg.values()
            mask = idx[0] != idx[1]
            Wavg = torch.sparse_coo_tensor(idx[:, mask], val[mask], Wavg.shape)
            self._W_cpu = Wavg.to(dtype=torch.float32, device="cpu").coalesce()
            self._is_sparse = True
        else:
            Wd = 0.5 * (Wc + Wc.transpose(0,1))
            Wd = Wd.clone()
            Wd.fill_diagonal_(0.0)
            self._W_cpu = Wd.to(dtype=torch.float32, device="cpu")
            self._is_sparse = False

        # Precompute degree on CPU
        if self._is_sparse:
            # degree = sum_j W_ij
            idx = self._W_cpu.indices()
            val = self._W_cpu.values()
            B = int(self._W_cpu.shape[0])
            deg = torch.zeros(B, dtype=torch.float32)
            deg.index_add_(0, idx[0], val)
        else:
            deg = self._W_cpu.sum(dim=1).to(dtype=torch.float32, device="cpu")
        self._deg_cpu = deg.clamp_min(0.0)

        # Per-device cache for W/derived mats
        self._cache = {}  # dev -> dict(keys: 'W', 'Dinv', 'Dminushalf')

    # ------- optional halo (general W is nonlocal; we default to none) -------
    def halo(self, roles):
        return {}

    # ------------------------------- internals --------------------------------
    def _to_device_cached(self, dev: torch.device, dtype_r: torch.dtype):
        """
        Move W and degree-derived vectors to device 'dev' and cache.
        """
        key = (dev.type, dev.index if dev.type == "cuda" else -1, dtype_r)
        cache = self._cache.get(key)
        if cache is not None:
            return cache

        cache = {}
        if self._is_sparse:
            cache["W"] = self._W_cpu.to(device=dev)
        else:
            cache["W"] = self._W_cpu.to(device=dev)
        deg = self._deg_cpu.to(device=dev, dtype=dtype_r)
        cache["deg"] = deg

        norm = (self.params.normalize or "none").lower()
        if norm == "rw":
            # D^{-1}
            cache["Dinv"] = (1.0 / deg.clamp_min(1e-20))
        elif norm == "sym":
            # D^{-1/2}
            cache["Dminushalf"] = torch.rsqrt(deg.clamp_min(1e-20))
        self._cache[key] = cache
        return cache

    @staticmethod
    def _mm_real_on_complex(W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Compute W @ X when W is real and X may be complex, returning a complex result.
        Shapes: W: (B,B), X: (B,N).
        """
        if X.is_complex():
            Yr = W @ X.real
            Yi = W @ X.imag
            return torch.complex(Yr, Yi)
        else:
            return W @ X

    @staticmethod
    def _spmm_real_on_complex(Wc: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Sparse COO spmm when W is sparse (real) and X may be complex.
        """
        if X.is_complex():
            Yr = torch.sparse.mm(Wc, X.real)
            Yi = torch.sparse.mm(Wc, X.imag)
            return torch.complex(Yr, Yi)
        else:
            return torch.sparse.mm(Wc, X)

    def _apply_L(self, cache: dict, X: torch.Tensor) -> torch.Tensor:
        """
        Y = L @ X with normalization mode.
        cache: dict from _to_device_cached
        X: (B, N), complex ok
        return Y: (B, N), same dtype as X
        """
        norm = (self.params.normalize or "none").lower()
        W = cache["W"]
        deg = cache["deg"]

        if norm == "none":
            # (D - W) X  = deg[:,None]*X - W@X
            Y = (deg.view(-1, 1) * X)
            Z = self._spmm_real_on_complex(W, X) if W.is_sparse else self._mm_real_on_complex(W, X)
            Y = Y - Z
            return Y

        if norm == "rw":
            # L_rw X = X - D^{-1} W X
            Z = self._spmm_real_on_complex(W, X) if W.is_sparse else self._mm_real_on_complex(W, X)
            Dinv = cache["Dinv"].view(-1, 1)
            return X - (Dinv * Z)

        # 'sym'
        # L_sym X = X - D^{-1/2} W D^{-1/2} X
        Dmh = cache["Dminushalf"].view(-1, 1)
        X1 = Dmh * X
        Z = self._spmm_real_on_complex(W, X1) if W.is_sparse else self._mm_real_on_complex(W, X1)
        return X - (Dmh * Z)

    # -------------------------------- API -------------------------------------
    def energy_grad(self, ctx: RegContext) -> torch.Tensor:
        """
        Compute E and accumulate grad into ctx.g on the local interior frames only.
        """
        lam = float(self.params.weight)
        if lam == 0.0:
            return torch.zeros((), device=ctx.device, dtype=ctx.dtype_r)

        assert getattr(ctx, "ws", None) is not None and getattr(ctx, "shard_index", None) is not None, \
               "GraphLaplacian expects ctx.ws and ctx.shard_index for sharded access"

        ws = ctx.ws
        i  = ctx.shard_index
        dev = ctx.device
        interior = ctx.write_interior_slice or (slice(None),) * ctx.x.ndim

        # Gather all frames to current device (kept compact by chunking later)
        x_all = []
        for _, j in ws.iter_shards():
            xj = ws.get("x", j)
            x_all.append(xj.to(dev, non_blocking=True))
        X = torch.cat(x_all, dim=0)             # (B, C, H, W, ...)
        B = int(X.shape[0])
        N = int(X.numel() // B)
        X2 = X.reshape(B, N)

        cache = self._to_device_cached(dev, ctx.dtype_r)

        # Chunked matmul over columns to limit peak VRAM
        chunk = max(1, int(self.params.mm_chunk_cols))
        Y2 = torch.empty_like(X2)
        E_val = torch.zeros((), device=dev, dtype=ctx.dtype_r)

        # global row bounds for this shard's interior
        sh = ws.shard_for_index(i)
        row0 = int(sh.b_start)
        row1 = int(sh.b_stop)

        lam_t = torch.as_tensor(lam, device=dev, dtype=ctx.dtype_r)

        for j0 in range(0, N, chunk):
            j1 = min(N, j0 + chunk)
            Xc = X2[:, j0:j1]                              # (B, m)
            Yc = self._apply_L(cache, Xc)                  # (B, m)
            Y2[:, j0:j1] = Yc

            # Energy contribution (local rows only)
            # E_chunk = (λ/2) * sum_{t in interior} Re( conj(X)*Y )[t, :]
            Xi = Xc[row0:row1, :]
            Yi = Yc[row0:row1, :]
            # Re(conj(X) * Y) sums over complex entries
            ed = (Xi.conj() * Yi).real.sum()
            E_val = E_val + (0.5 * lam_t * ed)

        # Accumulate gradient on interior only: g[interior] += λ * Y[rows]
        g_int = ctx.g[interior]
        Yi_full = Y2[row0:row1, :].reshape(g_int.shape)    # same shape as interior
        if g_int.is_complex():
            g_int.add_(lam_t.to(dtype=g_int.real.dtype) * Yi_full)
        else:
            # (rare) real-valued field
            g_int.add_(lam_t * Yi_full.real)

        return E_val if E_val.dtype == ctx.dtype_r else E_val.to(ctx.dtype_r)

    def add_diag(self, ctx: RegContext) -> bool:
        """
        Exact, degree‑aware diagonal:
          • unnormalized: diag += λ * deg
          • normalized (rw/sym): diag += λ * 1
        Supports optional temporal scaling (ws.scale.inv_for_shard) like TVND.add_diag.
        """
        if ctx.diag is None or self.params.weight == 0.0:
            return False

        dev = ctx.device
        Dint = ctx.diag[ctx.write_interior_slice or (slice(None),) * ctx.diag.ndim]
        cache = self._to_device_cached(dev, Dint.dtype)
        lam = torch.as_tensor(self.params.weight, device=dev, dtype=Dint.dtype)

        norm = (self.params.normalize or "none").lower()
        if norm == "none":
            v = cache["deg"]           # per-frame degrees
        else:
            v = torch.ones_like(cache["deg"])  # diag(L)=1 for rw and sym

        # Optional temporal scaling (B_loc,1,1,...)
        inv_s2 = None
        ws = getattr(ctx, "ws", None)
        i  = getattr(ctx, "shard_index", None)
        if ws is not None and i is not None and hasattr(ws, "scale"):
            sh = ws.shard_for_index(i)
            inv_s = ws.scale.inv_for_shard(sh, anchor=Dint)   # (B_loc,1,1,...)
            inv_s2 = inv_s * inv_s

        # Add per-frame values to Dint
        n = int(Dint.shape[0])
        # This shard’s global extent
        sh = ctx.ws.shard_for_index(ctx.shard_index)
        row0 = int(sh.b_start)
        row1 = row0 + n
        v_loc = v[row0:row1].view(n, *([1]*(Dint.ndim-1))) * lam
        if inv_s2 is not None:
            v_loc = v_loc * inv_s2
        Dint.add_(v_loc)
        return True

    # Optional scalar majorizer (fallback when used under a mapping)
    def majorizer_diag(self, ctx: RegContext):
        # We prefer the exact degree-aware add_diag; return 0 to avoid double add.
        return torch.zeros((), device=ctx.device, dtype=ctx.dtype_r)
