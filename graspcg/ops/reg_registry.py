# graspcg/ops/reg_registry.py
from functools import wraps

# energy/gradient handlers:   handler(ws) -> float energy, writes ∇ into ws.g (in-place)
REG_HANDLERS: dict[str, callable] = {}

def register(name: str):
    """Decorator:  @register('id')   puts the function into REG_HANDLERS."""
    def _decor(fn):
        REG_HANDLERS[name] = fn
        return wraps(fn)(fn)
    return _decor

# diagonal helpers for preconditioner: helper(ws, diag) -> adds in-place to diag
DIAG_HELPERS: dict[str, callable] = {}

def register_diag(key: str):
    """Decorator: register a diagonal‑contribution helper."""
    def _decor(fn):
        DIAG_HELPERS[key] = fn
        return fn
    return _decor
# --- shard‑explicit diagonal helpers ---------------------------------
DIAG_SHARD_HELPERS: dict[str, callable] = {}

def register_diag_shard(key: str):
    """
    Decorator for per‑shard diagonal builder:
        fn(ws, sh, diag)  # diag belongs to this shard
    """
    def _decor(fn):
        DIAG_SHARD_HELPERS[key] = fn
        return fn
    return _decor
# statistics helpers for init scaling / continuation: helper(ws, xs, **kw) -> (eps, sigma)
STATS_HELPERS: dict[str, callable] = {}

def register_stats(key: str):
    """Decorator: register a helper that returns (eps, sigma) for initialisation/continuation."""
    def _decor(fn):
        STATS_HELPERS[key] = fn
        return fn
    return _decor
