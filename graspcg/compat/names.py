# graspcg/compat/names.py
"""
Canonical-and-alias names for problem-agnostic buffer keys.

We keep canonical names *and* legacy aliases active simultaneously. The
Workspace installs aliases so `ws.get("x", i)` and `ws.get("var", i)` (etc.)
return the exact same underlying buffer object.
"""

from __future__ import annotations

# Canonical names → list of aliases that should point to the same storage
CANONICAL: dict[str, list[str]] = {
    # primary variable & its numerics
    "var":  ["x"],
    "grad": ["g"],
    "dir":  ["dx"],
    "diag": ["diag"],  # keep identical for clarity (still round-trips)

    # acquired signal
    "data": ["y"],

    # line-search caches (forward op on var/dir, residual)
    "fwd_var_sh": ["Ax_sh"],
    "fwd_dir_sh": ["Ad_sh"],
    "resid_sh":   ["r_sh"],
}

# Convenience: build a reverse alias→canonical map
ALIAS_TO_CANON: dict[str, str] = {}
for canon, aliases in CANONICAL.items():
    for a in [canon] + list(aliases):
        ALIAS_TO_CANON.setdefault(a, canon)
