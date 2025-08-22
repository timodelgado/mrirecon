#!/usr/bin/env python3
"""
Patch NUFFT consistency across front-end and backends.

- Switch factory to use api.NSNUFFT
- Pass traj_units config to adapters in api.py
- TorchKb: allow ω/DCFs expansion when B_eff is a multiple of prepared B
- CUFI: guard DCF shape, fix ntr==1 branch in AH, disable Dynamo on A/AH
- op.NSNUFFT: fix missing return in A() (optional safety)

Run from repo root:
  python tools/patch_nufft_consistency.py
"""
from __future__ import annotations
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]  # repo root if script is tools/patch_*.py

FILES = {
    "factory": ROOT / "mrirecon" / "graspcg" / "nufft" / "factory.py",
    "api":     ROOT / "mrirecon" / "graspcg" / "nufft" / "api.py",
    "kb":      ROOT / "mrirecon" / "graspcg" / "nufft" / "backends" / "torchkb.py",
    "cufi":    ROOT / "mrirecon" / "graspcg" / "nufft" / "backends" / "cufi.py",
    "op":      ROOT / "mrirecon" / "graspcg" / "nufft" / "op.py",
}

def backup(path: Path):
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)

def patch_file(path: Path, do_patch):
    txt = path.read_text()
    new = do_patch(txt)
    if new is not None and new != txt:
        backup(path)
        path.write_text(new)
        return True
    return False

def subn(txt, pattern, repl, flags=0):
    return re.subn(pattern, repl, txt, flags=flags)

def patch_factory(txt: str) -> str:
    changed = False
    # import redirection
    new, n = subn(txt, r"from\s+\.op\s+import\s+NSNUFFT", "from .api import NSNUFFT")
    if n: changed = True; txt = new
    return txt if changed else txt

def patch_api_units(txt: str) -> str:
    changed = False
    # TorchKb: set traj_units from config
    new, n = subn(
        txt,
        r"(be\s*=\s*TorchKbNUFFTAdapter\(ndim=ndim\)\s*\n)",
        r"\1            be.traj_units = self.cfg.traj_units\n",
    )
    if n: changed = True; txt = new
    # CUFI: set traj_units from config (if adapter exposes the field)
    new, n = subn(
        txt,
        r"(be\s*=\s*CuFiNUFFTAdapter\([^\)]*\)\s*\n)",
        r"\1            try:\n                be.traj_units = self.cfg.traj_units\n            except Exception:\n                pass\n",
    )
    if n: changed = True; txt = new
    return txt if changed else txt

def patch_kb_expand(txt: str) -> str:
    changed = False
    # _expand_omega
    patt = r"def\s+_expand_omega\(self,\s*Bx:\s*int\)\s*->\s*torch\.Tensor:\s*\n(?:\s.*\n)+?"
    if re.search(patt, txt):
        body = """
    def _expand_omega(self, Bx: int) -> torch.Tensor:
        \"\"\"Expand ω to effective batch Bx.
        If prepared B == Bx: return as-is.
        If prepared B == 1:  broadcast to Bx.
        If Bx is an integer multiple of prepared B: repeat_interleave.
        Else: raise.
        \"\"\"
        assert self._traj_BndK is not None
        om = self._traj_BndK
        Bom = int(om.shape[0])
        if Bom == Bx:
            return om
        if Bom == 1 and Bx > 1:
            return om.expand(Bx, -1, -1)
        if Bx % Bom == 0:
            r = Bx // Bom
            return om.repeat_interleave(r, dim=0)
        raise ValueError(f\"omega batch {Bom} incompatible with requested {Bx}. Prepare with matching B or fold like→batch only when Bx is a multiple of B.\")
"""
        new, n = subn(txt, patt, body, flags=re.M)
        if n: changed = True; txt = new
    # _expand_dcf
    patt = r"def\s+_expand_dcf\(self,\s*Bx:\s*int,\s*K:\s*int\)\s*->\s*Optional\[torch\.Tensor\]:\s*\n(?:\s.*\n)+?"
    if re.search(patt, txt):
        body = """
    def _expand_dcf(self, Bx: int, K: int) -> Optional[torch.Tensor]:
        \"\"\"Expand DCF to (Bx,K) if needed; None if not set.\"\"\"
        dw = self._dcf_BK
        if dw is None:
            return None
        if int(dw.shape[-1]) != K:
            return None
        Bd = int(dw.shape[0])
        if Bd == Bx:
            return dw
        if Bd == 1 and Bx > 1:
            return dw.expand(Bx, -1)
        if Bx % Bd == 0:
            r = Bx // Bd
            return dw.repeat_interleave(r, dim=0)
        raise ValueError(f\"dcf batch {Bd} incompatible with requested {Bx} for K={K}.\")
"""
        new, n = subn(txt, patt, body, flags=re.M)
        if n: changed = True; txt = new
    return txt if changed else txt

def patch_cufi_disable_and_dcf(txt: str) -> str:
    changed = False
    # Add dynamo disable helper once after imports
    if "from torch._dynamo import disable as _dynamo_disable" not in txt:
        new, n = subn(
            txt,
            r"(try:\s*\n\s*from cufinufft import Plan as _CuFiPlan[^\n]*\n\s*_HAVE_CUFI\s*=\s*True\s*\nexcept[^\n]+\n\s*_HAVE_CUFI\s*=\s*False\s*\n)",
            r"\1\n# Mark CUFI call-sites as non-traceable for torch.compile\ntry:\n    from torch._dynamo import disable as _dynamo_disable\nexcept Exception:\n    def _dynamo_disable(fn):\n        return fn\n",
            flags=re.M,
        )
        if n: changed = True; txt = new

    # Decorate A/AH with @_dynamo_disable
    new, n = subn(txt, r"(@torch\.no_grad\(\)\s*\n\s*def\s+A\()", r"@_dynamo_disable\n    @torch.no_grad()\n    def A(", flags=re.M)
    if n: changed = True; txt = new
    new, n = subn(txt, r"(@torch\.no_grad\(\)\s*\n\s*def\s+AH\()", r"@_dynamo_disable\n    @torch.no_grad()\n    def AH(", flags=re.M)
    if n: changed = True; txt = new

    # Guard DCF shape in forward (A): add 'and int(self._dcf_BK.shape[1]) == K'
    new, n = subn(
        txt,
        r"if\s*\(self\._dcf_BK\s+is\s+not\s+None\)\s*and\s*self\.apply_dcf_in_fwd:",
        "if (self._dcf_BK is not None) and self.apply_dcf_in_fwd and int(self._dcf_BK.shape[1]) == K:",
    )
    if n: changed = True; txt = new

    # Guard DCF shape in adjoint (AH) ntr==C branch
    new, n = subn(
        txt,
        r"if\s*\(self\._dcf_BK\s+is\s+not\s+None\)\s*and\s*self\.apply_dcf_in_adj:",
        "if (self._dcf_BK is not None) and self.apply_dcf_in_adj and int(self._dcf_BK.shape[1]) == K:",
    )
    if n: changed = True; txt = new

    # Fix ntr==1 branch: handle dcf None and shape guard
    patt = r"elif\s+ntr\s*==\s*1:\s*\n(?:\s.*\n)+?x_out\[b\]\.copy_\(\s*x_accum\s*\)\s*\n"
    if re.search(patt, txt):
        body = """elif ntr == 1:
                x_accum = torch.zeros((1,) + spatial, device=y.device, dtype=y.dtype)
                for c in range(C):
                    if (self._dcf_BK is not None) and self.apply_dcf_in_adj and int(self._dcf_BK.shape[1]) == K:
                        cnu[0].copy_(y[b, c] * self._dcf_BK[b])
                    else:
                        cnu[0].copy_(y[b, c])
                    plan1.execute(cnu, grid)
                    x_accum.add_(grid * torch.conj(self._maps[c]).view(1, *spatial))
                x_out[b].copy_(x_accum)
"""
        new, n = subn(txt, patt, body)
        if n: changed = True; txt = new

    return txt if changed else txt

def patch_op_return(txt: str) -> str:
    changed = False
    # Ensure NSNUFFT.A returns its value (fix earlier double-call & missing return)
    patt = r"def\s+A\(self,\s*x:\s*torch\.Tensor,\s*out:\s*Optional\[torch\.Tensor\]=None\)\s*->\s*torch\.Tensor:\s*\n(?:\s.*\n)+?def\s+AH"
    if re.search(patt, txt):
        body = r"""def A(self, x: torch.Tensor, out: Optional[torch.Tensor]=None) -> torch.Tensor:
        # Re-prepare CUFI for like dims before calling, if needed
        if getattr(self._impl, 'backend_name', '') in ('cufinufft', 'cufi'):
            C = int(self.maps.shape[0])
            expect_no_like = 2 + (len(self._im_shape) - 1)   # B + C + spatial dims
            if x.ndim == expect_no_like + 1:
                L = int(x.shape[1])
                like_prod = C * L
                if getattr(self._impl, '_n_trans', C) != like_prod:
                    self._impl.prepare(self._im_shape, self.traj, self.dcf, self.maps, self.dtype_c, self.device, like_prod=like_prod)
        y = self._impl.A(x, out=out)
        return y

    def AH(self, y: torch.Tensor, out: Optional[torch.Tensor]=None) -> torch.Tensor:
        # For CUFI, if y has a like dim (B,L,C,K), re-plan with like_prod=C×L
        if getattr(self._impl, 'backend_name', '') in ('cufinufft', 'cufi'):
            C = int(self.maps.shape[0])
            if y.ndim == 4:
                L = int(y.shape[1])
                like_prod = C * L
                if getattr(self._impl, '_n_trans', C) != like_prod:
                    self._impl.prepare(self._im_shape, self.traj, self.dcf, self.maps, self.dtype_c, self.device, like_prod=like_prod)
        x = self._impl.AH(y, out=out)
        return x

    def """
        new, n = subn(txt, patt, body, flags=re.M)
        if n: changed = True; txt = new
    return txt if changed else txt

def main():
    modified = []

    if patch_file(FILES["factory"], patch_factory):
        modified.append(FILES["factory"].relative_to(ROOT))

    if patch_file(FILES["api"], patch_api_units):
        modified.append(FILES["api"].relative_to(ROOT))

    if patch_file(FILES["kb"], patch_kb_expand):
        modified.append(FILES["kb"].relative_to(ROOT))

    if patch_file(FILES["cufi"], patch_cufi_disable_and_dcf):
        modified.append(FILES["cufi"].relative_to(ROOT))

    # Optional safety for legacy op.py
    try:
        if patch_file(FILES["op"], patch_op_return):
            modified.append(FILES["op"].relative_to(ROOT))
    except FileNotFoundError:
        pass

    if not modified:
        print("No changes applied (files already patched).")
    else:
        print("Patched files:")
        for p in modified:
            print("  -", p)

if __name__ == "__main__":
    sys.exit(main() or 0)
