#%%
#!/usr/bin/env python3
"""
Apply NUFFT backend + operator patches matching the current codebase.

- graspcg/nufft/backends/torchkb.py
- graspcg/nufft/backends/cufi.py
- graspcg/nufft/op.py

Creates .bak backups before modifying.
"""

from __future__ import annotations
import sys
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0] if Path(__file__).name != "apply_nufft_patches.py" else Path(__file__).resolve().parent
PKG = ROOT / "graspcg" / "nufft"

def backup(p: Path):
    bak = p.with_suffix(p.suffix + ".bak")
    if not bak.exists():
        bak.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[backup] {p} -> {bak}")

def patch_torchkb():
    p = PKG / "backends" / "torchkb.py"
    if not p.exists():
        print(f"[skip] {p} not found")
        return
    txt = p.read_text(encoding="utf-8")
    orig = txt

    # 1) Add _dev/_dtype fields if missing
    if "_dev:" not in txt or "_dtype:" not in txt:
        txt = txt.replace(
            "_alpha_scalar: float | None = None",
            "_alpha_scalar: float | None = None\n    _dev: torch.device | None = None\n    _dtype: torch.dtype = torch.complex64"
        )

    # 2) Fix scale_traj body: normalize to (B, ndim, K) with robust transpose logic
    def replace_scale_traj(m):
        body = (
            "    def scale_traj(self, traj: torch.Tensor, im_shape: Tuple[int, ...]) -> torch.Tensor:\n"
            "        \"\"\"\n"
            "        Expect input traj from the user in normalized cycles/pixel ∈ [-0.5,0.5].\n"
            "        Convert to radians in [-π,π] per dimension, batched shape (B, ndim, K).\n"
            "        If you already store radians, return traj.to(dtype=torch.float32).\n"
            "        \"\"\"\n"
            "        if traj.ndim != 3:\n"
            "            raise ValueError('traj must be (B, K, ndim) or (B, ndim, K)')\n"
            "        # Normalize to (B, ndim, K)\n"
            "        if traj.shape[1] == self.ndim:\n"
            "            t = traj\n"
            "        elif traj.shape[2] == self.ndim:\n"
            "            t = traj.transpose(1, 2).contiguous()\n"
            "        else:\n"
            "            raise ValueError('traj must be (B, K, ndim) or (B, ndim, K)')\n"
            "        return 2.0 * torch.pi * t.to(dtype=torch.float32)\n"
        )
        return body

    txt = re.sub(
        r"def\s+scale_traj\([\s\S]*?\)\s*->\s*torch\.Tensor:\s*[\s\S]*?return\s+.*?\n",
        replace_scale_traj, txt, count=1
    )

    # 3) Prepare: set _dev/_dtype and keep tensors contiguous
    def replace_prepare(m):
        body = (
            "    def prepare(self, im_shape: Tuple[int, ...], traj: torch.Tensor, dcf: Optional[torch.Tensor], maps: torch.Tensor, dtype: torch.dtype, device: torch.device) -> None:\n"
            "        self._im_shape = tuple((int(s) for s in im_shape))\n"
            "        self._dev = device\n"
            "        self._dtype = dtype\n"
            "        self._maps = maps.to(device=device, dtype=dtype, non_blocking=True).contiguous()\n"
            "        self._traj_BndK = self.scale_traj(traj, self._im_shape).to(device, non_blocking=True)\n"
            "        self._dcf_BK = None if dcf is None else dcf.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()\n"
            "        self._K = int(self._traj_BndK.shape[-1])\n"
            "        from torchkbnufft import KbNufft, KbNufftAdjoint\n"
            "        spatial = self._im_shape[1:]\n"
            "        dtype_r = torch.float32 if dtype == torch.complex64 else torch.float64\n"
            "        self._fwd = KbNufft(im_size=spatial, grid_size=None, numpoints=6).to(device=device, dtype=dtype_r)\n"
            "        self._adj = KbNufftAdjoint(im_size=spatial, grid_size=None, numpoints=6).to(device=device, dtype=dtype_r)\n"
        )
        return body

    txt = re.sub(
        r"def\s+prepare\([\s\S]*?\)\s*->\s*None:\s*[\s\S]*?self\._adj\s*=.*\n",
        replace_prepare, txt, count=1
    )

    # 4) diag_AHA_profile: use actual tensor device, not _dev
    def replace_profile(m):
        body = (
            "    @torch.no_grad()\n"
            "    def diag_AHA_profile(self) -> torch.Tensor:\n"
            "        \"\"\"\n"
            "        Per-frame proxy α_b ≈ sum_k dcf[b,k] (or K if dcf is None).\n"
            "        Constant across frames when frames have identical sampling/DCF.\n"
            "        \"\"\"\n"
            "        assert self._traj_BndK is not None\n"
            "        B, _, K = self._traj_BndK.shape\n"
            "        dev = self._traj_BndK.device\n"
            "        if self._dcf_BK is None:\n"
            "            return torch.full((B,), float(K), device=dev, dtype=torch.float32)\n"
            "        dcf = self._dcf_BK\n"
            "        if dcf.ndim == 1:\n"
            "            val = dcf.sum().to(torch.float32)\n"
            "            return val.expand(B).to(dev)\n"
            "        return dcf.sum(dim=1).to(torch.float32).to(dev)\n"
        )
        return body

    txt = re.sub(
        r"@torch\.no_grad\(\)\s*def\s+diag_AHA_profile\([\s\S]*?\):\s*[\s\S]*?return\s+.*\n",
        replace_profile, txt, count=1
    )

    # 5) diag_AHA_scalar: strengthen assert to require _dev
    txt = txt.replace(
        "assert hasattr(self, \"_im_shape\") and self._im_shape is not None",
        "assert self._im_shape is not None and self._dev is not None"
    )

    if txt != orig:
        backup(p)
        p.write_text(txt, encoding="utf-8")
        print(f"[patch] {p}")
    else:
        print(f"[ok] {p} already patched")


def patch_cufi():
    p = PKG / "backends" / "cufi.py"
    if not p.exists():
        print(f"[skip] {p} not found")
        return
    txt = p.read_text(encoding="utf-8")
    orig = txt

    # Add a defensive __del__ to destroy plans safely if not present
    if "__del__(" not in txt:
        insert_after = "self._plans_t1 = plans_t1\n"
        destructor = (
            "\n    def __del__(self):\n"
            "        # Defensive: cufinufft.Plan.__del__ can raise if __init__ never completed.\n"
            "        try:\n"
            "            if self._plans_t1 is not None:\n"
            "                for p in self._plans_t1:\n"
            "                    try:\n"
            "                        p.destroy()\n"
            "                    except Exception:\n"
            "                        pass\n"
            "            if self._plans_t2 is not None:\n"
            "                for p in self._plans_t2:\n"
            "                    try:\n"
            "                        p.destroy()\n"
            "                    except Exception:\n"
            "                        pass\n"
            "        except Exception:\n"
            "            pass\n"
            "        finally:\n"
            "            self._plans_t1 = None\n"
            "            self._plans_t2 = None\n"
        )
        if insert_after in txt:
            txt = txt.replace(insert_after, insert_after + destructor)
        else:
            # fallback: append to end of file
            txt = txt.rstrip() + "\n" + destructor

    # Ensure Plan(...) uses only keyword n_trans (your code already does; keep it)
    # Nothing to change here unless there is a positional n_trans in your local variant.

    if txt != orig:
        backup(p)
        p.write_text(txt, encoding="utf-8")
        print(f"[patch] {p}")
    else:
        print(f"[ok] {p} already patched")


def patch_op():
    p = PKG / "op.py"
    if not p.exists():
        print(f"[skip] {p} not found")
        return
    txt = p.read_text(encoding="utf-8")
    orig = txt

    # Prefer backend alpha; then fallback to robust median
    if "Prefer backend-calibrated α" not in txt:
        txt = txt.replace(
            "    def diag_AHA_scalar(self, device: Optional[torch.device]=None) -> torch.Tensor:\n"
            "        \"\"\"\n"
            "        Alpha from AH(ones) vs sum|S|^2 ratio (median over a safe mask).\n"
            "        Cached as a CPU scalar (float32).\n"
            "        \"\"\"\n",
            "    def diag_AHA_scalar(self, device: Optional[torch.device]=None) -> torch.Tensor:\n"
            "        \"\"\"\n"
            "        Prefer backend-calibrated α if available; fallback to robust median of\n"
            "        AH(ones)/sum|S|^2. Cached as a CPU scalar (float32).\n"
            "        \"\"\"\n"
        )
        # Inject backend try path just after the cache check
        txt = txt.replace(
            "        if self._alpha is not None:\n"
            "            return self._alpha.to(device or 'cpu')\n",
            "        if self._alpha is not None:\n"
            "            return self._alpha.to(device or 'cpu')\n"
            "        # 1) Try backend-specific calibrated scalar (fast & stable)\n"
            "        if hasattr(self._impl, 'diag_AHA_scalar'):\n"
            "            try:\n"
            "                alpha = float(self._impl.diag_AHA_scalar())\n"
            "                self._alpha = torch.tensor(alpha, dtype=self.dtype_r, device='cpu')\n"
            "                return self._alpha.to(device or 'cpu')\n"
            "            except Exception:\n"
            "                pass\n"
        )

    # Add diag_AHA_profile passthrough if missing
    if "def diag_AHA_profile(" not in txt:
        txt = txt.rstrip() + (
            "\n\n    @torch.no_grad()\n"
            "    def diag_AHA_profile(self) -> Optional[torch.Tensor]:\n"
            "        \"\"\"Optional per-frame α_b from the backend; None if unsupported.\"\"\"\n"
            "        if hasattr(self._impl, 'diag_AHA_profile'):\n"
            "            try:\n"
            "                return self._impl.diag_AHA_profile()\n"
            "            except Exception:\n"
            "                return None\n"
            "        return None\n"
        )

    if txt != orig:
        backup(p)
        p.write_text(txt, encoding="utf-8")
        print(f"[patch] {p}")
    else:
        print(f"[ok] {p} already patched")


def main():
    # Try to locate repo root by presence of 'graspcg/nufft'
    if not PKG.exists():
        print(f"[error] Could not find package folder at {PKG}")
        sys.exit(1)

    patch_torchkb()
    patch_cufi()
    patch_op()
    print("\nDone. Re-run your tests.\n")


if __name__ == "__main__":
    main()
