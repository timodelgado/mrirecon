#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply Option B (strong simplification) to the MRI recon library.

Changes (idempotent):
  • Standardize workspace keys to: var / dir / grad / diag (solver, objective, manager)
  • Switch to generic stats (no legacy keys): obj/total, term/data/energy,
    term/reg/total/energy, term/reg/<name>/energy, slope/dir, probe/{q,mad}/<name>
  • Remove residual weighting W from Objective (and drop `weight_mode`)
  • Make RegManager probe- and policy-agnostic (no TV checks/histograms)
  • Add sample_probe() to TVND and MappedRegularizer; Manager uses it if present
  • StatsBoard: replace tv_* knobs with probe_* knobs (no aliases)

Backups:
  • Writes <file>.bak the first time a file is changed.

Validation:
  • After patching, attempts AST-parse of modified files to catch syntax errors.
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


# ─────────────────────────── utilities ───────────────────────────

def find_first(paths: List[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None

def search_repo_for(basedir: Path, filename: str) -> List[Path]:
    return [p for p in basedir.rglob(filename) if p.is_file()]

def backup_once(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)

def replace_exact(s: str, before: str, after: str, tag: str, log: List[str]) -> str:
    if before in s and after not in s:
        s = s.replace(before, after)
        log.append(f"  - {tag}: replaced exact block")
    else:
        if after in s:
            log.append(f"  - {tag}: already present")
        else:
            log.append(f"  - {tag}: not found (skipped)")
    return s

def replace_regex(s: str, pattern: str, repl, tag: str, log: List[str], flags=re.S|re.M) -> str:
    if re.search(pattern, s, flags):
        s_new, n = re.subn(pattern, repl, s, flags=flags)
        if n > 0:
            log.append(f"  - {tag}: {n} replacement(s)")
            return s_new
    else:
        log.append(f"  - {tag}: pattern not found (skipped)")
    return s

def write_if_changed(path: Path, src: str, orig: str, log: List[str]) -> None:
    if src != orig:
        backup_once(path)
        path.write_text(src, encoding="utf-8")
        log.append(f"✓ Patched {path}")
    else:
        log.append(f"= No changes needed in {path}")


# ─────────────────────────── patches ───────────────────────────

def patch_stats_board(path: Path, log: List[str]) -> None:
    s = path.read_text(encoding="utf-8"); orig = s
    # Replace tv_* knobs with probe_* knobs; update comment example
    s = replace_regex(
        s,
        r"self\.tv_percentile\s*:\s*float\s*=\s*0\.90\s*\n\s*self\.tv_sample_K\s*:\s*int\s*=\s*4096",
        "self.probe_percentile: float = 0.90\n        self.probe_sample_K: int = 4096",
        "stats_board: tv_* -> probe_* knobs",
        log)
    s = replace_regex(
        s,
        r'Feature toggles\s*\(e\.g\.,\s*"[^"]*"\)',
        'Feature toggles (e.g., "term_reg_energy", "probe_quantile", "probe_mad")',
        "stats_board: toggle comment example",
        log)
    write_if_changed(path, s, orig, log)


def patch_objective(path: Path, log: List[str]) -> None:
    s = path.read_text(encoding="utf-8"); orig = s

    # Drop Literal import
    s = replace_regex(s,
        r"from\s+typing\s+import\s+([^#\n]*?)\bLiteral\b\s*,?\s*",
        lambda m: "from typing import " + ", ".join([x.strip() for x in m.group(1).split(",") if x.strip()]) + "\n",
        "objective: remove Literal from typing import",
        log)

    # Remove weight_mode from ObjectiveConfig + its docstring bullet
    s = replace_regex(
        s,
        r"•\s*weight_mode\s*:.*?\n",
        "",
        "objective: drop weight_mode docstring",
        log)
    s = replace_regex(
        s,
        r"\s*weight_mode\s*:\s*[^=\n]+\s*=\s*['\"][^'\"]+['\"]\s*\n",
        "",
        "objective: drop weight_mode field",
        log)

    # Remove residual weighting W: kernel signature/body + call-sites + _W + set_weight
    s = replace_regex(
        s,
        r"def\s+_ker_residual_energy_\(\s*r:\s*torch\.Tensor,\s*Ax:\s*torch\.Tensor,\s*Ad:\s*torch\.Tensor,\s*y:\s*torch\.Tensor,\s*t:\s*torch\.Tensor,\s*W:\s*Optional\[torch\.Tensor\]\s*=\s*None\)\s*->\s*torch\.Tensor:\s*[\s\S]*?return\s*\(r\.conj\(\)\s*\*\s*r\)\.real\.sum\(\)\s*",
        (
            "def _ker_residual_energy_(r: torch.Tensor,\n"
            "                          Ax: torch.Tensor,\n"
            "                          Ad: torch.Tensor,\n"
            "                          y: torch.Tensor,\n"
            "                          t: torch.Tensor) -> torch.Tensor:\n"
            "    \"\"\"\n"
            "    In-place residual + energy on device.\n"
            "      r = Ax + t*Ad − y\n"
            "      return ||r||^2  (0-D REAL tensor)\n"
            "    \"\"\"\n"
            "    r.copy_(Ad)\n"
            "    r.mul_(t)\n"
            "    r.add_(Ax)\n"
            "    r.sub_(y)\n"
            "    return (r.conj() * r).real.sum()\n"
        ),
        "objective: remove W from residual kernel",
        log)

    s = replace_exact(
        s,
        "e = self._ker_residual_energy(r, Ax, Ad, y_slice, t_dev, self._W)",
        "e = self._ker_residual_energy(r, Ax, Ad, y_slice, t_dev)",
        "objective: drop W at sharded callsite",
        log)
    s = replace_exact(
        s,
        "e = self._ker_residual_energy(r, self._Ax_g, self._Ad_g, self.y, t_y, self._W)",
        "e = self._ker_residual_energy(r, self._Ax_g, self._Ad_g, self.y, t_y)",
        "objective: drop W at global callsite",
        log)

    s = replace_regex(
        s,
        r"\n\s*self\._W\s*:\s*Optional\[torch\.Tensor\]\s*=\s*None\s*\n",
        "\n",
        "objective: remove _W attribute",
        log)

    s = replace_regex(
        s,
        r"def\s+set_weight\(self,\s*W:\s*Optional\[torch\.Tensor\]\)\s*->\s*None:\s*[\s\S]*?(?=\n\s*@|\n\s*def\s+)",
        (
            "def set_weight(self, W: Optional[torch.Tensor]) -> None:\n"
            "        raise RuntimeError(\"Residual weighting was removed; apply DCF/prewhitening inside A/AH.\")\n"
        ),
        "objective: replace set_weight with hard error",
        log)

    # Stats: generic names only
    s = replace_regex(
        s,
        r'sb\.scalar_slot\("E_data",\s*([^)]+)\)\.add_\(',
        r'sb.scalar_slot("term/data/energy", \1).add_(',
        "objective: write term/data/energy (sharded)",
        log)
    s = replace_regex(
        s,
        r'sb\.scalar_slot\("gdot",\s*([^)]+)\)\.add_\(',
        r'sb.scalar_slot("slope/dir", \1).add_(',
        "objective: write slope/dir",
        log)
    s = replace_regex(
        s,
        r'sb\.scalar_slot\("f_total",\s*([^)]+)\)\.add_\(',
        r'sb.scalar_slot("obj/total", \1).add_(',
        "objective: write obj/total",
        log)

    # Workspace buffers: x/g/dx -> var/grad/dir
    s = replace_regex(s, r'ws\.bind\(i,\s*"x",\s*"dx"\)', 'ws.bind(i, "var", "dir")',
                      "objective: ws.bind(x,dx)->(var,dir)", log)
    s = replace_regex(s, r'ws\.get\("g",\s*i\)', 'ws.get("grad", i)',
                      "objective: ws.get('g')->'grad'", log)
    s = replace_regex(s, r'ws\.bind\(i,\s*"g",\s*"dx"\)', 'ws.bind(i, "grad", "dir")',
                      "objective: ws.bind(g,dx)->(grad,dir)", log)
    s = replace_regex(s, r'ws\.get\("x",\s*i\)', 'ws.get("var", i)',
                      "objective: ws.get('x')->'var'", log)
    s = replace_regex(s, r'ws\.get\("dx",\s*i\)', 'ws.get("dir", i)',
                      "objective: ws.get('dx')->'dir'", log)

    # Clear grad loop
    s = replace_regex(s, r'ws\.get\("g",\s*i\)\.zero_\(\)', 'ws.get("grad", i).zero_()',
                      "objective: zero grad", log)

    write_if_changed(path, s, orig, log)


def patch_manager(path: Path, log: List[str]) -> None:
    s = path.read_text(encoding="utf-8"); orig = s

    # Strip TV/ndops imports in energy_and_grad (we'll replace the whole method)
    s = replace_regex(s, r"@torch\.no_grad\(\)\s*def\s+energy_and_grad\(self,\s*ws\)\s*->\s*torch\.Tensor:[\s\S]*?return\s+total",
        (
            "@torch.no_grad()\n"
            "def energy_and_grad(self, ws) -> torch.Tensor:\n"
            "        \"\"\"\n"
            "        Total regularization energy; accumulates ∂E/∂x into ws.grad.\n"
            "        Generic stats only:\n"
            "          • per‑reg energy → 'term/reg/<name>/energy' (if enabled)\n"
            "          • total reg energy → 'term/reg/total/energy'\n"
            "          • probes (quantile/MAD) → 'probe/q/<name>' / 'probe/mad/<name>'\n"
            "        \"\"\"\n"
            "        import contextlib\n"
            "        from .base import RegContext\n"
            "\n"
            "        roles = ws.plan.roles_image\n"
            "        halo  = self._aggregate_halo(roles)\n"
            "        n_shards = self._num_shards(ws)\n"
            "\n"
            "        # Resolve axes once per reg (compile-static)\n"
            "        reg_axes = []\n"
            "        for reg in self._regs:\n"
            "            axes_spec = getattr(getattr(reg, \"params\", None), \"axes\", \"spatial\")\n"
            "            reg_axes.append(self._resolve_axes(axes_spec, roles))\n"
            "\n"
            "        sb = getattr(ws, \"stats\", None)\n"
            "        collect_termE = bool(sb and sb.enabled.get(\"term_reg_energy\", False))\n"
            "        collect_q     = bool(sb and sb.enabled.get(\"probe_quantile\", False))\n"
            "        collect_mad   = bool(sb and sb.enabled.get(\"probe_mad\", False))\n"
            "        qv_target = float(getattr(sb, \"probe_percentile\", 0.90)) if sb is not None else 0.90\n"
            "        sample_K  = int(getattr(sb, \"probe_sample_K\", 4096)) if sb is not None else 4096\n"
            "        sample_K  = max(1, sample_K)\n"
            "\n"
            "        dev_order = []\n"
            "        dev_accum: Dict = {}\n"
            "        samples_by_reg: Dict = {}\n"
            "\n"
            "        for sh, i in ws.iter_shards():\n"
            "            dev = sh.device\n"
            "            if dev not in dev_accum:\n"
            "                x_dtype_r = ws.get(\"var\", i).real.dtype\n"
            "                dev_accum[dev] = torch.zeros((), device=dev, dtype=x_dtype_r)\n"
            "                dev_order.append(dev)\n"
            "\n"
            "            stream = ws.arena.stream_for(dev) if getattr(dev, \"type\", \"cpu\") == \"cuda\" else None\n"
            "            ctx_mgr = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()\n"
            "            with ctx_mgr:\n"
            "                x = ws.get(\"var\", i)\n"
            "                g = ws.get(\"grad\", i)\n"
            "                D = ws.get(\"diag\", i) if getattr(ws, \"has\", lambda _: False)(\"diag\") else None\n"
            "\n"
            "                x_ext, interior = self._with_halo_x(ws, i, n_shards, halo, anchor=x)\n"
            "\n"
            "                for reg, axes in zip(self._regs, reg_axes):\n"
            "                    ctx = RegContext(\n"
            "                        x=x_ext, g=g, diag=D, roles_image=roles, device=dev,\n"
            "                        dtype_c=x.dtype, dtype_r=g.real.dtype,\n"
            "                        axes_resolver=lambda spec: self._resolve_axes(spec, roles),\n"
            "                        arena=ws.arena, write_interior_slice=interior,\n"
            "                        ws=ws, shard_index=i, halo_map=halo,\n"
            "                    )\n"
            "\n"
            "                    fn = self._compiled_fixed_axes(reg, axes, self._kernel_signature_axes(reg, axes, ctx))\n"
            "                    e_shard = fn(ctx)\n"
            "                    dev_accum[dev].add_(e_shard)\n"
            "\n"
            "                    if collect_termE and sb is not None:\n"
            "                        sb.scalar_slot(f\"term/reg/{reg.name}/energy\", dev, e_shard.dtype).add_(e_shard)\n"
            "\n"
            "                    if (collect_q or collect_mad) and sb is not None:\n"
            "                        smp_fn = getattr(reg, \"sample_probe\", None)\n"
            "                        if callable(smp_fn):\n"
            "                            K_shard = max(1, (sample_K + n_shards - 1) // max(1, n_shards))\n"
            "                            smp = smp_fn(ctx, axes, K_shard)\n"
            "                            if (smp is not None) and int(smp.numel()) > 0:\n"
            "                                samples_by_reg.setdefault(reg.name, []).append(smp)\n"
            "\n"
            "        if not dev_order:\n"
            "            return torch.zeros((), device=\"cpu\", dtype=torch.float32)\n"
            "\n"
            "        primary = dev_order[0]\n"
            "        dtype_r = next(iter(dev_accum.values())).dtype\n"
            "        total = torch.zeros((), device=primary, dtype=dtype_r)\n"
            "        for dev in dev_order:\n"
            "            v = dev_accum[dev]\n"
            "            total.add_(v if dev == primary else v.to(primary, non_blocking=True))\n"
            "\n"
            "        if sb is not None:\n"
            "            sb.scalar_slot(\"term/reg/total/energy\", primary, dtype_r).add_(total)\n"
            "\n"
            "            if (collect_q or collect_mad) and samples_by_reg:\n"
            "                for name, chunks in samples_by_reg.items():\n"
            "                    if not chunks:\n"
            "                        continue\n"
            "                    if len(chunks) == 1 and chunks[0].device == primary:\n"
            "                        S = chunks[0]\n"
            "                    else:\n"
            "                        S = torch.cat([c if c.device == primary else c.to(primary, non_blocking=True) for c in chunks], dim=0)\n"
            "                    if int(S.numel()) == 0:\n"
            "                        continue\n"
            "                    if collect_q:\n"
            "                        qv = torch.quantile(S, qv_target)\n"
            "                        sb.scalar_slot(f\"probe/q/{name}\", primary, qv.dtype).add_(qv)\n"
            "                    if collect_mad:\n"
            "                        med = torch.quantile(S, 0.5)\n"
            "                        mad = torch.quantile((S - med).abs(), 0.5)\n"
            "                        sb.scalar_slot(f\"probe/mad/{name}\", primary, mad.dtype).add_(mad)\n"
            "\n"
            "        return total\n"
        ),
        "manager: replace energy_and_grad with generic/probe version",
        log)

    # Replace x/g/dx -> var/grad/dir elsewhere in manager (add_diag, _with_halo_x)
    s = replace_regex(s, r'ws\.get\("x",\s*i\)', 'ws.get("var", i)',
                      "manager: ws.get('x')->'var'", log)
    s = replace_regex(s, r'ws\.get\("x",\s*shard_idx\)', 'ws.get("var", shard_idx)',
                      "manager: _with_halo_x read", log)
    s = replace_regex(s, r'ws\.get\("x",\s*shard_idx\s*-\s*1\)', 'ws.get("var", shard_idx - 1)',
                      "manager: _with_halo_x prev", log)
    s = replace_regex(s, r'ws\.get\("x",\s*shard_idx\s*\+\s*1\)', 'ws.get("var", shard_idx + 1)',
                      "manager: _with_halo_x next", log)
    s = replace_regex(s, r'ws\.get\("g",\s*i\)', 'ws.get("grad", i)',
                      "manager: ws.get('g')->'grad'", log)

    # Stats key names: E_reg/* -> term/reg/*/energy ; E_reg_total -> term/reg/total/energy
    s = replace_regex(s, r'sb\.scalar_slot\("E_reg_total",\s*([^,]+),\s*([^)]+)\)\.add_\(',
                      r'sb.scalar_slot("term/reg/total/energy", \1, \2).add_(',
                      "manager: write term/reg/total/energy", log)
    s = replace_regex(s, r'sb\.scalar_slot\(f"E_reg/\{reg\.name\}"\s*,\s*dev\s*,\s*e_shard\.dtype\)\.add_\(',
                      r'sb.scalar_slot(f"term/reg/{reg.name}/energy", dev, e_shard.dtype).add_(',
                      "manager: write term/reg/<name>/energy", log)

    # Remove any leftover tv_quantile code paths (safely scrub toggles)
    s = replace_regex(s, r'enabled\.get\("tv_quantile"[^\)]*\)', 'enabled.get("probe_quantile", False)',
                      "manager: scrub tv_quantile toggle", log)

    write_if_changed(path, s, orig, log)


def patch_mapping(path: Path, log: List[str]) -> None:
    s = path.read_text(encoding="utf-8"); orig = s

    # Remove the early duplicate add_diag stub (identity‑only)
    s = replace_regex(
        s,
        r"\n\s*def\s+add_diag\(self,\s*ctx:\s*RegContext\)\s*->\s*None:\s*\n\s*#.*?return\s+None\s*\n",
        "\n    # (removed earlier duplicate add_diag stub; operator‑aware version below remains)\n",
        "mapping: remove early add_diag stub",
        log)

    # Replace quantile_sample_shard(...) with sample_probe(..., K_shard) (no q)
    s = replace_regex(
        s,
        r"def\s+quantile_sample_shard\(self,\s*ctx:\s*RegContext,\s*axes:\s*Tuple\[int,\s*\.\.\.\],\s*K_shard:\s*int,\s*q:\s*float\)\s*->\s*Optional\[torch\.Tensor\]:[\s\S]*?return\s+sample\s*",
        (
            "def sample_probe(self, ctx: RegContext, axes: Tuple[int, ...], K_shard: int) -> Optional[torch.Tensor]:\n"
            "        \"\"\"\n"
            "        Return a small 1‑D tensor of energy‑density samples from the mapped field u.\n"
            "        Default implementation (TV-like): uniform striding over per-voxel TV energy.\n"
            "        \"\"\"\n"
            "        interior = ctx.write_interior_slice or (slice(None),) * ctx.x.ndim\n"
            "        u_ext = self.op.forward_apply(ctx.x, ctx, interior)\n"
            "        grads = [fwd_diff(u_ext, ax) for ax in axes]\n"
            "\n"
            "        eps  = torch.as_tensor(getattr(getattr(self.inner, \"params\", None), \"eps\", 0.0),\n"
            "                               device=u_ext.device, dtype=ctx.dtype_r)\n"
            "        isot = bool(getattr(getattr(self.inner, \"params\", None), \"isotropic\", True))\n"
            "\n"
            "        e_den_full = tv_iso_energy(grads, eps) if isot else tv_aniso_energy(grads, eps)\n"
            "        e_den = e_den_full[interior] if interior is not None else e_den_full\n"
            "        flat = e_den.reshape(-1)\n"
            "        nvox = int(flat.numel())\n"
            "        if nvox == 0:\n"
            "            return None\n"
            "\n"
            "        K = max(1, int(K_shard))\n"
            "        stride = max(1, nvox // K)\n"
            "        sample = flat[::stride]\n"
            "        if int(sample.numel()) > K:\n"
            "            sample = sample[:K]\n"
            "        return sample\n"
        ),
        "mapping: quantile_sample_shard -> sample_probe",
        log)

    write_if_changed(path, s, orig, log)


def patch_tvnd(path: Path, log: List[str]) -> None:
    s = path.read_text(encoding="utf-8"); orig = s

    if "def sample_probe(" not in s:
        # Insert after energy_grad_fixed_axes()
        s = replace_regex(
            s,
            r"(def\s+energy_grad_fixed_axes\(self,\s*ctx:\s*RegContext,\s*axes:\s*Tuple\[int,\s*\.\.\.\]\)\s*->\s*torch\.Tensor:[\s\S]*?return\s+E[^\n]*\n)",
            r"\1\n"
            "    def sample_probe(self, ctx: RegContext, axes: Tuple[int, ...], K_shard: int):\n"
            "        \"\"\"Uniform-stride samples from TV energy density on ctx.x (halo-aware).\"\"\"\n"
            "        dev, dr = ctx.device, ctx.dtype_r\n"
            "        x_ext = ctx.x\n"
            "        interior = ctx.write_interior_slice or (slice(None),) * x_ext.ndim\n"
            "        grads = [fwd_diff(x_ext, ax) for ax in axes]\n"
            "        eps = torch.tensor(self.params.eps, device=dev, dtype=dr)\n"
            "        if self.params.isotropic:\n"
            "            e_den_full = tv_iso_energy(grads, eps)\n"
            "        else:\n"
            "            e_den_full = tv_aniso_energy(grads, eps)\n"
            "        e_den = e_den_full[interior] if interior is not None else e_den_full\n"
            "        flat = e_den.reshape(-1)\n"
            "        nvox = int(flat.numel())\n"
            "        if nvox == 0:\n"
            "            return None\n"
            "        K = max(1, int(K_shard))\n"
            "        stride = max(1, nvox // K)\n"
            "        sample = flat[::stride]\n"
            "        if int(sample.numel()) > K:\n"
            "            sample = sample[:K]\n"
            "        return sample\n",
            "tv_nd: add sample_probe() to TVND",
            log)
    else:
        log.append("  - tv_nd: sample_probe already present")

    write_if_changed(path, s, orig, log)


def patch_policy(path: Path, log: List[str]) -> None:
    s = path.read_text(encoding="utf-8"); orig = s

    # Remove TV import
    s = replace_regex(s, r"\nfrom\s+\.\s*tv_nd\s+import\s+TVND[^\n]*\n", "\n# (generic policy; no reg-type imports)\n", "policy: remove TVND import", log)

    # Replace RegPolicyConfig with generic keys (quantile/MAD + term_reg_energy)
    s = replace_regex(
        s,
        r"@dataclass\s*\nclass\s+RegPolicyConfig:[\s\S]*?clear_after_read:\s*bool\s*=\s*True\s*",
        (
            "@dataclass\n"
            "class RegPolicyConfig:\n"
            "    # ---- Which stats to collect (generic) ----\n"
            "    probe_quantile_key: str = \"probe_quantile\"\n"
            "    probe_mad_key: str      = \"probe_mad\"\n"
            "    term_energy_key: str    = \"term_reg_energy\"\n"
            "\n"
            "    # ---- ε (Huber knee) updates ----\n"
            "    enable_eps_from_percentile: bool = True\n"
            "    eps_percentile: float = 0.90\n"
            "    enable_eps_from_mad: bool = True\n"
            "    mad_to_eps_scale: float = 1.4826\n"
            "    eps_floor: float = 1e-6\n"
            "    eps_ema_alpha: float = 0.6\n"
            "\n"
            "    # ---- λ from relative energy ratio (reg/data) ----\n"
            "    enable_lambda_from_ratio: bool = True\n"
            "    ratio_target_global: float | None = 0.10\n"
            "    ratio_target_per_reg: dict[str, float] | None = None\n"
            "    ratio_eta: float = 0.5\n"
            "    lambda_ema_alpha: float = 0.6\n"
            "\n"
            "    # ---- Preconditioner (diag) management ----\n"
            "    manage_preconditioner: bool = True\n"
            "    update_diag_after_change: bool = True\n"
            "    precond_refresh_every: int | None = None\n"
            "    precond_base: float = 1.0\n"
            "\n"
            "    # ---- Bookkeeping ----\n"
            "    clear_after_read: bool = True\n"
        ),
        "policy: replace RegPolicyConfig",
        log)

    # prepare_collection with generic toggles
    s = replace_regex(
        s,
        r"def\s+prepare_collection\(self,\s*ws,\s*regm\)\s*->\s*None:[\s\S]*?(?=\n\s*def\s+prime_from_stats)",
        (
            "def prepare_collection(self, ws, regm) -> None:\n"
            "        sb = getattr(ws, \"stats\", None)\n"
            "        if sb is None:\n"
            "            return\n"
            "\n"
            "        regs = list(getattr(regm, \"_regs\", []))\n"
            "        has_eps = any(hasattr(getattr(r, \"params\", None), \"eps\") for r in regs)\n"
            "        need_probes = has_eps and (self.cfg.enable_eps_from_percentile or self.cfg.enable_eps_from_mad)\n"
            "        self._sb_enable(sb, self.cfg.probe_quantile_key, need_probes)\n"
            "        self._sb_enable(sb, self.cfg.probe_mad_key,      need_probes)\n"
            "\n"
            "        need_term_energy = False\n"
            "        if self.cfg.enable_lambda_from_ratio:\n"
            "            if self.cfg.ratio_target_per_reg and len(self.cfg.ratio_target_per_reg) > 0:\n"
            "                need_term_energy = True\n"
            "            elif len(regs) > 1 and (self.cfg.ratio_target_global is not None):\n"
            "                need_term_energy = True\n"
            "        self._sb_enable(sb, self.cfg.term_energy_key, need_term_energy)\n"
            "\n"
            "        if self.cfg.clear_after_read:\n"
            "            if need_probes:\n"
            "                for r in regs:\n"
            "                    self._reset_scalar(sb, f\"probe/q/{r.name}\")\n"
            "                    self._reset_scalar(sb, f\"probe/mad/{r.name}\")\n"
            "            if need_term_energy:\n"
            "                for r in regs:\n"
            "                    self._reset_scalar(sb, f\"term/reg/{r.name}/energy\")\n"
            "            for k in (\"term/data/energy\", \"term/reg/total/energy\", \"slope/dir\", \"obj/total\"):\n"
            "                self._reset_scalar(sb, k)\n"
        ),
        "policy: generic prepare_collection",
        log)

    # Replace _update_from_stats_impl core (ε and λ with generic keys)
    s = replace_regex(
        s,
        r"def\s+_update_from_stats_impl\([^\)]*\)\s*->\s*bool:[\s\S]*?return\s+changed_any",
        (
            "def _update_from_stats_impl(self, ws, regm, *, use_ema: bool) -> bool:\n"
            "        sb = getattr(ws, \"stats\", None)\n"
            "        if sb is None:\n"
            "            return False\n"
            "\n"
            "        changed_any = False\n"
            "\n"
            "        # ε via probe quantile and/or MAD\n"
            "        if self.cfg.enable_eps_from_percentile or self.cfg.enable_eps_from_mad:\n"
            "            regs = list(getattr(regm, \"_regs\", []))\n"
            "            for reg in regs:\n"
            "                params = getattr(reg, \"params\", None)\n"
            "                if not hasattr(params, \"eps\"):\n"
            "                    continue\n"
            "                eps_est = 0.0\n"
            "                if self.cfg.enable_eps_from_percentile:\n"
            "                    eps_est = float(self._sb_read_scalar(sb, f\"probe/q/{reg.name}\"))\n"
            "                if (eps_est <= 0.0) and self.cfg.enable_eps_from_mad:\n"
            "                    mad = float(self._sb_read_scalar(sb, f\"probe/mad/{reg.name}\"))\n"
            "                    if mad > 0.0:\n"
            "                        eps_est = mad * float(self.cfg.mad_to_eps_scale)\n"
            "                if eps_est <= 0.0:\n"
            "                    continue\n"
            "                eps_est = max(eps_est, float(self.cfg.eps_floor))\n"
            "                eps_old = float(getattr(params, \"eps\", 0.0))\n"
            "                eps_new = self._ema(eps_old, eps_est, self.cfg.eps_ema_alpha) if use_ema else eps_est\n"
            "                if eps_new != eps_old:\n"
            "                    reg.params = replace(params, eps=eps_new)\n"
            "                    changed_any = True\n"
            "\n"
            "        # λ via energy ratio\n"
            "        if self.cfg.enable_lambda_from_ratio:\n"
            "            E_data = self._sb_read_scalar(sb, \"term/data/energy\")\n"
            "            if E_data <= 0.0:\n"
            "                E_data = 1e-30\n"
            "            regs = list(getattr(regm, \"_regs\", []))\n"
            "            many = len(regs) > 1\n"
            "            for reg in regs:\n"
            "                rho_t = None\n"
            "                if self.cfg.ratio_target_per_reg and reg.name in self.cfg.ratio_target_per_reg:\n"
            "                    rho_t = float(self.cfg.ratio_target_per_reg[reg.name])\n"
            "                elif (self.cfg.ratio_target_global is not None) and not many:\n"
            "                    rho_t = float(self.cfg.ratio_target_global)\n"
            "                if rho_t is None:\n"
            "                    continue\n"
            "                E_reg = 0.0\n"
            "                if self._sb_enabled(sb, self.cfg.term_energy_key):\n"
            "                    E_reg = self._sb_read_scalar(sb, f\"term/reg/{reg.name}/energy\")\n"
            "                if (E_reg <= 0.0) and not many:\n"
            "                    E_reg = self._sb_read_scalar(sb, \"term/reg/total/energy\")\n"
            "                if E_reg <= 0.0:\n"
            "                    continue\n"
            "                rho = E_reg / max(E_data, 1e-30)\n"
            "                lam_old = float(getattr(reg.params, \"weight\", 0.0))\n"
            "                mult = (rho_t / max(rho, 1e-30)) ** float(self.cfg.ratio_eta)\n"
            "                lam_est = lam_old * mult\n"
            "                lam_new = self._ema(lam_old, lam_est, self.cfg.lambda_ema_alpha) if use_ema else lam_est\n"
            "                if lam_new != lam_old:\n"
            "                    reg.params = replace(reg.params, weight=lam_new)\n"
            "                    changed_any = True\n"
            "\n"
            "        if changed_any and self.cfg.manage_preconditioner and self.cfg.update_diag_after_change:\n"
            "            self._rebuild_preconditioner(ws, regm)\n"
            "\n"
            "        if self.cfg.clear_after_read:\n"
            "            self._clear_consumed(sb, regm)\n"
            "        return changed_any\n"
        ),
        "policy: replace _update_from_stats_impl with generic version",
        log)

    # Replace _clear_consumed to generic
    s = replace_regex(
        s,
        r"def\s+_clear_consumed\(self,\s*sb,\s*regm\)\s*->\s*None:[\s\S]*?(?=\n\s*def\s+|$)",
        (
            "def _clear_consumed(self, sb, regm) -> None:\n"
            "        for k in (\"term/data/energy\", \"term/reg/total/energy\", \"slope/dir\", \"obj/total\"):\n"
            "            self._reset_scalar(sb, k)\n"
            "        if self._sb_enabled(sb, self.cfg.term_energy_key):\n"
            "            for r in getattr(regm, \"_regs\", []):\n"
            "                self._reset_scalar(sb, f\"term/reg/{r.name}/energy\")\n"
            "        if self._sb_enabled(sb, self.cfg.probe_quantile_key) or self._sb_enabled(sb, self.cfg.probe_mad_key):\n"
            "            for r in getattr(regm, \"_regs\", []):\n"
            "                self._reset_scalar(sb, f\"probe/q/{r.name}\")\n"
            "                self._reset_scalar(sb, f\"probe/mad/{r.name}\")\n"
        ),
        "policy: generic _clear_consumed",
        log)

    # Remove histogram helpers if present
    s = replace_regex(s, r"def\s+_sb_read_reg_hist[\s\S]*?return\s+None\s*", "", "policy: drop histogram helper 1", log)
    s = replace_regex(s, r"def\s+_sb_hist_percentile[\s\S]*?return\s+float\(sb\.hist_percentile", "", "policy: drop histogram helper 2", log)
    s = replace_regex(s, r"def\s+_reset_tv_hist[\s\S]*?pass\s*", "", "policy: drop histogram helper 3", log)

    write_if_changed(path, s, orig, log)


def patch_cg(path: Path, log: List[str]) -> None:
    s = path.read_text(encoding="utf-8"); orig = s

    # Bindings: g/dx/x -> grad/dir/var
    s = replace_exact(s, 'g, d, D = ws.bind(i, "g", "dx", "diag")', 'g, d, D = ws.bind(i, "grad", "dir", "diag")',
                      "cg: ws.bind(g,dx,diag)->(grad,dir,diag)", log)
    s = replace_exact(s, 'x, d = ws.bind(i, "x", "dx")', 'x, d = ws.bind(i, "var", "dir")',
                      "cg: ws.bind(x,dx)->(var,dir)", log)
    s = replace_regex(s, r'ws\.bind\(i,\s*"g",\s*"dx"\)', 'ws.bind(i, "grad", "dir")',
                      "cg: other ws.bind(g,dx)", log)

    write_if_changed(path, s, orig, log)


# ─────────────────────────── main ───────────────────────────

def ast_check(paths: List[Path], log: List[str]) -> None:
    errs: List[Tuple[Path, str]] = []
    for p in paths:
        try:
            ast.parse(p.read_text(encoding="utf-8"))
        except Exception as e:
            errs.append((p, f"{type(e).__name__}: {e}"))
    if errs:
        log.append("! Syntax check failed in:")
        for p, msg in errs:
            log.append(f"    {p}: {msg}")
    else:
        log.append("✓ AST syntax check passed for modified files")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=str, default=".", help="Repository root (default: .)")
    args = ap.parse_args()
    root = Path(args.repo).resolve()

    # Locate files (prefer canonical package paths)
    cand = {
        "cg": [
            root / "graspcg" / "solvers" / "cg.py",
            root / "solvers" / "cg.py",
            root / "cg.py",
        ],
        "stats": [
            root / "graspcg" / "regularization" / "stats_board.py",
            root / "regularization" / "stats_board.py",
            root / "stats_board.py",
        ],
        "objective": [
            root / "graspcg" / "ops" / "objective.py",
            root / "ops" / "objective.py",
            root / "objective.py",
        ],
        "manager": [
            root / "graspcg" / "regularization" / "manager.py",
            root / "regularization" / "manager.py",
            root / "manager.py",
        ],
        "mapping": [
            root / "graspcg" / "regularization" / "mapping.py",
            root / "regularization" / "mapping.py",
            root / "mapping.py",
        ],
        "tvnd": [
            root / "graspcg" / "regularization" / "tv_nd.py",
            root / "regularization" / "tv_nd.py",
            root / "tv_nd.py",
        ],
        "policy": [
            root / "graspcg" / "policies" / "reg_policies.py",
            root / "policies" / "reg_policies.py",
            root / "reg_policies.py",
        ],
    }

    files: Dict[str, Path] = {}
    for k, lst in cand.items():
        p = find_first(lst)
        if p is None:
            hits = search_repo_for(root, Path(lst[0]).name)
            p = hits[0] if hits else None
        files[k] = p

    print("Patch targets:")
    for k, p in files.items():
        print(f"  {k:10}: {p if p else '(not found)'}")

    log: List[str] = []
    modified: List[Path] = []

    # Apply patches
    if files["stats"]:
        patch_stats_board(files["stats"], log); modified.append(files["stats"])
    else:
        log.append("! stats_board.py not found")

    if files["objective"]:
        patch_objective(files["objective"], log); modified.append(files["objective"])
    else:
        log.append("! objective.py not found")

    if files["manager"]:
        patch_manager(files["manager"], log); modified.append(files["manager"])
    else:
        log.append("! manager.py not found")

    if files["mapping"]:
        patch_mapping(files["mapping"], log); modified.append(files["mapping"])
    else:
        log.append("! mapping.py not found")

    if files["tvnd"]:
        patch_tvnd(files["tvnd"], log); modified.append(files["tvnd"])
    else:
        log.append("! tv_nd.py not found")

    if files["policy"]:
        patch_policy(files["policy"], log); modified.append(files["policy"])
    else:
        log.append("! reg_policies.py not found")

    if files["cg"]:
        patch_cg(files["cg"], log); modified.append(files["cg"])
    else:
        log.append("! cg.py not found")

    # Syntax check
    ast_check([p for p in modified if p is not None], log)

    print("\nChange log:")
    for line in log:
        print(line)


if __name__ == "__main__":
    main()
