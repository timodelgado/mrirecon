#!/usr/bin/env python3
"""
CRISP v1 applier — Code Review Instruction Set for Patching
Applies a CRISP JSON document to a working directory.

Usage:
  python crisp_apply.py --review path/to/review.json --root /path/to/repo --in-place
  # omit --in-place for a dry run; a unified diff will be printed either way
"""
from __future__ import annotations
import argparse, json, os, re, glob, difflib, copy
from typing import Any, Dict, List, Tuple, Iterable

REGEX_FLAGS = {
    "IGNORECASE": re.IGNORECASE,
    "MULTILINE": re.MULTILINE,
    "DOTALL": re.DOTALL,
}

def _read_text(path: str) -> Tuple[str, str]:
    with open(path, "rb") as f:
        raw = f.read()
    # naive newline detection
    text = raw.decode("utf-8")
    eol = "\r\n" if "\r\n" in text and not "\n" in text.replace("\r\n", "") else "\n"
    return text, eol

def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)

def _unified_diff(a: str, b: str, path: str) -> str:
    return "\n".join(difflib.unified_diff(
        a.splitlines(), b.splitlines(), fromfile=f"a/{path}", tofile=f"b/{path}", lineterm=""
    ))

def _compile_matcher(m: Dict[str, Any]):
    mtype = m.get("type")
    pattern = m.get("pattern", "")
    if mtype == "regex":
        flags = 0
        for name in m.get("flags", []):
            flags |= REGEX_FLAGS.get(name, 0)
        rx = re.compile(pattern, flags)
        return ("regex", rx)
    elif mtype == "literal":
        return ("literal", pattern)
    else:
        raise ValueError("match.type must be 'literal' or 'regex'")

def _find_spans(text: str, matcher) -> List[Tuple[int, int]]:
    mtype, spec = matcher
    spans = []
    if mtype == "literal":
        start = 0
        lit = spec
        if lit == "":
            return []
        while True:
            i = text.find(lit, start)
            if i == -1:
                break
            spans.append((i, i + len(lit)))
            start = i + len(lit)
    else:
        for m in spec.finditer(text):
            spans.append(m.span())
    return spans

def _select_spans(spans: List[Tuple[int, int]], occ) -> List[Tuple[int, int]]:
    if not spans:
        return []
    if occ is None or occ == "all":
        return spans
    if isinstance(occ, int):
        idx = len(spans) + occ if occ < 0 else occ - 1
        return [spans[idx]] if 0 <= idx < len(spans) else []
    if isinstance(occ, list):
        out = []
        for o in occ:
            idx = len(spans) + o if o < 0 else o - 1
            if 0 <= idx < len(spans):
                out.append(spans[idx])
        # ensure uniqueness and original order
        uniq = []
        for s in spans:
            if s in out and s not in uniq:
                uniq.append(s)
        return uniq
    raise ValueError("occurrence must be 'all', int, or list[int]")

def _pattern_present(text: str, ensure: Dict[str, Any]) -> bool:
    mtype = ensure.get("type")
    pat = ensure.get("pattern", "")
    if mtype == "literal":
        return pat in text
    elif mtype == "regex":
        return re.search(pat, text) is not None
    else:
        raise ValueError("ensure_absent.type must be 'literal' or 'regex'")

def _apply_action_to_text(text: str, action: Dict[str, Any]) -> Tuple[str, bool, str]:
    """
    Returns (new_text, changed?, message)
    """
    op = action.get("op")
    if op in ("append", "prepend"):
        content = action.get("content", "")
        ensure = action.get("ensure_absent")
        if ensure and _pattern_present(text, ensure):
            return text, False, f"{op}: skipped (ensure_absent matched)"
        new_text = (content + text) if op == "prepend" else (text + ("" if text.endswith("\n") or content.startswith("\n") else "\n") + content)
        return new_text, True, f"{op}: added {len(content)} chars"

    if op in ("replace", "insert_before", "insert_after", "delete"):
        match = action.get("match")
        if not match:
            raise ValueError(f"{op} requires a 'match'")
        matcher = _compile_matcher(match)
        spans = _find_spans(text, matcher)
        sel = _select_spans(spans, match.get("occurrence"))
        if not sel:
            mode = action.get("if_no_match", "error")
            if mode == "skip":
                return text, False, f"{op}: skipped (no match)"
            if mode == "append_end" and op in ("insert_before","insert_after"):
                content = action.get("content", "")
                return text + ("" if text.endswith("\n") or content.startswith("\n") else "\n") + content, True, f"{op}: appended to end (no match)"
            raise ValueError(f"{op}: no match and if_no_match!=skip/append_end")
        # Apply from end to start to keep indices stable
        new_text = text
        changed = False
        for start, end in reversed(sel):
            if op == "replace":
                rep = action.get("replacement", "")
                new_text = new_text[:start] + rep + new_text[end:]
                changed = True
            elif op == "insert_before":
                content = action.get("content", "")
                new_text = new_text[:start] + content + new_text[start:]
                changed = True
            elif op == "insert_after":
                content = action.get("content", "")
                new_text = new_text[:end] + content + new_text[end:]
                changed = True
            elif op == "delete":
                new_text = new_text[:start] + new_text[end:]
                changed = True
        return new_text, changed, f"{op}: applied to {len(sel)} occurrence(s)"

    raise ValueError(f"Unknown text op: {op}")

def _apply_json_set(root: str, entry: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
    rel = entry["path"]
    path = os.path.join(root, rel)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pointer = entry["json_pointer"]
    if not pointer.startswith("/"):
        raise ValueError("json_pointer must start with '/'")
    segments = [s for s in pointer.split("/")[1:] if s != ""]
    cur = data
    for i, seg in enumerate(segments):
        last = i == len(segments) - 1
        is_index = seg.isdigit()
        if last:
            if isinstance(cur, list) and is_index:
                idx = int(seg)
                # extend list if needed
                if idx >= len(cur):
                    cur.extend([None] * (idx - len(cur) + 1))
                cur[idx] = entry["value"]
            else:
                if not isinstance(cur, dict):
                    raise ValueError(f"json_set: cannot set key '{seg}' under non-object")
                cur[seg] = entry["value"]
        else:
            if isinstance(cur, list) and is_index:
                idx = int(seg)
                if idx >= len(cur):  # extend sparse
                    cur.extend([{}] * (idx - len(cur) + 1))
                if cur[idx] is None:
                    cur[idx] = {}
                cur = cur[idx]
            else:
                if seg not in cur or cur[seg] is None:
                    cur[seg] = {}
                cur = cur[seg]
    new_text = json.dumps(data, indent=2) + "\n"
    old_text, _ = _read_text(path)
    diff = _unified_diff(old_text, new_text, rel)
    if not dry_run:
        _write_text(path, new_text)
    return {"file": rel, "changed": (old_text != new_text), "diff": diff, "status": "ok"}

def _resolve_targets(root: str, target: Any) -> List[str]:
    targets = target if isinstance(target, list) else [target]
    paths: List[str] = []
    for t in targets:
        matches = glob.glob(os.path.join(root, t), recursive=True)
        if matches:
            paths.extend(matches)
    # Return repo-relative paths
    rels = []
    for p in sorted(set(paths)):
        rels.append(os.path.relpath(p, root))
    return rels

def apply_crisp(crisp: Dict[str, Any], root: str = ".", dry_run: bool = True) -> Dict[str, Any]:
    if crisp.get("format") != "CRISP" or crisp.get("version") != 1:
        raise ValueError("Unsupported CRISP document (expect format=CRISP, version=1)")
    results: List[Dict[str, Any]] = []
    errors: List[str] = []

    for entry in crisp.get("changes", []):
        # File-ops shortcut entries (no 'target')
        if "op" in entry and entry.get("op") in ("create_file","delete_file","rename_file","json_set"):
            op = entry["op"]
            if op == "create_file":
                rel = entry["path"]
                path = os.path.join(root, rel)
                if os.path.exists(path) and not entry.get("overwrite", False):
                    results.append({"file": rel, "changed": False, "diff": "", "status": "exists"})
                    continue
                old_text = "" if not os.path.exists(path) else _read_text(path)[0]
                new_text = entry.get("content", "")
                diff = _unified_diff(old_text, new_text, rel)
                if not dry_run:
                    _write_text(path, new_text)
                results.append({"file": rel, "changed": True, "diff": diff, "status": "ok"})
                continue
            if op == "delete_file":
                rel = entry["path"]
                path = os.path.join(root, rel)
                if not os.path.exists(path):
                    results.append({"file": rel, "changed": False, "diff": "", "status": "missing"})
                    continue
                old_text, _ = _read_text(path)
                diff = _unified_diff(old_text, "", rel)
                if not dry_run:
                    os.remove(path)
                results.append({"file": rel, "changed": True, "diff": diff, "status": "ok"})
                continue
            if op == "rename_file":
                src = entry["path"]; dst = entry["to"]
                src_path = os.path.join(root, src)
                dst_path = os.path.join(root, dst)
                if not os.path.exists(src_path):
                    results.append({"file": src, "changed": False, "diff": "", "status": "missing"})
                    continue
                if os.path.exists(dst_path) and not entry.get("overwrite", False):
                    results.append({"file": dst, "changed": False, "diff": "", "status": "exists"})
                    continue
                # diff note: rename has no textual diff; we report a header-only hint
                diff = f"--- a/{src}\n+++ b/{dst}\n# file renamed"
                if not dry_run:
                    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
                    os.replace(src_path, dst_path)
                results.append({"file": f"{src} -> {dst}", "changed": True, "diff": diff, "status": "ok"})
                continue
            if op == "json_set":
                try:
                    results.append(_apply_json_set(root, entry, dry_run))
                except Exception as e:
                    errors.append(f"json_set {entry.get('path')}: {e}")
                continue

        # Text edits for matching targets
        target = entry.get("target")
        actions = entry.get("actions", [])
        if not target or not actions:
            errors.append("entry missing 'target' or 'actions'")
            continue
        paths = _resolve_targets(root, target)
        if not paths:
            results.append({"file": str(target), "changed": False, "diff": "", "status": "no_targets"})
            continue

        for rel in paths:
            abs_path = os.path.join(root, rel)
            try:
                original, eol = _read_text(abs_path)
                text = original
                applied_msgs: List[str] = []
                for action in actions:
                    text, changed, msg = _apply_action_to_text(text, action)
                    applied_msgs.append(msg)
                diff = "" if text == original else _unified_diff(original, text, rel)
                if not dry_run and text != original:
                    # normalize to original EOL where possible
                    text = text.replace("\n", eol)
                    _write_text(abs_path, text)
                results.append({
                    "file": rel, "changed": (text != original), "diff": diff,
                    "status": "ok", "log": applied_msgs
                })
            except Exception as e:
                errors.append(f"{rel}: {e}")

    return {"review_id": crisp.get("review_id"), "results": results, "errors": errors}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--review", required=True, help="Path to CRISP JSON")
    p.add_argument("--root", default=".", help="Repo root to apply against")
    p.add_argument("--in-place", action="store_true", help="Apply changes (default: dry-run)")
    args = p.parse_args()

    with open(args.review, "r", encoding="utf-8") as f:
        crisp = json.load(f)
    outcome = apply_crisp(crisp, root=args.root, dry_run=(not args.in_place))

    # Print diffs and a compact summary
    any_diff = False
    for r in outcome["results"]:
        if r.get("diff"):
            any_diff = True
            print(r["diff"])
    if not any_diff:
        print("# No textual diffs (dry-run or no changes).")

    print("\n# Summary")
    for r in outcome["results"]:
        flag = "CHANGED" if r.get("changed") else "OK"
        path = r.get("file")
        status = r.get("status", "")
        print(f"- [{flag}] {path} ({status})")
        if r.get("log"):
            for msg in r["log"]:
                print(f"    • {msg}")

    if outcome["errors"]:
        print("\n# Errors")
        for e in outcome["errors"]:
            print(f"- {e}")

if __name__ == "__main__":
    main()
