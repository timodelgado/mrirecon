#%%
import re
from pathlib import Path

def apply_diff(file_path: str, diff: str) -> None:
    """
    Apply a simple unified diff hunk to a file.
    Expects diff lines starting with '-' (remove), '+' (add), or ' ' (context).
    """
    file_path = Path(file_path)
    lines = file_path.read_text().splitlines()

    # Break diff into minus/plus/context lines
    diff_lines = diff.strip().splitlines()

    # Collect removed/added parts (ignore '+'/'-' markers for comparison)
    to_remove = [l[1:] for l in diff_lines if l.startswith("-")]
    to_add    = [l[1:] for l in diff_lines if l.startswith("+")]
    context   = [l[1:] for l in diff_lines if l.startswith(" ")]

    # Find where to apply: search for a block that matches the context+removed
    block_pattern = context[:]
    block_pattern.extend(to_remove)

    joined_text = "\n".join(lines)
    pattern_text = "\n".join(block_pattern)

    if pattern_text not in joined_text:
        raise ValueError("Could not locate patch context in file")

    # Replace block with context+added
    replacement = "\n".join(context + to_add)
    new_text = joined_text.replace(pattern_text, replacement, 1)

    file_path.write_text(new_text)
    print(f"Applied diff to {file_path}")


diff = """\
-        if self._alpha_scalar is not None:
-            return float(self._alpha_scalar)
-
-        assert self._im_shape is not None and self._dev is not None
-        spatial = tuple(self._im_shape[1:])
-        ctr = tuple(s // 2 for s in spatial)
-
-        # Probe shapes: our forward may expect (B,C,...) or (B,1,...)
-        tried = []
-        for C_try in [self._im_shape[0], 1]:
-            try:
-                x0 = torch.zeros((1, C_try) + spatial, device=self._dev, dtype=self._dtype)
-                x0[0, 0][ctr] = 1.0 + 0.0j  # unit delta in first channel
-                y = self.A(x0)   # must work
-                z = self.AH(y)
-                zc = z[0, 0][ctr].real.item()
-                # normalize by coil SoS at center if maps exist
-                if getattr(self, "_maps", None) is not None and self._maps.numel() > 0:
-                    sos = (self._maps.abs() ** 2).sum(dim=0)[ctr].clamp_min(1e-20).item()
-                else:
-                    sos = 1.0
-                alpha = float(zc) / float(sos)
-                self._alpha_scalar = alpha
-                return alpha
-            except Exception as e:
-                tried.append((C_try, str(e)))
-                continue
-
-        # If both shapes failed, surface the info
-        raise RuntimeError(f"diag_AHA_scalar() shape probe failed; tried {tried}")
+        if self._alpha_scalar is not None:
+            return float(self._alpha_scalar)
+
+        assert self._im_shape is not None and self._dev is not None
+        spatial = tuple(self._im_shape[1:])
+        ctr = tuple(s // 2 for s in spatial)
+
+        # Temporarily restrict omega (and DCF) to a single frame so batch dims match B=1 probe
+        traj_save, dcf_save = self._traj_BndK, self._dcf_BK
+        if self._traj_BndK.shape[0] != 1:
+            self._traj_BndK = self._traj_BndK[:1].contiguous()
+            if self._dcf_BK is not None and self._dcf_BK.ndim == 2:
+                self._dcf_BK = self._dcf_BK[:1].contiguous()
+        try:
+            tried = []
+            for C_try in [int(self._im_shape[0]), 1]:
+                try:
+                    x0 = torch.zeros((1, C_try) + spatial, device=self._dev, dtype=self._dtype)
+                    x0[0, 0][ctr] = 1.0 + 0.0j
+                    y = self.A(x0)
+                    z = self.AH(y)
+                    zc = z[0, 0][ctr].real.item()
+                    sos = (self._maps.abs() ** 2).sum(dim=0)[ctr].clamp_min(1e-20).item() if (self._maps is not None) else 1.0
+                    self._alpha_scalar = float(zc) / float(sos)
+                    return float(self._alpha_scalar)
+                except Exception as e:
+                    tried.append((C_try, str(e)))
+                    continue
+            raise RuntimeError(f"diag_AHA_scalar() shape probe failed; tried {tried}")
+        finally:
+            # Always restore multi-frame omega/DCF
+            self._traj_BndK = traj_save
+            self._dcf_BK = dcf_save

"""

apply_diff("graspcg/nufft/backends/torchkb.py", diff)
