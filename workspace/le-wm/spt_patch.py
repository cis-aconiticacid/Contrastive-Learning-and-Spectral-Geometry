"""Patch stable_pretraining/module.py to handle single optimizer.

Lightning's `self.optimizers()` returns a single object when there's only
one optimizer; spt's on_train_start uses `len(optimizers)` which fails on
a non-list. Wrap in a list before len.
"""
import sys
import stable_pretraining
from pathlib import Path

p = Path(stable_pretraining.__file__).parent / "module.py"
s = p.read_text()
old = '        optimizers = self.optimizers()\n        logging.info(f"`self.optimizers() gave us {len(optimizers)} optimizers")'
new = '        optimizers = self.optimizers()\n        if not isinstance(optimizers, (list, tuple)):\n            optimizers = [optimizers]\n        logging.info(f"`self.optimizers() gave us {len(optimizers)} optimizers")'
if new in s:
    print("already patched")
elif old in s:
    p.write_text(s.replace(old, new))
    print(f"patched {p}")
else:
    print(f"PATTERN NOT FOUND in {p}")
    sys.exit(1)
