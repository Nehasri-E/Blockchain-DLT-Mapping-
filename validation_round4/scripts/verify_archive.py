import hashlib
import sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
man = pd.read_csv(ROOT / "ARCHIVE_MANIFEST.csv")
missing, changed = [], []
for r in man.itertuples():
    p = ROOT / r.relative_path
    if not p.exists():
        missing.append(r.relative_path)
    elif hashlib.sha256(p.read_bytes()).hexdigest() != r.sha256:
        changed.append(r.relative_path)
print(f"checked {len(man)} files")
print(f"missing {len(missing)}; changed {len(changed)}")
for x in (missing + changed)[:20]:
    print("  ", x)
sys.exit(1 if (missing or changed) else 0)
