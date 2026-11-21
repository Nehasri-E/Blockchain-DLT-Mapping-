"""Task 8: consolidated archive with checksums, then verified from a clean extract."""
import sys
import pickle
import hashlib
import shutil
import zipfile
import subprocess
import pandas as pd
from pathlib import Path

ROOT = Path("/home/claude/LDA_Project")
R4 = ROOT / "validation_round4"
STAGE = Path("/tmp/claude-0/-home-claude/b86f9bac-1ee2-5191-9fe6-401f33bd8b7b/scratchpad/archive")
OUTDIR = Path("/mnt/user-data/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)
ZIP = OUTDIR / "LDA_Project_FINAL_verified.zip"

SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints"}
SKIP_SUFFIX = {".pyc"}

(R4 / "scripts" / "verify_archive.py").write_text('''import hashlib
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
''')
(ROOT / "README_FINAL.md").touch()

files = []
for p in sorted(ROOT.rglob("*")):
    if not p.is_file():
        continue
    rel = p.relative_to(ROOT)
    if any(part in SKIP_DIRS for part in rel.parts) or p.suffix in SKIP_SUFFIX:
        continue
    if str(rel) == "ARCHIVE_MANIFEST.csv":
        continue
    files.append(rel)
N = len(files)

(ROOT / "README_FINAL.md").write_text(f"""# Blockchain literature topic modelling: final verified archive

{N} files, built after four rounds of validation.

## Read first

    validation_round4/documentation/FINAL_FINDINGS.md
    validation_round4/documentation/FINAL_METHODS.md
    validation_round4/documentation/FINAL_LIMITATIONS.md
    validation_round4/documentation/FINAL_REPRODUCTION_GUIDE.md

## Headline

Primary model is NMF K=6 on TF-IDF, not the originally planned LDA K=12.
Three themes are defensible: two strongly supported, one with qualification.
No corpus-wide temporal claim is supported under the primary inference procedure.

## Verifying this archive

    python validation_round4/scripts/verify_archive.py

Checks every file in ARCHIVE_MANIFEST.csv against its SHA-256 and exits non-zero
on any mismatch. Results for this build are in
validation_round4/ARCHIVE_VERIFICATION.csv.

## Structure

    scripts/                  baseline pipeline, 01 to 08
    data/                     interim and processed
    models/                   baseline LDA candidates
    outputs/                  baseline topic assignments and trends
    documentation/            baseline manifest and methods
    validation_round1/        cap sensitivity, seed stability, FDR trends
    validation_round2/        convergence, priors, K sweep, bootstrap
    validation_round3/        consensus, NMF, corrected inference
    validation_round4/        final consolidation
    ARCHIVE_MANIFEST.csv      {N} files with SHA-256

## Status

The computational analysis is reproducible and extensively self-audited.
The manuscript is not ready. FINAL_LIMITATIONS.md section 10 states why.
""")


def stage_of(p):
    for k in ["validation_round1", "validation_round2", "validation_round3",
              "validation_round4"]:
        if p.startswith(k):
            return k
    return "baseline"


rows = []
for rel in files:
    b = (ROOT / rel).read_bytes()
    rows.append({"relative_path": str(rel), "bytes": len(b),
                 "sha256": hashlib.sha256(b).hexdigest(),
                 "stage": stage_of(str(rel))})
man = pd.DataFrame(rows)
man.to_csv(ROOT / "ARCHIVE_MANIFEST.csv", index=False)

print("ARCHIVE CONTENTS")
print(man.groupby("stage").agg(files=("relative_path", "size"),
                               MB=("bytes", lambda s: round(s.sum() / 1e6, 1)))
      .to_string())
print(f"\nTOTAL {len(man)} files, {man.bytes.sum()/1e6:.1f} MB uncompressed")

if ZIP.exists():
    ZIP.unlink()
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for rel in files:
        z.write(ROOT / rel, f"LDA_Project/{rel}")
    z.write(ROOT / "ARCHIVE_MANIFEST.csv", "LDA_Project/ARCHIVE_MANIFEST.csv")
print(f"\nZIP {ZIP} = {ZIP.stat().st_size/1e6:.1f} MB")

if STAGE.exists():
    shutil.rmtree(STAGE)
STAGE.mkdir(parents=True)
with zipfile.ZipFile(ZIP) as z:
    z.extractall(STAGE)
E = STAGE / "LDA_Project"

checks = []
em = pd.read_csv(E / "ARCHIVE_MANIFEST.csv")
missing = [r.relative_path for r in em.itertuples() if not (E / r.relative_path).exists()]
changed = [r.relative_path for r in em.itertuples()
           if (E / r.relative_path).exists()
           and hashlib.sha256((E / r.relative_path).read_bytes()).hexdigest() != r.sha256]
checks.append(("every manifested file present in the extract", not missing,
               f"{len(em)} files, {len(missing)} missing"))
checks.append(("every extracted file matches its SHA-256", not changed,
               f"{len(changed)} changed" + (f": {changed[:3]}" if changed else "")))

need = ["scripts/01_build_corpus.py", "data/processed/lda_master.parquet",
        "models/candidates.pkl", "outputs/papers_with_topics.csv",
        "documentation/FILE_MANIFEST.md",
        "validation_round1/outputs/C5_trend_headline_table.csv",
        "validation_round1/outputs/D2_seed_stability_summary.csv",
        "validation_round2/outputs/T3_stability_by_config.csv",
        "validation_round2/outputs/T11_k_sweep_summary.csv",
        "validation_round3/outputs/N2_nmf_stability_by_k.csv",
        "validation_round4/FINAL_QC_CHECKLIST.csv",
        "validation_round4/FINAL_TABLES/Table7_claim_to_file.csv",
        "validation_round4/documentation/FINAL_FINDINGS.md",
        "validation_round4/documentation/FINAL_METHODS.md",
        "validation_round4/documentation/FINAL_LIMITATIONS.md",
        "validation_round4/documentation/FINAL_REPRODUCTION_GUIDE.md",
        "README_FINAL.md"]
absent = [n for n in need if not (E / n).exists()]
checks.append(("all load-bearing files reachable in the extract", not absent,
               f"{len(need)} checked, missing: {absent or 'none'}"))

n2 = pd.read_csv(E / "validation_round3/outputs/N2_nmf_stability_by_k.csv").set_index("k")
checks.append(("Round 3 NMF stability reads correctly from the extract",
               abs(float(n2.loc[6, "mean_matched_cosine"]) - 0.9415) < 0.001,
               f"K=6 mean matched cosine {n2.loc[6,'mean_matched_cosine']}"))
cl = pd.read_csv(E / "validation_round4/FINAL_TABLES/Table7_claim_to_file.csv")
checks.append(("no claim in the extract is a MISMATCH",
               int((cl.status == "MISMATCH").sum()) == 0,
               f"{len(cl)} claims, {int((cl.status=='VERIFIED').sum())} verified"))
qc = pd.read_csv(E / "validation_round4/FINAL_QC_CHECKLIST.csv")
checks.append(("Round 4 QC checklist in the extract is all PASS",
               int((qc.result == "FAIL").sum()) == 0,
               f"{int((qc.result=='PASS').sum())} of {len(qc)} PASS"))
with open(E / "models/candidates.pkl", "rb") as f:
    cand = pickle.load(f)
checks.append(("pickled LDA models load from the extract", isinstance(cand, dict),
               f"keys {sorted(cand.keys())}"))
mp = pd.read_parquet(E / "data/processed/lda_master.parquet")
checks.append(("master parquet loads with 987 papers", len(mp) == 987,
               f"{len(mp)} rows, {mp.shape[1]} columns"))
figs = sorted((E / "validation_round4/FINAL_FIGURES").glob("*.png"))
checks.append(("all six final figures present and non-empty",
               len(figs) == 6 and all(f.stat().st_size > 20000 for f in figs),
               f"{len(figs)} figures"))

import re
bm = re.findall(r"^\|\s*`([^`]+)`\s*\|[^|]*\|\s*`([0-9a-f]{16})`\s*\|",
                (E / "documentation/FILE_MANIFEST.md").read_text(), re.M)
idx = {}
for q in E.rglob("*"):
    if q.is_file() and "validation_round" not in str(q.relative_to(E)):
        idx.setdefault(q.name, []).append(q)
bad = [n for n, h in bm
       if n != "FILE_MANIFEST.md"
       and not any(hashlib.sha256(c.read_bytes()).hexdigest()[:16] == h
                   for c in idx.get(Path(n).name, []))]
checks.append(("baseline unchanged against its own 73-file manifest", not bad,
               f"{len(bm)-1} of {len(bm)} checkable (FILE_MANIFEST.md cannot hash "
               f"itself), {len(bad)} mismatched"))

r = subprocess.run(["python3", str(E / "validation_round4/scripts/verify_archive.py")],
                   capture_output=True, text=True)
checks.append(("shipped verify_archive.py runs clean in the extract",
               r.returncode == 0, r.stdout.strip().replace("\n", "; ")))

res = pd.DataFrame([{"check": c, "result": "PASS" if ok else "FAIL", "detail": d}
                    for c, ok, d in checks])
res.to_csv(R4 / "ARCHIVE_VERIFICATION.csv", index=False)
print("\nCLEAN-EXTRACT VERIFICATION")
print(res.to_string(index=False, max_colwidth=62))
print(f"\n{int((res.result=='PASS').sum())} PASS / {int((res.result=='FAIL').sum())} FAIL")
sys.exit(1 if (res.result == "FAIL").any() else 0)
