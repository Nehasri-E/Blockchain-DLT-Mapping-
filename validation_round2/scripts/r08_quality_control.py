"""Quality control for validation round 2. Every check is an assertion on disk."""
import json
import hashlib
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from gensim.models import LdaModel
import sys

sys.path.insert(0, str(Path(__file__).parent))
from r00_common import R2, ROOT, SEEDS, load_corpus, compare, theta

OUT, FIG = R2 / "outputs", R2 / "figures"
R1 = ROOT / "validation_round1"
results = []


def check(name, ok, detail=""):
    results.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return ok


c = load_corpus("B250")
bow, dic, pids = c["bow"], c["dictionary"], c["paper_ids"]

print("1  BASELINE AND ROUND 1 ARTIFACTS UNMODIFIED")
man = (ROOT / "documentation" / "FILE_MANIFEST.md").read_text()
drift = [str(f.relative_to(ROOT)) for f in sorted(ROOT.rglob("*"))
         if f.is_file() and "validation_round" not in str(f)
         and f.name != "FILE_MANIFEST.md"
         and hashlib.sha256(f.read_bytes()).hexdigest()[:16] not in man]
check("72 original package files unmodified", not drift, str(drift) if drift else "")
r1man = pd.read_csv(R1 / "documentation" / "VALIDATION_FILE_MANIFEST.csv")
# Round 2 began when its first script was written. Nothing in Round 1 may have
# been touched after that instant.
r2_start = min(f.stat().st_mtime for f in (R2 / "scripts").glob("*.py"))
touched = [str(f.relative_to(R1)) for f in R1.rglob("*")
           if f.is_file() and f.stat().st_mtime > r2_start]
check(f"no Round 1 file modified after Round 2 began", not touched,
      str(touched[:5]) if touched else f"{len(list(R1.rglob('*')))} paths checked by mtime")

# Separately: Round 1's own manifest is known to be stale for exactly one file.
# QC_checklist.csv was rewritten 6 s AFTER that manifest was generated, during
# Round 1 itself. This is a Round 1 bookkeeping artifact, not a Round 2 edit.
r1drift = []
for _, r in r1man.iterrows():
    f = R1 / r["path"]
    if f.name == "VALIDATION_FILE_MANIFEST.csv":
        continue
    if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest()[:16] != r["sha256_16"]:
        r1drift.append(r["path"])
check("Round 1 manifest drift limited to the known stale QC_checklist entry",
      set(r1drift) <= {"outputs/QC_checklist.csv"},
      f"drift: {r1drift} (QC_checklist.csv was rewritten after the manifest, within Round 1)")

print("\n2  MODEL COUNTS AND CONFIG RECORDS")
counts = {d.name: len(list(d.glob("*.model")))
          for d in (R2 / "models").iterdir() if d.is_dir()}
check("convergence grid 30 models", counts.get("convergence") == 30, str(counts.get("convergence")))
check("prior grid 18 models", counts.get("priors") == 18, str(counts.get("priors")))
check("explicit eta grid 24 models", counts.get("eta_explicit") == 24, str(counts.get("eta_explicit")))
check("K sweep 66 models", counts.get("k_sweep") == 66, str(counts.get("k_sweep")))
jsons = sum(len(list(d.glob("*.json"))) for d in (R2 / "models").iterdir() if d.is_dir())
check("every model has a config/diagnostic JSON", jsons == sum(counts.values()),
      f"{jsons} json for {sum(counts.values())} models")

print("\n3  DISTRIBUTIONS SUM TO 1")
worst_tw = worst_dt = 0.0
for d in (R2 / "models").iterdir():
    if not d.is_dir():
        continue
    for f in sorted(d.glob("*.model"))[:6]:
        m = LdaModel.load(str(f))
        worst_tw = max(worst_tw, float(np.abs(m.get_topics().sum(1) - 1).max()))
check("topic-word rows sum to 1 (sampled across all grids)", worst_tw < 1e-6,
      f"max dev {worst_tw:.2e}")
for k in [4, 6, 7, 12]:
    m = LdaModel.load(str(R2 / "models/k_sweep" / f"k{k}_seed42.model"))
    worst_dt = max(worst_dt, float(np.abs(theta(m, bow, k).sum(1) - 1).max()))
check("document-topic rows sum to 1", worst_dt < 1e-4, f"max dev {worst_dt:.2e}")

print("\n4  TOPIC MATCHING IS ONE-TO-ONE")
bad = 0
for k in [4, 6, 7, 12]:
    ms = {s: LdaModel.load(str(R2 / "models/k_sweep" / f"k{k}_seed{s}.model")) for s in SEEDS}
    for s1, s2 in itertools.combinations(SEEDS, 2):
        _, cols, _ = compare(ms[s1], ms[s2], dic)
        if len(set(cols)) != k:
            bad += 1
check("all 60 K-sweep alignments are bijections", bad == 0, f"{bad} violations")
m = LdaModel.load(str(R2 / "models/k_sweep/k12_seed42.model"))
_, cols, sims = compare(m, m, dic)
check("self-alignment is identity at cosine 1.0",
      bool(np.allclose(sims, 1.0)) and list(cols) == list(range(12)))

print("\n5  DOCUMENT IDENTITY")
master = pd.read_parquet(ROOT / "data/processed/lda_master.parquet")
check("987 unique paper_ids, subset of master",
      len(pids) == len(set(pids)) == 987 and set(pids) <= set(master["Paper_ID"]))
rec = pd.read_csv(OUT / "T20_topic_recovery_table.csv")
check("recovery table has no duplicate (K, topic) rows",
      not rec.duplicated(["candidate_K", "topic_id"]).any(),
      f"{len(rec)} rows over {rec.candidate_K.nunique()} K values")

print("\n6  CONVERGENCE CLAIM IS SUPPORTED BY EVIDENCE")
st = pd.read_csv(OUT / "T3_stability_by_config.csv")
inv = pd.read_csv(OUT / "T2_solution_invariance_vs_anchor.csv")
c3 = float(inv[inv.config == "C3"]["cosine_vs_C4_anchor"].mean())
check("C3 invariance vs 200-pass anchor >= 0.99", c3 >= 0.99, f"{c3:.4f}")
check("all configs reach >=99.9% E-step convergence",
      bool((st["mean_estep_convergence"] >= 0.999).all()),
      f"min {st['mean_estep_convergence'].min():.5f}")
spread = float(st["mean_matched_cosine"].max() - st["mean_matched_cosine"].min())
check("stability spread across configs is small (supports 'not convergence')",
      spread < 0.05, f"spread {spread:.4f} over 20-200 passes")

print("\n7  ETA NO-OP FINDING VERIFIED")
a = LdaModel.load(str(R2 / "models/priors/P_base_sym_auto_seed42.model"))
b = LdaModel.load(str(R2 / "models/priors/P_eta_sym_sym_seed42.model"))
check("eta='auto' and eta='symmetric' give bit-identical topics",
      bool(np.array_equal(a.expElogbeta, b.expElogbeta)), "confirms T9 finding")
e = LdaModel.load(str(R2 / "models/eta_explicit/eta_1.0_seed42.model"))
check("explicit eta DOES change topics (control)",
      float(np.abs(a.get_topics() - e.get_topics()).max()) > 1e-3,
      f"max diff {float(np.abs(a.get_topics()-e.get_topics()).max()):.3e}")

print("\n8  CHANCE CORRECTION APPLIED WHERE SIMILARITY COULD BE INFLATED")
ks = pd.read_csv(OUT / "T11_k_sweep_summary.csv")
check("K sweep reports a random-pairing baseline",
      "random_pairing_cosine" in ks.columns and bool((ks.random_pairing_cosine > 0).all()))
et = pd.read_csv(OUT / "T23_eta_chance_corrected.csv")
check("eta sweep reports a random-pairing baseline",
      "random_pairing_cosine" in et.columns and bool((et.random_pairing_cosine > 0).all()))

print("\n9  CORRECTED STATISTICS")
cc = pd.read_csv(OUT / "T12_cap_vs_seed_corrected.csv")
check("exact permutation used all 2^6 = 64 sign assignments",
      bool((cc["n_permutations"] == 64).all()))
# CSV stores p rounded to 5 dp, so compare with a rounding tolerance.
check("no p-value below the 6-seed permutation floor",
      bool((cc["exact_signflip_p_one_sided"] >= round(1 / 64, 5) - 1e-9).all()),
      f"min p {cc['exact_signflip_p_one_sided'].min():.5f} equals the floor "
      f"{1/64:.6f} (stored at 5 dp)")
check("Round 1 descriptive effect sizes preserved unchanged",
      bool(np.allclose(
          pd.read_csv(OUT / "T15_round1_vs_corrected.csv")["round1_difference"],
          pd.read_csv(OUT / "T15_round1_vs_corrected.csv")["mean_paired_difference"],
          atol=1e-3)), "point estimates identical; only the inference changed")

print("\n10  FIGURES DERIVE FROM RESULT TABLES")
figs = sorted(FIG.glob("*.png"))
check("6 figures present", len(figs) == 6, ", ".join(f.name for f in figs))
src = (R2 / "scripts" / "r07_figures.py").read_text()
check("figure script loads no model objects", "LdaModel" not in src)
for t in ["T3_stability_by_config.csv", "T23_eta_chance_corrected.csv",
          "T11_k_sweep_summary.csv", "T12_cap_vs_seed_corrected.csv",
          "T16_trend_bootstrap_full.csv"]:
    check(f"figure script reads {t}", t in src)

print("\n11  OUTPUT COMPLETENESS")
csvs = sorted(OUT.glob("*.csv"))
check("machine-readable CSV for every major comparison", len(csvs) >= 20, f"{len(csvs)} CSVs")
for d in ["ROUND2_METHODS.md", "ROUND2_FINDINGS.md"]:
    check(f"{d} present", (R2 / "documentation" / d).exists())

qc = pd.DataFrame(results)
qc.to_csv(OUT / "QC_round2_checklist.csv", index=False)
nf = int((qc.status == "FAIL").sum())
print(f"\n{'='*70}\n{len(qc)-nf}/{len(qc)} checks passed, {nf} failed")
if nf:
    print(qc[qc.status == "FAIL"].to_string(index=False))
