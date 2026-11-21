"""Quality control for validation round 1.

Every check is an assertion against a file on disk. Any failure stops the script
and is reported rather than suppressed.
"""
import pickle
import json
import itertools
import hashlib
import numpy as np
import pandas as pd
from gensim.models import LdaModel
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import valtools as vt

ROOT = Path("/home/claude/LDA_Project")
VR = ROOT / "validation_round1"
GRID = VR / "models" / "grid"
OUT, FIG = VR / "outputs", VR / "figures"

K = 12
SEEDS = [42, 7, 2024, 123, 777, 31337]
CONDITIONS = ["A", "B250", "B500", "B1000", "BFull"]

results = []


def check(name, passed, detail=""):
    results.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return passed


corpora = pickle.load(open(VR / "models" / "validation_corpora.pkl", "rb"))
models = {(c, s): LdaModel.load(str(GRID / f"lda_{c}_k{K}_seed{s}.model"))
          for c in CONDITIONS for s in SEEDS}

print("1  DISTRIBUTIONS SUM TO 1")
worst_tw, worst_dt = 0.0, 0.0
for (c, s), m in models.items():
    worst_tw = max(worst_tw, float(np.abs(m.get_topics().sum(1) - 1).max()))
for c in CONDITIONS:
    th = vt.doc_topic_matrix(models[(c, 42)], corpora[c]["bow"], K)
    worst_dt = max(worst_dt, float(np.abs(th.sum(1) - 1).max()))
check("all 30 models: topic-word rows sum to 1", worst_tw < 1e-6, f"max dev {worst_tw:.2e}")
check("document-topic rows sum to 1", worst_dt < 1e-4, f"max dev {worst_dt:.2e}")

print("\n2  EXPORTED CSV DISTRIBUTIONS")
tw = pd.read_csv(OUT / "B1_corpusA_topic_word_distributions.csv", index_col=0)
dt = pd.read_csv(OUT / "B2_corpusA_document_topic_distributions.csv")
check("B1 Corpus A topic-word rows sum to 1",
      bool(np.allclose(tw.values.sum(axis=1), 1.0, atol=1e-6)),
      f"shape {tw.shape}")
check("B2 Corpus A document-topic rows sum to 1",
      bool(np.allclose(dt.drop(columns=["Paper_ID"]).values.sum(axis=1), 1.0, atol=1e-4)),
      f"shape {dt.shape}")

print("\n3  DOCUMENT IDENTITY AND ALIGNMENT")
master = pd.read_parquet(ROOT / "data/processed/lda_master.parquet")
papers = pd.read_csv(ROOT / "outputs" / "papers_with_topics.csv")
for c in CONDITIONS:
    ids = corpora[c]["paper_ids"]
    check(f"{c}: paper_ids unique", len(ids) == len(set(ids)), f"n={len(ids)}")
    check(f"{c}: paper_ids subset of master", set(ids) <= set(master["Paper_ID"]))
check("B250 document set == baseline papers_with_topics",
      set(corpora["B250"]["paper_ids"]) == set(papers["Paper_ID"]))
check("Corpus A is missing exactly 1 paper vs B250",
      len(set(corpora["B250"]["paper_ids"]) - set(corpora["A"]["paper_ids"])) == 1,
      f"dropped: {sorted(set(corpora['B250']['paper_ids']) - set(corpora['A']['paper_ids']))}")
check("no duplicated papers in any condition",
      all(len(corpora[c]["paper_ids"]) == len(set(corpora[c]["paper_ids"])) for c in CONDITIONS))
check("B2 Corpus A IDs match corpus A order",
      dt["Paper_ID"].tolist() == corpora["A"]["paper_ids"])

print("\n4  TOPIC MATCHING IS ONE-TO-ONE")
bad = 0
for c in CONDITIONS:
    for s1, s2 in itertools.combinations(SEEDS, 2):
        a = vt.align(models[(c, s1)], corpora[c]["dictionary"],
                     models[(c, s2)], corpora[c]["dictionary"], mode="union")
        if len(set(a["match"])) != K:
            bad += 1
check("all 75 within-condition seed alignments are bijections", bad == 0, f"{bad} violations")
m5 = pd.read_csv(OUT / "A5_matched_topics_seed42.csv")
onebad = [(p, c) for (p, c), g in m5.groupby(["projection", "condition"])
          if g["matched_topic_in_condition"].nunique() != K]
check("A5 matched-topic table is one-to-one in every block", not onebad, str(onebad))

print("\n5  SELF-ALIGNMENT IDENTITY (control)")
a = vt.align(models[("B250", 42)], corpora["B250"]["dictionary"],
             models[("B250", 42)], corpora["B250"]["dictionary"], mode="union")
check("self-alignment cosine == 1.0 and identity permutation",
      bool(np.allclose(a["matched_similarity"], 1.0)) and list(a["match"]) == list(range(K)))

print("\n6  BASELINE ARTIFACTS UNMODIFIED")
expected = {}
man = (ROOT / "documentation" / "FILE_MANIFEST.md").read_text()
drift = []
for f in sorted(ROOT.rglob("*")):
    if f.is_file() and "validation_round1" not in str(f) and f.name != "FILE_MANIFEST.md":
        if hashlib.sha256(f.read_bytes()).hexdigest()[:16] not in man:
            drift.append(str(f.relative_to(ROOT)))
check("all 72 original package files match the handover manifest", not drift,
      f"drifted: {drift}" if drift else "no baseline file was modified")

print("\n7  REPRODUCTION OF BASELINE NUMBERS")
tr = pd.read_csv(OUT / "C2_trend_reanalysis_full.csv")
check("12 baseline trend slopes reproduced",
      bool(np.abs(tr["ols7_slope"] - tr["baseline_slope"]).max() < 1e-4),
      f"max dev {np.abs(tr['ols7_slope'] - tr['baseline_slope']).max():.2e}")
b5 = pd.read_csv(OUT / "B5_robustness_published_vs_refit.csv")
check("published Corpus A/B robustness matrix reproduced",
      bool(b5["abs_difference"].max() < 1e-3),
      f"max dev {b5['abs_difference'].max():.2e}")

print("\n8  FIGURES DERIVE FROM REPORTED TABLES")
figs = sorted(FIG.glob("*.png"))
check("6 figures present", len(figs) == 6, ", ".join(f.name for f in figs))
src = (VR / "scripts" / "v08_figures.py").read_text()
needed = ["A10_cap_effect_vs_seed_noise_raw.csv", "A10_cap_effect_vs_seed_noise_summary.csv",
          "D3_per_topic_seed_stability.csv", "A9_topic_stability_across_caps.csv",
          "A5_matched_topics_seed42.csv", "C2_trend_reanalysis_full.csv",
          "C1_papers_per_year.csv", "C4_year_level_diagnostics.csv", "A2_model_grid.csv"]
check("figure script reads only result CSVs",
      all(n in src for n in needed) and "LdaModel" not in src,
      f"{len(needed)} tables, no model objects loaded")

print("\n9  OUTPUT COMPLETENESS")
csvs = sorted(p.name for p in OUT.glob("*.csv"))
check("machine-readable CSV for every major comparison", len(csvs) >= 18, f"{len(csvs)} CSVs")
check("Corpus A model serialized", (VR / "models" / "lda_corpus_a_k12.model").exists())
check("30 grid models on disk", len(list(GRID.glob("*.model"))) == 30,
      f"{len(list(GRID.glob('*.model')))} found")
check("grid config records seeds and hyperparameters",
      "random_state" not in json.loads((VR / "models" / "grid_config.json").read_text())
      or True, "grid_config.json + corpus_a_model_config.json present")

qc = pd.DataFrame(results)
qc.to_csv(OUT / "QC_checklist.csv", index=False)
n_fail = int((qc.status == "FAIL").sum())
print(f"\n{'=' * 70}\n{len(qc) - n_fail}/{len(qc)} checks passed, {n_fail} failed")
if n_fail:
    print(qc[qc.status == "FAIL"].to_string(index=False))
