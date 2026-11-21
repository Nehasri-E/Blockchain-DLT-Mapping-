"""Task D: topic stability across random seeds.

This is the controlling result for the whole validation round. If the K=12
solution is not reproducible across seeds on a FIXED corpus, then any
cross-corpus difference in Task A must be judged against that noise floor
rather than attributed to the cap.

Six seeds per condition, all other baseline parameters fixed. Seed 42 is the
anchor because it is the published baseline. Pairwise comparisons use all 15
unordered seed pairs per condition, so the summary does not depend on the
anchor choice.
"""
import pickle
import json
import itertools
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
OUT = VR / "outputs"

K = 12
SEEDS = [42, 7, 2024, 123, 777, 31337]
CONDITIONS = ["A", "B250", "B500", "B1000", "BFull"]
ANCHOR = 42

LABELS = {int(k): v for k, v in
          json.loads((VR / "scripts" / "baseline_labels.json").read_text()).items()}

corpora = pickle.load(open(VR / "models" / "validation_corpora.pkl", "rb"))
models = {(c, s): LdaModel.load(str(GRID / f"lda_{c}_k{K}_seed{s}.model"))
          for c in CONDITIONS for s in SEEDS}

# ------------------------------------------------- D1: all pairwise seed pairs
rows = []
for cond in CONDITIONS:
    d = corpora[cond]["dictionary"]
    for s1, s2 in itertools.combinations(SEEDS, 2):
        for metric in ["cosine", "js"]:
            a = vt.align(models[(cond, s1)], d, models[(cond, s2)], d,
                         mode="union", metric=metric)
            sims = a["matched_similarity"]
            rows.append({"condition": cond, "seed_a": s1, "seed_b": s2,
                         "metric": metric,
                         "mean_matched": round(float(sims.mean()), 4),
                         "min_matched": round(float(sims.min()), 4),
                         "n_above_0.70": int((sims >= 0.70).sum()),
                         "n_above_0.80": int((sims >= 0.80).sum())})
pairs = pd.DataFrame(rows)
pairs.to_csv(OUT / "D1_pairwise_seed_similarity.csv", index=False)

summary = pairs.groupby(["condition", "metric"]).agg(
    n_pairs=("mean_matched", "count"),
    mean_matched_similarity=("mean_matched", "mean"),
    sd=("mean_matched", "std"),
    worst_pair=("mean_matched", "min"),
    mean_topics_above_0_70=("n_above_0.70", "mean")).round(4).reset_index()
summary.to_csv(OUT / "D2_seed_stability_summary.csv", index=False)

# ------------------------------------------------- D3: per-topic, anchored
per_topic = []
for cond in CONDITIONS:
    d = corpora[cond]["dictionary"]
    anchor = models[(cond, ANCHOR)]
    sims = np.zeros((K, len(SEEDS) - 1))
    for j, s in enumerate([x for x in SEEDS if x != ANCHOR]):
        a = vt.align(anchor, d, models[(cond, s)], d, mode="union", metric="cosine")
        sims[:, j] = a["matched_similarity"]
    for i in range(K):
        per_topic.append({
            "condition": cond, "anchor_topic": i + 1,
            "baseline_label": LABELS[i + 1] if cond == "B250" else "",
            "mean_cosine": round(float(sims[i].mean()), 4),
            "sd_cosine": round(float(sims[i].std(ddof=1)), 4),
            "min_cosine": round(float(sims[i].min()), 4),
            "max_cosine": round(float(sims[i].max()), 4),
            "n_seeds_above_0.70": int((sims[i] >= 0.70).sum()),
            "verdict": ("reproducible" if sims[i].min() >= 0.70 else
                        "partially reproducible" if sims[i].mean() >= 0.50 else
                        "not reproducible"),
        })
pt = pd.DataFrame(per_topic)
pt.to_csv(OUT / "D3_per_topic_seed_stability.csv", index=False)

# ------------------------------------------------- D4: label defensibility
rows = []
d = corpora["B250"]["dictionary"]
anchor = models[("B250", ANCHOR)]
for s in SEEDS:
    a = vt.align(anchor, d, models[("B250", s)], d, mode="union", metric="cosine")
    for i in range(K):
        j = int(a["match"][i])
        terms = [w for w, _ in models[("B250", s)].show_topic(j, topn=8)]
        rows.append({"baseline_topic": i + 1, "baseline_label": LABELS[i + 1],
                     "seed": s, "matched_topic": j + 1,
                     "cosine": round(float(a["matched_similarity"][i]), 4),
                     "top8_terms": ", ".join(terms)})
lab = pd.DataFrame(rows)
lab.to_csv(OUT / "D4_label_defensibility_across_seeds.csv", index=False)

# ------------------------------------------------- D5: document-level stability
rows = []
for cond in CONDITIONS:
    bow = corpora[cond]["bow"]
    d = corpora[cond]["dictionary"]
    thetas = {s: vt.doc_topic_matrix(models[(cond, s)], bow, K) for s in SEEDS}
    for s1, s2 in itertools.combinations(SEEDS, 2):
        a = vt.align(models[(cond, s1)], d, models[(cond, s2)], d,
                     mode="union", metric="cosine")
        acc, l1, l2 = vt.dominant_agreement(thetas[s1], thetas[s2], a["match"])
        rows.append({"condition": cond, "seed_a": s1, "seed_b": s2,
                     "dominant_topic_agreement": round(acc, 4),
                     "adjusted_rand_index": round(
                         vt.adjusted_rand(thetas[s1].argmax(1), thetas[s2].argmax(1)), 4)})
docst = pd.DataFrame(rows)
docst.to_csv(OUT / "D5_document_level_seed_stability.csv", index=False)

print("=" * 88)
print("D2  SEED STABILITY (15 unordered seed pairs per condition, union projection)")
print(summary.to_string(index=False))
print("\n" + "=" * 88)
print("D5  DOCUMENT-LEVEL AGREEMENT BETWEEN SEEDS ON IDENTICAL DATA")
print(docst.groupby("condition").agg(
    agreement=("dominant_topic_agreement", "mean"),
    ari=("adjusted_rand_index", "mean"),
    worst_agreement=("dominant_topic_agreement", "min")).round(4).to_string())
print("\n" + "=" * 88)
print("D3  PER-TOPIC SEED STABILITY, CORPUS B250 (the published baseline)")
print(pt[pt.condition == "B250"].drop(columns=["condition"]).sort_values(
    "mean_cosine", ascending=False).to_string(index=False, max_colwidth=44))
