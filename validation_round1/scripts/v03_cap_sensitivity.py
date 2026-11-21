"""Task A: conclusion-cap sensitivity analysis.

Reference model throughout is B250 / seed 42, which is the published baseline
(bit-identical to models/candidates.pkl, verified in v02).

Every cross-condition comparison is done at matched seed, so the only thing
varying within a comparison is the cap. Seed variation is quantified separately
in Task D and is used here as the yardstick for "is a cap effect larger than
noise?".
"""
import pickle
import json
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
REF = "B250"
REF_SEED = 42

LABELS = json.loads((VR / "scripts" / "baseline_labels.json").read_text())
LABELS = {int(k): v for k, v in LABELS.items()}

corpora = pickle.load(open(VR / "models" / "validation_corpora.pkl", "rb"))
models = {(c, s): LdaModel.load(str(GRID / f"lda_{c}_k{K}_seed{s}.model"))
          for c in CONDITIONS for s in SEEDS}
print(f"Loaded {len(models)} models")

theta = {}
for (c, s), m in models.items():
    theta[(c, s)] = vt.doc_topic_matrix(m, corpora[c]["bow"], K)
print("Computed document-topic matrices")

ref_model = models[(REF, REF_SEED)]
ref_dict = corpora[REF]["dictionary"]
ref_ids = corpora[REF]["paper_ids"]
ref_theta = theta[(REF, REF_SEED)]

# ---------------------------------------------------------------- A4, A5
rows, matched_rows = [], []
sim_store = {}
for mode in ["union", "intersection"]:
    for cond in CONDITIONS:
        for seed in SEEDS:
            if (cond, seed) == (REF, REF_SEED):
                continue
            a = vt.align(ref_model, ref_dict, models[(cond, seed)],
                         corpora[cond]["dictionary"], mode=mode, metric="cosine")
            sims = a["matched_similarity"]
            rows.append({
                "projection": mode, "condition": cond, "seed": seed,
                "shared_vocab": a["vocab_size"],
                "mean_matched_cosine": round(float(sims.mean()), 4),
                "median_matched_cosine": round(float(np.median(sims)), 4),
                "min_matched_cosine": round(float(sims.min()), 4),
                "n_topics_above_0.70": int((sims >= 0.70).sum()),
                "n_topics_above_0.50": int((sims >= 0.50).sum()),
            })
            if seed == REF_SEED:
                sim_store[(mode, cond)] = a
                for i in range(K):
                    matched_rows.append({
                        "projection": mode, "condition": cond,
                        "ref_topic": i + 1, "ref_label": LABELS[i + 1],
                        "matched_topic_in_condition": int(a["match"][i]) + 1,
                        "cosine": round(float(sims[i]), 4),
                    })

pd.DataFrame(rows).to_csv(OUT / "A4_alignment_all_seeds.csv", index=False)
matched = pd.DataFrame(matched_rows)
matched.to_csv(OUT / "A5_matched_topics_seed42.csv", index=False)

for mode, a in [(k[0], v) for k, v in sim_store.items() if k[1] == "BFull"]:
    pd.DataFrame(a["similarity"].round(4),
                 index=[f"B250_T{i+1}" for i in range(K)],
                 columns=[f"BFull_T{j+1}" for j in range(K)]).to_csv(
        OUT / f"A6_similarity_matrix_B250_vs_BFull_{mode}.csv")

for cond in CONDITIONS:
    if cond == REF:
        continue
    pd.DataFrame(sim_store[("union", cond)]["similarity"].round(4),
                 index=[f"B250_T{i+1}" for i in range(K)],
                 columns=[f"{cond}_T{j+1}" for j in range(K)]).to_csv(
        OUT / f"A6_similarity_matrix_B250_vs_{cond}_union.csv")

# ---------------------------------------------------------------- A7 prevalence
prev_rows = []
ref_prev = ref_theta.mean(0)
for cond in CONDITIONS:
    for seed in SEEDS:
        if (cond, seed) == (REF, REF_SEED):
            cols, p = np.arange(K), ref_prev
        else:
            a = vt.align(ref_model, ref_dict, models[(cond, seed)],
                         corpora[cond]["dictionary"], mode="union", metric="cosine")
            cols = a["match"]
            p = theta[(cond, seed)].mean(0)[cols]
        for i in range(K):
            prev_rows.append({"condition": cond, "seed": seed, "ref_topic": i + 1,
                              "ref_label": LABELS[i + 1],
                              "prevalence": round(float(p[i]), 5),
                              "baseline_prevalence": round(float(ref_prev[i]), 5),
                              "delta_vs_baseline": round(float(p[i] - ref_prev[i]), 5)})
prev = pd.DataFrame(prev_rows)
prev.to_csv(OUT / "A7_prevalence_by_condition_long.csv", index=False)

pt = prev.groupby(["ref_topic", "ref_label", "condition"])["prevalence"].mean().unstack()
pt = pt[CONDITIONS].round(4).reset_index()
pt.to_csv(OUT / "A7_prevalence_comparison_wide.csv", index=False)

# ---------------------------------------------------------------- A8 agreement
agree_rows = []
for cond in CONDITIONS:
    for seed in SEEDS:
        if (cond, seed) == (REF, REF_SEED):
            continue
        shared = [p for p in corpora[cond]["paper_ids"] if p in set(ref_ids)]
        ri = {p: i for i, p in enumerate(ref_ids)}
        ci = {p: i for i, p in enumerate(corpora[cond]["paper_ids"])}
        tr = ref_theta[[ri[p] for p in shared]]
        tc = theta[(cond, seed)][[ci[p] for p in shared]]
        a = vt.align(ref_model, ref_dict, models[(cond, seed)],
                     corpora[cond]["dictionary"], mode="union", metric="cosine")
        acc, lref, lcmp = vt.dominant_agreement(tr, tc, a["match"])
        agree_rows.append({
            "condition": cond, "seed": seed, "shared_documents": len(shared),
            "dominant_topic_agreement": round(acc, 4),
            "cohens_kappa": round(vt.cohens_kappa(lref, lcmp, K), 4),
            "adjusted_rand_index": round(vt.adjusted_rand(tr.argmax(1), tc.argmax(1)), 4),
            "mean_abs_theta_diff": round(float(np.abs(tr - tc[:, a["match"]]).mean()), 5),
        })
agree = pd.DataFrame(agree_rows)
agree.to_csv(OUT / "A8_dominant_topic_agreement.csv", index=False)

# ---------------------------------------------------------------- A9 stability
stab_rows = []
for i in range(K):
    sub = matched[(matched.projection == "union") & (matched.ref_topic == i + 1)]
    bconds = sub[sub.condition != "A"]["cosine"]
    stab_rows.append({
        "ref_topic": i + 1, "ref_label": LABELS[i + 1],
        "baseline_prevalence": round(float(ref_prev[i]), 4),
        "mean_cosine_across_B_caps": round(float(bconds.mean()), 4),
        "min_cosine_across_B_caps": round(float(bconds.min()), 4),
        "cosine_vs_corpusA": round(float(sub[sub.condition == "A"]["cosine"].iloc[0]), 4),
        "verdict": ("stable" if bconds.min() >= 0.70 else
                    "moderate" if bconds.min() >= 0.50 else "cap-sensitive"),
    })
stab = pd.DataFrame(stab_rows).sort_values("mean_cosine_across_B_caps", ascending=False)
stab.to_csv(OUT / "A9_topic_stability_across_caps.csv", index=False)

print("\n" + "=" * 90)
print("ALIGNMENT TO BASELINE (B250 seed 42), seed-matched, union projection")
piv = pd.DataFrame(rows)
print(piv[piv.projection == "union"].groupby("condition").agg(
    mean_cos=("mean_matched_cosine", "mean"), sd=("mean_matched_cosine", "std"),
    min_cos=("min_matched_cosine", "mean")).round(4).to_string())
print("\nIntersection projection")
print(piv[piv.projection == "intersection"].groupby("condition").agg(
    mean_cos=("mean_matched_cosine", "mean"), sd=("mean_matched_cosine", "std")
).round(4).to_string())
print("\n" + "=" * 90)
print("DOMINANT-TOPIC AGREEMENT vs baseline")
print(agree.groupby("condition").agg(
    agreement=("dominant_topic_agreement", "mean"), ari=("adjusted_rand_index", "mean"),
    kappa=("cohens_kappa", "mean"), docs=("shared_documents", "first")).round(4).to_string())
print("\n" + "=" * 90)
print("PER-TOPIC STABILITY ACROSS CAP CONDITIONS")
print(stab.to_string(index=False, max_colwidth=44))
print("\n" + "=" * 90)
print("PREVALENCE BY CONDITION (mean over 6 seeds, matched to baseline topics)")
print(pt.to_string(index=False, max_colwidth=44))
