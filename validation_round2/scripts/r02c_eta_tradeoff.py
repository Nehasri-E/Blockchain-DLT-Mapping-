"""Task 2, control analysis: is the eta=1.0 stability gain real or geometric?

A larger eta smooths topic-word distributions. Smoother topics are more similar
to each other, and more similar topics are easier to match, so matched cosine
can rise for purely geometric reasons without any genuine improvement in
recovering the same structure. This is the same trap flagged for the K sweep.

Two controls:

  chance baseline   mean cosine under RANDOM topic pairing at the same eta. The
                    gap between matched and random is the recovery that is not
                    explained by topics having become mutually similar.

  ARI               permutation-invariant and computed on hard document
                    assignments, so it cannot be inflated by smoothing of the
                    topic-word distributions. It is the cleanest stability
                    signal across eta.

Interpretation change is assessed by matching each eta setting's topics back to
the baseline eta=1/K solution and reporting the top terms side by side.
"""
import sys
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from gensim.models import LdaModel

sys.path.insert(0, str(Path(__file__).parent))
from r00_common import R2, SEEDS, load_corpus, compare

OUT = R2 / "outputs"
MODELS = R2 / "models" / "eta_explicit"
K = 12
ETAS = {"eta_0.01": 0.01, "eta_0.0833_1overK": 1.0 / K, "eta_0.3": 0.3, "eta_1.0": 1.0}
BASE = "eta_0.0833_1overK"
rng = np.random.default_rng(11)

c = load_corpus("B250")
bow, dic = c["bow"], c["dictionary"]

summ = pd.read_csv(OUT / "T8_eta_explicit_stability.csv").set_index("eta_setting")
rows = []
for name in ETAS:
    models = {s: LdaModel.load(str(MODELS / f"{name}_seed{s}.model")) for s in SEEDS}
    rand = []
    for s1, s2 in itertools.combinations(SEEDS, 2):
        S, _, _ = compare(models[s1], models[s2], dic)
        rand += [float(S[np.arange(K), rng.permutation(K)].mean()) for _ in range(200)]
    rmean = float(np.mean(rand))
    matched = float(summ.loc[name, "mean_matched_cosine"])
    rows.append({
        "eta_setting": name, "eta_value": ETAS[name],
        "mean_matched_cosine": matched,
        "random_pairing_cosine": round(rmean, 4),
        "recovery_above_chance": round(matched - rmean, 4),
        "mean_ari": float(summ.loc[name, "mean_ari"]),
        "mean_dominant_agreement": float(summ.loc[name, "mean_dominant_agreement"]),
        "mean_distinctiveness": float(summ.loc[name, "mean_distinctiveness"]),
        "mean_reproducible_topics": float(summ.loc[name, "mean_reproducible_topics"]),
        "mean_c_v": float(summ.loc[name, "mean_c_v"]),
        "mean_u_mass": float(summ.loc[name, "mean_u_mass"]),
    })
ctrl = pd.DataFrame(rows)
ctrl.to_csv(OUT / "T23_eta_chance_corrected.csv", index=False)

base_m = LdaModel.load(str(MODELS / f"{BASE}_seed42.model"))
irows = []
for name in ETAS:
    m = LdaModel.load(str(MODELS / f"{name}_seed42.model"))
    _, cols, sims = compare(base_m, m, dic)
    for t in range(K):
        irows.append({
            "eta_setting": name, "baseline_topic": t + 1,
            "matched_topic": int(cols[t]) + 1,
            "cosine_to_baseline": round(float(sims[t]), 4),
            "top12_terms": ", ".join(w for w, _ in m.show_topic(int(cols[t]), topn=12)),
        })
pd.DataFrame(irows).to_csv(OUT / "T24_eta_interpretation_shift.csv", index=False)

print("=" * 120)
print("ETA TRADE-OFF, CHANCE-CORRECTED")
print(ctrl[["eta_setting", "eta_value", "mean_matched_cosine", "random_pairing_cosine",
            "recovery_above_chance", "mean_ari", "mean_distinctiveness",
            "mean_reproducible_topics"]].to_string(index=False))
print("\nInterpretation: matched cosine rises with eta, but so does the chance baseline,")
print("because smoother topics are mutually more similar. recovery_above_chance and ARI")
print("are the signals that cannot be inflated that way.")
print("\n" + "=" * 120)
print("TOP TERMS AT eta=1.0 vs BASELINE eta=1/K (seed 42, matched)")
sub = pd.DataFrame(irows)
print(sub[sub.eta_setting == "eta_1.0"][
    ["baseline_topic", "matched_topic", "cosine_to_baseline", "top12_terms"]].to_string(
    index=False, max_colwidth=86))
