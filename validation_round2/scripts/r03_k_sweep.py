"""Task 3: is K=12 too granular?

Training configuration is C3 (100 passes, 1000 iterations), proven converged in
Task 1. Six seeds per K. alpha=symmetric, eta=1/K explicit (equivalent to the
baseline's eta='auto' for the fitted topics, see r02b, but stated explicitly so
the setting is unambiguous and reproducible outside gensim).

Interpretive caution built into the reporting: matched cosine rises mechanically
as K falls, because fewer topics means broader distributions that are easier to
pair. mean_matched_cosine is therefore reported alongside a null baseline, the
mean cosine under RANDOM topic pairing at the same K. The gap between matched
and random is the part that reflects genuine recovery rather than the geometry
of having fewer, broader topics.
"""
import json
import sys
import time
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

sys.path.insert(0, str(Path(__file__).parent))
from r00_common import (R2, SEEDS, load_corpus, train, compare, stability_block,
                        distinctiveness, theta)

OUT = R2 / "outputs"
MODELS = R2 / "models" / "k_sweep"
MODELS.mkdir(parents=True, exist_ok=True)

KS = [4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16]
PASSES, ITERS = 100, 1000
SMALL_TOPIC = 0.02

budget = float(sys.argv[1]) if len(sys.argv) > 1 else 1e9
c = load_corpus("B250")
bow, dic, toks = c["bow"], c["dictionary"], c["tokens"]
rng = np.random.default_rng(7)

start = time.time()
rows = []
for k in KS:
    for s in SEEDS:
        path, meta = MODELS / f"k{k}_seed{s}.model", MODELS / f"k{k}_seed{s}.json"
        if path.exists() and meta.exists():
            continue
        if time.time() - start > budget:
            done = len(list(MODELS.glob("*.model")))
            print(f"budget reached: {done}/{len(KS)*len(SEEDS)} models trained")
            sys.exit(0)
        m, d = train(bow, dic, k, s, PASSES, ITERS, alpha="symmetric", eta=1.0 / k)
        m.save(str(path))
        d["bound_per_word"] = round(float(m.log_perplexity(bow)), 6)
        d["c_v"] = round(CoherenceModel(model=m, texts=toks, dictionary=dic,
                                        coherence="c_v", processes=1).get_coherence(), 5)
        d["u_mass"] = round(CoherenceModel(model=m, corpus=bow, dictionary=dic,
                                           coherence="u_mass", processes=1).get_coherence(), 5)
        d["distinctiveness"] = round(distinctiveness(m), 5)
        th = theta(m, bow, k)
        prev = th.mean(0)
        d["n_small_topics"] = int((prev < SMALL_TOPIC).sum())
        d["min_prevalence"] = round(float(prev.min()), 5)
        d["max_prevalence"] = round(float(prev.max()), 5)
        d["prevalence_gini"] = round(float(
            (np.abs(prev[:, None] - prev[None, :]).sum()) / (2 * k * k * prev.mean())), 5)
        d["n_empty_dominant"] = int(sum(
            (th.argmax(1) == t).sum() == 0 for t in range(k)))
        meta.write_text(json.dumps(d))
        print(f"  K={k:<3} seed {s:<6} {d['seconds']:>5.1f}s  C_v {d['c_v']:.4f}  "
              f"small topics {d['n_small_topics']}")

n_done = len(list(MODELS.glob("*.model")))
if n_done < len(KS) * len(SEEDS):
    print(f"incomplete: {n_done}/{len(KS)*len(SEEDS)}")
    sys.exit(0)

print("\nAll models trained. Computing stability metrics.")
summ, per_model = [], []
for k in KS:
    models = {s: LdaModel.load(str(MODELS / f"k{k}_seed{s}.model")) for s in SEEDS}
    metas = {s: json.loads((MODELS / f"k{k}_seed{s}.json").read_text()) for s in SEEDS}
    for s in SEEDS:
        per_model.append({"k": k, "seed": s, **metas[s]})
    blk = stability_block(models, bow, dic, k)

    rand = []
    for s1, s2 in itertools.combinations(SEEDS, 2):
        S, _, _ = compare(models[s1], models[s2], dic, "cosine")
        rand += [float(S[np.arange(k), rng.permutation(k)].mean()) for _ in range(200)]
    rand_mean = float(np.mean(rand))

    md = pd.DataFrame([metas[s] for s in SEEDS])
    summ.append({
        "k": k,
        "mean_matched_cosine": blk["mean_matched_cosine"],
        "sd_matched_cosine": blk["sd_matched_cosine"],
        "random_pairing_cosine": round(rand_mean, 4),
        "recovery_above_chance": round(blk["mean_matched_cosine"] - rand_mean, 4),
        "mean_reproducible_topics": blk["mean_reproducible_topics"],
        "pct_reproducible_topics": round(100 * blk["mean_reproducible_topics"] / k, 1),
        "mean_dominant_agreement": blk["mean_dominant_agreement"],
        "mean_ari": blk["mean_ari"],
        "sd_ari": blk["sd_ari"],
        "mean_c_v": round(float(md.c_v.mean()), 4),
        "sd_c_v": round(float(md.c_v.std(ddof=1)), 4),
        "mean_u_mass": round(float(md.u_mass.mean()), 4),
        "mean_distinctiveness": round(float(md.distinctiveness.mean()), 4),
        "mean_bound_per_word": round(float(md.bound_per_word.mean()), 5),
        "mean_n_small_topics": round(float(md.n_small_topics.mean()), 2),
        "mean_min_prevalence": round(float(md.min_prevalence.mean()), 4),
        "mean_prevalence_gini": round(float(md.prevalence_gini.mean()), 4),
        "mean_seconds": round(float(md.seconds.mean()), 1),
    })
    print(f"  K={k:<3} matched {blk['mean_matched_cosine']:.4f}  "
          f"chance {rand_mean:.4f}  above-chance {blk['mean_matched_cosine']-rand_mean:.4f}  "
          f"ARI {blk['mean_ari']:.4f}")

pd.DataFrame(per_model).to_csv(OUT / "T10_k_sweep_per_model.csv", index=False)
sm = pd.DataFrame(summ)
sm.to_csv(OUT / "T11_k_sweep_summary.csv", index=False)

print("\n" + "=" * 126)
print("K SWEEP: STABILITY")
print(sm[["k", "mean_matched_cosine", "sd_matched_cosine", "random_pairing_cosine",
          "recovery_above_chance", "mean_reproducible_topics", "pct_reproducible_topics",
          "mean_dominant_agreement", "mean_ari"]].to_string(index=False))
print("\n" + "=" * 126)
print("K SWEEP: FIT, DISTINCTIVENESS AND TOPIC SIZE")
print(sm[["k", "mean_c_v", "sd_c_v", "mean_u_mass", "mean_distinctiveness",
          "mean_bound_per_word", "mean_n_small_topics", "mean_min_prevalence",
          "mean_prevalence_gini"]].to_string(index=False))
