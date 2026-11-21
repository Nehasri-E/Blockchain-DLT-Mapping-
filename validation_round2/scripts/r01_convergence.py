"""Task 1: is the K=12 seed instability caused by insufficient convergence?

Three independent lines of evidence are collected, because "training finished"
is not evidence of convergence:

  E1  E-step convergence. gensim logs, per chunk, how many documents reached
      their own variational convergence criterion before hitting `iterations`.
      If that rate is below 100%, the per-document posterior was truncated and
      `iterations` is too low.

  E2  Objective plateau. The variational lower bound per word. If more passes
      buy no bound improvement, the M-step has stopped moving.

  E3  Solution invariance under more training. For a FIXED seed, compare the
      topic-word distributions at configuration X against a longer-trained
      anchor (C4, 200 passes / 1000 iterations). A converged model should be
      essentially unchanged by further training. This is the strongest test,
      because E1 and E2 can both look healthy while the optimiser sits in a
      seed-dependent local optimum.

The distinction that matters: E1-E3 are about CONVERGENCE (does one run reach
its own optimum). Seed stability is about IDENTIFIABILITY (do different runs
reach the SAME optimum). A model can be perfectly converged and perfectly
unstable, and that is the hypothesis this script is built to detect.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from r00_common import (R2, SEEDS, load_corpus, train, compare, stability_block)

OUT = R2 / "outputs"
MODELS = R2 / "models" / "convergence"
OUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

K = 12
CONFIGS = {
    "baseline": (20, 200),
    "C1": (50, 500),
    "C2": (100, 500),
    "C3": (100, 1000),
    "C4_anchor": (200, 1000),
}

c = load_corpus("B250")
bow, dic = c["bow"], c["dictionary"]
print(f"Corpus B250: {len(bow)} documents, {len(dic)} terms, K={K}\n")

from gensim.models import LdaModel

models, diag_rows = {}, []
for cfg, (p, it) in CONFIGS.items():
    for s in SEEDS:
        path = MODELS / f"{cfg}_seed{s}.model"
        meta = MODELS / f"{cfg}_seed{s}.json"
        if path.exists() and meta.exists():
            m = LdaModel.load(str(path))
            d = json.loads(meta.read_text())
        else:
            m, d = train(bow, dic, K, s, p, it)
            m.save(str(path))
            d["bound_per_word"] = round(float(m.log_perplexity(bow)), 6)
            meta.write_text(json.dumps(d))
        models[(cfg, s)] = m
        diag_rows.append({"config": cfg, "passes": p, "iterations": it, "seed": s, **d})
        print(f"  {cfg:>10} seed {s:<6} {d['seconds']:>5.1f}s  "
              f"E-step convergence {d['final_pass_convergence_rate']:.3f}  "
              f"bound/word {d['bound_per_word']:.4f}")

diag = pd.DataFrame(diag_rows)
diag.to_csv(OUT / "T1_convergence_diagnostics_per_model.csv", index=False)

# ------------------------------------------------- E3 solution invariance
inv_rows = []
for cfg in CONFIGS:
    if cfg == "C4_anchor":
        continue
    for s in SEEDS:
        _, _, sims = compare(models[(cfg, s)], models[("C4_anchor", s)], dic)
        inv_rows.append({"config": cfg, "seed": s,
                         "cosine_vs_C4_anchor": round(float(sims.mean()), 4),
                         "min_topic_cosine": round(float(sims.min()), 4)})
inv = pd.DataFrame(inv_rows)
inv.to_csv(OUT / "T2_solution_invariance_vs_anchor.csv", index=False)

# ------------------------------------------------- stability per config
stab_rows = []
for cfg in CONFIGS:
    mm = {s: models[(cfg, s)] for s in SEEDS}
    blk = stability_block(mm, bow, dic, K)
    sub = diag[diag.config == cfg]
    stab_rows.append({
        "config": cfg, "passes": CONFIGS[cfg][0], "iterations": CONFIGS[cfg][1],
        "mean_seconds": round(float(sub["seconds"].mean()), 1),
        "mean_estep_convergence": round(float(sub["final_pass_convergence_rate"].mean()), 5),
        "mean_bound_per_word": round(float(sub["bound_per_word"].mean()), 5),
        "sd_bound_per_word": round(float(sub["bound_per_word"].std(ddof=1)), 5),
        "mean_cosine_vs_anchor": (round(float(inv[inv.config == cfg]["cosine_vs_C4_anchor"].mean()), 4)
                                  if cfg != "C4_anchor" else 1.0),
        **blk})
stab = pd.DataFrame(stab_rows)
stab.to_csv(OUT / "T3_stability_by_config.csv", index=False)

base = stab[stab.config == "baseline"].iloc[0]
delta = stab.assign(
    d_cosine=(stab["mean_matched_cosine"] - base["mean_matched_cosine"]).round(4),
    d_agreement=(stab["mean_dominant_agreement"] - base["mean_dominant_agreement"]).round(4),
    d_ari=(stab["mean_ari"] - base["mean_ari"]).round(4),
    d_bound=(stab["mean_bound_per_word"] - base["mean_bound_per_word"]).round(5),
)[["config", "passes", "iterations", "mean_seconds", "mean_matched_cosine", "d_cosine",
   "mean_dominant_agreement", "d_agreement", "mean_ari", "d_ari",
   "mean_bound_per_word", "d_bound", "mean_cosine_vs_anchor"]]
delta.to_csv(OUT / "T4_config_vs_baseline_delta.csv", index=False)

print("\n" + "=" * 112)
print("E1  E-STEP CONVERGENCE AND E2  OBJECTIVE PLATEAU")
print(stab[["config", "passes", "iterations", "mean_seconds", "mean_estep_convergence",
            "mean_bound_per_word", "sd_bound_per_word"]].to_string(index=False))

print("\n" + "=" * 112)
print("E3  SOLUTION INVARIANCE: cosine against the 200-pass/1000-iteration anchor, same seed")
print(inv.groupby("config").agg(mean=("cosine_vs_C4_anchor", "mean"),
                                sd=("cosine_vs_C4_anchor", "std"),
                                worst=("cosine_vs_C4_anchor", "min")).round(4).to_string())

print("\n" + "=" * 112)
print("SEED STABILITY BY CONFIGURATION (15 seed pairs each)")
print(stab[["config", "mean_matched_cosine", "sd_matched_cosine", "mean_reproducible_topics",
            "mean_dominant_agreement", "mean_ari"]].to_string(index=False))

print("\n" + "=" * 112)
print("CHANGE VS BASELINE")
print(delta[["config", "d_cosine", "d_agreement", "d_ari", "d_bound"]].to_string(index=False))
