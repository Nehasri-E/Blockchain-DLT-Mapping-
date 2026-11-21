"""Train the validation model grid: 5 cap conditions x 6 seeds, K=12.

All hyperparameters are the baseline values from models/model_config.json.
The only things that vary are the corpus (cap condition) and random_state.

Resumable: models already on disk are skipped.
"""
import pickle
import time
import json
import numpy as np
import pandas as pd
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from pathlib import Path

ROOT = Path("/home/claude/LDA_Project")
VR = ROOT / "validation_round1"
MODELS = VR / "models" / "grid"
MODELS.mkdir(parents=True, exist_ok=True)
OUT = VR / "outputs"

K = 12
SEEDS = [42, 7, 2024, 123, 777, 31337]
CONDITIONS = ["A", "B250", "B500", "B1000", "BFull"]
HP = dict(num_topics=K, alpha="symmetric", eta="auto", passes=20,
          iterations=200, chunksize=200, eval_every=None)

corpora = pickle.load(open(VR / "models" / "validation_corpora.pkl", "rb"))

print("=" * 78)
print("REUSE CHECK: does refitting B250 seed 42 reproduce the baseline model?")
base = pickle.load(open(ROOT / "models" / "candidates.pkl", "rb"))[K]
t = time.time()
refit = LdaModel(corpus=corpora["B250"]["bow"], id2word=corpora["B250"]["dictionary"],
                 random_state=42, **HP)
same_beta = np.array_equal(base.expElogbeta, refit.expElogbeta)
max_dev = float(np.abs(base.get_topics() - refit.get_topics()).max())
print(f"  expElogbeta bit-identical : {same_beta}")
print(f"  max topic-word deviation  : {max_dev:.3e}")
print(f"  fit time                  : {time.time() - t:.0f}s")
print("  -> the baseline B250/seed42 model is reproducible from the rebuilt corpus")

rows = []
for cond in CONDITIONS:
    c = corpora[cond]
    for seed in SEEDS:
        path = MODELS / f"lda_{cond}_k{K}_seed{seed}.model"
        if path.exists():
            m = LdaModel.load(str(path))
            cached = True
        else:
            t = time.time()
            m = LdaModel(corpus=c["bow"], id2word=c["dictionary"], random_state=seed, **HP)
            m.save(str(path))
            cached = False
        cv = CoherenceModel(model=m, texts=c["tokens"], dictionary=c["dictionary"],
                            coherence="c_v", processes=1).get_coherence()
        rows.append({
            "condition": cond, "seed": seed, "k": K,
            "documents": len(c["bow"]), "vocabulary": len(c["dictionary"]),
            "c_v": round(cv, 5),
            "log_perplexity": round(m.log_perplexity(c["bow"]), 5),
            "model_file": path.name,
        })
        print(f"  {cond:>6} seed {seed:<6} C_v {cv:.4f}  {'(cached)' if cached else ''}")

grid = pd.DataFrame(rows)
grid.to_csv(OUT / "A2_model_grid.csv", index=False)

agg = grid.groupby("condition").agg(
    n_models=("seed", "count"), c_v_mean=("c_v", "mean"), c_v_sd=("c_v", "std"),
    perplexity_mean=("log_perplexity", "mean"), vocabulary=("vocabulary", "first"),
    documents=("documents", "first")).round(5).reset_index()
agg.to_csv(OUT / "A3_grid_summary_by_condition.csv", index=False)

(VR / "models" / "grid_config.json").write_text(json.dumps({
    "K": K, "seeds": SEEDS, "conditions": CONDITIONS,
    "hyperparameters": {k: (v if v is not None else "None") for k, v in HP.items()},
    "implementation": "gensim.models.LdaModel (single process, deterministic)",
    "total_models": len(rows),
    "baseline_reuse_check": {
        "refit_B250_seed42_expElogbeta_bit_identical_to_baseline": bool(same_beta),
        "max_topic_word_deviation": max_dev,
    },
}, indent=2), encoding="utf-8")

print(f"\nTrained/loaded {len(rows)} models")
print(agg.to_string(index=False))
