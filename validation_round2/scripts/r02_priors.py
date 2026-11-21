"""Task 2: does the eta prior drive the instability?

Training configuration is C3 (100 passes, 1000 iterations), the configuration
Task 1 showed to be converged (cosine 0.9992 against a 200-pass anchor).

Three prior settings, six seeds each:
  P_base  alpha=symmetric, eta=auto       (the baseline prior)
  P_eta   alpha=symmetric, eta=symmetric
  P_alpha alpha=asymmetric, eta=auto

Selection is NOT on coherence. A prior is judged on whether it makes the
solution reproducible across seeds, with coherence and distinctiveness reported
alongside so the trade-off is visible.
"""
import json
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

sys.path.insert(0, str(Path(__file__).parent))
from r00_common import (R2, SEEDS, load_corpus, train, stability_block, distinctiveness)

OUT = R2 / "outputs"
MODELS = R2 / "models" / "priors"
MODELS.mkdir(parents=True, exist_ok=True)

K, PASSES, ITERS = 12, 100, 1000
PRIORS = {
    "P_base_sym_auto": ("symmetric", "auto"),
    "P_eta_sym_sym": ("symmetric", "symmetric"),
    "P_alpha_asym_auto": ("asymmetric", "auto"),
}

budget = float(sys.argv[1]) if len(sys.argv) > 1 else 1e9
c = load_corpus("B250")
bow, dic, toks = c["bow"], c["dictionary"], c["tokens"]

start = time.time()
models, rows = {}, []
for name, (a, e) in PRIORS.items():
    for s in SEEDS:
        path = MODELS / f"{name}_seed{s}.model"
        meta = MODELS / f"{name}_seed{s}.json"
        if path.exists() and meta.exists():
            m, d = LdaModel.load(str(path)), json.loads(meta.read_text())
        else:
            if time.time() - start > budget:
                print(f"budget reached; {len(list(MODELS.glob('*.model')))}/18 models done")
                sys.exit(0)
            m, d = train(bow, dic, K, s, PASSES, ITERS, alpha=a, eta=e)
            m.save(str(path))
            d["bound_per_word"] = round(float(m.log_perplexity(bow)), 6)
            d["c_v"] = round(CoherenceModel(model=m, texts=toks, dictionary=dic,
                                            coherence="c_v", processes=1).get_coherence(), 5)
            d["u_mass"] = round(CoherenceModel(model=m, corpus=bow, dictionary=dic,
                                               coherence="u_mass", processes=1).get_coherence(), 5)
            d["distinctiveness"] = round(distinctiveness(m), 5)
            meta.write_text(json.dumps(d))
        models[(name, s)] = m
        rows.append({"prior": name, "alpha": a, "eta": e, "seed": s, **d})
        print(f"  {name:<20} seed {s:<6} C_v {d['c_v']:.4f}  JS-dist {d['distinctiveness']:.4f}")

per_model = pd.DataFrame(rows)
per_model.to_csv(OUT / "T5_prior_per_model.csv", index=False)

summ = []
for name, (a, e) in PRIORS.items():
    mm = {s: models[(name, s)] for s in SEEDS}
    blk = stability_block(mm, bow, dic, K)
    sub = per_model[per_model.prior == name]
    summ.append({"prior": name, "alpha": a, "eta": e,
                 "mean_c_v": round(float(sub["c_v"].mean()), 4),
                 "sd_c_v": round(float(sub["c_v"].std(ddof=1)), 4),
                 "mean_u_mass": round(float(sub["u_mass"].mean()), 4),
                 "mean_distinctiveness": round(float(sub["distinctiveness"].mean()), 4),
                 "mean_bound_per_word": round(float(sub["bound_per_word"].mean()), 5),
                 "mean_estep_convergence": round(float(sub["final_pass_convergence_rate"].mean()), 5),
                 **blk})
s = pd.DataFrame(summ)
s.to_csv(OUT / "T6_prior_stability_summary.csv", index=False)

print("\n" + "=" * 118)
print("PRIOR SENSITIVITY at C3 (100 passes, 1000 iterations), 6 seeds, 15 pairs each")
print(s[["prior", "alpha", "eta", "mean_matched_cosine", "sd_matched_cosine",
         "mean_reproducible_topics", "mean_dominant_agreement", "mean_ari"]].to_string(index=False))
print("\nReported alongside, NOT used for selection:")
print(s[["prior", "mean_c_v", "mean_u_mass", "mean_distinctiveness",
         "mean_bound_per_word", "mean_estep_convergence"]].to_string(index=False))
