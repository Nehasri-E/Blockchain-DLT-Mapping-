"""Task 2, extended: genuine eta sensitivity using explicit numeric priors.

r02_priors.py ran the comparison as specified and found eta='auto' and
eta='symmetric' to be bit-identical. That is not a coding error. In gensim
4.4.0:

  * LdaState.get_lambda() returns state.eta + state.sstats, and
    get_Elogbeta() = dirichlet_expectation(get_lambda()). The topics are
    therefore a function of state.eta.
  * init_dir_prior returns 1.0/num_topics for BOTH 'auto' and 'symmetric', so
    state.eta starts identical under either setting.
  * LdaModel.update_eta assigns to self.eta (the MODEL's eta). It never writes
    to self.state.eta. The learned eta consequently never feeds back into the
    quantity that produces the topics.

Net effect: eta='auto' changes only log_perplexity (whose bound uses
model.eta), never the fitted topic-word distributions. Verified directly:
explicit eta=0.01 moves the topics by 3.4e-02 while 'auto' and 'symmetric'
differ by exactly 0.0.

This script therefore sweeps explicit numeric eta to answer the question the
brief intended.
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
from r00_common import R2, SEEDS, load_corpus, train, stability_block, distinctiveness

OUT = R2 / "outputs"
MODELS = R2 / "models" / "eta_explicit"
MODELS.mkdir(parents=True, exist_ok=True)

K, PASSES, ITERS = 12, 100, 1000
ETAS = {"eta_0.01": 0.01, "eta_0.0833_1overK": 1.0 / K, "eta_0.3": 0.3, "eta_1.0": 1.0}

budget = float(sys.argv[1]) if len(sys.argv) > 1 else 1e9
c = load_corpus("B250")
bow, dic, toks = c["bow"], c["dictionary"], c["tokens"]

start = time.time()
models, rows = {}, []
for name, e in ETAS.items():
    for s in SEEDS:
        path, meta = MODELS / f"{name}_seed{s}.model", MODELS / f"{name}_seed{s}.json"
        if path.exists() and meta.exists():
            m, d = LdaModel.load(str(path)), json.loads(meta.read_text())
        else:
            if time.time() - start > budget:
                print(f"budget reached: {len(list(MODELS.glob('*.model')))}/{len(ETAS)*len(SEEDS)}")
                sys.exit(0)
            m, d = train(bow, dic, K, s, PASSES, ITERS, alpha="symmetric", eta=e)
            m.save(str(path))
            d["bound_per_word"] = round(float(m.log_perplexity(bow)), 6)
            d["c_v"] = round(CoherenceModel(model=m, texts=toks, dictionary=dic,
                                            coherence="c_v", processes=1).get_coherence(), 5)
            d["u_mass"] = round(CoherenceModel(model=m, corpus=bow, dictionary=dic,
                                               coherence="u_mass", processes=1).get_coherence(), 5)
            d["distinctiveness"] = round(distinctiveness(m), 5)
            meta.write_text(json.dumps(d))
        models[(name, s)] = m
        rows.append({"eta_setting": name, "eta_value": e, "seed": s, **d})

pm = pd.DataFrame(rows)
pm.to_csv(OUT / "T7_eta_explicit_per_model.csv", index=False)

summ = []
for name, e in ETAS.items():
    blk = stability_block({s: models[(name, s)] for s in SEEDS}, bow, dic, K)
    sub = pm[pm.eta_setting == name]
    summ.append({"eta_setting": name, "eta_value": e,
                 "mean_c_v": round(float(sub.c_v.mean()), 4),
                 "mean_u_mass": round(float(sub.u_mass.mean()), 4),
                 "mean_distinctiveness": round(float(sub.distinctiveness.mean()), 4),
                 "mean_bound_per_word": round(float(sub.bound_per_word.mean()), 5), **blk})
s_ = pd.DataFrame(summ)
s_.to_csv(OUT / "T8_eta_explicit_stability.csv", index=False)

print("=" * 118)
print("EXPLICIT ETA SWEEP at C3 (100 passes / 1000 iterations), 6 seeds, K=12")
print(s_[["eta_setting", "eta_value", "mean_matched_cosine", "sd_matched_cosine",
          "mean_reproducible_topics", "mean_dominant_agreement", "mean_ari"]].to_string(index=False))
print("\nReported alongside, not used for selection:")
print(s_[["eta_setting", "mean_c_v", "mean_u_mass", "mean_distinctiveness",
          "mean_bound_per_word"]].to_string(index=False))

note = {
    "finding": "In gensim 4.4.0, eta='auto' is a no-op for fitted topic-word distributions.",
    "mechanism": ("LdaState.get_lambda() = state.eta + state.sstats determines the topics. "
                  "init_dir_prior returns 1.0/num_topics for both 'auto' and 'symmetric', so "
                  "state.eta is identical. LdaModel.update_eta writes to model.eta only and "
                  "never to state.eta, so the learned prior never affects the topics."),
    "evidence": {
        "auto_vs_symmetric_max_topic_diff": 0.0,
        "auto_vs_explicit_0.01_max_topic_diff": 0.03409,
        "expElogbeta_bit_identical_auto_vs_symmetric": True,
        "what_does_change": "log_perplexity only, because the bound uses model.eta",
    },
    "consequence_for_baseline": ("The published model's eta='auto' is equivalent to "
                                 "eta='symmetric'. The 312-model baseline tuning grid varied "
                                 "eta over {symmetric, auto}, so that grid dimension produced "
                                 "identical topics and differed only in reported perplexity. "
                                 "The effective grid was half its nominal size."),
}
(OUT / "T9_eta_auto_noop_finding.json").write_text(json.dumps(note, indent=2))
print("\nWrote T9_eta_auto_noop_finding.json")
