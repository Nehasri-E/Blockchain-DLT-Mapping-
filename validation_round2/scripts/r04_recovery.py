"""Task 4: topic recovery and interpretability across seeds.

For each candidate K, topics are anchored on seed 42 and matched to every other
seed by Hungarian assignment. A topic counts as RECURRING when the same
substantive theme reappears across seeds, judged by two independent signals
that must agree:

  lexical   mean matched cosine across the 5 comparison seeds, plus the size of
            the stable term core (terms in the anchor's top 20 that also appear
            in the matched topic's top 20 in at least 4 of 5 other seeds).

  documental  whether the same papers are assigned. Measured as the mean
            Jaccard overlap of the top 25 papers by topic share between the
            anchor topic and its match in each other seed.

Lexical agreement without documental agreement is the signature of a broad
catch-all topic: the words recur because they are corpus-generic, while the
documents it claims shift from run to run. Both are therefore reported and the
catch-all flag fires when cosine is respectable but document overlap is not.

Catch-all detection also uses topic specificity: the mean of the top-20 term
probabilities relative to those terms' corpus frequency. A topic whose top
terms are simply the corpus's most frequent words is a catch-all regardless of
how stably it recurs.

Proposed interpretations are written from the recurring term core AND the
representative papers, never from the term list alone. Baseline labels are
deliberately NOT reused; they are shown in a separate column for comparison
only.
"""
import json
import sys
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from gensim.models import LdaModel

sys.path.insert(0, str(Path(__file__).parent))
from r00_common import R2, ROOT, SEEDS, load_corpus, compare, theta

OUT = R2 / "outputs"
MODELS = R2 / "models" / "k_sweep"
ANCHOR = 42
TOPN = 20
TOPDOCS = 25
CANDIDATE_KS = [int(a) for a in sys.argv[1:]] or [6, 8, 12]

c = load_corpus("B250")
bow, dic, pids = c["bow"], c["dictionary"], c["paper_ids"]
meta = pd.read_parquet(ROOT / "data/processed/lda_master.parquet").set_index("Paper_ID")
LABELS = {int(k): v for k, v in json.loads(
    (ROOT / "validation_round1/scripts/baseline_labels.json").read_text()).items()}

cf = np.array([dic.cfs[i] for i in range(len(dic))], dtype=float)
cf = cf / cf.sum()

rows, term_rows = [], []
for k in CANDIDATE_KS:
    models = {s: LdaModel.load(str(MODELS / f"k{k}_seed{s}.model")) for s in SEEDS}
    th = {s: theta(models[s], bow, k) for s in SEEDS}
    others = [s for s in SEEDS if s != ANCHOR]

    matches, sims = {}, {}
    for s in others:
        _, cols, sim = compare(models[ANCHOR], models[s], dic)
        matches[s], sims[s] = cols, sim

    for t in range(k):
        anchor_terms = [w for w, _ in models[ANCHOR].show_topic(t, topn=TOPN)]
        anchor_probs = dict(models[ANCHOR].show_topic(t, topn=TOPN))

        counts = {}
        for s in others:
            mt = int(matches[s][t])
            for w, _ in models[s].show_topic(mt, topn=TOPN):
                counts[w] = counts.get(w, 0) + 1
        core = sorted([w for w in anchor_terms if counts.get(w, 0) >= 4],
                      key=lambda w: -anchor_probs[w])
        drift = [w for w in anchor_terms if counts.get(w, 0) <= 1]

        top_idx = np.argsort(-th[ANCHOR][:, t])[:TOPDOCS]
        anchor_docs = {pids[i] for i in top_idx}
        jac = []
        for s in others:
            mt = int(matches[s][t])
            d2 = {pids[i] for i in np.argsort(-th[s][:, mt])[:TOPDOCS]}
            jac.append(len(anchor_docs & d2) / len(anchor_docs | d2))

        tid = [dic.token2id[w] for w in anchor_terms if w in dic.token2id]
        lift = float(np.mean([anchor_probs[dic[i]] / max(cf[i], 1e-12) for i in tid]))

        cos_mean = float(np.mean([sims[s][t] for s in others]))
        cos_min = float(np.min([sims[s][t] for s in others]))
        jac_mean = float(np.mean(jac))
        prev = float(th[ANCHOR][:, t].mean())

        stability = ("reproducible" if cos_min >= 0.70 else
                     "partial" if cos_mean >= 0.50 else "unstable")
        catchall = bool(cos_mean >= 0.50 and jac_mean < 0.25) or lift < 2.0

        reps = []
        for i in top_idx[:3]:
            r = meta.loc[pids[i]]
            reps.append(f"{str(r['Title_Clean'])[:72]} ({int(r['Year'])}, "
                        f"{str(r['Journal'])[:28]})")

        rows.append({
            "candidate_K": k, "topic_id": t + 1,
            "prevalence": round(prev, 4),
            "mean_cosine_across_seeds": round(cos_mean, 4),
            "min_cosine_across_seeds": round(cos_min, 4),
            "stable_term_core_size": len(core),
            "mean_top25_doc_jaccard": round(jac_mean, 4),
            "term_lift_vs_corpus": round(lift, 2),
            "stability": stability,
            "catch_all_flag": catchall,
            "stable_core_terms": ", ".join(core[:12]),
            "drifting_terms": ", ".join(drift[:8]),
            "top20_terms_anchor": ", ".join(anchor_terms),
            "representative_papers": " | ".join(reps),
            "baseline_label_for_reference": LABELS.get(t + 1, "") if k == 12 else "",
        })

        for s in SEEDS:
            mt = t if s == ANCHOR else int(matches[s][t])
            term_rows.append({
                "candidate_K": k, "anchor_topic": t + 1, "seed": s,
                "matched_topic": mt + 1,
                "cosine": 1.0 if s == ANCHOR else round(float(sims[s][t]), 4),
                "top20": ", ".join(w for w, _ in models[s].show_topic(mt, topn=TOPN)),
            })

rec = pd.DataFrame(rows)
rec.to_csv(OUT / "T20_topic_recovery_table.csv", index=False)
pd.DataFrame(term_rows).to_csv(OUT / "T21_matched_top20_by_seed.csv", index=False)

fate = []
for k in CANDIDATE_KS:
    sub = rec[rec.candidate_K == k]
    fate.append({
        "candidate_K": k,
        "reproducible": int((sub.stability == "reproducible").sum()),
        "partial": int((sub.stability == "partial").sum()),
        "unstable": int((sub.stability == "unstable").sum()),
        "catch_all_flagged": int(sub.catch_all_flag.sum()),
        "mean_doc_jaccard": round(float(sub.mean_top25_doc_jaccard.mean()), 4),
        "mean_term_lift": round(float(sub.term_lift_vs_corpus.mean()), 2),
        "mean_core_size": round(float(sub.stable_term_core_size.mean()), 1),
        "defensible_topics": int(((sub.stability != "unstable") & ~sub.catch_all_flag).sum()),
    })
f = pd.DataFrame(fate)
f.to_csv(OUT / "T22_topic_fate_summary.csv", index=False)

print("=" * 128)
print("TOPIC FATE BY CANDIDATE K")
print(f.to_string(index=False))
for k in CANDIDATE_KS:
    print("\n" + "=" * 128)
    print(f"K={k}  RECOVERY DETAIL (anchor seed 42)")
    print(rec[rec.candidate_K == k][
        ["topic_id", "prevalence", "mean_cosine_across_seeds", "stable_term_core_size",
         "mean_top25_doc_jaccard", "term_lift_vs_corpus", "stability", "catch_all_flag",
         "stable_core_terms"]].to_string(index=False, max_colwidth=58))
