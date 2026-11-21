"""Task 5: sensitivity of the consensus result. Kept separate from the main analysis.

Four perturbations, each applied independently to the Round 3 consensus:

  S1 similarity threshold       swept 0.40 to 0.80 (already in C4; re-expressed
                                here in terms of which themes survive).
  S2 supporting-seed requirement swept 3 to 6 of 6.
  S3 influential journals        drop the top 1, 3 and 5 journals by paper count
                                 and refit LDA K=6 x 6 seeds on the surviving
                                 documents.
  S4 corpus definition           repeat the whole consensus on BFull
                                 (uncapped conclusions) and compare themes.

Design note for S3: documents are dropped from the existing BoW while the
dictionary is held fixed. Refitting filter_extremes on the reduced corpus would
change the vocabulary as well, confounding "do these journals carry the theme"
with "does the vocabulary change". Holding the vocabulary fixed isolates the
question actually being asked.
"""
import json
import sys
import time
import itertools
from collections import Counter
import numpy as np
import pandas as pd
from pathlib import Path
from gensim.models import LdaModel

sys.path.insert(0, str(Path(__file__).parent))
from s00_common import (R3, ROOT, SEEDS, load_corpus, load_k6_models, theta,
                        cosine_matrix, hungarian, term_lift)

OUT = R3 / "outputs"
MODELS = R3 / "models" / "sensitivity"
MODELS.mkdir(parents=True, exist_ok=True)
K, TOPN, TOPDOCS = 6, 20, 25
HP = dict(num_topics=K, alpha="symmetric", eta=1.0 / K, passes=100,
          iterations=1000, chunksize=200, eval_every=None)
budget = float(sys.argv[1]) if len(sys.argv) > 1 else 1e9

base = load_corpus("B250")
dic = base["dictionary"]
vocab = [dic[i] for i in range(len(dic))]
cfs = np.array([dic.cfs[i] for i in range(len(dic))], float)
cfs_norm = cfs / cfs.sum()
meta = pd.read_parquet(ROOT / "data/processed/lda_master.parquet").set_index("Paper_ID")
ref_cons = pd.read_csv(OUT / "C10_consensus_topics_seedpure.csv")


def consensus_from(models_by_seed, bow, pids, anchor=42):
    TW = np.vstack([models_by_seed[s].get_topics() for s in SEEDS])
    TH = {s: theta(models_by_seed[s], bow, K) for s in SEEDS}
    labels = [(s, t) for s in SEEDS for t in range(K)]
    idx_of = {l: i for i, l in enumerate(labels)}
    S = cosine_matrix(TW, TW)
    np.fill_diagonal(S, 1.0)
    a_idx = [idx_of[(anchor, t)] for t in range(K)]
    grouping = {(anchor, t): t for t in range(K)}
    for s in SEEDS:
        if s == anchor:
            continue
        b_idx = [idx_of[(s, t)] for t in range(K)]
        cols, _ = hungarian(S[np.ix_(a_idx, b_idx)])
        for g in range(K):
            grouping[(s, int(cols[g]))] = g
    out = []
    for g in range(K):
        mem = [idx_of[l] for l in labels if grouping[l] == g]
        w = TW[mem].mean(0)
        w = w / w.sum()
        terms = [vocab[i] for i in np.argsort(-w)[:TOPN]]
        cnt = Counter()
        for i in mem:
            cnt.update([vocab[j] for j in np.argsort(-TW[i])[:TOPN]])
        core = [t for t in terms if cnt[t] >= 4]
        th_cols = [TH[labels[i][0]][:, labels[i][1]] for i in mem]
        td = [set(np.argsort(-tc)[:TOPDOCS]) for tc in th_cols]
        jac = [len(a & b) / len(a | b) for a, b in itertools.combinations(td, 2)]
        within = [S[i, j] for i, j in itertools.combinations(mem, 2)]
        out.append({"group": g + 1, "weights": w,
                    "mean_within_cosine": float(np.mean(within)),
                    "min_within_cosine": float(np.min(within)),
                    "mean_doc_jaccard": float(np.mean(jac)),
                    "prevalence": float(np.mean(th_cols, 0).mean()),
                    "term_lift": term_lift(w, vocab, cfs_norm, TOPN),
                    "core_size": len(core), "core": core, "terms": terms})
    return out


ref = consensus_from(load_k6_models(), base["bow"], base["paper_ids"])
REF_W = np.vstack([r["weights"] for r in ref])

# ----------------------------------------------------------------- S2 seeds
memb = pd.read_csv(OUT / "C11_consensus_membership_seedpure.csv")
s2 = []
for req in [3, 4, 5, 6]:
    q = []
    for _, r in ref_cons.iterrows():
        ok = (r.n_supporting_seeds >= req and r.stable_core_size >= 4 and
              r.mean_top25_doc_jaccard >= 0.15 and r.term_lift_vs_corpus >= 2.0 and
              r.min_within_cosine >= 0.50)
        if ok:
            q.append(int(r.consensus_topic))
    s2.append({"min_supporting_seeds": req, "n_qualifying": len(q),
               "qualifying_topics": ", ".join(f"CT{x}" for x in sorted(q))})
pd.DataFrame(s2).to_csv(OUT / "S2_supporting_seed_sensitivity.csv", index=False)

# ------------------------------------------------------------ S3 journals
jc = meta.loc[base["paper_ids"]]["Journal"].value_counts()
rows_s3, start = [], time.time()
for drop_n in [1, 3, 5]:
    drop = set(jc.head(drop_n).index)
    keep = [i for i, p in enumerate(base["paper_ids"])
            if meta.loc[p, "Journal"] not in drop]
    bow = [base["bow"][i] for i in keep]
    pids = [base["paper_ids"][i] for i in keep]
    mods = {}
    for s in SEEDS:
        path = MODELS / f"dropj{drop_n}_seed{s}.model"
        if path.exists():
            mods[s] = LdaModel.load(str(path))
        else:
            if time.time() - start > budget:
                print(f"budget reached during S3 drop_n={drop_n}")
                sys.exit(0)
            m = LdaModel(corpus=bow, id2word=dic, random_state=s, **HP)
            m.save(str(path))
            mods[s] = m
    sub = consensus_from(mods, bow, pids)
    SUB_W = np.vstack([r["weights"] for r in sub])
    Sm = cosine_matrix(REF_W, SUB_W)
    cols, sims = hungarian(Sm)
    for g in range(K):
        rows_s3.append({
            "dropped_journals": drop_n, "documents_remaining": len(keep),
            "ref_consensus_topic": g + 1,
            "matched_topic": int(cols[g]) + 1,
            "cosine_to_reference": round(float(sims[g]), 4),
            "sub_mean_within_cosine": round(sub[int(cols[g])]["mean_within_cosine"], 4),
            "sub_doc_jaccard": round(sub[int(cols[g])]["mean_doc_jaccard"], 4),
            "sub_term_lift": round(sub[int(cols[g])]["term_lift"], 2),
            "sub_core_size": sub[int(cols[g])]["core_size"],
            "sub_core": ", ".join(sub[int(cols[g])]["core"][:10])})
    print(f"  S3 drop top {drop_n} journals ({len(keep)} docs): "
          f"mean cosine to reference {sims.mean():.4f}")
pd.DataFrame(rows_s3).to_csv(OUT / "S3_journal_exclusion_sensitivity.csv", index=False)

# --------------------------------------------------------------- S4 BFull
bf = load_corpus("BFull")
mods = {}
for s in SEEDS:
    path = MODELS / f"bfull_seed{s}.model"
    if path.exists():
        mods[s] = LdaModel.load(str(path))
    else:
        if time.time() - start > budget:
            print("budget reached during S4")
            sys.exit(0)
        m = LdaModel(corpus=bf["bow"], id2word=bf["dictionary"], random_state=s, **HP)
        m.save(str(path))
        mods[s] = m

TWb = np.vstack([mods[s].get_topics() for s in SEEDS])
labels_b = [(s, t) for s in SEEDS for t in range(K)]
Sb = cosine_matrix(TWb, TWb)
np.fill_diagonal(Sb, 1.0)
a_idx = [labels_b.index((42, t)) for t in range(K)]
grouping = {(42, t): t for t in range(K)}
for s in SEEDS:
    if s == 42:
        continue
    b_idx = [labels_b.index((s, t)) for t in range(K)]
    cols, _ = hungarian(Sb[np.ix_(a_idx, b_idx)])
    for g in range(K):
        grouping[(s, int(cols[g]))] = g

bvocab = [bf["dictionary"][i] for i in range(len(bf["dictionary"]))]
bcons = []
for g in range(K):
    mem = [labels_b.index(l) for l in labels_b if grouping[l] == g]
    w = TWb[mem].mean(0)
    w = w / w.sum()
    bcons.append({"terms": {bvocab[i]: float(w[i]) for i in np.argsort(-w)[:400]},
                  "top": [bvocab[i] for i in np.argsort(-w)[:TOPN]]})

from s00_common import align_union
ref_dicts = [{vocab[i]: float(r["weights"][i]) for i in np.argsort(-r["weights"])[:400]}
             for r in ref]
Sx, colsx, simsx = align_union(ref_dicts, [b["terms"] for b in bcons])
s4 = [{"ref_consensus_topic": g + 1,
       "matched_BFull_topic": int(colsx[g]) + 1,
       "cosine_B250_vs_BFull": round(float(simsx[g]), 4),
       "BFull_top12": ", ".join(bcons[int(colsx[g])]["top"][:12])} for g in range(K)]
pd.DataFrame(s4).to_csv(OUT / "S4_bfull_corpus_sensitivity.csv", index=False)

print("\n" + "=" * 120)
print("S2  SUPPORTING-SEED REQUIREMENT")
print(pd.DataFrame(s2).to_string(index=False))
print("\n" + "=" * 120)
print("S3  JOURNAL EXCLUSION: cosine of each reference consensus topic to its match")
piv = pd.DataFrame(rows_s3).pivot(index="ref_consensus_topic", columns="dropped_journals",
                                  values="cosine_to_reference")
print(piv.to_string())
print("\n" + "=" * 120)
print("S4  B250 vs BFull CONSENSUS")
print(pd.DataFrame(s4).to_string(index=False, max_colwidth=70))
