"""Task 2-3 continued: seed-pure consensus, and diagnosis of the impure clusters.

s01 showed that average-linkage hierarchical clustering at cosine 0.55 produces
two clusters (C5, C2) with seven members, i.e. one seed contributed two topics.
The brief requires that a consensus group not contain multiple topics from the
same seed without clear justification. Two things are done here:

A. DIAGNOSIS of each impure cluster. For the seed that contributed two topics,
   the mutual cosine of its two topics is compared against the cluster's mean
   within-cluster cosine. If the two same-seed topics are at least as similar to
   each other as the cluster is internally, that seed simply split one theme and
   merging is justified. If they are not, the cluster is genuinely conflating
   two things and must be split.

B. PRIMARY CONSENSUS by a procedure that enforces seed uniqueness by
   construction: anchor-based Hungarian matching (M1). Each of the six anchors
   yields six groups of exactly six topics, one per seed. The six anchor
   solutions are compared; the modal (most frequently recurring) grouping is
   taken as the consensus, and the per-anchor agreement is reported so the
   reader can see how anchor-dependent it is.

Every consensus theme is then validated on the full battery required by Task 3.
"""
import json
import sys
import itertools
from collections import Counter
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from s00_common import (R3, ROOT, SEEDS, load_corpus, load_k6_models, theta,
                        cosine_matrix, js_matrix, hungarian, term_lift)

OUT = R3 / "outputs"
K, TOPN, TOPDOCS = 6, 20, 25

c = load_corpus("B250")
bow, dic, pids = c["bow"], c["dictionary"], c["paper_ids"]
vocab = [dic[i] for i in range(len(dic))]
cfs = np.array([dic.cfs[i] for i in range(len(dic))], float)
cfs_norm = cfs / cfs.sum()
meta = pd.read_parquet(ROOT / "data/processed/lda_master.parquet").set_index("Paper_ID")

models = load_k6_models()
TW = np.vstack([models[s].get_topics() for s in SEEDS])
TH = {s: theta(models[s], bow, K) for s in SEEDS}
labels = [(s, t) for s in SEEDS for t in range(K)]
idx_of = {l: i for i, l in enumerate(labels)}
S = cosine_matrix(TW, TW)
SJ = js_matrix(TW, TW)
np.fill_diagonal(S, 1.0)

# --------------------------------------------------------------- A. diagnosis
memb = pd.read_csv(OUT / "C6_consensus_membership.csv")
clus = pd.read_csv(OUT / "C5_consensus_clusters.csv")
diag = []
for cid in clus[clus.seed_impure]["cluster_id"]:
    mem = memb[memb.cluster_id == cid]
    within = float(clus[clus.cluster_id == cid]["mean_within_cosine"].iloc[0])
    for s, g in mem.groupby("seed"):
        if len(g) < 2:
            continue
        for (_, a), (_, b) in itertools.combinations(g.iterrows(), 2):
            ia, ib = idx_of[(s, a.topic - 1)], idx_of[(s, b.topic - 1)]
            diag.append({
                "cluster_id": cid, "seed": s,
                "topic_a": a.topic, "topic_b": b.topic,
                "same_seed_cosine": round(float(S[ia, ib]), 4),
                "cluster_mean_within_cosine": round(within, 4),
                "justified_merge": bool(S[ia, ib] >= within),
                "verdict": ("one seed split a single theme; merge justified"
                            if S[ia, ib] >= within else
                            "two distinct topics conflated; cluster should split"),
                "topic_a_top8": a.top10, "topic_b_top8": b.top10,
            })
dg = pd.DataFrame(diag)
dg.to_csv(OUT / "C8_seed_impurity_diagnosis.csv", index=False)
print("A. SEED-IMPURITY DIAGNOSIS")
print(dg[["cluster_id", "seed", "topic_a", "topic_b", "same_seed_cosine",
          "cluster_mean_within_cosine", "justified_merge", "verdict"]].to_string(index=False))

# ------------------------------------------------- B. anchor-based consensus
solutions = {}
for anchor in SEEDS:
    a_idx = [idx_of[(anchor, t)] for t in range(K)]
    grouping = {(anchor, t): t for t in range(K)}
    for s in SEEDS:
        if s == anchor:
            continue
        b_idx = [idx_of[(s, t)] for t in range(K)]
        cols, _ = hungarian(S[np.ix_(a_idx, b_idx)])
        for g in range(K):
            grouping[(s, int(cols[g]))] = g
    solutions[anchor] = grouping

pair_co = Counter()
for grouping in solutions.values():
    for a, b in itertools.combinations(labels, 2):
        if grouping[a] == grouping[b]:
            pair_co[(a, b)] += 1

ref = solutions[42]
stab = []
for g in range(K):
    mem = [l for l in labels if ref[l] == g]
    co = [pair_co[tuple(sorted((a, b)))] / len(SEEDS)
          for a, b in itertools.combinations(mem, 2)]
    stab.append({"group": g + 1, "mean_co_membership_across_anchors": round(float(np.mean(co)), 3)})
pd.DataFrame(stab).to_csv(OUT / "C9_anchor_solution_stability.csv", index=False)
print(f"\nB. ANCHOR CONSENSUS: mean co-membership across the 6 anchor solutions "
      f"{np.mean([s['mean_co_membership_across_anchors'] for s in stab]):.3f}")

rows, member_rows = [], []
for g in range(K):
    mem = [idx_of[l] for l in labels if ref[l] == g]
    seeds_in = [labels[i][0] for i in mem]
    assert len(set(seeds_in)) == len(seeds_in) == len(SEEDS), "anchor group not seed-pure"

    w = TW[mem].mean(0)
    w = w / w.sum()
    terms = [vocab[i] for i in np.argsort(-w)[:TOPN]]
    counts = Counter()
    for i in mem:
        counts.update([vocab[j] for j in np.argsort(-TW[i])[:TOPN]])
    core = [t for t in terms if counts[t] >= 4]

    th_cols = [TH[labels[i][0]][:, labels[i][1]] for i in mem]
    cons_theta = np.mean(th_cols, 0)
    topdocs = [set(np.argsort(-tc)[:TOPDOCS]) for tc in th_cols]
    jac = [len(a & b) / len(a | b) for a, b in itertools.combinations(topdocs, 2)]

    within = [S[i, j] for i, j in itertools.combinations(mem, 2)]
    wjs = [SJ[i, j] for i, j in itertools.combinations(mem, 2)]
    other = [i for i in range(len(labels)) if i not in mem]
    between = float(S[np.ix_(mem, other)].mean())

    reps = []
    for i in np.argsort(-cons_theta)[:4]:
        r = meta.loc[pids[i]]
        reps.append(f"{str(r['Title_Clean'])[:68]} ({int(r['Year'])}, {str(r['Journal'])[:24]})")

    co = float(np.mean([pair_co[tuple(sorted((labels[i], labels[j])))] / len(SEEDS)
                        for i, j in itertools.combinations(mem, 2)]))
    rows.append({
        "consensus_topic": g + 1, "n_supporting_seeds": len(set(seeds_in)),
        "mean_within_cosine": round(float(np.mean(within)), 4),
        "min_within_cosine": round(float(np.min(within)), 4),
        "mean_within_js": round(float(np.mean(wjs)), 4),
        "mean_between_cosine": round(between, 4),
        "separation": round(float(np.mean(within)) - between, 4),
        "mean_top25_doc_jaccard": round(float(np.mean(jac)), 4),
        "min_top25_doc_jaccard": round(float(np.min(jac)), 4),
        "consensus_prevalence": round(float(cons_theta.mean()), 4),
        "term_lift_vs_corpus": round(term_lift(w, vocab, cfs_norm, TOPN), 2),
        "stable_core_size": len(core),
        "anchor_co_membership": round(co, 3),
        "stable_core_terms": ", ".join(core[:14]),
        "consensus_top20_terms": ", ".join(terms),
        "representative_papers": " | ".join(reps),
    })
    for i in mem:
        s, t = labels[i]
        member_rows.append({"consensus_topic": g + 1, "seed": s, "topic": t + 1,
                            "cosine_to_consensus": round(float(
                                cosine_matrix(TW[i:i + 1], w[None, :])[0, 0]), 4),
                            "top10": ", ".join(vocab[j] for j in np.argsort(-TW[i])[:10])})

cons = pd.DataFrame(rows)
cons["qualifies"] = ((cons.stable_core_size >= 4) &
                     (cons.mean_top25_doc_jaccard >= 0.15) &
                     (cons.term_lift_vs_corpus >= 2.0) &
                     (cons.min_within_cosine >= 0.50))
cons = cons.sort_values("mean_within_cosine", ascending=False)
cons.to_csv(OUT / "C10_consensus_topics_seedpure.csv", index=False)
pd.DataFrame(member_rows).to_csv(OUT / "C11_consensus_membership_seedpure.csv", index=False)

print("\n" + "=" * 132)
print("SEED-PURE CONSENSUS TOPICS (anchor-based, 6 members each, one per seed)")
print(cons[["consensus_topic", "mean_within_cosine", "min_within_cosine", "mean_within_js",
            "separation", "mean_top25_doc_jaccard", "consensus_prevalence",
            "term_lift_vs_corpus", "stable_core_size", "anchor_co_membership",
            "qualifies"]].to_string(index=False))
print("\nStable cores and proposed reading:")
for _, r in cons.iterrows():
    print(f"\n  CT{r.consensus_topic} ({'QUALIFIES' if r.qualifies else 'fails'}, "
          f"prev {r.consensus_prevalence:.3f}, doc-Jaccard {r.mean_top25_doc_jaccard:.3f}, "
          f"lift {r.term_lift_vs_corpus:.2f})")
    print(f"    core: {r.stable_core_terms}")
    print(f"    reps: {r.representative_papers[:200]}")

json.dump({"procedure": "anchor-based Hungarian, seed-uniqueness by construction",
           "anchor_used_for_reference": 42,
           "mean_co_membership_across_anchors": round(float(np.mean(
               [s['mean_co_membership_across_anchors'] for s in stab])), 4),
           "qualification_rule": {
               "stable_core_size": ">= 4", "mean_top25_doc_jaccard": ">= 0.15",
               "term_lift_vs_corpus": ">= 2.0", "min_within_cosine": ">= 0.50"},
           "n_qualifying": int(cons.qualifies.sum())},
          open(OUT / "C12_seedpure_config.json", "w"), indent=2)
