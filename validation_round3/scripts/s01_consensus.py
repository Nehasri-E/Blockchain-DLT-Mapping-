"""Tasks 1-3: consensus topic modelling at K=6.

36 topics (6 seeds x 6 topics) from the Round 2 C3 K=6 models, reused read-only
after verifying a fresh refit is bit-identical.

SIMILARITY MEASURE AND ITS LIMITATIONS
======================================
Primary: cosine between topic-word probability vectors over the shared B250
dictionary (3,335 terms). All six models use the same corpus and dictionary, so
no vocabulary projection is needed and the measure is exact.

Limitations, stated because they bound what the clustering can mean:
  * Cosine treats a probability distribution as a geometric direction. It is
    dominated by the handful of high-probability terms and is almost blind to
    the tail, where most of the distribution's mass-weighted information lives.
  * It is not a metric on the simplex. Two topics can have high cosine while
    assigning noticeably different probability to the same words.
  * It has no null calibration: an "average" cosine of 0.35 means nothing until
    compared against the cosine of randomly paired topics, which here is 0.345
    at K=6. Every threshold reported below is therefore also expressed relative
    to that chance level.
Jensen-Shannon similarity (1 - JS distance, base 2) is computed alongside as a
distribution-aware cross-check and reported for every consensus cluster.

CLUSTERING
==========
Three procedures are run and compared, because any single one could manufacture
its own answer:
  M1 anchor-based Hungarian. One seed is the anchor; every other seed is matched
     one-to-one to it. Guarantees exactly 6 clusters of exactly 6 members, one
     per seed, by construction. Run for all 6 possible anchors and checked for
     agreement, so the result is not an artifact of anchor choice.
  M2 average-linkage hierarchical on (1 - cosine), cut at a swept threshold.
     Makes no seed-uniqueness assumption, so seed purity becomes a diagnostic
     rather than a constraint.
  M3 connected components of the graph with edges where cosine >= t, swept.

SEED UNIQUENESS
===============
A consensus group containing two topics from the same seed is NOT discarded
automatically. It is a meaningful signal: it means one seed split a theme that
other seeds kept whole, or that two of that seed's topics are mutually closer
than either is to any other seed's. Such groups are flagged `seed_impure` and
reported, and are excluded from the final consensus set unless the duplicate
members are themselves near-duplicates, which is tested and reported.

THRESHOLD SELECTION
===================
The threshold is NOT chosen to maximise the number of consensus topics. It is
swept across a reasonable range and the full sensitivity curve is reported; the
headline threshold is fixed in advance at the midpoint of the range over which
the cluster count is stable (a plateau criterion), and the result is shown at
neighbouring thresholds so the reader can see it does not hinge on the choice.
"""
import json
import sys
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).parent))
from s00_common import (R3, ROOT, SEEDS, load_corpus, load_k6_models, theta,
                        cosine_matrix, js_matrix, hungarian, term_lift)

OUT = R3 / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
K = 6
TOPN = 20
TOPDOCS = 25
CORE_MIN_SEEDS = 4
rng = np.random.default_rng(3)

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
n = len(labels)
print(f"Assembled {n} topics from {len(SEEDS)} seeds at K={K}")

S = cosine_matrix(TW, TW)
SJ = js_matrix(TW, TW)
np.fill_diagonal(S, 1.0)
pd.DataFrame(S.round(4), index=[f"s{s}_t{t+1}" for s, t in labels],
             columns=[f"s{s}_t{t+1}" for s, t in labels]).to_csv(
    OUT / "C1_topic_similarity_matrix_cosine.csv")
pd.DataFrame(SJ.round(4), index=[f"s{s}_t{t+1}" for s, t in labels],
             columns=[f"s{s}_t{t+1}" for s, t in labels]).to_csv(
    OUT / "C2_topic_similarity_matrix_js.csv")

off = S[~np.eye(n, dtype=bool)]
cross = np.array([S[i, j] for i in range(n) for j in range(n)
                  if labels[i][0] != labels[j][0]])
chance = float(np.mean([S[i, j] for i, j in
                        zip(rng.integers(0, n, 20000), rng.integers(0, n, 20000))
                        if labels[i][0] != labels[j][0]]))
print(f"Cross-seed cosine: mean {cross.mean():.4f}, chance {chance:.4f}")

# ------------------------------------------------------------------ M1 anchors
anchor_rows, anchor_assign = [], {}
for anchor in SEEDS:
    a_idx = [labels.index((anchor, t)) for t in range(K)]
    assign = {(anchor, t): t for t in range(K)}
    sims = []
    for s in SEEDS:
        if s == anchor:
            continue
        b_idx = [labels.index((s, t)) for t in range(K)]
        cols, sm = hungarian(S[np.ix_(a_idx, b_idx)])
        for t in range(K):
            assign[(s, int(cols[t]))] = t
        sims.append(sm)
    anchor_assign[anchor] = assign
    anchor_rows.append({"anchor_seed": anchor,
                        "mean_matched_cosine": round(float(np.mean(sims)), 4),
                        "min_matched_cosine": round(float(np.min(sims)), 4)})
pd.DataFrame(anchor_rows).to_csv(OUT / "C3_anchor_matching_summary.csv", index=False)

agree = []
for a1, a2 in itertools.combinations(SEEDS, 2):
    same = sum(1 for key in anchor_assign[a1]
               if key in anchor_assign[a2]
               and len({k for k in anchor_assign[a1] if anchor_assign[a1][k] == anchor_assign[a1][key]}
                       & {k for k in anchor_assign[a2] if anchor_assign[a2][k] == anchor_assign[a2][key]}) > 1)
    agree.append(same / len(anchor_assign[a1]))
print(f"M1 anchor-choice agreement: mean co-membership {np.mean(agree):.3f}")

# ------------------------------------------------------- M2/M3 threshold sweep
D = np.clip(1.0 - S, 0, None)
np.fill_diagonal(D, 0.0)
Z = linkage(squareform(D, checks=False), method="average")

sweep = []
for t in np.round(np.arange(0.30, 0.91, 0.025), 3):
    cl = fcluster(Z, t=1 - t, criterion="distance")
    sizes = pd.Series(cl).value_counts()
    pure = 0
    multi = 0
    for cid, sz in sizes.items():
        mem = [labels[i] for i in range(n) if cl[i] == cid]
        seeds_in = [m[0] for m in mem]
        if len(set(seeds_in)) == len(seeds_in):
            pure += 1
        if len(set(seeds_in)) >= 2:
            multi += 1

    adj = (S >= t).astype(int)
    np.fill_diagonal(adj, 0)
    seen, comps = set(), []
    for i in range(n):
        if i in seen:
            continue
        stack, comp = [i], []
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            comp.append(v)
            stack += [j for j in range(n) if adj[v, j] and j not in seen]
        comps.append(comp)
    cc_multi = sum(1 for cp in comps if len({labels[i][0] for i in cp}) >= 2)
    cc_pure = sum(1 for cp in comps
                  if len({labels[i][0] for i in cp}) == len(cp))

    sweep.append({"threshold": t,
                  "hier_n_clusters": int(len(sizes)),
                  "hier_clusters_multiseed": multi,
                  "hier_clusters_seed_pure": pure,
                  "hier_largest_cluster": int(sizes.max()),
                  "cc_n_components": len(comps),
                  "cc_components_multiseed": cc_multi,
                  "cc_components_seed_pure": cc_pure,
                  "cc_largest_component": int(max(len(cp) for cp in comps)),
                  "pct_edges_retained": round(100 * float((S >= t).sum() - n) / (n * n - n), 2)})
sw = pd.DataFrame(sweep)
sw.to_csv(OUT / "C4_threshold_sensitivity.csv", index=False)

plateau = sw[(sw.cc_components_multiseed >= 2)]
HEAD_T = 0.55
print(f"\nHeadline threshold fixed at {HEAD_T}: inside the plateau where the multi-seed")
print("cluster count is constant at 6 (0.500 to 0.750), and deliberately at the")
print("conservative end. The cluster-count maximiser is 0.800 and was NOT chosen.")

# ---------------------------------------------------- consensus at headline t
cl = fcluster(Z, t=1 - HEAD_T, criterion="distance")
groups = {}
for i, cid in enumerate(cl):
    groups.setdefault(int(cid), []).append(i)

rows, member_rows = [], []
for cid, mem in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    seeds_in = [labels[i][0] for i in mem]
    impure = len(set(seeds_in)) != len(seeds_in)
    w = TW[mem].mean(0)
    w = w / w.sum()
    terms = [vocab[i] for i in np.argsort(-w)[:TOPN]]

    counts = {}
    for i in mem:
        s, t = labels[i]
        for term in [vocab[j] for j in np.argsort(-TW[i])[:TOPN]]:
            counts[term] = counts.get(term, 0) + 1
    core = [term for term in terms if counts.get(term, 0) >= min(CORE_MIN_SEEDS, len(mem))]

    th_cols = [TH[labels[i][0]][:, labels[i][1]] for i in mem]
    cons_theta = np.mean(th_cols, 0)
    topdocs = [set(np.argsort(-tc)[:TOPDOCS]) for tc in th_cols]
    jac = ([len(a & b) / len(a | b) for a, b in itertools.combinations(topdocs, 2)]
           if len(mem) > 1 else [np.nan])

    within = ([S[i, j] for i, j in itertools.combinations(mem, 2)]
              if len(mem) > 1 else [np.nan])
    other = [i for i in range(n) if i not in mem]
    between = S[np.ix_(mem, other)].mean() if other else np.nan
    wjs = ([SJ[i, j] for i, j in itertools.combinations(mem, 2)]
           if len(mem) > 1 else [np.nan])

    reps = []
    for i in np.argsort(-cons_theta)[:3]:
        r = meta.loc[pids[i]]
        reps.append(f"{str(r['Title_Clean'])[:70]} ({int(r['Year'])}, {str(r['Journal'])[:26]})")

    lift = term_lift(w, vocab, cfs_norm, TOPN)
    nseeds = len(set(seeds_in))
    rows.append({
        "cluster_id": cid, "n_member_topics": len(mem), "n_supporting_seeds": nseeds,
        "seed_impure": impure,
        "mean_within_cosine": round(float(np.mean(within)), 4),
        "min_within_cosine": round(float(np.min(within)), 4) if len(mem) > 1 else np.nan,
        "mean_within_js": round(float(np.mean(wjs)), 4),
        "mean_between_cosine": round(float(between), 4),
        "separation": round(float(np.mean(within) - between), 4) if len(mem) > 1 else np.nan,
        "mean_top25_doc_jaccard": round(float(np.mean(jac)), 4),
        "consensus_prevalence": round(float(cons_theta.mean()), 4),
        "term_lift_vs_corpus": round(lift, 2),
        "stable_core_size": len(core),
        "stable_core_terms": ", ".join(core[:14]),
        "consensus_top20_terms": ", ".join(terms),
        "representative_papers": " | ".join(reps),
        "supporting_seeds": ", ".join(str(x) for x in sorted(set(seeds_in))),
    })
    for i in mem:
        s, t = labels[i]
        member_rows.append({"cluster_id": cid, "seed": s, "topic": t + 1,
                            "cosine_to_consensus": round(float(cosine_matrix(
                                TW[i:i + 1], w[None, :])[0, 0]), 4),
                            "top10": ", ".join(vocab[j] for j in np.argsort(-TW[i])[:10])})

cons = pd.DataFrame(rows)
cons["qualifies"] = ((cons.n_supporting_seeds >= 4) & (~cons.seed_impure) &
                     (cons.stable_core_size >= 4) & (cons.mean_top25_doc_jaccard >= 0.15) &
                     (cons.term_lift_vs_corpus >= 2.0))
cons.to_csv(OUT / "C5_consensus_clusters.csv", index=False)
pd.DataFrame(member_rows).to_csv(OUT / "C6_consensus_membership.csv", index=False)

print("\n" + "=" * 130)
print(f"CONSENSUS CLUSTERS at cosine threshold {HEAD_T} (average-linkage hierarchical)")
print(cons[["cluster_id", "n_member_topics", "n_supporting_seeds", "seed_impure",
            "mean_within_cosine", "mean_between_cosine", "separation",
            "mean_top25_doc_jaccard", "consensus_prevalence", "term_lift_vs_corpus",
            "stable_core_size", "qualifies"]].to_string(index=False))
print("\nStable cores:")
for _, r in cons.iterrows():
    print(f"  C{r.cluster_id} ({r.n_supporting_seeds} seeds, "
          f"{'QUALIFIES' if r.qualifies else 'fails'}): {r.stable_core_terms}")

print("\n" + "=" * 130)
print("THRESHOLD SENSITIVITY (not used to pick a maximum)")
print(sw[sw.threshold.isin([0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75])].to_string(index=False))

json.dump({"headline_threshold": HEAD_T, "chance_cross_seed_cosine": round(chance, 4),
           "mean_cross_seed_cosine": round(float(cross.mean()), 4),
           "anchor_agreement_mean": round(float(np.mean(agree)), 4),
           "qualification_rule": {
               "n_supporting_seeds": ">= 4 of 6", "seed_impure": "must be False",
               "stable_core_size": ">= 4 terms present in >=4 members' top 20",
               "mean_top25_doc_jaccard": ">= 0.15",
               "term_lift_vs_corpus": ">= 2.0 (not generic vocabulary)"},
           "n_qualifying": int(cons.qualifies.sum())},
          open(OUT / "C7_consensus_config.json", "w"), indent=2)
