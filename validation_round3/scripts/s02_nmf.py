"""Task 4: NMF as an independent comparator.

NMF is run on a TF-IDF matrix built over the SAME vocabulary as the LDA models
(the gensim B250 dictionary, 3,335 terms, same 987 documents, same tokenisation).
Holding the vocabulary fixed is what makes the later LDA-vs-NMF alignment
meaningful.

WHY RAW WEIGHTS ARE NOT COMPARED
================================
An LDA topic is a probability distribution over the vocabulary: rows sum to 1
and the values are P(word | topic). An NMF component is a non-negative loading
vector on TF-IDF features: it has no normalisation, no probabilistic meaning,
and its scale is arbitrary up to the H/W scaling indeterminacy of the
factorisation. Comparing the two elementwise, or comparing "top term weights",
would be comparing different quantities.

What IS comparable is DIRECTION in term space. Both sides are therefore
L2-normalised and compared by cosine over the shared vocabulary. This answers
"do these two topics point at the same words in the same relative proportions",
which is the substantive question, and nothing more. Jensen-Shannon is NOT used
across methods, because it requires both inputs to be distributions and an NMF
component is not one.

Initialisation: init='random' with six distinct random_state values. NMF's
default 'nndsvd' is deterministic from the data alone, so it would report
perfect stability by construction and tell us nothing about identifiability.
"""
import json
import sys
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_rand_score
from gensim.models.coherencemodel import CoherenceModel
from scipy.spatial.distance import jensenshannon

sys.path.insert(0, str(Path(__file__).parent))
from s00_common import (R3, ROOT, SEEDS, load_corpus, cosine_matrix, hungarian, term_lift)

OUT = R3 / "outputs"
MODELS = R3 / "models" / "nmf"
MODELS.mkdir(parents=True, exist_ok=True)
KS = [4, 6, 8]
TOPN, TOPDOCS = 20, 25

c = load_corpus("B250")
bow, dic, pids, toks = c["bow"], c["dictionary"], c["paper_ids"], c["tokens"]
vocab = [dic[i] for i in range(len(dic))]
vset = {w: i for i, w in enumerate(vocab)}
cfs = np.array([dic.cfs[i] for i in range(len(dic))], float)
cfs_norm = cfs / cfs.sum()
meta = pd.read_parquet(ROOT / "data/processed/lda_master.parquet").set_index("Paper_ID")

vec = TfidfVectorizer(vocabulary=vset, analyzer=lambda d: d, lowercase=False,
                      norm="l2", sublinear_tf=True)
X = vec.fit_transform(toks)
print(f"TF-IDF matrix {X.shape}, identical vocabulary to the LDA dictionary")

fits, rows = {}, []
for k in KS:
    for s in SEEDS:
        m = NMF(n_components=k, init="random", random_state=s, max_iter=1000,
                beta_loss="frobenius", solver="cd", tol=1e-5)
        W = m.fit_transform(X)
        H = m.components_
        fits[(k, s)] = (W, H, float(m.reconstruction_err_), int(m.n_iter_))
        top = [[vocab[i] for i in np.argsort(-H[t])[:TOPN]] for t in range(k)]
        cv = CoherenceModel(topics=top, texts=toks, dictionary=dic,
                            coherence="c_v", processes=1).get_coherence()
        um = CoherenceModel(topics=top, texts=toks, dictionary=dic, corpus=bow,
                            coherence="u_mass", processes=1).get_coherence()
        Hn = H / np.clip(H.sum(1, keepdims=True), 1e-12, None)
        dist = float(np.mean([jensenshannon(Hn[i], Hn[j], base=2)
                              for i, j in itertools.combinations(range(k), 2)]))
        rows.append({"k": k, "seed": s, "reconstruction_err": round(m.reconstruction_err_, 5),
                     "n_iter": int(m.n_iter_), "converged": bool(m.n_iter_ < 1000),
                     "c_v": round(cv, 5), "u_mass": round(um, 5),
                     "distinctiveness": round(dist, 5),
                     "mean_term_lift": round(float(np.mean(
                         [term_lift(H[t], vocab, cfs_norm, TOPN) for t in range(k)])), 2)})
        print(f"  NMF k={k} seed={s}  iters {m.n_iter_:>4}  C_v {cv:.4f}  dist {dist:.4f}")

per = pd.DataFrame(rows)
per.to_csv(OUT / "N1_nmf_per_model.csv", index=False)
np.save(MODELS / "tfidf_vocab_order.npy", np.array(vocab))
for (k, s), (W, H, err, it) in fits.items():
    np.save(MODELS / f"nmf_k{k}_seed{s}_H.npy", H)
    np.save(MODELS / f"nmf_k{k}_seed{s}_W.npy", W)

summ = []
for k in KS:
    cos_m, aris, agrees, nrepro, jacs = [], [], [], [], []
    for s1, s2 in itertools.combinations(SEEDS, 2):
        W1, H1, _, _ = fits[(k, s1)]
        W2, H2, _, _ = fits[(k, s2)]
        Sm = cosine_matrix(H1, H2)
        cols, sims = hungarian(Sm)
        cos_m.append(float(sims.mean()))
        nrepro.append(int((sims >= 0.70).sum()))
        a, b = W1.argmax(1), W2.argmax(1)
        back = {int(cc): i for i, cc in enumerate(cols)}
        agrees.append(float((a == np.array([back[int(x)] for x in b])).mean()))
        aris.append(float(adjusted_rand_score(a, b)))
        for t in range(k):
            d1 = set(np.argsort(-W1[:, t])[:TOPDOCS])
            d2 = set(np.argsort(-W2[:, int(cols[t])])[:TOPDOCS])
            jacs.append(len(d1 & d2) / len(d1 | d2))
    sub = per[per.k == k]
    summ.append({"k": k, "n_models": len(SEEDS), "n_pairs": len(cos_m),
                 "mean_matched_cosine": round(float(np.mean(cos_m)), 4),
                 "sd_matched_cosine": round(float(np.std(cos_m, ddof=1)), 4),
                 "mean_reproducible_topics": round(float(np.mean(nrepro)), 2),
                 "pct_reproducible": round(100 * float(np.mean(nrepro)) / k, 1),
                 "mean_dominant_agreement": round(float(np.mean(agrees)), 4),
                 "mean_ari": round(float(np.mean(aris)), 4),
                 "mean_top25_doc_jaccard": round(float(np.mean(jacs)), 4),
                 "mean_c_v": round(float(sub.c_v.mean()), 4),
                 "mean_u_mass": round(float(sub.u_mass.mean()), 4),
                 "mean_distinctiveness": round(float(sub.distinctiveness.mean()), 4),
                 "mean_term_lift": round(float(sub.mean_term_lift.mean()), 2)})
sm = pd.DataFrame(summ)
sm.to_csv(OUT / "N2_nmf_stability_by_k.csv", index=False)

print("\n" + "=" * 124)
print("NMF STABILITY ACROSS 6 RANDOM INITIALISATIONS")
print(sm.to_string(index=False))

# ---------------------------------------------------- NMF consensus at K=6
K6 = 6
ref_seed = 42
Wr, Hr, _, _ = fits[(K6, ref_seed)]
groups = {t: [(ref_seed, t)] for t in range(K6)}
for s in SEEDS:
    if s == ref_seed:
        continue
    W2, H2, _, _ = fits[(K6, s)]
    cols, _ = hungarian(cosine_matrix(Hr, H2))
    for t in range(K6):
        groups[t].append((s, int(cols[t])))

nrows, nmem = [], []
for g, mem in groups.items():
    Hs = np.vstack([fits[(K6, s)][1][t] for s, t in mem])
    Hn = Hs / np.clip(np.linalg.norm(Hs, axis=1, keepdims=True), 1e-12, None)
    cons = Hn.mean(0)
    terms = [vocab[i] for i in np.argsort(-cons)[:TOPN]]
    from collections import Counter
    cnt = Counter()
    for i in range(len(mem)):
        cnt.update([vocab[j] for j in np.argsort(-Hs[i])[:TOPN]])
    core = [t for t in terms if cnt[t] >= 4]
    within = [cosine_matrix(Hs[i:i + 1], Hs[j:j + 1])[0, 0]
              for i, j in itertools.combinations(range(len(mem)), 2)]
    Ws = [fits[(K6, s)][0][:, t] for s, t in mem]
    td = [set(np.argsort(-w)[:TOPDOCS]) for w in Ws]
    jac = [len(a & b) / len(a | b) for a, b in itertools.combinations(td, 2)]
    cons_w = np.mean([w / max(w.sum(), 1e-12) for w in Ws], 0)
    reps = []
    for i in np.argsort(-cons_w)[:3]:
        r = meta.loc[pids[i]]
        reps.append(f"{str(r['Title_Clean'])[:66]} ({int(r['Year'])})")
    nrows.append({"nmf_topic": g + 1, "n_supporting_inits": len(mem),
                  "mean_within_cosine": round(float(np.mean(within)), 4),
                  "min_within_cosine": round(float(np.min(within)), 4),
                  "mean_top25_doc_jaccard": round(float(np.mean(jac)), 4),
                  "prevalence_share": round(float(cons_w.sum() / len(cons_w) * len(cons_w)) /
                                            len(cons_w), 4),
                  "term_lift_vs_corpus": round(term_lift(cons, vocab, cfs_norm, TOPN), 2),
                  "stable_core_size": len(core),
                  "stable_core_terms": ", ".join(core[:14]),
                  "top20_terms": ", ".join(terms),
                  "representative_papers": " | ".join(reps)})
    for s, t in mem:
        nmem.append({"nmf_topic": g + 1, "seed": s, "component": t + 1})
nc = pd.DataFrame(nrows)
nc["qualifies"] = ((nc.stable_core_size >= 4) & (nc.mean_top25_doc_jaccard >= 0.15) &
                   (nc.term_lift_vs_corpus >= 2.0) & (nc.min_within_cosine >= 0.50))
nc.to_csv(OUT / "N3_nmf_consensus_k6.csv", index=False)
pd.DataFrame(nmem).to_csv(OUT / "N4_nmf_consensus_membership.csv", index=False)

print("\n" + "=" * 124)
print("NMF CONSENSUS AT K=6 (anchor-matched across 6 initialisations)")
print(nc[["nmf_topic", "mean_within_cosine", "min_within_cosine", "mean_top25_doc_jaccard",
          "term_lift_vs_corpus", "stable_core_size", "qualifies"]].to_string(index=False))
for _, r in nc.iterrows():
    print(f"  N{r.nmf_topic} ({'QUALIFIES' if r.qualifies else 'fails'}): {r.stable_core_terms}")

json.dump({"vectoriser": "TfidfVectorizer(norm=l2, sublinear_tf=True) on the gensim B250 "
                         "dictionary vocabulary, identical tokens to LDA",
           "nmf": {"init": "random", "solver": "cd", "beta_loss": "frobenius",
                   "max_iter": 1000, "tol": 1e-5, "random_states": SEEDS},
           "why_random_init": "nndsvd is deterministic from the data, so it would report "
                              "perfect stability by construction",
           "comparison_rule": "L2-normalised cosine over the shared vocabulary; raw LDA "
                              "probabilities and NMF loadings are never compared directly"},
          open(OUT / "N5_nmf_config.json", "w"), indent=2)
