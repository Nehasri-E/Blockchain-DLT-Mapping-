"""Task 4 final part: do LDA and NMF recover the same substantive themes?

Alignment is between the LDA K=6 seed-pure CONSENSUS topics and the NMF K=6
consensus topics, over the shared vocabulary, by L2-normalised cosine. As stated
in s02, raw LDA probabilities and NMF loadings are never compared: only the
direction of each topic in term space.

Convergent validity is claimed for a theme only when all three hold:
  * the two methods' topics align at cosine >= 0.50 under one-to-one matching
  * the two methods' stable term cores overlap substantially
  * the two methods select overlapping representative papers
The third is the strictest and the most informative, because two topics can
share vocabulary while describing different literatures.
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from s00_common import (R3, ROOT, SEEDS, load_corpus, load_k6_models, theta,
                        cosine_matrix, hungarian)

OUT = R3 / "outputs"
MODELS = R3 / "models" / "nmf"
K, TOPDOCS = 6, 25

c = load_corpus("B250")
bow, dic, pids = c["bow"], c["dictionary"], c["paper_ids"]
vocab = [dic[i] for i in range(len(dic))]
meta = pd.read_parquet(ROOT / "data/processed/lda_master.parquet").set_index("Paper_ID")

lda_mem = pd.read_csv(OUT / "C11_consensus_membership_seedpure.csv")
lda_cons = pd.read_csv(OUT / "C10_consensus_topics_seedpure.csv").set_index("consensus_topic")
models = load_k6_models()
TH = {s: theta(models[s], bow, K) for s in SEEDS}

L = np.zeros((K, len(vocab)))
Ltheta = np.zeros((len(bow), K))
for g in range(1, K + 1):
    mem = lda_mem[lda_mem.consensus_topic == g]
    L[g - 1] = np.mean([models[r.seed].get_topics()[r.topic - 1] for r in mem.itertuples()], 0)
    Ltheta[:, g - 1] = np.mean([TH[r.seed][:, r.topic - 1] for r in mem.itertuples()], 0)

nmf_mem = pd.read_csv(OUT / "N4_nmf_consensus_membership.csv")
nmf_cons = pd.read_csv(OUT / "N3_nmf_consensus_k6.csv").set_index("nmf_topic")
N = np.zeros((K, len(vocab)))
Nw = np.zeros((len(bow), K))
for g in range(1, K + 1):
    mem = nmf_mem[nmf_mem.nmf_topic == g]
    Hs, Ws = [], []
    for r in mem.itertuples():
        H = np.load(MODELS / f"nmf_k6_seed{r.seed}_H.npy")[r.component - 1]
        W = np.load(MODELS / f"nmf_k6_seed{r.seed}_W.npy")[:, r.component - 1]
        Hs.append(H / max(np.linalg.norm(H), 1e-12))
        Ws.append(W / max(W.sum(), 1e-12))
    N[g - 1] = np.mean(Hs, 0)
    Nw[:, g - 1] = np.mean(Ws, 0)

S = cosine_matrix(L, N)
cols, sims = hungarian(S)
pd.DataFrame(S.round(4), index=[f"LDA_CT{i+1}" for i in range(K)],
             columns=[f"NMF_N{j+1}" for j in range(K)]).to_csv(
    OUT / "X1_lda_vs_nmf_similarity.csv")


def core(s):
    return set(str(s).split(", ")) if isinstance(s, str) else set()


rows = []
for g in range(K):
    n = int(cols[g]) + 1
    lc = core(lda_cons.loc[g + 1, "stable_core_terms"])
    nc = core(nmf_cons.loc[n, "stable_core_terms"])
    ld = set(np.argsort(-Ltheta[:, g])[:TOPDOCS])
    nd = set(np.argsort(-Nw[:, n - 1])[:TOPDOCS])
    jac = len(ld & nd) / len(ld | nd)
    jterm = len(lc & nc) / max(len(lc | nc), 1)
    conv = bool(sims[g] >= 0.50 and jterm >= 0.30 and jac >= 0.20)
    reps = [str(meta.loc[pids[i], "Title_Clean"])[:62] for i in sorted(ld & nd)][:3]
    rows.append({
        "lda_consensus_topic": g + 1, "nmf_consensus_topic": n,
        "cosine": round(float(sims[g]), 4),
        "term_core_jaccard": round(jterm, 3),
        "shared_core_terms": ", ".join(sorted(lc & nc)),
        "top25_paper_jaccard": round(jac, 3),
        "n_shared_papers": len(ld & nd),
        "lda_qualifies": bool(lda_cons.loc[g + 1, "qualifies"]),
        "nmf_qualifies": bool(nmf_cons.loc[n, "qualifies"]),
        "convergent_validity": conv,
        "shared_representative_papers": " | ".join(reps),
    })
x = pd.DataFrame(rows).sort_values("cosine", ascending=False)
x.to_csv(OUT / "X2_lda_vs_nmf_crosswalk.csv", index=False)

r2themes = {
    "Cryptocurrency and token markets": ["cryptocurrency", "bitcoin", "market", "token", "price", "investor", "return"],
    "Pricing, channel strategy and consumer platforms": ["platform", "consumer", "product", "manufacturer", "retailer", "cost", "strategy"],
}
rt = []
for name, keys in r2themes.items():
    ks = set(keys)
    lbest = max(range(K), key=lambda g: len(core(lda_cons.loc[g + 1, "stable_core_terms"]) & ks))
    nbest = max(range(K), key=lambda g: len(core(nmf_cons.loc[g + 1, "stable_core_terms"]) & ks))
    rt.append({
        "round2_theme": name,
        "lda_consensus_topic": lbest + 1,
        "lda_key_terms_matched": len(core(lda_cons.loc[lbest + 1, "stable_core_terms"]) & ks),
        "lda_qualifies": bool(lda_cons.loc[lbest + 1, "qualifies"]),
        "lda_doc_jaccard": float(lda_cons.loc[lbest + 1, "mean_top25_doc_jaccard"]),
        "lda_min_within_cosine": float(lda_cons.loc[lbest + 1, "min_within_cosine"]),
        "nmf_consensus_topic": nbest + 1,
        "nmf_key_terms_matched": len(core(nmf_cons.loc[nbest + 1, "stable_core_terms"]) & ks),
        "nmf_qualifies": bool(nmf_cons.loc[nbest + 1, "qualifies"]),
        "nmf_doc_jaccard": float(nmf_cons.loc[nbest + 1, "mean_top25_doc_jaccard"]),
        "cross_method_cosine": float(S[lbest, nbest]),
    })
rtd = pd.DataFrame(rt)
rtd.to_csv(OUT / "X3_round2_theme_confirmation.csv", index=False)

print("=" * 128)
print("LDA K=6 CONSENSUS vs NMF K=6 CONSENSUS (one-to-one, L2-normalised cosine)")
print(x[["lda_consensus_topic", "nmf_consensus_topic", "cosine", "term_core_jaccard",
         "top25_paper_jaccard", "n_shared_papers", "lda_qualifies", "nmf_qualifies",
         "convergent_validity"]].to_string(index=False))
print("\nShared stable cores:")
for _, r in x.iterrows():
    print(f"  CT{r.lda_consensus_topic} <-> N{r.nmf_consensus_topic} "
          f"(cos {r.cosine:.2f}, {'CONVERGENT' if r.convergent_validity else 'not convergent'}): "
          f"{r.shared_core_terms[:110]}")
print("\n" + "=" * 128)
print("DO THE TWO ROUND 2 THEMES SURVIVE IN BOTH METHODS?")
print(rtd.to_string(index=False, max_colwidth=50))

json.dump({"alignment": "L2-normalised cosine over the shared 3,335-term vocabulary",
           "convergent_validity_rule": {"cosine": ">= 0.50", "term_core_jaccard": ">= 0.30",
                                        "top25_paper_jaccard": ">= 0.20"},
           "n_convergent": int(x.convergent_validity.sum())},
          open(OUT / "X4_crosswalk_config.json", "w"), indent=2)
