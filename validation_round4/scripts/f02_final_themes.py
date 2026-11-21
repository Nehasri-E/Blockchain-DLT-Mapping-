"""Task 2: finalise the three research themes from NMF K=6, with LDA as comparator.

Also tests whether the outlier initialisation (seed 777, which lands in a
different basin at K=6) contaminates the NMF consensus. If the five-seed
dominant basin alone gives the same themes, the consensus is not an artifact of
averaging across two different optima.
"""
import json
import sys
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon

ROOT = Path("/home/claude/LDA_Project")
R3 = ROOT / "validation_round3"
R4 = ROOT / "validation_round4"
OUT, TAB = R4 / "outputs", R4 / "FINAL_TABLES"
TAB.mkdir(parents=True, exist_ok=True)
M = R3 / "models" / "nmf"
SEEDS = [42, 7, 2024, 123, 777, 31337]
BASIN = [42, 7, 2024, 123, 31337]
K, TOPN, TOPDOCS = 6, 20, 25

sys.path.insert(0, str(R3 / "scripts"))
from s00_common import load_corpus

c = load_corpus("B250")
bow, dic, pids = c["bow"], c["dictionary"], c["paper_ids"]
vocab = [dic[i] for i in range(len(dic))]
cfs = np.array([dic.cfs[i] for i in range(len(dic))], float)
cfs_norm = cfs / cfs.sum()
meta = pd.read_parquet(ROOT / "data/processed/lda_master.parquet").set_index("Paper_ID")

nmem = pd.read_csv(R3 / "outputs" / "N4_nmf_consensus_membership.csv")
n3 = pd.read_csv(R3 / "outputs" / "N3_nmf_consensus_k6.csv").set_index("nmf_topic")
x2 = pd.read_csv(R3 / "outputs" / "X2_lda_vs_nmf_crosswalk.csv")
c10 = pd.read_csv(R3 / "outputs" / "C10_consensus_topics_seedpure.csv").set_index("consensus_topic")
s3 = pd.read_csv(R3 / "outputs" / "S3_journal_exclusion_sensitivity.csv")
s4 = pd.read_csv(R3 / "outputs" / "S4_bfull_corpus_sensitivity.csv").set_index("ref_consensus_topic")


def cosm(A, B):
    An = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
    Bn = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-12, None)
    return An @ Bn.T


def build_consensus(seed_list):
    ref = seed_list[0]
    Hr = np.load(M / f"nmf_k6_seed{ref}_H.npy")
    groups = {t: [(ref, t)] for t in range(K)}
    for s in seed_list[1:]:
        H2 = np.load(M / f"nmf_k6_seed{s}_H.npy")
        r, cc = linear_sum_assignment(-cosm(Hr, H2))
        for t in range(K):
            groups[t].append((s, int(cc[t])))
    out = {}
    for g, mem in groups.items():
        Hs = np.vstack([np.load(M / f"nmf_k6_seed{s}_H.npy")[t] for s, t in mem])
        Hn = Hs / np.clip(np.linalg.norm(Hs, axis=1, keepdims=True), 1e-12, None)
        # Raw W columns carry the topic's actual mass. Normalising each column
        # before averaging would force every topic to an identical share (1/K),
        # which is what a first version of this script did.
        Ws = [np.load(M / f"nmf_k6_seed{s}_W.npy")[:, t] for s, t in mem]
        Wraw = np.mean(Ws, 0)
        Wn = Wraw / max(Wraw.sum(), 1e-12)
        out[g] = {"H": Hn.mean(0), "Hs": Hs, "W": Wn, "Wraw": Wraw, "mem": mem,
                  "within": [cosm(Hs[i:i+1], Hs[j:j+1])[0, 0]
                             for i, j in itertools.combinations(range(len(mem)), 2)]}
    return out


full = build_consensus(SEEDS)
basin = build_consensus(BASIN)

Sb = cosm(np.vstack([full[g]["H"] for g in range(K)]),
          np.vstack([basin[g]["H"] for g in range(K)]))
r, cc = linear_sum_assignment(-Sb)
basin_rows = [{"full_consensus_topic": g + 1, "basin_only_topic": int(cc[g]) + 1,
               "cosine_full_vs_basin": round(float(Sb[g, cc[g]]), 4),
               "full_min_within": round(float(np.min(full[g]["within"])), 4),
               "basin_min_within": round(float(np.min(basin[int(cc[g])]["within"])), 4)}
              for g in range(K)]
pd.DataFrame(basin_rows).to_csv(OUT / "F4_nmf_outlier_seed_sensitivity.csv", index=False)
print("OUTLIER-SEED CHECK: consensus from all 6 vs the 5-seed dominant basin")
print(pd.DataFrame(basin_rows).to_string(index=False))

WMAT = np.vstack([full[g]["Wraw"] for g in range(K)]).T
DOM = np.bincount(WMAT.argmax(1), minlength=K)
Hall = np.vstack([full[g]["H"] for g in range(K)])
Hn = Hall / np.clip(Hall.sum(1, keepdims=True), 1e-12, None)
distinct = {g: float(np.mean([jensenshannon(Hn[g], Hn[j], base=2)
                              for j in range(K) if j != g])) for g in range(K)}

THEMES = {
    2: ("Cryptocurrency and token markets", "Strongly supported"),
    3: ("Pricing, channel strategy and consumer platforms", "Strongly supported"),
    4: ("Firm-level adoption and performance outcomes", "Supported with qualification"),
}
NON_THEMES = {
    1: ("Generic operations and management vocabulary",
        "term lift 1.61 is below the 2.0 generic-vocabulary floor; its LDA counterpart "
        "shares only 2 of 25 representative papers with it. Shared words, different "
        "literatures. Retained in the descriptive model as a background component, not "
        "presented as a substantive theme."),
    5: ("Governance, digital platforms and NFTs",
        "qualifies on NMF's own criteria, but its LDA counterpart (CT6) aligns at only "
        "cosine 0.596 with a term-core overlap of 0.059, so it has no cross-method "
        "confirmation. Reported as an NMF-only component pending independent support."),
    6: ("System architecture and security design",
        "qualifies on NMF's own criteria and aligns with LDA CT5 at cosine 0.791, but "
        "representative-paper overlap is 0.190, just below the 0.20 convergent-validity "
        "floor. The two methods agree on the vocabulary but not sufficiently on which "
        "papers belong. Reported as an NMF-only component."),
}

rows = []
for g in range(K):
    nt = g + 1
    H = full[g]["H"]
    terms = [vocab[i] for i in np.argsort(-H)[:TOPN]]
    from collections import Counter
    cnt = Counter()
    for s, t in full[g]["mem"]:
        Hs = np.load(M / f"nmf_k6_seed{s}_H.npy")[t]
        cnt.update([vocab[j] for j in np.argsort(-Hs)[:TOPN]])
    corelist = [t for t in terms if cnt[t] >= 4]

    W = full[g]["W"]
    top = np.argsort(-W)[:TOPDOCS]
    reps = []
    for i in top[:5]:
        rmeta = meta.loc[pids[i]]
        reps.append(f"{str(rmeta['Title_Clean'])[:80]} ({int(rmeta['Year'])})")

    xw = x2[x2.nmf_consensus_topic == nt]
    lda_ct = int(xw.lda_consensus_topic.iloc[0]) if len(xw) else None
    jmin = float(s3[s3.ref_consensus_topic == lda_ct]["cosine_to_reference"].min()) if lda_ct else np.nan
    bfull = float(s4.loc[lda_ct, "cosine_B250_vs_BFull"]) if lda_ct else np.nan

    is_theme = nt in THEMES
    label, strength = THEMES.get(nt, NON_THEMES.get(nt))[:2] if is_theme else \
        (NON_THEMES[nt][0], "Not presented as a validated theme")

    rows.append({
        "nmf_topic": nt, "final_label": label,
        "evidence_strength": strength,
        "presented_as_theme": is_theme,
        "prevalence_share_of_W": round(float(full[g]["Wraw"].sum() /
                                               sum(full[q]["Wraw"].sum() for q in range(K))), 4),
        "mean_doc_loading": round(float(full[g]["Wraw"].mean()), 5),
        "n_papers_dominant": int(DOM[g]),
        "mean_within_cosine": round(float(np.mean(full[g]["within"])), 4),
        "min_within_cosine": round(float(np.min(full[g]["within"])), 4),
        "distinctiveness_js": round(distinct[g], 4),
        "term_lift_vs_corpus": float(n3.loc[nt, "term_lift_vs_corpus"]),
        "stable_core_size": len(corelist),
        "doc_overlap_across_inits": float(n3.loc[nt, "mean_top25_doc_jaccard"]),
        "lda_counterpart": f"CT{lda_ct}" if lda_ct else "",
        "lda_nmf_cosine": float(xw.cosine.iloc[0]) if len(xw) else np.nan,
        "term_core_overlap": float(xw.term_core_jaccard.iloc[0]) if len(xw) else np.nan,
        "representative_paper_overlap": float(xw.top25_paper_jaccard.iloc[0]) if len(xw) else np.nan,
        "convergent_validity": bool(xw.convergent_validity.iloc[0]) if len(xw) else False,
        "journal_sensitivity_min_cosine": round(jmin, 4),
        "corpus_sensitivity_bfull_cosine": round(bfull, 4),
        "top20_terms": ", ".join(terms),
        "stable_term_core": ", ".join(corelist),
        "representative_papers": " | ".join(reps),
        "reason_if_not_a_theme": NON_THEMES[nt][1] if not is_theme else "",
    })

fin = pd.DataFrame(rows).sort_values(["presented_as_theme", "lda_nmf_cosine"],
                                     ascending=[False, False])
fin.to_csv(TAB / "Table2_final_themes.csv", index=False)
fin.to_csv(OUT / "F5_final_themes_full.csv", index=False)

print("\n" + "=" * 132)
print("FINAL THEMES (NMF K=6 primary, LDA comparator)")
print(fin[["nmf_topic", "final_label", "evidence_strength", "presented_as_theme",
           "prevalence_share_of_W", "lda_nmf_cosine", "term_core_overlap",
           "representative_paper_overlap", "convergent_validity",
           "journal_sensitivity_min_cosine"]].to_string(index=False, max_colwidth=46))
print("\nNON-THEME COMPONENTS, retained in the descriptive model:")
for _, r in fin[~fin.presented_as_theme].iterrows():
    print(f"\n  N{r.nmf_topic} {r.final_label}")
    print(f"    {r.reason_if_not_a_theme}")
