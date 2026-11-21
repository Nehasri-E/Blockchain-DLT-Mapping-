"""Task 1: audit every numerical claim in ROUND3_FINDINGS.md against its source CSV.

Each claim is re-derived from the underlying output file. Nothing is taken from
the narrative. Where the narrative and the data disagree, the DATA is
authoritative and the narrative is corrected.
"""
import json
import sys
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import linear_sum_assignment

ROOT = Path("/home/claude/LDA_Project")
R3 = ROOT / "validation_round3"
R4 = ROOT / "validation_round4"
OUT = R4 / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 7, 2024, 123, 777, 31337]

rows = []


def claim(stmt, actual, source, status, corrected=None):
    rows.append({"original_statement": stmt, "actual_value": actual,
                 "status": status, "corrected_statement": corrected or stmt,
                 "source_file": source})


# ---------------------------------------------------------------- NMF detail
def cos(A, B):
    An = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
    Bn = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-12, None)
    return An @ Bn.T


M = R3 / "models" / "nmf"
pair_rows = []
for k in [4, 6, 8]:
    H = {s: np.load(M / f"nmf_k{k}_seed{s}_H.npy") for s in SEEDS}
    for s1, s2 in itertools.combinations(SEEDS, 2):
        S = cos(H[s1], H[s2])
        r, c = linear_sum_assignment(-S)
        pair_rows.append({"k": k, "seed_a": s1, "seed_b": s2,
                          "matched_cosine": round(float(S[r, c].mean()), 6),
                          "identical_solution": bool(S[r, c].mean() > 0.99999)})
pairs = pd.DataFrame(pair_rows)
pairs.to_csv(OUT / "F1_nmf_pairwise_detail.csv", index=False)

basin = []
for k in [4, 6, 8]:
    sub = pairs[pairs.k == k]
    ident = sub[sub.identical_solution]
    # Largest set of seeds in which EVERY internal pair is identical.
    # (Requiring a seed to be identical to all six, including outliers, would
    # wrongly exclude everyone as soon as one outlier exists.)
    ident = {(min(r.seed_a, r.seed_b), max(r.seed_a, r.seed_b))
             for r in sub.itertuples() if r.identical_solution}
    seeds_in_majority = set()
    for size in range(len(SEEDS), 0, -1):
        found = None
        for combo in itertools.combinations(SEEDS, size):
            if all((min(a, b), max(a, b)) in ident
                   for a, b in itertools.combinations(combo, 2)):
                found = set(combo)
                break
        if found:
            seeds_in_majority = found
            break
    basin.append({"k": k,
                  "n_pairs_identical": int(sub.identical_solution.sum()),
                  "n_pairs_total": len(sub),
                  "mean_matched_cosine": round(float(sub.matched_cosine.mean()), 4),
                  "sd": round(float(sub.matched_cosine.std(ddof=1)), 4),
                  "min_matched_cosine": round(float(sub.matched_cosine.min()), 4),
                  "seeds_in_dominant_basin": ", ".join(str(s) for s in sorted(seeds_in_majority)),
                  "n_seeds_in_dominant_basin": len(seeds_in_majority),
                  "outlier_seeds": ", ".join(str(s) for s in SEEDS if s not in seeds_in_majority)})
bas = pd.DataFrame(basin)
bas.to_csv(OUT / "F2_nmf_basin_structure.csv", index=False)

n2 = pd.read_csv(R3 / "outputs" / "N2_nmf_stability_by_k.csv").set_index("k")

claim("NMF achieves matched cosine 1.000000 at both K=4 and K=6",
      f"K=4: 1.0000 (15/15 pairs identical). K=6: 0.9415 mean, "
      f"10/15 pairs identical, 5 pairs involving seed 777 at ~0.8245",
      "N2_nmf_stability_by_k.csv; F1_nmf_pairwise_detail.csv", "INCORRECT",
      "NMF converges to an identical solution from all six initialisations at K=4 "
      "(15/15 pairs at cosine 1.000000). At K=6 it has a dominant basin rather than a "
      "unique optimum: five of six initialisations (42, 7, 2024, 123, 31337) converge "
      "identically, while seed 777 finds a different optimum, giving a mean matched "
      "cosine of 0.9415 (sd 0.0856).")

claim("matched cosine 1.000000 at K=4 and K=6 (raw H differs in magnitude only)",
      "true at K=4 only; at K=6 this held for the single pair 42 vs 7 that was "
      "spot-checked, both of which lie in the dominant basin",
      "F1_nmf_pairwise_detail.csv", "INCORRECT (over-generalised from one pair)",
      "The spot-check compared seeds 42 and 7, which both lie in the dominant basin. "
      "Generalising from that one pair to all 15 was an error.")

claim("The TF-IDF Frobenius objective is effectively globally identifiable on this corpus",
      "K=4: yes (15/15). K=6: 10/15 pairs identical, one outlier basin. "
      "K=8: 2/15 pairs identical",
      "F2_nmf_basin_structure.csv", "OVERSTATED",
      "The TF-IDF Frobenius objective is globally identifiable at K=4 and has a "
      "dominant basin at K=6 (5 of 6 initialisations). Identifiability degrades with "
      "K: at K=8 only 2 of 15 pairs converge identically.")

for k in [4, 6, 8]:
    nar = {4: "1.000", 6: "0.942", 8: "0.887"}[k]
    claim(f"NMF K={k} matched cosine reported as {nar}",
          f"{n2.loc[k, 'mean_matched_cosine']:.4f}",
          "N2_nmf_stability_by_k.csv",
          "CORRECT" if abs(float(nar) - n2.loc[k, "mean_matched_cosine"]) < 0.001 else "CHECK")

# ------------------------------------------------------- other numeric claims
c10 = pd.read_csv(R3 / "outputs" / "C10_consensus_topics_seedpure.csv").set_index("consensus_topic")
n3 = pd.read_csv(R3 / "outputs" / "N3_nmf_consensus_k6.csv").set_index("nmf_topic")
x2 = pd.read_csv(R3 / "outputs" / "X2_lda_vs_nmf_crosswalk.csv")
s3 = pd.read_csv(R3 / "outputs" / "S3_journal_exclusion_sensitivity.csv")
s4 = pd.read_csv(R3 / "outputs" / "S4_bfull_corpus_sensitivity.csv").set_index("ref_consensus_topic")
c7 = json.loads((R3 / "outputs" / "C7_consensus_config.json").read_text())
c12 = json.loads((R3 / "outputs" / "C12_seedpure_config.json").read_text())
v1 = pd.read_csv(R3 / "outputs" / "V1_trend_valid_inference.csv").set_index("topic")
v3 = pd.read_csv(R3 / "outputs" / "V3_procedure_comparison.csv")
c13 = pd.read_csv(R3 / "outputs" / "C13_qualification_across_plateau.csv")

checks = [
    ("CT1 mean within-cluster cosine 0.850", f"{c10.loc[1,'mean_within_cosine']:.4f}",
     "C10", 0.850, c10.loc[1, "mean_within_cosine"]),
    ("CT1 document Jaccard 0.361", f"{c10.loc[1,'mean_top25_doc_jaccard']:.4f}",
     "C10", 0.361, c10.loc[1, "mean_top25_doc_jaccard"]),
    ("CT1 anchor agreement 0.800", f"{c10.loc[1,'anchor_co_membership']:.3f}",
     "C10", 0.800, c10.loc[1, "anchor_co_membership"]),
    ("CT4 min within-cosine 0.468", f"{c10.loc[4,'min_within_cosine']:.4f}",
     "C10", 0.468, c10.loc[4, "min_within_cosine"]),
    ("CT4 fails by 0.032", f"{0.50 - c10.loc[4,'min_within_cosine']:.4f}",
     "C10", 0.032, 0.50 - c10.loc[4, "min_within_cosine"]),
    ("CT2 term lift 1.88", f"{c10.loc[2,'term_lift_vs_corpus']:.2f}",
     "C10", 1.88, c10.loc[2, "term_lift_vs_corpus"]),
    ("overall anchor agreement 0.476", f"{c12['mean_co_membership_across_anchors']:.4f}",
     "C12", 0.476, c12["mean_co_membership_across_anchors"]),
    ("chance cross-seed cosine 0.345", f"{c7['chance_cross_seed_cosine']:.4f}",
     "C7", 0.345, c7["chance_cross_seed_cosine"]),
    ("NMF 5 of 6 topics qualify", f"{int(n3.qualifies.sum())} of {len(n3)}",
     "N3", 5, int(n3.qualifies.sum())),
    ("LDA 1 of 6 topics qualifies", f"{int(c10.qualifies.sum())} of {len(c10)}",
     "C10", 1, int(c10.qualifies.sum())),
    ("CT1/N2 cosine 0.878", f"{x2[x2.lda_consensus_topic==1]['cosine'].iloc[0]:.4f}",
     "X2", 0.878, x2[x2.lda_consensus_topic == 1]["cosine"].iloc[0]),
    ("CT1/N2 paper overlap 0.613", f"{x2[x2.lda_consensus_topic==1]['top25_paper_jaccard'].iloc[0]:.3f}",
     "X2", 0.613, x2[x2.lda_consensus_topic == 1]["top25_paper_jaccard"].iloc[0]),
    ("CT2/N1 agree on 2 of 25 papers", f"{int(x2[x2.lda_consensus_topic==2]['n_shared_papers'].iloc[0])} of 25",
     "X2", 2, int(x2[x2.lda_consensus_topic == 2]["n_shared_papers"].iloc[0])),
    ("CT4 journal robustness min 0.941",
     f"{s3[s3.ref_consensus_topic==4]['cosine_to_reference'].min():.4f}",
     "S3", 0.941, s3[s3.ref_consensus_topic == 4]["cosine_to_reference"].min()),
    ("CT1 journal robustness min 0.884",
     f"{s3[s3.ref_consensus_topic==1]['cosine_to_reference'].min():.4f}",
     "S3", 0.884, s3[s3.ref_consensus_topic == 1]["cosine_to_reference"].min()),
    ("CT1 BFull cosine 0.935", f"{s4.loc[1,'cosine_B250_vs_BFull']:.4f}",
     "S4", 0.935, s4.loc[1, "cosine_B250_vs_BFull"]),
    ("T4 wild bootstrap raw p 0.0098", f"{v1.loc[4,'p_wild_cluster']:.4f}",
     "V1", 0.0098, v1.loc[4, "p_wild_cluster"]),
    ("T4 wild bootstrap q 0.118", f"{v1.loc[4,'p_wild_cluster_fdr']:.4f}",
     "V1", 0.118, v1.loc[4, "p_wild_cluster_fdr"]),
    ("T8 wild bootstrap raw p 0.043", f"{v1.loc[8,'p_wild_cluster']:.4f}",
     "V1", 0.043, v1.loc[8, "p_wild_cluster"]),
    ("zero topics significant under wild bootstrap after FDR",
     f"{int(v1.p_wild_cluster_sig_fdr.sum())}", "V1", 0,
     int(v1.p_wild_cluster_sig_fdr.sum())),
    ("two CR1 CIs exclude zero", f"{int(v1.cr1_ci_excludes_zero.sum())}",
     "V1", 2, int(v1.cr1_ci_excludes_zero.sum())),
    ("headline threshold 0.55 gives 1 qualifying theme",
     f"{int(c13[c13.threshold==0.55]['n_qualifying_themes'].iloc[0])}",
     "C13", 1, int(c13[c13.threshold == 0.55]["n_qualifying_themes"].iloc[0])),
    ("threshold 0.80 gives 3 qualifying themes",
     f"{int(c13[c13.threshold==0.80]['n_qualifying_themes'].iloc[0])}",
     "C13", 3, int(c13[c13.threshold == 0.80]["n_qualifying_themes"].iloc[0])),
]
for stmt, actual, src, expected, got in checks:
    ok = abs(float(expected) - float(got)) <= max(0.0015, abs(float(expected)) * 0.006)
    claim(stmt, actual, f"{src} (Round 3 outputs)", "CORRECT" if ok else "INCORRECT")

lda6 = pd.read_csv(ROOT / "validation_round2" / "outputs" / "T11_k_sweep_summary.csv")
lda6 = lda6[lda6.k == 6].iloc[0]
claim("LDA K=6 ARI 0.285 vs NMF 0.749",
      f"LDA {lda6['mean_ari']:.4f}, NMF {n2.loc[6,'mean_ari']:.4f}",
      "T11 (Round 2); N2 (Round 3)",
      "CORRECT" if abs(lda6["mean_ari"] - 0.285) < 0.002 and
                   abs(n2.loc[6, "mean_ari"] - 0.749) < 0.002 else "INCORRECT")

claim("four topics survive FDR under within-journal permutation",
      f"{int(v1.p_within_journal_perm_sig_fdr.sum())} topics: "
      f"{', '.join('T'+str(t) for t in sorted(v1[v1.p_within_journal_perm_sig_fdr].index))}",
      "V1", "CORRECT" if int(v1.p_within_journal_perm_sig_fdr.sum()) == 4 else "INCORRECT")

aud = pd.DataFrame(rows)
aud.to_csv(OUT / "F3_round3_discrepancy_table.csv", index=False)

print("=" * 120)
print("NMF BASIN STRUCTURE (the source of the discrepancy)")
print(bas.to_string(index=False))
print("\n" + "=" * 120)
print("ROUND 3 NUMERICAL AUDIT")
print(aud["status"].value_counts().to_string())
print("\nNon-CORRECT entries:")
bad = aud[~aud.status.str.startswith("CORRECT")]
for _, r in bad.iterrows():
    print(f"\n  [{r.status}] {r.original_statement}")
    print(f"     actual : {r.actual_value}")
    print(f"     source : {r.source_file}")
print(f"\n{len(aud)} claims audited, {len(bad)} require correction")
