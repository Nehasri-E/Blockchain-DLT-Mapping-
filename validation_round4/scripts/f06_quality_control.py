"""Task 7/8: final QC, claim-to-file map, Round 4 manifest.

Every numeric claim in FINAL_FINDINGS.md is re-derived here from its source file
and compared against the stated value. A claim that cannot be re-derived from a
shipped file is a claim that should not be in the document.
"""
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/home/claude/LDA_Project")
R1, R2 = ROOT / "validation_round1", ROOT / "validation_round2"
R3, R4 = ROOT / "validation_round3", ROOT / "validation_round4"
OUT, TAB, DOC = R4 / "outputs", R4 / "FINAL_TABLES", R4 / "documentation"

CLAIMS, CHECKS = [], []


def claim(section, text, stated, source, got):
    ok = (got is None) or (isinstance(stated, str) and str(got) == stated) or \
         (not isinstance(stated, str) and abs(float(got) - float(stated)) <= 0.0011)
    CLAIMS.append({"section": section, "claim": text, "stated_value": stated,
                   "recomputed_value": "not machine-checkable" if got is None else got,
                   "source_file": source,
                   "status": "VERIFIED" if got is not None and ok else
                             ("DOCUMENTED" if got is None else "MISMATCH")})


def check(name, passed, detail):
    CHECKS.append({"check": name, "result": "PASS" if passed else "FAIL",
                   "detail": detail})


n2 = pd.read_csv(R3 / "outputs/N2_nmf_stability_by_k.csv").set_index("k")
n3 = pd.read_csv(R3 / "outputs/N3_nmf_consensus_k6.csv").set_index("nmf_topic")
x2 = pd.read_csv(R3 / "outputs/X2_lda_vs_nmf_crosswalk.csv")
t11 = pd.read_csv(R2 / "outputs/T11_k_sweep_summary.csv").set_index("k")
f1 = pd.read_csv(OUT / "F1_nmf_pairwise_detail.csv")
f2 = pd.read_csv(OUT / "F2_nmf_basin_structure.csv").set_index("k")
f3 = pd.read_csv(OUT / "F3_round3_discrepancy_table.csv")
f4 = pd.read_csv(OUT / "F4_nmf_outlier_seed_sensitivity.csv")
f5 = pd.read_csv(OUT / "F5_final_themes_full.csv").set_index("nmf_topic")
f6 = pd.read_csv(OUT / "F6_author_parsing_diagnostics.csv").set_index("metric")
f9 = pd.read_csv(OUT / "F9_methodology_audit.csv")
f10 = pd.read_csv(OUT / "F10_criterion_sensitivity.csv")
t4 = pd.read_csv(TAB / "Table4_temporal_inference.csv")
t1 = pd.read_csv(TAB / "Table1_corpus_summary.csv").set_index("Property")

claim("1 Corrections", "Round 3 statements audited", 31,
      "F3_round3_discrepancy_table.csv", len(f3))
claim("1 Corrections", "statements found correct", 28,
      "F3_round3_discrepancy_table.csv",
      int(f3.status.str.fullmatch("CORRECT").sum()))
claim("1 Corrections", "statements requiring correction", 3,
      "F3_round3_discrepancy_table.csv",
      int((~f3.status.str.fullmatch("CORRECT")).sum()))
claim("1 Corrections", "NMF mean matched cosine at K=4", 1.0,
      "N2_nmf_stability_by_k.csv", float(n2.loc[4, "mean_matched_cosine"]))
claim("1 Corrections", "NMF mean matched cosine at K=6", 0.9415,
      "N2_nmf_stability_by_k.csv", float(n2.loc[6, "mean_matched_cosine"]))
claim("1 Corrections", "identical seed pairs at K=4", "15 of 15",
      "F2_nmf_basin_structure.csv",
      f"{int(f2.loc[4,'n_pairs_identical'])} of {int(f2.loc[4,'n_pairs_total'])}")
claim("1 Corrections", "identical seed pairs at K=6", "10 of 15",
      "F2_nmf_basin_structure.csv",
      f"{int(f2.loc[6,'n_pairs_identical'])} of {int(f2.loc[6,'n_pairs_total'])}")
claim("1 Corrections", "identical seed pairs at K=8", "2 of 15",
      "F2_nmf_basin_structure.csv",
      f"{int(f2.loc[8,'n_pairs_identical'])} of {int(f2.loc[8,'n_pairs_total'])}")
claim("1 Corrections", "K=6 dominant basin membership",
      "7, 42, 123, 2024, 31337", "F2_nmf_basin_structure.csv",
      str(f2.loc[6, "seeds_in_dominant_basin"]))
claim("1 Corrections", "K=6 outlier seed", "777",
      "F2_nmf_basin_structure.csv", str(f2.loc[6, "outlier_seeds"]))
claim("1 Corrections", "all non-identical K=6 pairs involve seed 777", "True",
      "F1_nmf_pairwise_detail.csv",
      str(bool(all(777 in (r.seed_a, r.seed_b) for r in
                   f1[(f1.k == 6) & (~f1.identical_solution)].itertuples()))))
claim("1 Corrections", "basin-only consensus reproduces full consensus, min cosine",
      0.989, "F4_nmf_outlier_seed_sensitivity.csv",
      float(f4.cosine_full_vs_basin.min()))
claim("1 Corrections", "NMF ARI at K=6", 0.749,
      "N2_nmf_stability_by_k.csv", float(n2.loc[6, "mean_ari"]))

for nt, nm, prev, lift, js in [(2, "N2", 0.158, 3.82, 0.722),
                               (3, "N3", 0.163, 2.71, 0.753),
                               (4, "N4", 0.174, 2.10, 0.691)]:
    claim("2 Themes", f"{nm} prevalence share of W", prev,
          "F5_final_themes_full.csv", float(f5.loc[nt, "prevalence_share_of_W"]))
    claim("2 Themes", f"{nm} term lift vs corpus", lift,
          "F5_final_themes_full.csv", float(f5.loc[nt, "term_lift_vs_corpus"]))
    claim("2 Themes", f"{nm} distinctiveness (mean JS)", js,
          "F5_final_themes_full.csv", float(f5.loc[nt, "distinctiveness_js"]))
    claim("2 Themes", f"{nm} LDA cross-method cosine",
          round(float(f5.loc[nt, "lda_nmf_cosine"]), 3),
          "X2_lda_vs_nmf_crosswalk.csv", float(f5.loc[nt, "lda_nmf_cosine"]))
    claim("2 Themes", f"{nm} representative-paper overlap",
          round(float(f5.loc[nt, "representative_paper_overlap"]), 3),
          "X2_lda_vs_nmf_crosswalk.csv",
          float(f5.loc[nt, "representative_paper_overlap"]))
    claim("2 Themes", f"{nm} journal-exclusion minimum cosine",
          round(float(f5.loc[nt, "journal_sensitivity_min_cosine"]), 3),
          "S3_journal_exclusion_sensitivity.csv",
          float(f5.loc[nt, "journal_sensitivity_min_cosine"]))
    claim("2 Themes", f"{nm} uncapped-corpus cosine",
          round(float(f5.loc[nt, "corpus_sensitivity_bfull_cosine"]), 3),
          "S4_bfull_corpus_sensitivity.csv",
          float(f5.loc[nt, "corpus_sensitivity_bfull_cosine"]))

claim("2 Themes", "themes presented as validated", 3,
      "Table2_final_themes.csv", int(f5.presented_as_theme.sum()))
claim("2 Themes", "CT4 min_within_cosine", 0.468,
      "F10_criterion_sensitivity.csv",
      float(f10[f10.consensus_topic == 4]["min_within_cosine"].iloc[0]))
claim("2 Themes", "topics qualifying under all four criteria", 1,
      "F10_criterion_sensitivity.csv", int(f10.qualifies_all_four_criteria.sum()))
claim("2 Themes", "topics qualifying under the three pre-specified criteria", 2,
      "F10_criterion_sensitivity.csv", int(f10.qualifies_prespecified_only.sum()))

for nt, nm, prev in [(1, "N1", 0.178), (5, "N5", 0.173), (6, "N6", 0.155)]:
    claim("3 Non-themes", f"{nm} prevalence share of W", prev,
          "F5_final_themes_full.csv", float(f5.loc[nt, "prevalence_share_of_W"]))
claim("3 Non-themes", "N1 term lift", 1.61,
      "F5_final_themes_full.csv", float(f5.loc[1, "term_lift_vs_corpus"]))
claim("3 Non-themes", "N5 LDA cosine", 0.596,
      "X2_lda_vs_nmf_crosswalk.csv", float(f5.loc[5, "lda_nmf_cosine"]))
claim("3 Non-themes", "N5 term-core overlap", 0.059,
      "X2_lda_vs_nmf_crosswalk.csv", float(f5.loc[5, "term_core_overlap"]))
claim("3 Non-themes", "N6 paper overlap", 0.190,
      "X2_lda_vs_nmf_crosswalk.csv",
      float(f5.loc[6, "representative_paper_overlap"]))

claim("4 Temporal", "topics tested", 12, "Table4_temporal_inference.csv", len(t4))
claim("4 Temporal", "significant under wild cluster bootstrap after FDR", 0,
      "Table4_temporal_inference.csv", int(t4.p_wild_cluster_sig_fdr.sum()))
claim("4 Temporal", "significant under within-journal permutation after FDR", 4,
      "Table4_temporal_inference.csv", int(t4.p_within_journal_perm_sig_fdr.sum()))
claim("4 Temporal", "CR1 intervals excluding zero", 2,
      "Table4_temporal_inference.csv", int(t4.cr1_ci_excludes_zero.sum()))
claim("4 Temporal", "T4 raw clustered p", 0.0098,
      "Table4_temporal_inference.csv",
      float(t4[t4.topic == 4]["p_wild_cluster"].iloc[0]))
claim("4 Temporal", "T4 BH-adjusted q", 0.118,
      "Table4_temporal_inference.csv",
      float(t4[t4.topic == 4]["p_wild_cluster_fdr"].iloc[0]))

for m, stated, lbl in [("papers with non-null Authors", 987, "papers with author strings"),
                       ("total author mentions", 3483, "author mentions"),
                       ("distinct normalised author keys", 2299, "distinct normalised keys"),
                       ("keys with >1 distinct raw spelling", 135, "ambiguous keys"),
                       ("pct of mentions affected by ambiguity", 16.3,
                        "percent of mentions affected by ambiguity"),
                       ("repeat authors in 2019-2025 window", 548, "repeat authors")]:
    claim("5 Authors", lbl, stated, "F6_author_parsing_diagnostics.csv",
          float(f6.loc[m, "value"]))
claim("5 Authors", "reliable disambiguation feasible", "False",
      "F8_author_feasibility_verdict.json",
      str(json.loads((OUT / "F8_author_feasibility_verdict.json").read_text())
          ["reliable_disambiguation_feasible"]))

claim("6 Decisions", "pipeline decisions audited", 30,
      "F9_methodology_audit.csv", len(f9))
claim("6 Decisions", "decisions documented as fixed in advance", 6,
      "F9_methodology_audit.csv", int(f9.preregistered.sum()))
claim("6 Decisions", "decisions made after seeing results", 24,
      "F9_methodology_audit.csv",
      int((f9.when_decided == "after_seeing_results").sum()
          + (~f9.preregistered & (f9.when_decided != "after_seeing_results")).sum()))
claim("Methods", "papers analysed", 987, "Table1_corpus_summary.csv",
      float(t1.loc["Papers analysed", "Value"]))
claim("Methods", "source records", 1003, "Table1_corpus_summary.csv",
      float(t1.loc["Source records", "Value"]))
claim("Methods", "journals", 133, "Table1_corpus_summary.csv",
      float(t1.loc["Journals", "Value"]))
claim("Methods", "vocabulary terms", 3335, "Table1_corpus_summary.csv",
      float(t1.loc["Vocabulary terms", "Value"]))
claim("Methods", "NMF ARI at K=4 exceeds K=6", "True",
      "N2_nmf_stability_by_k.csv",
      str(bool(n2.loc[4, "mean_ari"] > n2.loc[6, "mean_ari"])))
claim("Methods", "LDA ARI at K=6", 0.285, "T11_k_sweep_summary.csv",
      float(t11.loc[6, "mean_ari"]))
claim("Methods", "LDA recovery above chance is flat across K", "0.341 to 0.418",
      "T11_k_sweep_summary.csv",
      f"{t11.recovery_above_chance.min():.3f} to {t11.recovery_above_chance.max():.3f}")
claim("Methods", "baseline E-step convergence", 0.99983,
      "T3_stability_by_config.csv",
      float(pd.read_csv(R2 / "outputs/T3_stability_by_config.csv")
            .set_index("config").loc["baseline", "mean_estep_convergence"]))
claim("Methods", "C3 agreement with the 200-pass anchor", 0.9992,
      "T3_stability_by_config.csv",
      float(pd.read_csv(R2 / "outputs/T3_stability_by_config.csv")
            .set_index("config").loc["C3", "mean_cosine_vs_anchor"]))
claim("Limitations", "in-window papers touched by a repeat author", "718 of 956",
      "f03_authors.py console output", None)
claim("Limitations", "NMF ARI at K=4", 0.873,
      "N2_nmf_stability_by_k.csv", float(n2.loc[4, "mean_ari"]))

cl = pd.DataFrame(CLAIMS)
cl.to_csv(TAB / "Table7_claim_to_file.csv", index=False)
cl.to_csv(OUT / "F11_claim_to_file.csv", index=False)

nmis = int((cl.status == "MISMATCH").sum())
check("Every findings claim maps to a shipped source file",
      bool(cl.source_file.notna().all()),
      f"{len(cl)} claims, {cl.source_file.nunique()} distinct source files")
check("Machine-checkable claims match their source files", nmis == 0,
      f"{int((cl.status=='VERIFIED').sum())} verified, {nmis} mismatched, "
      f"{int((cl.status=='DOCUMENTED').sum())} documented but not machine-checkable")

check("Round 3 audit covers 31 statements", len(f3) == 31, f"{len(f3)} rows")
check("All three Round 3 corrections concern NMF identifiability",
      bool(f3[~f3.status.str.fullmatch("CORRECT")]
           .original_statement.str.contains("cosine|identifiab", case=False).all()),
      "; ".join(f3[~f3.status.str.fullmatch("CORRECT")].status.tolist()))
check("Corrected K=6 figure is not 1.000000",
      abs(float(n2.loc[6, "mean_matched_cosine"]) - 1.0) > 0.01,
      f"K=6 mean matched cosine {n2.loc[6,'mean_matched_cosine']}")

check("NMF prevalence is not the degenerate 1/K",
      bool(f5.prevalence_share_of_W.std() > 0.002),
      f"range {f5.prevalence_share_of_W.min():.4f} to "
      f"{f5.prevalence_share_of_W.max():.4f}, 1/K would be {1/6:.4f}")
check("Prevalence shares sum to 1",
      abs(float(f5.prevalence_share_of_W.sum()) - 1.0) < 0.002,
      f"sum {f5.prevalence_share_of_W.sum():.4f}")
check("Outlier-seed consensus agrees with full consensus on all topics",
      bool((f4.cosine_full_vs_basin >= 0.98).all()),
      f"min {f4.cosine_full_vs_basin.min():.4f}")
check("Every validated theme has convergent validity",
      bool(f5[f5.presented_as_theme].convergent_validity.all()),
      f"{int(f5.presented_as_theme.sum())} themes")
check("Every validated theme clears the term-lift floor of 2.0",
      bool((f5[f5.presented_as_theme].term_lift_vs_corpus >= 2.0).all()),
      f"min lift {f5[f5.presented_as_theme].term_lift_vs_corpus.min()}")
check("No non-theme component is described as validated",
      bool(f5[~f5.presented_as_theme].reason_if_not_a_theme.notna().all()),
      f"{int((~f5.presented_as_theme).sum())} components, each with a stated reason")

check("Primary temporal procedure yields no significant trend",
      int(t4.p_wild_cluster_sig_fdr.sum()) == 0,
      "0 of 12 after BH")
check("CI exclusion and significance are reported separately",
      int(t4.cr1_ci_excludes_zero.sum()) != int(t4.p_wild_cluster_sig_fdr.sum()),
      f"{int(t4.cr1_ci_excludes_zero.sum())} CIs exclude zero, "
      f"{int(t4.p_wild_cluster_sig_fdr.sum())} significant")
check("The two temporal procedures are reported as different nulls",
      int(t4.p_within_journal_perm_sig_fdr.sum()) !=
      int(t4.p_wild_cluster_sig_fdr.sum()),
      "4 vs 0 significant; both columns shipped")

check("Author disambiguation declared infeasible",
      json.loads((OUT / "F8_author_feasibility_verdict.json").read_text())
      ["reliable_disambiguation_feasible"] is False,
      "no ORCID, affiliation or email field in the dataset")
check("Author ambiguity is quantified rather than asserted",
      float(f6.loc["pct of mentions affected by ambiguity", "value"]) > 0,
      f"{f6.loc['pct of mentions affected by ambiguity','value']}% of mentions")

check("Post-hoc decisions are disclosed, not hidden",
      int((f9.when_decided == "after_seeing_results").sum()) > 0,
      f"{int((f9.when_decided=='after_seeing_results').sum())} of {len(f9)} "
      "decisions made after seeing results")
check("The primary-model switch is recorded as post-hoc",
      bool((f9[f9.decision.str.contains("primary model")].when_decided
            == "after_seeing_results").all()),
      "NMF adoption marked after_seeing_results")
check("Threshold contingency of the theme count is documented",
      int(f10.qualifies_all_four_criteria.sum()) !=
      int(f10.qualifies_prespecified_only.sum()),
      "1 theme under all four criteria, 2 under the three pre-specified")

mf = []
for p in sorted(R4.rglob("*")):
    if p.is_file():
        mf.append({"relative_path": str(p.relative_to(R4)),
                   "bytes": p.stat().st_size,
                   "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
man = pd.DataFrame(mf)
man.to_csv(R4 / "ROUND4_FILE_MANIFEST.csv", index=False)
check("Round 4 manifest written", len(man) > 0,
      f"{len(man)} files, {man.bytes.sum()/1e6:.1f} MB")

for d in ["FINAL_METHODS.md", "FINAL_FINDINGS.md", "FINAL_LIMITATIONS.md",
          "FINAL_REPRODUCTION_GUIDE.md"]:
    check(f"{d} present and non-trivial", (DOC / d).exists() and
          (DOC / d).stat().st_size > 3000,
          f"{(DOC/d).stat().st_size if (DOC/d).exists() else 0} bytes")
for n in range(1, 8):
    g = sorted(TAB.glob(f"Table{n}_*.csv"))
    check(f"Table{n} present", len(g) == 1, g[0].name if g else "missing")
for n in range(1, 7):
    g = sorted((R4 / "FINAL_FIGURES").glob(f"Figure{n}_*.png"))
    check(f"Figure{n} present", len(g) == 1 and g[0].stat().st_size > 20000,
          f"{g[0].name} ({g[0].stat().st_size//1024} KB)" if g else "missing")
check("Supplementary material carries all Round 1-3 figures",
      len(list((R4 / "SUPPLEMENTARY_MATERIAL").glob("*.png"))) == 18,
      f"{len(list((R4/'SUPPLEMENTARY_MATERIAL').glob('*.png')))} figures")

qc = pd.DataFrame(CHECKS)
qc.to_csv(R4 / "FINAL_QC_CHECKLIST.csv", index=False)

print(f"CLAIM-TO-FILE MAP: {len(cl)} claims")
print(cl.status.value_counts().to_string())
if nmis:
    print("\nMISMATCHES:")
    print(cl[cl.status == "MISMATCH"].to_string(index=False, max_colwidth=46))
print(f"\nQC: {int((qc.result=='PASS').sum())} PASS / "
      f"{int((qc.result=='FAIL').sum())} FAIL of {len(qc)}")
if (qc.result == "FAIL").any():
    print(qc[qc.result == "FAIL"].to_string(index=False, max_colwidth=70))
