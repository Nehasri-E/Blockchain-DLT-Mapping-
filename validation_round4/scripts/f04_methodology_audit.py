"""Task 5: end-to-end methodological audit, with explicit post-hoc disclosure.

Every pipeline decision is recorded with:
  when_decided   before_data / before_modelling / after_seeing_results
  preregistered  whether documentation establishes it was fixed in advance
  impact         what the result would have been otherwise, where testable

A decision is marked "after_seeing_results" whenever the analyst had already
seen output bearing on it, regardless of whether it was written into a script
before that script ran. The stricter reading is used deliberately.
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/home/claude/LDA_Project")
R3, R4 = ROOT / "validation_round3", ROOT / "validation_round4"
OUT, TAB = R4 / "outputs", R4 / "FINAL_TABLES"

D = []


def dec(stage, decision, value, when, prereg, rationale, impact=""):
    D.append({"stage": stage, "decision": decision, "value": value,
              "when_decided": when, "preregistered": prereg,
              "rationale": rationale, "impact_if_changed": impact})


dec("Corpus", "Source file", "EDA_INPUT.csv; LDA_INPUT.csv discarded",
    "before_modelling", False,
    "LDA_INPUT rows offset by one after row 314; title join showed text byte-identical. "
    "Decision made from a data audit, before any model was fitted.",
    "Zipping the two files by position would have mislabelled 687 of 1,002 papers.")
dec("Corpus", "Exclusion rules",
    "drop 3 Excel spill rows, 3 correction notices, 5 duplicates, 6 abstracts under 50 words",
    "before_modelling", False,
    "Each rule derived from a data defect found during the audit, not from model output.",
    "1,003 raw to 987 analysed.")
dec("Corpus", "Duplicate handling", "by DOI then normalised title; keep longer conclusion",
    "before_modelling", False, "5 duplicate DOIs and 1 duplicate title found.", "")
dec("Corpus", "Missing-text handling",
    "abstract under 50 words excludes the paper; missing conclusion keeps it (abstract only)",
    "before_modelling", False,
    "4 of 987 papers enter with abstract only.", "")
dec("Text", "Mojibake repair", "hybrid UTF-8/CP1252 decode + ftfy, up to 3 passes",
    "before_modelling", False, "Source is a mixed byte stream; naive read crashes.",
    "77 suspect cells to 0.")
dec("Text", "Lemma repair map", "datum->data, cryptocurrencie->cryptocurrency, plus 8 more",
    "after_seeing_results", False,
    "Found by inspecting the FIRST K=12 topic solution, which showed data (153) and "
    "datum (915) competing as separate vocabulary entries.",
    "The full 312-model grid was re-run after the fix; the pre-fix grid is retained as "
    "tuning_results_SUPERSEDED_pre_lemma_fix.csv.")
dec("Text", "Stopword list", "standard English + academic boilerplate + 'blockchain'",
    "before_modelling", False,
    "'blockchain' appears in nearly every document and discriminates nothing. "
    "12 domain terms explicitly protected.", "")
dec("Text", "Conclusion cap", "250 words",
    "before_modelling", False,
    "Chosen by the analyst from a measured options table after the user declined to "
    "pick. Conclusions carried 77.3% of tokens uncapped; 250 brings this to 55%.",
    "Round 1 showed topic structure is insensitive from 250 to uncapped; the "
    "250-vs-500 contrast is indistinguishable from seed noise.")
dec("Text", "Phrase detection", "gensim Phrases bigram (12, 12.0) then trigram (8, 10.0)",
    "before_modelling", False, "Standard settings, not tuned against results.", "")
dec("Vocabulary", "filter_extremes", "no_below=5, no_above=0.50",
    "before_modelling", False, "10,873 to 3,335 terms.", "")
dec("Vocabulary", "Refit per corpus condition", "yes, for every cap condition",
    "before_modelling", True,
    "Phrase statistics and filter_extremes are corpus-relative; freezing them would "
    "make the vocabulary a function of one arbitrary cap. Stated in Round 1 methods "
    "before the cap sensitivity was run.", "")
dec("LDA", "Implementation", "LdaModel, not LdaMulticore",
    "after_seeing_results", False,
    "LdaMulticore proved non-deterministic despite random_state; two identical runs "
    "differed by up to 1 percentage point in topic prevalence.",
    "Defect fix. LdaModel is bit-identical across runs.")
dec("LDA", "eta specified explicitly as 1/K", "1/K numeric, not 'auto'",
    "after_seeing_results", False,
    "eta='auto' was found to be a no-op in gensim 4.4.0: it initialises to 1/K and "
    "update_eta never writes to state.eta, so the topics are identical to 'symmetric'.",
    "The original 312-model grid varied eta over {symmetric, auto}, so that dimension "
    "produced identical topics. The effective grid was half its nominal size.")
dec("LDA", "Training configuration C3", "100 passes, 1000 iterations",
    "after_seeing_results", False,
    "Selected from a convergence diagnostic sweep. It is the cheapest configuration "
    "with demonstrated convergence (cosine 0.9992 against a 200-pass anchor), not the "
    "one with the best stability number.",
    "Stability is flat across 20-200 passes, so this choice does not drive any result.")
dec("LDA", "K=12 in the baseline", "12",
    "after_seeing_results", False,
    "C_v was flat across K=4 to 12, so the selection criterion (distinctiveness and "
    "interpretability) was itself chosen after coherence failed to discriminate.",
    "Round 2 found K=12 least reproducible of all K tested; it is NOT carried forward.")
dec("NMF", "Adopted as the primary model", "NMF K=6 on TF-IDF",
    "after_seeing_results", False,
    "MAJOR POST-HOC MODEL SWITCH. LDA was the planned method. It was abandoned as "
    "primary only after systematic testing across 138 models established "
    "non-identifiability (ARI 0.285 at K=6).",
    "This must be stated plainly in any paper. The switch is defensible as a documented "
    "response to a measured failure, not as a search for a better-looking result.")
dec("NMF", "init='random' with 6 seeds", "random, not nndsvd",
    "before_modelling", True,
    "nndsvd is deterministic from the data and would report perfect stability by "
    "construction. Stated in the Round 3 methods before the NMF runs.", "")
dec("NMF", "TF-IDF settings", "sublinear_tf=True, L2 norm, LDA vocabulary held fixed",
    "before_modelling", True,
    "Holding the vocabulary fixed is what makes the cross-method alignment meaningful.", "")
dec("NMF", "K=6", "6",
    "after_seeing_results", False,
    "Inherited from the Round 2 LDA K sweep, where K=6 had the C_v peak and no "
    "degenerate topics. NMF's own K was not independently optimised; only 4, 6 and 8 "
    "were tested as the brief specified.",
    "NMF is more stable at K=4 (ARI 0.873) than at K=6 (0.749). K=6 was not chosen "
    "for stability.")
dec("Alignment", "Hungarian one-to-one", "scipy linear_sum_assignment",
    "before_modelling", True,
    "Greedy best-match permits many-to-one and inflates agreement. Stated in Round 1.",
    "The published Corpus A/B robustness figure used max(axis=1) and double-counted: "
    "7/12 by that procedure, 6/12 one-to-one.")
dec("Alignment", "L2-normalised cosine across methods", "cosine on shared vocabulary",
    "before_modelling", True,
    "LDA rows are probabilities, NMF loadings are not. Only direction is comparable.", "")
dec("Consensus", "Qualification criteria 1-3",
    "core>=4 terms, doc Jaccard>=0.15, term lift>=2.0",
    "before_modelling", True,
    "Written into s01 and recorded in C7_consensus_config.json before the seed-pure "
    "analysis was run.", "")
dec("Consensus", "Qualification criterion 4: min_within_cosine >= 0.50",
    "0.50",
    "after_seeing_results", False,
    "ADDED IN s01b, AFTER s01 had already produced hierarchical results. It is absent "
    "from C7 (the s01 config) and present in C12 (the s01b config). It is the ONLY "
    "criterion the pricing theme (CT4) fails, at 0.468.",
    "Under the three criteria actually pre-specified in s01, CT4 QUALIFIES. The fourth "
    "criterion is the sole reason it was reported as a near-miss.")
dec("Consensus", "Headline similarity threshold", "0.55",
    "after_seeing_results", False,
    "Chosen after seeing the threshold sweep. It sits on the 0.50-0.75 plateau at the "
    "conservative end. The cluster-count maximiser is 0.80 and was not chosen.",
    "0.55 yields the FEWEST qualifying themes available on the plateau (1); "
    "0.625-0.75 yields 2 and 0.80 yields 3.")
dec("Consensus", "Convergent-validity thresholds",
    "cosine>=0.50, term Jaccard>=0.30, paper Jaccard>=0.20",
    "after_seeing_results", False,
    "Set in s05 after the LDA and NMF consensus results were already known.",
    "System architecture (N6) fails only the paper criterion, at 0.190 against 0.20. "
    "A threshold of 0.15 would admit it as a fourth theme.")
dec("Prevalence", "NMF prevalence from raw W column mass", "raw, not column-normalised",
    "after_seeing_results", False,
    "A first version of the Round 4 script normalised each W column before summing, "
    "forcing every topic to exactly 1/6. Corrected on inspection.",
    "Corrected values range 0.155 to 0.178.")
dec("Temporal", "Window 2019-2025", "7 annual observations",
    "before_modelling", False,
    "2017 (6 papers), 2018 (7) and partial 2026 (18) excluded as too thin to plot. "
    "Window never widened to chase significance.", "")
dec("Temporal", "Primary inference", "wild cluster bootstrap-t, journal clusters, null imposed",
    "after_seeing_results", False,
    "Adopted in Round 3 after the Round 2 procedure was found invalid: a bootstrap "
    "tail proportion centred on the observed estimate is not a p-value.",
    "Defect fix. Validated with a positive control before its null results were "
    "accepted.")
dec("Temporal", "Multiple-testing correction", "Benjamini-Hochberg across 12 topics",
    "before_modelling", False,
    "Applied only to valid p-value families; the withdrawn tail proportions are "
    "excluded from correction.", "")
dec("Authors", "No author-level clustering", "not performed",
    "after_seeing_results", False,
    "Measured in Round 4: strings parse for 100% of papers, but 16.3% of author "
    "mentions are affected by surname-initial ambiguity and the dataset has no "
    "affiliation, ORCID or email field.",
    "548 repeat authors touch 718 of 956 in-window papers, so the unmodelled "
    "dependence is large.")

aud = pd.DataFrame(D)
aud.to_csv(TAB / "Table5_methodological_audit.csv", index=False)
aud.to_csv(OUT / "F9_methodology_audit.csv", index=False)

post = aud[aud.when_decided == "after_seeing_results"]
prereg = aud[aud.preregistered]

print("=" * 120)
print("METHODOLOGICAL AUDIT")
print(f"  decisions recorded              : {len(aud)}")
print(f"  made before any modelling       : {int((aud.when_decided != 'after_seeing_results').sum())}")
print(f"  made AFTER seeing results       : {len(post)}")
print(f"  documentation establishes prereg: {len(prereg)}")
print("\nPOST-HOC DECISIONS (must be disclosed in any paper):")
for _, r in post.iterrows():
    print(f"\n  [{r.stage}] {r.decision} = {r.value}")
    print(f"      {r.rationale}")
    if r.impact_if_changed:
        print(f"      IMPACT: {r.impact_if_changed}")

print("\n" + "=" * 120)
print("CRITERION-SENSITIVITY: which themes qualify under the PRE-SPECIFIED criteria only?")
c10 = pd.read_csv(R3 / "outputs" / "C10_consensus_topics_seedpure.csv")
c10["qualifies_prespecified_only"] = ((c10.stable_core_size >= 4) &
                                      (c10.mean_top25_doc_jaccard >= 0.15) &
                                      (c10.term_lift_vs_corpus >= 2.0))
cmp = c10[["consensus_topic", "stable_core_size", "mean_top25_doc_jaccard",
           "term_lift_vs_corpus", "min_within_cosine", "qualifies",
           "qualifies_prespecified_only"]].rename(
    columns={"qualifies": "qualifies_all_four_criteria"})
cmp.to_csv(OUT / "F10_criterion_sensitivity.csv", index=False)
print(cmp.to_string(index=False))
print(f"\n  under all four criteria        : {int(cmp.qualifies_all_four_criteria.sum())} qualify")
print(f"  under the three pre-specified  : {int(cmp.qualifies_prespecified_only.sum())} qualify")
