# Validation round 1: findings

**Bottom line for the research evaluator.** The 250-word conclusion cap is not
the problem. The K=12 topic solution is. Two models trained on identical data
differing only in random seed recover each other's topics at a mean matched
cosine of **0.561** and agree on the dominant topic for only **33%** of papers.
Against that noise floor, every conclusion-cap condition tested sits within
1.65 standard deviations. The cap question is therefore largely moot: the model
is not stable enough for the cap to be the thing that matters.

The temporal trend findings are in better shape. All four originally reported
trends survive Benjamini-Hochberg correction, with consistent sign and
magnitude across four estimators.

---

## 1. Is the 250-word cap defensible?

**Yes, but only because the choice barely matters, which is not the reason the
baseline gave for it.**

The baseline justified 250 words on the grounds that it balances conclusion and
abstract token share. That is a statement about input composition, not about
results. Tested empirically against the seed-noise floor:

| Contrast | Mean matched cosine | Distance from noise floor | Mann-Whitney p | Separated? |
|---|---|---|---|---|
| noise floor (B250 vs B250, seed varies) | 0.5610 (sd 0.0376) | reference | | |
| B250 vs **B500** | 0.5677 | **-0.18 SD** | 0.715 | no |
| B250 vs **BFull** | 0.5261 | +0.93 SD | 0.047 | marginal |
| B250 vs **B1000** | 0.5155 | +1.21 SD | 0.007 | yes, small |
| B250 vs **A** (no conclusions) | 0.4989 | +1.65 SD | 0.009 | yes, small |

Reading this honestly:

- Moving the cap from 250 to 500 words changes the topic solution **less than
  re-running with a different seed does**. The two distributions overlap
  completely.
- Removing conclusions entirely (Corpus A) is the largest perturbation tested,
  and it still moves the solution only 1.65 noise standard deviations.
- No cap condition reaches 2 SD from the floor.

**Recommendation.** Keep 250 words, but change the justification in the paper.
Defend it as *"topic structure is insensitive to the cap across 250 to
uncapped, so 250 was retained for length comparability"*, with this sensitivity
table as the evidence. Do not defend it as the uniquely correct choice; the
data do not support that and a reviewer testing 500 words would find no
difference.

### A caution on coherence

C_v rises monotonically with cap: A 0.353, B250 0.406, B500 0.417, B1000 0.412,
BFull 0.432. This looks like an argument for dropping the cap, and it is not.
Each C_v is computed on its own corpus with its own vocabulary and sliding
windows, so longer documents mechanically supply more co-occurrence evidence.
**These numbers are not comparable across conditions** and `V6` is labelled to
say so. They are reported because withholding them would be selective.

---

## 2. Which topics are stable and which are cap-sensitive?

From `A9_topic_stability_across_caps.csv` (minimum matched cosine across the
three alternative B conditions, seed 42):

| Verdict | Topics |
|---|---|
| **stable** (min >= 0.70) | T4 Pricing and Channel Strategy (0.77), T6 Cryptocurrency Markets and Token Pricing (0.75) |
| **moderate** (0.50-0.70) | T8 Sustainable Adoption, T2 Supply Chain Operations, T10 Governance and Ecosystems, T11 NFTs and Marketing |
| **cap-sensitive** (< 0.50) | T5 System Architecture, T7 Smart Contract Protocols, T3 FinTech and Payments, T9 Trust and Transparency, T12 Risk Management and Accounting, T1 Empirical Market and Disclosure Effects |

Six of twelve topics are cap-sensitive. T1 (0.15 mean) and T12 (0.20 mean) are
barely recoverable at all once the cap changes.

Prevalence is similarly unsettled. T11 NFTs and Marketing ranges from 4.5% in
Corpus A to 18.8% in B1000, a 14.3 point swing. T8 ranges 7.4% to 15.4%. Any
statement in the paper of the form "topic X accounts for Y% of the literature"
is conditional on the cap to a degree the baseline did not disclose.

**However**, by Task D these same topics are also the seed-unstable ones. The
cap-sensitivity ranking and the seed-instability ranking correlate strongly:
T4 is top of both stability lists, T1 and T12 bottom of both. This is the
signature of topics that are weakly identified in the data, not of topics that
respond specifically to conclusion text.

---

## 3. Is the 12-topic solution stable across random seeds?

**No. This is the most serious finding in this round.**

Six seeds, corpus fixed at B250, everything else at baseline:

| Measure | Value |
|---|---|
| Mean matched cosine, 15 seed pairs | **0.561** (sd 0.038, worst pair 0.509) |
| Topics clearing cosine 0.70 per pair | **3.3 of 12** on average |
| Dominant-topic agreement between seeds | **0.331** |
| Adjusted Rand Index between seeds | **0.192** |

Per topic, against the published seed-42 baseline:

| Verdict | Count | Topics |
|---|---|---|
| reproducible (min >= 0.70) | **1** | T4 Pricing and Channel Strategy (mean 0.90, min 0.81) |
| partially reproducible | 6 | T5, T8, T6, T2, T11, T9 |
| not reproducible (mean < 0.50) | **5** | T10 (0.47), T3 (0.41), T7 (0.41), T1 (0.31), T12 (0.25) |

T12 Risk Management, Accounting and Credit ranges from cosine 0.014 to 0.634
depending on seed: in some runs it does not exist as a recognizable topic at
all. It is also the smallest topic (2.0% prevalence, 8 papers dominant).

This pattern holds in every corpus condition, so it is not an artifact of the
cap: Corpus A is worst (0.478), B1000 best (0.572), all in the same poor band.

**What this means for the labels.** Eleven of the twelve published labels
describe a topic that a reader re-running the pipeline with a different seed
would not recover in the same form. `D4_label_defensibility_across_seeds.csv`
prints the top 8 terms of the matched topic in each seed so this can be
inspected topic by topic. Per the brief, **no label has been changed in this
round.**

---

## 4. Has the Corpus A model been saved?

**Yes.** `validation_round1/models/lda_corpus_a_k12.model`, with dictionary,
config JSON, topic-word distributions (12 x 2,308), document-topic
distributions (986 x 12) and top-30 terms per topic.

It is bit-identical to `grid/lda_A_k12_seed42.model`, and the rebuilt Corpus A
is asserted equal to the baseline Corpus A on all five structural checks.

**It reproduces the published robustness result.** Maximum deviation from
`outputs/corpus_robustness.csv` is **4.98e-04**, which is the 3-decimal rounding
in the published file. The published claim of 7 of 12 topics at cosine >= 0.50
is confirmed exactly.

**One methodological correction to flag, not applied.** The published figure
used `max(axis=1)`, which is many-to-one: **6 Corpus A topics are claimed by
more than one Corpus B topic**, so the published 7/12 double-counts. Under
proper one-to-one Hungarian matching the figure is **6 of 12** and the mean
best match falls from 0.518 to 0.452. Both are in
`B5_robustness_published_vs_refit.csv`. The published number has not been
overwritten; this requires an evaluator decision (section 7).

---

## 5. Which temporal trends survive FDR correction?

**All four originally reported trends survive**, and this is the most robust
part of the existing analysis.

| Topic | Slope/yr | 95% CI | raw p | **BH q** | Theil-Sen | Permutation p |
|---|---|---|---|---|---|---|
| T4 Pricing and Channel Strategy | +0.0187 | [0.0076, 0.0298] | 0.0076 | **0.030** | +0.0185 | 0.0097 |
| T8 Sustainable Adoption and Firm Performance | +0.0110 | [0.0084, 0.0135] | 0.0001 | **0.001** | +0.0112 | 0.0009 |
| T7 Smart Contract Protocols and Schemes | -0.0078 | [-0.0125, -0.0032] | 0.0075 | **0.030** | -0.0064 | 0.0072 |
| T9 Trust, Transparency and Sector Applications | -0.0084 | [-0.0145, -0.0024] | 0.0161 | **0.048** | -0.0072 | 0.0061 |

The other eight topics are not significant before or after correction.
T9 is marginal (q = 0.048) and would not survive a stricter threshold.

### Where the estimators disagree

| Estimator | significant raw | significant after BH |
|---|---|---|
| OLS-7 (baseline) | 4/12 | **4/12** |
| Permutation | 4/12 | 4/12 |
| Document-level OLS (n=956) | 5/12 | 5/12 (adds T5) |
| **WLS-7** (weighted by papers/year) | 4/12 | **1/12** |

All four estimators agree on **sign and magnitude** for all four trends. The
disagreement is purely about multiplicity-corrected significance: under WLS the
three weaker trends land at q = 0.061 to 0.065, just outside 0.05, leaving only
T8 significant under every estimator.

**T8 Sustainable Adoption and Firm Performance is the only trend significant
under all four estimators.** The other three are directionally consistent but
their statistical significance is estimator-dependent, and the paper should say
so rather than reporting the OLS result alone.

### Why seven annual observations is weak

Papers per year: 2019 **38**, 2020 94, 2021 102, 2022 135, 2023 179, 2024 191,
2025 217. The window holds 956 of 987 papers (96.9%).

- n=7 with 2 estimated parameters leaves 5 residual degrees of freedom. CIs are
  wide and a single year can move a slope materially.
- The annual means are not equally precise: 2019's mean rests on 38 papers,
  2025's on 217. Unweighted OLS treats them as equally informative. This is
  exactly why WLS and OLS diverge.
- **10 of 12 topics have at least one influential year** by Cook's distance
  > 4/n. 2019 is influential for 7 topics and 2020 for 2. The smallest, least
  precise year is doing a disproportionate amount of the work.
- Encouragingly, the two strongest trends (T4, T8) have **no** influential
  years, and residual normality is not rejected for any topic (Shapiro-Wilk
  p from 0.157 to 0.980). Durbin-Watson flags mild negative autocorrelation for
  T8 (3.25) and T12 (3.21).

The window was **not changed**. It remains defensible: 2017 (6 papers), 2018 (7)
and 2026 (18, partial year) are too thin to plot and their inclusion would make
the influential-observation problem worse.

### Research attention is not adoption

Topic prevalence measures the share of published research text devoted to a
theme. A rising T8 means the literature is writing more about sustainable
adoption; it is not evidence that firms are adopting blockchain more. Research
attention can lead practice, lag it, or respond to funding and editorial
fashion independent of it. This distinction must be maintained in the paper.
Nothing in this validation round licenses an adoption claim.

---

## 6. Limitations of this round

1. **Root cause not diagnosed.** This round establishes that the K=12 solution
   is seed-unstable. It does not establish why. The plausible causes (K too
   large for 987 short documents; 20 passes insufficient for convergence;
   `eta='auto'` interacting with a small corpus) are untested. Section 7 lists
   the diagnostics that would settle it.
2. **K was not revisited.** Out of scope per the brief, and the 312-model grid
   was not re-run. But if instability is driven by K being too large, the right
   fix is a smaller K, and that reopens a decision this round deliberately left
   alone.
3. **Cap contrasts are not pure.** B250(s) vs X(s) holds the seed value fixed
   but the two models still initialize differently because the corpora differ.
   The cap effect is therefore an upper bound. Since the conclusion is that the
   cap effect is *small*, this works in the conservative direction.
4. **Document-level OLS treats papers as independent.** Journal and author
   clustering is ignored, so its p-values are an upper bound on evidence. It is
   reported as one of four estimators, never as the headline.
5. **The 0.70 reproducibility threshold is conventional, not derived.** All
   per-topic similarities are in `D3` so the verdicts can be recomputed under a
   different threshold.
6. **Only one K and one alpha/eta setting were tested for stability.** Seed
   instability might be specific to `eta='auto'`.
7. **No held-out evaluation.** Perplexity remains in-sample throughout, as in
   the baseline.

---

## 7. Decisions requiring the research evaluator

1. **Is a mean matched cosine of 0.56 across seeds publishable?** My assessment
   is no, not without either a stability fix or an explicit stability section
   reporting these numbers. This is the central question of the round.
2. **Which remedy for seed instability?** Options, in increasing cost: (a)
   report the instability honestly and restrict interpretation to the topics
   that survive, which is T4 plus arguably T5, T8, T6, T2; (b) raise passes and
   iterations substantially and re-test convergence; (c) reduce K, which
   reopens model selection; (d) average over seeds via a consensus or
   ensemble topic model. I recommend running (b) as a diagnostic first because
   it is cheap and distinguishes under-convergence from genuine
   non-identifiability.
3. **The 7/12 vs 6/12 robustness figure.** The published 7/12 comes from a
   many-to-one procedure with 6 collisions. Correcting it to the one-to-one
   6/12 is more defensible but changes a published number. Baseline files are
   untouched pending this decision.
4. **How to justify the 250-word cap in the paper.** Recommended wording is in
   section 1. Needs sign-off since it replaces the balance argument.
5. **Whether to report WLS alongside OLS for the trends.** Doing so is more
   honest and weakens three of four headline trends to "directionally
   consistent, significance estimator-dependent". I recommend reporting it.
6. **Whether to retain topic labels for the 5 non-reproducible topics.** Not
   changed in this round per the brief. T1, T3, T7, T10 and T12 carry labels
   that a re-run would not reproduce.

---

## 8. Overall assessment

| Component | Status |
|---|---|
| Data pipeline and corpus construction | **Sound.** B250 and Corpus A both reproduce bit-for-bit from the baseline artifacts |
| Conclusion-cap choice | **Not a threat.** Insensitive from 250 words to uncapped; justification needs rewording |
| Corpus A robustness result | **Reproduced** to 5e-04; procedure has a correctable many-to-one flaw |
| Temporal trends | **Robust in direction and magnitude**; four survive FDR; significance of three is estimator-dependent; n=7 remains a real weakness |
| Topic solution at K=12 | **Not reproducible.** 1 of 12 topics stable across seeds; blocks publication in current form |

The existing findings are **not yet robust enough to support a research
publication** as they stand. The blocker is seed instability, not any of the
assumptions this round was commissioned to test. The data work underneath is
sound and the trend analysis is largely defensible; the topic model needs to be
either stabilized or reported with its instability made explicit.
