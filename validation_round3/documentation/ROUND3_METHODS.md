# Validation round 3: methods

Everything here was produced by `validation_round3/scripts/`. Nothing in the
baseline project, Round 1 or Round 2 was created, modified or deleted. Verified
three ways: SHA-256 of all 72 baseline files against
`documentation/FILE_MANIFEST.md`, and mtime checks that no file under
`validation_round1/` or `validation_round2/` was touched after Round 3 began.

Software unchanged: Python 3.13.18, gensim 4.4.0, scikit-learn 1.9.1,
numpy 2.5.3, scipy 1.18.1, statsmodels 0.15.0, pandas 3.0.5, matplotlib 3.11.2.

---

## Task 1: the 36 topics

Corpus B250, 987 documents, 3,335-term gensim dictionary. Six seeds: 42, 7,
2024, 123, 777, 31337. Configuration C3: K=6, passes=100, iterations=1000,
alpha=symmetric, eta=1/6, chunksize=200.

**Model reuse.** Round 2's K-sweep already fitted exactly this configuration at
K=6. Those models are reused read-only after verifying that a fresh refit is
**bit-identical** (`expElogbeta` equal, max topic-word deviation 0.0). The check
is re-asserted in QC section 2. Refitting would have produced the same objects
and consumed compute for nothing.

---

## Task 2: consensus construction

### Similarity measure and its limitations

Primary measure: **cosine between topic-word probability vectors** over the
shared dictionary. All six models use one corpus and one dictionary, so no
vocabulary projection is needed and the measure is exact.

Limitations, stated because they bound what the clustering can mean:

- Cosine treats a probability distribution as a geometric direction. It is
  dominated by the few high-probability terms and nearly blind to the tail.
- It is not a metric on the simplex. Two topics can score high while assigning
  materially different probability to the same words.
- It has **no null calibration**. A cosine of 0.35 means nothing until compared
  against randomly paired topics, which here average **0.345**. Every threshold
  below is therefore also expressed relative to that chance level.

**Jensen-Shannon similarity** (1 minus JS distance, base 2) is computed
alongside as a distribution-aware cross-check and reported for every consensus
cluster (`C2`, and the `mean_within_js` column of `C10`).

### Three clustering procedures

Run and compared, because any single procedure could manufacture its own answer.

**M1 anchor-based Hungarian** (`s01b`, the primary). One seed is the anchor;
every other seed is matched one-to-one to it. This guarantees exactly six groups
of exactly six topics, one per seed, **by construction**, so the seed-uniqueness
requirement is satisfied structurally rather than by filtering. Run for all six
possible anchors and checked for agreement, so the result is not an artifact of
anchor choice. Mean co-membership across the six anchor solutions: **0.476**,
reported because it is low and that matters.

**M2 average-linkage hierarchical** on (1 - cosine), cut at a swept threshold.
Makes no seed-uniqueness assumption, so seed purity becomes a diagnostic rather
than a constraint.

**M3 connected components** of the graph with edges at cosine >= t, swept.

### Seed uniqueness: diagnosed, not assumed

A group containing two topics from one seed is **not discarded automatically**.
It is a signal: either that seed split a theme other seeds kept whole, or two of
its topics are mutually closer than either is to any other seed's.

The test applied: compare the two same-seed topics' mutual cosine against the
cluster's mean within-cluster cosine. If the pair is at least as similar to each
other as the cluster is internally, that seed split one theme and merging is
justified. Otherwise the cluster conflates two things.

Both impure clusters failed this test decisively (same-seed cosines 0.311 and
0.269 against cluster means of 0.734 and 0.797), so both were judged
conflations. Recorded in `C8` with verdicts.

### Threshold selection, and why it is not threshold-shopping

The multi-seed cluster count is **constant at 6 across the whole range 0.500 to
0.750**. The headline threshold was fixed at **0.55**, inside that plateau and
deliberately at its conservative end.

The cluster-count maximiser is **0.800**, which yields 8 multi-seed clusters and
3 qualifying themes. It was **not** chosen. `C13` re-runs the full qualification
test at every threshold on the plateau and at the maximiser, and the result is
reported in full rather than summarised: 0.50-0.60 gives 1 qualifying theme,
0.625-0.75 gives 2, and 0.80 gives 3. The headline choice therefore produces the
**fewest** qualifying themes available, which QC section 6 asserts.

Note also that the primary consensus (M1) has **no threshold at all**, so the
threshold affects only the secondary hierarchical view.

### Qualification criteria

Fixed before inspecting the seed-pure results, and applied uniformly:

| Criterion | Threshold | Purpose |
|---|---|---|
| supporting seeds | >= 4 of 6 | multiple independent runs |
| seed purity | required | no double-counting one run |
| stable term core | >= 4 terms in >= 4 members' top 20 | stable lexical core |
| representative-paper overlap | mean top-25 Jaccard >= 0.15 | the same papers, not just the same words |
| term lift over corpus frequency | >= 2.0 | not generic corpus vocabulary |
| minimum within-cluster cosine | >= 0.50 | no weak member propping up a mean |

---

## Task 3: validation battery

For every consensus topic: consensus top 20 terms, stable term core,
representative papers, pairwise top-25 document Jaccard across seeds,
prevalence, supporting-seed count, within- and between-cluster cosine and their
separation, term lift, and anchor co-membership. In `C10`.

Comparison with the two Round 2 themes is in `X3`, and is deliberately
constructed so that the evidence can disagree: themes are located by key-term
matching, then judged on the same criteria as everything else.

---

## Task 4: NMF

TF-IDF (`sublinear_tf=True`, L2 norm) over the **identical vocabulary and
tokens** as the LDA models. Holding the vocabulary fixed is what makes the later
alignment meaningful. K in {4, 6, 8}, six initialisations each.

**Initialisation is `init='random'` with six distinct `random_state` values.**
The default `nndsvd` is deterministic from the data alone and would report
perfect stability by construction, which would tell us nothing about
identifiability. This matters because NMF turns out to be highly stable, and
that finding would be worthless if the initialisation were fixed. Verified
directly that the six initialisations genuinely differ (max |H0| difference
0.078 to 0.096) while converging to the same solution.

### Why raw LDA and NMF weights are never compared

An LDA topic is a probability distribution: rows sum to 1, values are
P(word | topic). An NMF component is a non-negative loading vector on TF-IDF
features with no normalisation, no probabilistic meaning, and arbitrary scale up
to the W/H scaling indeterminacy. Comparing them elementwise would compare
different quantities.

What is comparable is **direction in term space**. Both sides are L2-normalised
and compared by cosine over the shared vocabulary. This answers "do these two
topics point at the same words in the same relative proportions" and nothing
more. Jensen-Shannon is **not** used across methods, because it requires both
inputs to be distributions and an NMF component is not one.

### Convergent validity

Claimed only when all three hold: alignment cosine >= 0.50, term-core Jaccard
>= 0.30, and representative-paper Jaccard >= 0.20. The third is the strictest
and most informative, because two topics can share vocabulary while describing
different literatures.

---

## Task 5: sensitivity (kept separate)

| Check | Construction |
|---|---|
| S1 threshold | swept 0.30 to 0.90 (`C4`), qualification re-tested across the plateau (`C13`) |
| S2 supporting seeds | requirement swept 3 to 6 of 6 (`S2`) |
| S3 influential journals | drop the top 1, 3 and 5 journals by paper count, refit LDA K=6 x 6 seeds, re-derive consensus (`S3`) |
| S4 corpus definition | repeat the whole consensus on BFull, uncapped conclusions (`S4`) |

Design note for S3: documents are dropped while the **dictionary is held
fixed**. Refitting `filter_extremes` on the reduced corpus would change the
vocabulary too, confounding "do these journals carry the theme" with "does the
vocabulary change". Holding the vocabulary fixed isolates the question asked.

---

## Task 6: corrected trend inference

### What was wrong in Round 2

Round 2 computed, per topic, the proportion of journal-cluster bootstrap
resamples falling on the opposite side of zero from the point estimate, doubled
it, and called that a p-value.

**That is not a p-value.** The bootstrap distribution is centred on the
*observed* slope, not on the null. Its tail mass describes the estimate's
precision, not the probability of data this extreme under H0: slope = 0. Using
it as a p-value conflates a confidence statement with a test, and is
anti-conservative in exactly this regime of few effective clusters and a
heavy-tailed cluster-size distribution.

The Round 2 **confidence intervals are unaffected** by this error and are
retained as descriptive uncertainty. Only the p-values are withdrawn. They are
carried in the output as `r2_B2_tailprop_WITHDRAWN` so the audit trail is
complete, and BH is **not** applied to them.

### What is done instead

Inference moves to the document level, where the clustering actually lives, and
the null is imposed explicitly. Model:
`share_ij = a + b * (year_ij - mean_year) + e_ij` over the 956 in-window papers,
132 journal clusters.

- **V1 CR1 cluster-robust standard errors** by journal, giving a descriptive
  95% CI and a t-statistic. Reported but never used alone, because CR1 is known
  to be anti-conservative with unbalanced clusters.
- **V2 wild cluster bootstrap-t**, Rademacher weights, null imposed (Cameron,
  Gelbach and Miller 2008), 9,999 replications. The restricted model with b = 0
  is fitted, residuals are resampled at the journal level with +/-1 weights, a
  null distribution of the t-statistic is built, and the observed t is compared
  against it. **This is the primary p-value.**
- **V3 within-journal year permutation**, 9,999 replications. Year labels are
  permuted among papers inside each journal, destroying any year-share
  association while preserving journal composition and each journal's year
  profile. Distribution-free cross-check.

**Positive control.** The wild cluster bootstrap was validated before its null
results were accepted: pure noise gives p = 0.728, a planted 0.02/yr trend gives
p = 0.0005, a planted 0.05/yr trend gives p = 0.0005, and a purely
between-journal planted signal gives p = 0.0005. The procedure has power; its
null findings are real.

BH correction is applied **only** to V2 and V3.

### Limitations of journal clustering

Journal is a proxy for the real dependence structure, not the structure itself.
Authors publish across journals, so author-level correlation leaks across
cluster boundaries and is not absorbed. Research groups produce correlated
series of papers on one theme in several outlets. Special issues create
within-journal, within-year clumping that journal clustering treats as ordinary.
The `Authors` field exists in the data but was never parsed into disambiguated
author identities, so **author-level clustering could not be implemented** and is
named as the main remaining gap rather than silently omitted.

---

## Quality control

38 assertions, all passing, in `outputs/QC_round3_checklist.csv`. Covers
immutability of all earlier rounds by hash and mtime; legitimacy of the model
reuse; distributions summing to 1; symmetry, diagonal and range of the 36x36
similarity matrix; seed uniqueness and completeness of the primary consensus;
that impure clusters were diagnosed rather than silently dropped; that the
headline threshold is **not** the cluster-count maximiser and yields the fewest
qualifying themes on the plateau; NMF's random initialisation and convergence;
that the cross-method comparison uses L2-normalised cosine on a shared
vocabulary and that convergent validity requires words **and** papers; that the
withdrawn Round 2 tail proportions are retained but excluded from BH; that CI
exclusion of zero is reported separately from significance; and that the figure
script loads no model objects, so figures and tables cannot disagree.

---

## Reproduction

```bash
cd LDA_Project
python validation_round3/scripts/s01_consensus.py            # ~3 min
python validation_round3/scripts/s01b_consensus_seedpure.py  # ~3 min
python validation_round3/scripts/s01c_threshold_plateau.py   # ~2 min
python validation_round3/scripts/s02_nmf.py                  # ~4 min
python validation_round3/scripts/s03_sensitivity.py 99999    # ~12 min, resumable
python validation_round3/scripts/s04_trend_inference.py      # ~9 min
python validation_round3/scripts/s05_lda_vs_nmf.py           # ~2 min
python validation_round3/scripts/s06_figures.py              # ~1 min
python validation_round3/scripts/s07_quality_control.py      # ~3 min
```

`s03` takes a training time budget in seconds and skips models already on disk.
Paths are rooted at `/home/claude/LDA_Project`; change `ROOT` in
`s00_common.py` to relocate. Random seeds: model seeds as listed;
bootstrap/permutation seed 20261116; chance-pairing seeds 3 (consensus), 7
(K sweep), 11 (eta).
