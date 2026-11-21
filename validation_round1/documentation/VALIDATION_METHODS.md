# Validation round 1: methods

Everything in `validation_round1/` was produced by the scripts in
`validation_round1/scripts/`. No file outside that directory was created,
modified or deleted. This is verified, not asserted: QC check 6 recomputes the
SHA-256 of all 72 original package files and compares them against
`documentation/FILE_MANIFEST.md`.

Software: Python 3.13.18, gensim 4.4.0, spaCy 3.8.18 with en_core_web_sm 3.8.0,
numpy 2.5.3, scipy 1.18.1, statsmodels 0.15.0, scikit-learn 1.9.1,
pandas 3.0.5, matplotlib 3.11.2. Identical to the baseline environment except
statsmodels, which the baseline did not use.

---

## 1. Corpus construction (`v01_build_corpora.py`)

Five conditions built from `data/processed/lda_master.parquet` (987 papers):

| Condition | Text | Conclusion cap |
|---|---|---|
| A | title + abstract | n/a |
| B250 | title + abstract + conclusion | 250 words (published baseline) |
| B500 | title + abstract + conclusion | 500 words |
| B1000 | title + abstract + conclusion | 1,000 words |
| BFull | title + abstract + conclusion | none |

The cap is applied to the **raw conclusion text before normalization**, exactly
as `scripts/03_preprocess.py` does. Capping lemmatized tokens instead would not
be the same operation and was not done.

### What is held fixed

spaCy POS-filtered lemmatization (NOUN, PROPN, ADJ, VERB), the `LEMMA_FIX` map,
the `GENERIC` stopword list, the `DOMAIN_KEEP` protections, the `normalize()`
regexes, and the source table. These are copied verbatim from
`scripts/03_preprocess.py`.

### What is refitted per corpus, and why

- gensim `Phrases` bigram (min_count=12, threshold=12.0)
- gensim `Phrases` trigram (min_count=8, threshold=10.0)
- gensim `Dictionary` and `filter_extremes(no_below=5, no_above=0.50)`
- the document retention filter (keep documents with 10 or more BoW entries)

These are **corpus-relative** quantities. `min_count` counts occurrences in the
corpus being fitted; `no_above=0.50` is a proportion of that corpus's documents.
Freezing B250's phraser and vocabulary and applying them to BFull would make
every condition's vocabulary a function of one arbitrary cap choice, which is
precisely the assumption under test. Refitting is the correct design, and it is
why the conditions have different vocabulary sizes (2,308 to 4,839 terms), which
in turn is why cross-vocabulary topic alignment is required.

### Baseline reproduction check

Rebuilt B250 is asserted equal to the baseline `data/processed/corpora.pkl`
Corpus B on document count, paper_ids, vocabulary size, vocabulary hash, token
lists and BoW. All six pass. Rebuilt Corpus A is likewise asserted equal to the
baseline Corpus A (`v05`, step 1). The replication of baseline preprocessing is
therefore demonstrated rather than assumed, which is what licenses reusing the
published model as a member of this grid.

---

## 2. Model grid (`v02_train_grid.py`)

5 conditions x 6 seeds = **30 models**, all at the baseline configuration:
`gensim.models.LdaModel`, K=12, alpha=symmetric, eta=auto, passes=20,
iterations=200, chunksize=200, eval_every=None.

Seeds: **42, 7, 2024, 123, 777, 31337**. The brief asked for at least three in
Task A and at least five in Task D; six are used throughout because a fit takes
about 7 seconds, so the extra seeds cost under four minutes in total and
materially tighten the stability estimates.

The 312-model baseline grid was **not** re-run. It answered a different question
(choice of K) and nothing in this round revisits that choice.

`LdaMulticore` is not used anywhere in this round. Refitting B250 seed 42 from
the rebuilt corpus reproduces the published model with `expElogbeta`
bit-identical and maximum topic-word deviation 0.000e+00.

---

## 3. Topic alignment (`valtools.py`)

Topic numbers are arbitrary and differ between runs. In this corpus the naive
diagonal of a B250 seed-42 vs seed-7 similarity matrix averages 0.18 while the
properly matched value is 0.61, so comparing topic *k* to topic *k* across
models would be meaningless. Every comparison in this round is matched first.

### Handling different vocabularies

Two projections are computed for every comparison, because neither alone is
adequate:

| Projection | Construction | Bias |
|---|---|---|
| union | pad both topic-word distributions to the union vocabulary with zeros | pessimistic: probability mass on terms absent from the other model can never be matched |
| intersection | restrict to shared terms, renormalize each topic to sum to 1 | optimistic: discards exactly the terms that make the corpora differ |

The true similarity is bracketed by the pair. In practice they agree closely
here (for example B250 vs BFull: 0.4950 union, 0.5179 intersection), so the
conclusions do not depend on the choice.

### Matching

One-to-one assignment via the Hungarian algorithm
(`scipy.optimize.linear_sum_assignment`) maximizing total similarity. Greedy
best-match is **not** used for the new analyses: it permits several source
topics to claim one target and inflates apparent agreement. Bijectivity is
asserted in code and re-verified across all 75 within-condition seed alignments
in QC check 4.

Both cosine similarity and Jensen-Shannon similarity (1 minus JS distance,
base 2) are computed. Cosine is used for headline numbers because it is the
measure the baseline used; JS is reported alongside in `D1`. JS values are
systematically lower (B250 seed pairs: 0.32 JS against 0.56 cosine) because JS
penalizes differences across the full support, but the two rank conditions
identically.

### Control

Self-alignment of a model against itself returns cosine exactly 1.000000 and the
identity permutation (QC check 5). A random-pairing control on B250 seed 42 vs
seed 7 averages 0.203 against the Hungarian 0.606, confirming the matching is
doing real work.

---

## 4. Task A: cap sensitivity (`v03`, `v07`)

Reference is B250 seed 42, the published baseline.

Cross-condition comparisons are **seed-matched**: B250(s) against X(s) for the
same s, so seed is held constant and only the cap varies. The two models still
have different random initializations because the corpora differ in size and
vocabulary, so this contrast is "cap plus residual initialization noise" and is
a conservative upper bound on the cap effect.

The controlling analysis in `v07` compares that distribution against the
**seed-noise floor**: B250(seed i) against B250(seed j) over all 15 unordered
seed pairs, where the corpus is fixed and only the seed varies. A cap effect
that does not separate from this floor is not a meaningful modelling decision.
Separation is tested with a one-sided Mann-Whitney U test and reported as both
a p-value and a difference in noise standard deviations, with Cliff's delta as
a non-parametric effect size.

Document-level agreement uses three measures on the shared document set:
raw dominant-topic agreement after applying the Hungarian matching, Cohen's
kappa on the same mapped labels, and the Adjusted Rand Index, which is
permutation invariant and therefore needs no matching at all.

---

## 5. Task B: Corpus A model (`v05`)

Fitted at the baseline configuration with seed 42 and serialized to
`validation_round1/models/lda_corpus_a_k12.model`. Confirmed bit-identical to
the grid's `lda_A_k12_seed42.model`.

The published robustness comparison is reproduced using **the baseline's own
procedure**: cosine on a union vocabulary, best match per row via
`max(axis=1)`. This is done to validate the refit against the published
numbers, not because the procedure is correct. The Hungarian one-to-one figure
is reported **alongside** it in `B5`, never substituted for it. Per the brief,
no baseline output is replaced.

---

## 6. Task C: trend re-analysis (`v06`)

Starts from `outputs/papers_with_topics.csv`. The baseline model is not
refitted. The baseline window 2019-2025 is **not changed**; the annual counts
that motivate the window are reported in `C1` so the reader can judge it.

Four estimators per topic:

| Estimator | n | Assumption it relaxes or adds |
|---|---|---|
| OLS-7 | 7 annual means | the baseline procedure, reproduced for verification |
| WLS-7 | 7 annual means, weighted by papers per year | annual means from 38 and from 217 papers are not equally precise; unweighted OLS assumes they are |
| Theil-Sen | 7 annual means | median of pairwise slopes; robust to one influential year, assumes only monotonic association. Kendall tau reported with it |
| OLS-doc | 987 paper-level observations | uses all data rather than 7 summary points, but assumes papers are independent, ignoring journal and author clustering |

A permutation test (20,000 permutations of the year labels, seed 20261113,
two-sided on absolute slope) is reported as a distribution-free alternative to
the OLS t-test, which is the right instinct at n=7.

**Multiplicity.** 12 topics are tested at once. Benjamini-Hochberg FDR control
at q=0.05 is applied within each estimator family separately. Both raw and
adjusted p-values are reported for every topic, so nothing is hidden.

**Diagnostics.** Per topic: Durbin-Watson on the residuals, Shapiro-Wilk
normality, leverage, and Cook's distance against the conventional 4/n threshold
(0.571 at n=7). Influential years are named rather than counted.

**Interpretation discipline.** Topic prevalence is the share of research text
devoted to a theme. It measures **research attention**, not real-world
blockchain adoption, deployment or spend. A rising topic means the literature
is writing more about it, which may lead, lag or contradict practice. This
distinction is maintained in the findings and no adoption claim is made
anywhere.

---

## 7. Task D: seed stability (`v04`)

All 15 unordered seed pairs per condition (75 comparisons), so the summary does
not depend on which seed is treated as the anchor. Per-topic figures are
anchored on seed 42 because that is the published baseline and the one whose
labels are in question.

Thresholds, stated in advance and applied uniformly:

- **reproducible**: minimum matched cosine across all five comparison seeds >= 0.70
- **partially reproducible**: mean >= 0.50 but minimum < 0.70
- **not reproducible**: mean < 0.50

0.70 is a conventional working threshold for "the same topic", not a derived
constant. It is stated here so the verdicts can be recomputed under a different
threshold from `D3_per_topic_seed_stability.csv`.

Label defensibility is assessed in `D4`, which prints the top 8 terms of the
topic matched to each baseline topic in every seed, so a reader can judge
whether the published label still describes the matched topic.

---

## 8. Quality control (`v09_quality_control.py`)

30 assertions, all passing, written to `outputs/QC_checklist.csv`:
distributions sum to 1 across all 30 models and the exported CSVs; paper IDs are
unique, subset the master table, and align across every comparison; all 75
alignments are bijections; the self-alignment control returns the identity; all
72 original package files are unmodified by SHA-256; the 12 baseline trend
slopes and the published robustness matrix are reproduced; and the figure script
is verified to read only result CSVs and to load no model objects, so figures
and tables cannot disagree.

---

## 9. Reproduction

```bash
cd LDA_Project
python validation_round1/scripts/v01_build_corpora.py      # ~3 min
python validation_round1/scripts/v02_train_grid.py         # ~6 min, resumable
python validation_round1/scripts/v03_cap_sensitivity.py    # ~4 min
python validation_round1/scripts/v04_seed_stability.py     # ~6 min
python validation_round1/scripts/v05_corpus_a_model.py     # ~1 min
python validation_round1/scripts/v06_trend_reanalysis.py   # ~2 min
python validation_round1/scripts/v07_cap_vs_noise.py       # ~2 min
python validation_round1/scripts/v08_figures.py            # ~30 s
python validation_round1/scripts/v09_quality_control.py    # ~3 min
```

Scripts use absolute paths rooted at `/home/claude/LDA_Project`. Change `ROOT`
at the top of each script to relocate. `v02` skips models already on disk.
