# Final methods

This is the method as it should be written up: the pipeline that produces the
final results, in the order a reader needs it, with every decision marked as
pre-specified or post-hoc. It supersedes the methods sections of Rounds 1 to 3,
which are retained unmodified in their own trees.

Three things about this method are unusual and are stated here rather than
buried:

1. The primary model is **NMF, not LDA**. LDA was the planned method. It was
   demoted after it proved non-identifiable on this corpus.
2. **K=6, not K=12**. The original baseline used K=12, which Round 2 showed was
   the least reproducible K tested.
3. Several thresholds that determine the headline theme count were set **after
   the relevant results were visible**. They are listed in section 9 and in
   `FINAL_TABLES/Table5_methodological_audit.csv`.

---

## 1. Data source and linkage

Two files were supplied. `EDA_INPUT.csv` (1,003 rows, 10 columns) carries the
metadata and the text. `LDA_INPUT.csv` carries a 3-column text variant.

`LDA_INPUT.csv` was **discarded**. Its rows are offset by one relative to
`EDA_INPUT.csv` from row 314 onward. A title-based join showed the text fields
to be byte-identical where they align, so nothing is lost by dropping it, but
combining the two by row position would have mislabelled **687 of 1,002 papers**.
No stable key (DOI, Row_UID) was present in both files.

A guide PDF accompanying the data describes a different file again: 1,010 rows,
38 columns, with pre-computed quality-control flags. That file was never
supplied. All quality control reported here was computed from scratch.

## 2. Corpus construction

Cascade from 1,003 source records to 987 analysed papers, recorded row by row in
`data/interim/reconciliation.csv`:

| Step | Removed | Remaining |
|---|---|---|
| Source records | | 1,003 |
| Excel spill rows (text fragments in metadata columns) | 3 | 1,000 |
| Correction and erratum notices | 3 | 997 |
| Duplicates (DOI, then normalised title) | 5 | 992 |
| Abstracts under 50 words | 6 | **987** |

For duplicates the record with the longer conclusion text was kept. A paper with
a usable abstract but no conclusion was retained; a paper with no usable abstract
was dropped regardless of its conclusion.

**Scope**: the 987 papers span 133 journals and publication years 2017 to 2026.
The journal composition is operations, supply chain and management research. This
is **not** a sample of the general blockchain and DLT literature, and the write-up
must be scoped accordingly.

## 3. Text preparation

Applied to the concatenation of title, abstract and conclusion.

- **Encoding repair**: hybrid UTF-8 / CP1252 byte decoding followed by `ftfy`,
  up to three passes. Reduced 77 cells with mojibake signatures to 0.
- **Conclusion cap**: 250 words. Conclusions are much longer than abstracts and
  uncapped text lets a handful of long papers dominate the vocabulary. Round 1
  tested caps of 250, 500, 1000 and uncapped and found the topic structure
  insensitive across that range (effect at most 1.65 noise standard deviations),
  so the choice does not drive any result.
- **Lemmatization**: spaCy `en_core_web_sm`, retaining NOUN, PROPN, ADJ and VERB.
- **Lemma repair map** (post-hoc, see section 9): spaCy lemmatizes `data` to
  `datum`, which had split one concept into two competing vocabulary entries
  (`data` 153, `datum` 915). A small repair map corrects this and three similar
  cases. The full 312-model tuning grid was re-run after the fix; the pre-fix
  grid is preserved as `tuning_results_SUPERSEDED_pre_lemma_fix.csv`.
- **Stopwords**: standard English, plus academic boilerplate (`paper`,
  `study`, `research`, `result`, `approach`), plus corpus-defining terms
  (`blockchain`, `technology`) which appear in nearly every document and carry
  no discriminating information.
- **Phrases**: gensim `Phrases` bigram pass (min_count 12, threshold 12.0) then
  trigram pass (min_count 8, threshold 10.0), fitted on the training corpus
  only.
- **Vocabulary**: `Dictionary.filter_extremes(no_below=5, no_above=0.50)`,
  yielding **3,335 terms**. Median 191 tokens per document.

The dictionary is refit for every corpus condition rather than shared, so cap
comparisons are not confounded by a fixed vocabulary chosen under one cap.

## 4. Primary model: NMF

- Input: `TfidfVectorizer(norm="l2", sublinear_tf=True)` over the gensim B250
  vocabulary held fixed, so NMF and LDA see exactly the same 3,335 terms.
- `sklearn.decomposition.NMF(n_components=6, init="random", solver="cd",
  beta_loss="frobenius", max_iter=1000, tol=1e-5)`.
- Six initialisations: seeds 42, 7, 2024, 123, 777, 31337.

`init="random"` is deliberate. The deterministic `nndsvd` initialisation would
return the same answer every time and so would make a stability analysis
vacuous. Random starts are what allow multimodality to be detected.

**Identifiability** (`outputs/F1_nmf_pairwise_detail.csv`,
`outputs/F2_nmf_basin_structure.csv`):

| K | Pairs converging to an identical solution | Structure |
|---|---|---|
| 4 | 15 of 15 | globally identifiable |
| 6 | 10 of 15 | dominant basin of 5 seeds; seed 777 finds another optimum |
| 8 | 2 of 15 | no dominant basin |

K=6 is therefore reported as a dominant basin, not a unique optimum. A consensus
built from the five-seed basin alone reproduces the six-seed consensus at cosine
0.989 to 1.000 (`outputs/F4_nmf_outlier_seed_sensitivity.csv`), so the outlier
initialisation does not shape the themes.

## 5. Comparator model: LDA

gensim `LdaModel` (single-process), 100 passes, 1000 iterations, `alpha="auto"`,
`eta` set to the explicit numeric value 1/K, K=6, six seeds matching NMF.

Two implementation facts matter and are both defect fixes:

- **`LdaMulticore` is not reproducible even with `random_state` fixed.** Two
  runs with identical parameters differed by up to 1 percentage point in topic
  prevalence. All final models use single-process `LdaModel`, verified
  bit-identical across repeated runs.
- **`eta="auto"` was a no-op for topics.** Verified against gensim source:
  `init_dir_prior` returns 1/K for both `"auto"` and `"symmetric"`, and
  `update_eta` writes to `model.eta` rather than `model.state.eta`. The eta
  dimension of the original 312-model tuning grid therefore varied nothing. All
  final models set eta numerically.

## 6. Topic alignment and consensus

- One-to-one matching by the Hungarian algorithm
  (`scipy.optimize.linear_sum_assignment`) on an L2-normalised cosine
  similarity matrix of term distributions. One-to-one matching prevents the
  inflation that arises when several topics in one run are all allowed to claim
  the same best match in another.
- Agreement between runs is additionally reported as **Adjusted Rand Index** on
  hard document assignments. ARI is permutation-invariant and cannot be raised
  by smoothing or renormalisation, so it is the figure to trust when the cosine
  and ARI diverge.
- Consensus term distributions are the mean of matched, L2-normalised rows.
  Consensus prevalence is the **raw** W column mass, not column-normalised; the
  normalised version forces every topic to exactly 1/K and is meaningless.
- Chance baselines are computed by random topic pairing so that every similarity
  figure can be read against what random alignment would give.

## 7. Method comparison

Reported in `outputs/F5_final_themes_full.csv` and
`validation_round3/outputs/X2_lda_vs_nmf_crosswalk.csv`:

- Stability at K=6: NMF **ARI 0.749** against LDA **0.285**.
- Cross-method agreement per topic on three independent axes: cosine of term
  distributions, Jaccard overlap of stable term cores, and Jaccard overlap of
  the top 25 representative papers.

The paper-overlap axis is the one that matters and the one that fails most
often. Two methods can agree on a topic's vocabulary while disagreeing about
which papers belong to it, and vocabulary agreement alone is not convergent
validity.

## 8. Temporal inference

Window 2019 to 2025: **7 annual observations**, 956 papers. The corpus grows from
38 papers in 2019 to 217 in 2025 and that growth is unevenly distributed across
journals, which is why the choice of null hypothesis changes the answer.

- **Primary**: wild cluster bootstrap-t (Cameron, Gelbach and Miller) on the
  paper-level regression of topic share on year, clustered by journal, Rademacher
  weights, null imposed, 10,000 replications. Benjamini-Hochberg across 12
  topics.
- **Secondary**: within-journal year permutation, which holds each journal's
  composition and year profile fixed.
- Cluster-robust CR1 confidence intervals are reported alongside, in a
  **separate column**, because a CI excluding zero is not a significance result.

Results: 0 of 12 topics significant under the primary procedure; 4 of 12 under
within-journal permutation. These test different nulls. See `FINAL_FINDINGS.md`
section 4.

The bootstrap replaced a Round 2 procedure that reported a bootstrap tail
proportion as a p-value. That was invalid. The replacement was validated with a
positive control before its null results were accepted: pure noise returns
p = 0.728 and a planted slope of 0.02 per year returns p = 0.0005.

## 9. Pre-specified versus post-hoc

Of 30 audited decisions, **6 are documented as fixed in advance** and 24 are
not. A decision is marked post-hoc whenever output bearing on it had already
been seen, even if it was written into a script before that script ran. The
strict reading is used on purpose.

Fixed in advance: dictionary refit per corpus condition; NMF `init="random"`
with 6 seeds; TF-IDF settings; Hungarian one-to-one matching; L2-normalised
cosine across methods; consensus qualification criteria 1 to 3.

The post-hoc decisions that **change the headline count** are:

| Decision | Value | If changed |
|---|---|---|
| Consensus criterion 4, `min_within_cosine` | 0.50 | Added after the first consensus run. It is the only criterion CT4 fails (0.468). Under the three criteria actually pre-specified, **2 LDA themes qualify, not 1** |
| Headline similarity threshold | 0.55 | On the plateau this yields the fewest qualifying themes available, so it is conservative, but it was chosen with the plateau in view |
| Convergent-validity paper-overlap floor | 0.20 | N6 fails only this, at 0.190. A floor of 0.15 would admit a **fourth theme** |
| Primary model | NMF K=6 | The planned model was LDA K=12 |

The remaining post-hoc decisions are defect fixes (`LdaModel`, numeric eta, the
lemma map, raw W prevalence, the bootstrap) whose effect is to correct an error
rather than to select a favourable result. They are listed with that
distinction in `FINAL_TABLES/Table5_methodological_audit.csv`.

## 10. Reproducibility

Every final model is deterministic given its seed and re-runs bit-identically.
One caveat is documented: `LdaModel.get_document_topics` seeds its gamma from a
mutable `random_state`, which pickling preserves and `save()`/`load()` does not,
so document-topic distributions recovered from a `.model` file can differ in the
fourth decimal place from the pickled object. Model parameters are bit-identical
either way. Always recover theta from the pickle.

`FINAL_REPRODUCTION_GUIDE.md` gives the run order and expected outputs.
`FINAL_QC_CHECKLIST.csv` lists every automated check and its result.
`FINAL_TABLES/Table7_claim_to_file.csv` maps every claim in `FINAL_FINDINGS.md`
to the file that supports it.
