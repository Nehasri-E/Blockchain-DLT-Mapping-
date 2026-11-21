# Validation round 2: methods

Everything here was produced by `validation_round2/scripts/`. Nothing outside
that directory was created, modified or deleted. Verified two ways: SHA-256 of
all 72 original package files against `documentation/FILE_MANIFEST.md`, and an
mtime check that no file under `validation_round1/` was touched after Round 2
began. Both pass.

Software unchanged from Round 1: Python 3.13.18, gensim 4.4.0, spaCy 3.8.18 with
en_core_web_sm 3.8.0, numpy 2.5.3, scipy 1.18.1, statsmodels 0.15.0,
scikit-learn 1.9.1, pandas 3.0.5, matplotlib 3.11.2.

Corpus throughout: B250 from `validation_round1/models/validation_corpora.pkl`,
987 documents, 3,335 terms. Seeds throughout: **42, 7, 2024, 123, 777, 31337**.
138 models trained, each with a sidecar JSON recording its configuration, timing
and convergence diagnostics.

---

## Four properties, kept separate

The brief asks that these not be treated as interchangeable. They are measured
by different instruments throughout:

| Property | Question | Instrument |
|---|---|---|
| **Convergence** | does ONE run reach its own optimum? | E-step convergence rate, variational bound, invariance under longer training |
| **Reproducibility** | do DIFFERENT runs reach the SAME optimum? | Hungarian-matched cosine across seeds, ARI, dominant-topic agreement |
| **Interpretability** | is a topic a coherent, specific theme? | stable term core, term lift over corpus frequency, representative-paper overlap |
| **Statistical robustness** | do the research findings survive proper inference? | FDR correction, cluster bootstrap, exact permutation |

A model can be perfectly converged and completely irreproducible. That turns out
to be exactly this corpus's situation.

---

## Task 1: convergence diagnostics (`r01_convergence.py`)

Five configurations x 6 seeds = 30 models. The brief specified four; a fifth,
**C4 (200 passes / 1000 iterations)**, was added as a convergence anchor, because
without a longer-trained reference there is no way to test whether a
configuration has stopped moving.

Three independent lines of evidence, because "training completed" is not
evidence of convergence:

**E1 E-step convergence.** gensim logs, per chunk at DEBUG level, how many
documents reached their own variational convergence criterion before hitting the
`iterations` cap (`r00_common.ConvergenceLog` captures these). A rate below 100%
means the per-document posterior was truncated and `iterations` is too low.

**E2 Objective plateau.** The variational lower bound per word
(`log_perplexity`). If extra passes buy no bound improvement, the M-step has
stopped moving.

**E3 Solution invariance.** For a FIXED seed, matched cosine between each
configuration and the C4 anchor. This is the strongest test: E1 and E2 can both
look healthy while the optimiser sits in a seed-dependent local optimum.
Criterion stated in advance: **>= 0.99 counts as converged**.

---

## Task 2: prior sensitivity (`r02_priors.py`, `r02b`, `r02c`)

Training at C3 (100 passes / 1000 iterations), the configuration Task 1 showed
converged.

### The comparison as specified, and why it is uninformative

`r02_priors.py` ran alpha=symmetric x {eta=auto, eta=symmetric} plus
alpha=asymmetric x eta=auto, 6 seeds each. The first two produced **bit-identical
topic-word matrices**. This is a property of gensim 4.4.0, not a coding error:

- `LdaState.get_lambda()` returns `state.eta + state.sstats`, and
  `get_Elogbeta() = dirichlet_expectation(get_lambda())`. The topics are a
  function of **state.eta**.
- `init_dir_prior` returns `1.0/num_topics` for BOTH `'auto'` and `'symmetric'`,
  so state.eta starts identical under either.
- `LdaModel.update_eta` assigns to `self.eta`, the MODEL's eta. It never writes
  to `self.state.eta`. The learned prior never reaches the quantity that
  produces the topics.

Net effect: **`eta='auto'` changes only `log_perplexity`** (whose bound uses
model.eta), never the fitted topics. Verified directly: `auto` vs `symmetric`
differ by exactly 0.0, while explicit `eta=0.01` moves the topics by 5.6e-02.
Recorded in `T9_eta_auto_noop_finding.json` and re-asserted in QC check 7.

This has a consequence for the baseline: the published model's `eta='auto'` is
equivalent to `eta='symmetric'`, and the 312-model Round 0 tuning grid varied eta
over {symmetric, auto}, so that grid dimension produced identical topics and
differed only in reported perplexity. The effective grid was half its nominal
size.

### The comparison that answers the intended question

`r02b_eta_explicit.py` sweeps explicit numeric eta in {0.01, 1/12, 0.3, 1.0},
6 seeds each.

`r02c_eta_tradeoff.py` adds the control that makes the sweep interpretable. A
larger eta smooths topic-word distributions; smoother topics are mutually more
similar; more similar topics are easier to match. Matched cosine can therefore
rise for purely geometric reasons. Two guards:

- **chance baseline**: mean cosine under RANDOM topic pairing at the same eta.
  `recovery_above_chance = matched - chance` is the part not explained by
  topics having become mutually similar.
- **ARI**: permutation-invariant, computed on hard document assignments, so it
  cannot be inflated by smoothing of the topic-word distributions.

Selection is explicitly **not** on coherence. C_v and u_mass are reported
alongside and are not used to choose.

---

## Task 3: K sensitivity (`r03_k_sweep.py`)

K in {4,5,6,7,8,9,10,11,12,14,16} x 6 seeds = 66 models at C3, alpha=symmetric,
eta=1/K stated explicitly (equivalent to the baseline's `eta='auto'`, per the
finding above, but unambiguous and portable outside gensim).

The same chance-correction applies and matters more here: matched cosine rises
mechanically as K falls, because fewer topics means broader distributions that
are easier to pair. The brief flags this and the design addresses it directly.
**`recovery_above_chance` and ARI are the primary stability readouts; raw matched
cosine is reported but never used alone.**

Reported per K: matched cosine and SD, chance baseline, recovery above chance,
mean and percentage of reproducible topics (cosine >= 0.70), dominant-topic
agreement, ARI, C_v, u_mass, mean pairwise JS distinctiveness, bound per word,
count of topics below 2% prevalence, minimum prevalence, and a Gini coefficient
of the prevalence distribution.

---

## Task 4: topic recovery and interpretability (`r04_recovery.py`)

Candidate K values 4, 6, 7 and 12, anchored on seed 42 and Hungarian-matched to
each of the other five seeds.

A topic counts as a genuinely recurring theme only when **two independent
signals agree**:

- **lexical**: mean matched cosine, plus the **stable term core**, the terms in
  the anchor's top 20 that also appear in the matched topic's top 20 in at least
  4 of the 5 other seeds.
- **documental**: mean Jaccard overlap of the **top 25 papers by topic share**
  between the anchor topic and its match in each other seed.

Lexical agreement without documental agreement is the signature of a broad
catch-all: the words recur because they are corpus-generic while the documents
the topic claims shift from run to run. The **catch-all flag** fires when
(cosine >= 0.50 and document Jaccard < 0.25), or when **term lift** (mean top-20
term probability relative to that term's corpus frequency) falls below 2.0,
which marks a topic whose top terms are simply the corpus's most frequent words.

`defensible_topics` counts topics that are not unstable AND not flagged
catch-all. Baseline labels are **not** reused; they appear in a separate
reference column only.

---

## Task 5: corrected cap-versus-seed comparison (`r05_cap_vs_seed_corrected.py`)

### What was wrong

Round 1 ran Mann-Whitney with 15 within-B250 seed-pair similarities against 6
seed-matched cap contrasts. Mann-Whitney assumes independence within each
sample. The 15 pairs come from only 6 seeds, so each seed appears in 5 pairs;
the effective sample size is nearer 6 than 15, and treating shared-seed pairs as
independent replicates understates variance and overstates significance.

A second dependence Round 1 did not note: the two samples are not independent of
**each other** either, since both derive from the same six B250 models.

### What is done instead

1. Descriptive distributions preserved unchanged and re-reported (`T14`).
2. Restructured as a **seed-level paired contrast**. For each seed s,
   `noise_s` = mean of the 5 pairs containing s; `cap_s` = cos(B250(s), X(s));
   `d_s = noise_s - cap_s`. One observation per seed removes the repeated-seed
   dependence. Residual dependence remains because noise_s shares pairs with
   noise_s'; this is stated, not assumed away.
3. **Exact paired sign-flip permutation** over all 2^6 = 64 sign assignments. No
   simulation error. **The smallest attainable one-sided p is 1/64 = 0.0156**,
   a hard floor reported with every result.
4. **Seed-level bootstrap** (20,000 resamples of the 6 seeds) for a percentile
   CI on the mean paired difference. With 6 clusters the CI is wide and coverage
   approximate; both stated.
5. Effect sizes: paired Cohen's dz and the count of seeds favouring the noise
   floor.

Round 1's CSVs are untouched; `T15` places old and new side by side.

---

## Task 6: trend uncertainty (`r06_trend_bootstrap.py`)

The 2019-2025 window is retained unchanged. The baseline topic model is held
fixed; only document sampling is resampled. 4,000 resamples per scheme.

| Scheme | Construction | What it addresses |
|---|---|---|
| **B1 iid** | resample the 956 in-window papers with replacement, recompute the 7 annual means, refit OLS-7 | propagates document sampling error into the annual means, which OLS-7 on fixed means ignores entirely. Assumes papers independent |
| **B2 journal cluster** | resample the 132 journals with replacement, take every paper in each drawn journal | papers in one journal share editorial scope and often authors. This is the scheme that addresses the dependence the brief flags in document-level OLS |

A year-stratified cluster bootstrap was considered and **rejected**: resampling
journals already perturbs year composition, and constraining it would
reintroduce the assumption under test.

Bootstrap p-values are two-sided by interval inversion, then BH FDR corrected
across the 12 topics within each scheme.

**Unquantified term, stated plainly**: these intervals describe sampling
uncertainty in the corpus, not uncertainty in the topic model. Given Task 1's
finding that the topic solution is seed-unstable, model uncertainty is the
larger missing term.

---

## Quality control (`r08_quality_control.py`)

34 assertions, all passing, in `outputs/QC_round2_checklist.csv`. Covers:
baseline and Round 1 immutability (hash and mtime), model and config-JSON counts
across all four grids, distributions summing to 1, bijectivity of all 60 K-sweep
alignments, the self-alignment identity control, document identity, the
convergence claim's evidential support, the eta no-op finding plus its positive
control, presence of chance baselines wherever similarity could be inflated, the
permutation floor, preservation of Round 1 point estimates, and verification
that the figure script loads no model objects so figures and tables cannot
disagree.

One known bookkeeping artifact is asserted explicitly rather than hidden: Round
1's own manifest is stale for `outputs/QC_checklist.csv`, which Round 1 rewrote
6 seconds after generating that manifest. The mtime check confirms Round 2 did
not touch it.

---

## Reproduction

```bash
cd LDA_Project
python validation_round2/scripts/r01_convergence.py          # ~13 min, resumable
python validation_round2/scripts/r02_priors.py               # ~8 min, resumable
python validation_round2/scripts/r02b_eta_explicit.py        # ~11 min, resumable
python validation_round2/scripts/r02c_eta_tradeoff.py        # ~3 min
python validation_round2/scripts/r03_k_sweep.py              # ~30 min, resumable
python validation_round2/scripts/r04_recovery.py 4 6 7 12    # ~4 min
python validation_round2/scripts/r05_cap_vs_seed_corrected.py  # ~3 min
python validation_round2/scripts/r06_trend_bootstrap.py      # ~5 min
python validation_round2/scripts/r07_figures.py              # ~1 min
python validation_round2/scripts/r08_quality_control.py      # ~5 min
```

Scripts taking a numeric argument accept a training time budget in seconds and
skip models already on disk, so a long grid can be run in chunks. Paths are
rooted at `/home/claude/LDA_Project`; change `ROOT` in `r00_common.py` to
relocate. Bootstrap seeds: 20261114 (Task 5), 20261115 (Task 6).
