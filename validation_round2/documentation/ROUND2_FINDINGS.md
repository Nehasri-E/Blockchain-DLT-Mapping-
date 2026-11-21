# Validation round 2: findings

**Bottom line.** The instability is **not** a convergence failure. The baseline
was already converged, and training five times longer does not improve seed
reproducibility at all. The K=12 posterior for this corpus is genuinely
multimodal: different seeds find different, equally good optima. Reducing K
helps far less than the raw numbers suggest once the mechanical component is
removed, and no K tested yields more than **two** topics that are both
reproducible and substantively specific.

The recommendation (section 7) is **not** to retain a 12-topic interpretation.

---

## 1. Task 1: convergence is not the cause

Three independent lines of evidence, 30 models:

| Config | passes/iters | E-step converged | Bound/word | Invariance vs 200-pass anchor | Matched cosine | ARI |
|---|---|---|---|---|---|---|
| baseline | 20/200 | 99.98% | -7.2583 | 0.901 | 0.5610 | 0.192 |
| C1 | 50/500 | 100% | -7.2572 | 0.971 | 0.5717 | 0.185 |
| C2 | 100/500 | 100% | -7.2505 | 0.969 | 0.5744 | 0.190 |
| C3 | 100/1000 | 100% | -7.2553 | **0.999** | 0.5616 | 0.178 |
| C4 anchor | 200/1000 | 100% | -7.2509 | 1.000 | 0.5627 | 0.182 |

**E1.** Even the baseline reaches 99.98% E-step convergence. The per-document
variational step was never meaningfully truncated, so `iterations=200` was not
the binding constraint.

**E2.** The bound improves by 0.0074 nats/word from baseline to the anchor,
which is smaller than the between-seed SD of the bound itself (0.010). The
objective has plateaued.

**E3.** C3 reproduces the 200-pass anchor at cosine **0.9992**. By any
reasonable criterion C3 is converged. The baseline at 0.901 is mildly
under-converged, which is worth knowing but is not the problem.

**And yet stability does not move.** Across a tenfold increase in training
(20 to 200 passes), matched cosine ranges 0.561 to 0.574 and ARI ranges 0.178 to
0.192. ARI is marginally **lower** at C3 than at baseline.

**Conclusion.** Convergence and reproducibility are different properties, and
this corpus separates them cleanly. The K=12 LDA posterior has many local optima
of near-identical quality; each seed finds a different one and stays there. More
compute buys a more precise answer to the wrong question.

**Best-supported configuration: C3 (100 passes, 1000 iterations)**, used for all
subsequent tasks. It is the cheapest configuration with demonstrated
convergence, not the one with the best stability number.

---

## 2. Task 2: the eta prior, and a defect in the baseline's configuration

### The comparison as specified is vacuous in gensim

`alpha=symmetric, eta=auto` and `alpha=symmetric, eta=symmetric` produce
**bit-identical topic-word matrices** (max difference exactly 0.0, `expElogbeta`
identical). This is a gensim 4.4.0 property, verified against the source and
with a positive control:

- the topics come from `state.eta + state.sstats`
- `init_dir_prior` returns `1/K` for both `'auto'` and `'symmetric'`
- `update_eta` writes to `model.eta`, never `state.eta`

So the learned prior never reaches the topics. Only `log_perplexity` changes.
Explicit `eta=0.01` moves the topics by 5.6e-02, confirming eta matters when it
actually propagates.

**Consequence for the project.** The published model's `eta='auto'` is
equivalent to `eta='symmetric'`. More importantly, the original 312-model tuning
grid varied eta over {symmetric, auto}: that dimension produced **identical
topics** and differed only in reported perplexity. The effective grid was half
its nominal size, and any perplexity comparison across that dimension compared
models with the same topics under different bound computations. This should be
corrected in the methods section of any paper.

### Genuine eta sensitivity, with the geometric component removed

| eta | Matched cosine | Chance | **Recovery above chance** | **ARI** | Distinctiveness | C_v |
|---|---|---|---|---|---|---|
| 0.01 | 0.5730 | 0.1901 | 0.3829 | 0.203 | 0.896 | 0.4067 |
| 1/12 (baseline) | 0.5616 | 0.1924 | 0.3692 | 0.178 | 0.874 | 0.4080 |
| 0.3 | 0.6498 | 0.2475 | 0.4023 | 0.332 | 0.748 | 0.4292 |
| 1.0 | 0.7017 | 0.3182 | 0.3835 | **0.483** | **0.537** | 0.4347 |

Read carefully, because the two signals disagree:

- **Recovery above chance is flat** (0.369 to 0.402). Essentially all of the
  rise in raw matched cosine from 0.56 to 0.70 is the chance baseline rising,
  because larger eta makes topics mutually more similar and therefore easier to
  pair. On the topic-word evidence, eta buys almost nothing.
- **ARI nearly triples** (0.178 to 0.483) and dominant-topic agreement rises
  0.335 to 0.605. ARI is permutation-invariant and computed on hard document
  assignments, so smoothing cannot inflate it. **The document-level stability
  gain is real.**

The price is severe: distinctiveness falls from 0.874 to **0.537**, and u_mass
degrades from -2.50 to -4.17. C_v rises, but C_v is known to reward smoother
topics and was explicitly not used for selection.

Interpretation also shifts: at eta=1.0, 8 of 12 topics match the baseline at
cosine >= 0.74, but four do not (T3 0.39, T7 0.26, T9 0.16, T12 0.16). The
eta=1.0 solution organises that part of the space differently rather than
reproducing the baseline more reliably.

**Verdict.** eta=1.0 buys document-level reproducibility by making topics
blurrier and more alike. That is a real option, with a real cost, and it is a
judgement call rather than a technical fix. `alpha=asymmetric` changes nothing
material (cosine 0.5484, ARI 0.180).

---

## 3. Task 3: is K=12 too granular?

| K | Matched cosine | Chance | **Recovery above chance** | **ARI** | % reproducible | C_v | Distinctiveness | Min prevalence |
|---|---|---|---|---|---|---|---|---|
| 4 | 0.8349 | 0.4877 | 0.3472 | **0.459** | 83.2% | 0.4286 | 0.719 | 0.174 |
| 5 | 0.7487 | 0.4077 | 0.3410 | 0.320 | 65.4% | 0.4189 | 0.780 | 0.127 |
| 6 | 0.7277 | 0.3450 | 0.3827 | 0.285 | 66.7% | **0.4377** | 0.811 | 0.100 |
| 7 | 0.7052 | 0.2981 | **0.4071** | 0.296 | 60.0% | 0.4314 | 0.832 | 0.073 |
| 8 | 0.6526 | 0.2728 | 0.3798 | 0.265 | 50.0% | 0.4268 | 0.844 | 0.065 |
| 9 | 0.6645 | 0.2462 | **0.4183** | 0.224 | 51.1% | 0.4267 | 0.855 | 0.059 |
| 10 | 0.5930 | 0.2263 | 0.3667 | 0.207 | 36.7% | 0.4140 | 0.861 | 0.045 |
| 11 | 0.5794 | 0.2073 | 0.3721 | 0.180 | 24.8% | 0.4079 | 0.870 | 0.041 |
| **12** | 0.5616 | 0.1956 | 0.3660 | 0.178 | 30.0% | 0.4080 | 0.874 | 0.030 |
| 14 | 0.5276 | 0.1671 | 0.3605 | 0.170 | 23.4% | 0.3997 | 0.883 | 0.021 |
| 16 | 0.5112 | 0.1469 | 0.3643 | 0.159 | 24.2% | 0.4026 | 0.892 | 0.017 |

**The headline warning in the brief is exactly right.** Matched cosine rises
monotonically as K falls (0.51 at K=16 to 0.83 at K=4), and that is almost
entirely mechanical: the chance baseline rises just as fast (0.15 to 0.49).
**Recovery above chance is flat across the whole range**, 0.341 to 0.418, with a
shallow maximum at K=9 (0.418) and K=7 (0.407). On the topic-word evidence
there is no K at which this corpus becomes reproducible.

ARI does fall genuinely with K (0.459 at K=4 to 0.159 at K=16) and is the one
signal that cannot be inflated. But even the best value, 0.459 at K=4, is
mediocre: two runs on identical data agree on well under half the structure.

Other considerations:
- **C_v peaks at K=6** (0.4377), then K=7 (0.4314). K=12 sits at 0.4080.
- **Distinctiveness rises monotonically with K**, which simply reflects narrower
  topics, and pulls the opposite way from stability.
- **Degenerate topics** appear only from K=12 upward (0.5 topics below 2%
  prevalence at K=12, 1.17 at K=16). K=12 is where the model starts producing
  topics too small to interpret: the baseline's T12 holds 8 papers.

**K=12 is not supported.** It is near the bottom on every stability measure, it
has the lowest coherence in the 4-12 range, and it is the first K to produce
degenerate topics. The evidence points to **K=6 or 7** on the balance of
coherence, recovery above chance, distinctiveness and absence of degenerate
topics. Not to K=4, whose ARI advantage comes with the lowest distinctiveness
and topics so broad they approach the corpus mean.

---

## 4. Task 4: what actually recurs

Dual criterion: a topic must be lexically stable **and** claim the same papers.
Lexical stability alone is the signature of a catch-all.

| Candidate K | Reproducible | Partial | Unstable | Catch-all flagged | Mean doc Jaccard | **Defensible topics** |
|---|---|---|---|---|---|---|
| 4 | 2 | 2 | 0 | 2 | 0.238 | **2** |
| 6 | 3 | 2 | 1 | 3 | 0.167 | **2** |
| 7 | 2 | 4 | 1 | 4 | 0.168 | **2** |
| 12 | 3 | 3 | 6 | 5 | 0.110 | **1** |

**No K yields more than two topics that are both reproducible and specific.**

The same two themes survive at every K tested:

| Theme | K=4 | K=6 | K=7 | K=12 | Evidence |
|---|---|---|---|---|---|
| **Cryptocurrency and token markets** | T1 (cos 0.82, Jac 0.28) | T1 (0.85, 0.32) | T1 (0.83, 0.29) | T6 (0.78, 0.14) | core terms: market, cryptocurrency, bitcoin, price, return, investor |
| **Pricing, channel strategy and consumer platforms** | T4 (0.96, 0.45) | T4 (0.84, 0.26) | T4 (0.92, 0.35) | T4 (0.90, 0.35) | core terms: platform, consumer, product, adoption, cost, strategy, manufacturer, retailer |

The second is the most robust object in the entire project: it is the only topic
reproducible at K=12, it survives every conclusion-cap condition in Round 1, and
it is the only temporal trend that survives journal-clustered inference
(section 5).

Everything else degrades. At K=12, five topics have a stable term core of 0 or 1
term: T3 and T9 share **no** term across seeds, T12 shares none and has a term
lift of 39.7 (its top terms are corpus-rare words that differ every run). The
mean top-25 document Jaccard at K=12 is **0.110**: even topics that look
lexically similar across seeds claim almost entirely different papers.

The catch-all pattern is visible at every K. At K=6, T2 (challenge, industry,
process, potential, business, management) has cosine 0.71 but document Jaccard
0.10 and term lift 2.30: generic operations-management vocabulary that recurs
because it is everywhere in the corpus, attached to a different set of papers
each run.

**Baseline labels were not reused.** Full term cores, drifting terms,
representative papers and per-seed top-20 lists are in `T20` and `T21`.

---

## 5. Task 5: corrected cap-versus-seed inference

Point estimates are unchanged. The inference is not.

| Comparison | Paired difference | 95% bootstrap CI | Seeds favouring noise | Exact p (one-sided) | Round 1 claimed |
|---|---|---|---|---|---|
| B250 vs A | 0.0621 | [0.030, 0.096] | 6/6 | **0.0156** | p=0.0092 |
| B250 vs B500 | -0.0067 | [-0.031, 0.013] | 3/6 | 0.6719 | p=0.715 |
| B250 vs B1000 | 0.0454 | [0.030, 0.065] | 6/6 | **0.0156** | p=0.0072 |
| B250 vs BFull | 0.0349 | [0.007, 0.062] | 5/6 | **0.0625** | p=0.047 |

Three things changed:

1. **Round 1's BFull claim does not survive.** It reported p=0.047 and called
   BFull "separated from noise". The exact paired test gives **p=0.0625**. That
   claim should be withdrawn.
2. **The two surviving results sit exactly on the permutation floor.** With 6
   seeds the smallest attainable one-sided p is 1/64 = 0.0156. "p=0.0156" here
   means only "all six seeds pointed the same way", which is the maximum
   evidence six seeds can supply. It is not strong evidence.
3. **The B500 null is unchanged and robust.** Moving the cap from 250 to 500
   words does nothing, by every method tried.

Round 1's substantive conclusion stands, with a better foundation: the
conclusion cap is not a material modelling decision. Its reported p-values
should not be cited.

---

## 6. Task 6: trend robustness under clustering

| Estimator | Significant after FDR | Topics |
|---|---|---|
| OLS-7 (Round 1) | 4 | T4, T7, T8, T9 |
| Theil-Sen (Round 1) | 4 | T4, T7, T8, T9 |
| WLS-7 (Round 1) | 1 | T8 |
| B1 iid document bootstrap | 5 | T4, T5, T7, T8, T9 |
| **B2 journal cluster bootstrap** | **1** | **T4** |

The corpus is heavily clustered: the largest journal supplies 12.6% of in-window
papers and the top 10 journals supply **59.1%**. There are 132 effective
clusters against 956 documents.

Once that clustering is respected, **only T4 Pricing and Channel Strategy
survives** (CI [0.0090, 0.0267], q=0.012). T8 Sustainable Adoption, which was
the only topic significant under all four Round 1 estimators, falls to q=0.168
under journal clustering: its apparent significance partly reflected a few
journals publishing a lot of sustainability work, not a corpus-wide shift.

**No topic is significant under all five estimators.** T4 fails WLS-7 after FDR
(q=0.065); T8 fails B2. The honest summary is that **T4 is the only trend with
broad support**, and even that is one estimator short of unanimous.

Seven annual observations remains the structural weakness. The bootstraps
address document and journal sampling, not the fact that the regression has 5
residual degrees of freedom, and not model uncertainty, which given section 1 is
the largest missing term of all.

**Research attention, not adoption.** Every statement above concerns the share
of published research text. None of it is evidence about blockchain deployment
in practice.

---

## 7. Final recommendation

### Candidate comparison

| Option | Reproducibility | Interpretability | Granularity | Verdict |
|---|---|---|---|---|
| **Retain K=12, better training** | ARI 0.178, 30% reproducible, 1 defensible topic | 5 topics with 0-1 stable terms; 1 degenerate topic | 12 themes | **Rejected.** Task 1 proves training is not the lever |
| **K=12 with eta=1.0** | ARI 0.483, 7.2 reproducible | distinctiveness 0.537, topics blurred toward each other; interpretation shifts | 12 nominal themes | **Rejected as primary.** Buys stability by erasing the distinctions that justify K=12 |
| **K=6 or 7** | ARI 0.285/0.296, 67%/60% reproducible, 2 defensible | C_v peak at K=6; no degenerate topics | 6-7 themes | **Best available**, but still only 2 defensible topics |
| **K=4** | ARI 0.459, 83% reproducible, 2 defensible | distinctiveness 0.719, topics near corpus mean | 4 broad themes | Most stable, least informative |
| **Report only reproducible topics** | by construction | 2 themes, both well-evidenced | 2 themes | **Recommended for substantive claims** |
| **Consensus/ensemble** | not tested | not tested | n/a | Proposed for Round 3 |

### What I recommend

**Do not publish a 12-topic interpretation of this corpus.** The evidence
against it is now threefold: it is the least reproducible K tested, it is the
first K to produce degenerate topics, and only one of its twelve topics is both
reproducible and specific.

**For the descriptive landscape mapping**, refit at **K=6** (C_v peak, no
degenerate topics, 67% of topics reproducible, distinctiveness still 0.81) and
present it explicitly as a coarse map, reporting the per-topic stability
statistics alongside every topic rather than in an appendix.

**For any substantive or inferential claim**, restrict to the **two
consistently defensible themes**: cryptocurrency and token markets, and pricing,
channel strategy and consumer platforms. These recur at every K, survive every
conclusion-cap condition, and in the case of the second, survive
journal-clustered trend inference. They can carry argument. The other topics
cannot.

**State the honest conclusion in the paper**: conventional LDA does not support
a stable fine-grained topic decomposition of this corpus. That is a legitimate
methodological finding about a 987-document corpus concentrated in one research
area, not a failure of the analysis. The corpus is small, thematically narrow
(blockchain in operations and supply chain management), and has a vocabulary of
3,335 terms over documents of median 191 tokens. These are conditions under
which LDA is known to be weakly identified.

### What I do not recommend

Selecting a favourable seed; presenting K=12 with stability relegated to a
footnote; or using eta=1.0 to reach a publishable stability number while still
claiming twelve distinct themes. The last is the most tempting and the least
defensible: it is stability bought by erasing the distinctions.

### Proposed Round 3, if the evaluator wants to continue

1. **Consensus/ensemble topic modelling** at K=6: cluster the 6x6=36 topics from
   six seeds, keep clusters with members from at least 5 seeds. This directly
   targets non-identifiability and is the one untested remedy.
2. **NMF with multiple initialisations** as a comparator. NMF on TF-IDF is often
   more stable than LDA on small corpora, and disagreement between the two would
   be informative about the corpus rather than the algorithm.
3. **Corpus enlargement**, which is the real fix. 987 documents is small for
   K=12 with a 3,335-term vocabulary.

---

## 8. Limitations

1. **Six seeds.** Adequate for the descriptive stability picture, at the exact
   permutation floor (p >= 0.0156) for inference. More seeds would tighten
   Task 5 but will not change the Task 1 conclusion, which rests on effect sizes.
2. **One corpus condition.** All of Round 2 is on B250. Round 1 showed the cap
   does not matter much, so this is reasonable, but the K sweep was not repeated
   on BFull.
3. **Consensus/ensemble not tested.** The most promising untested remedy.
4. **Trend bootstraps hold the topic model fixed.** Model uncertainty is not
   propagated and is the largest missing term.
5. **The catch-all flag uses stated thresholds** (document Jaccard 0.25, term
   lift 2.0) that are reasoned but not derived. All underlying values are in
   `T20` so the flags can be recomputed.
6. **The 0.70 reproducibility threshold** is conventional, carried over from
   Round 1 for comparability.
7. **alpha was tested only as symmetric vs asymmetric**, not swept numerically
   as eta was. The asymmetric result showed no material effect, but an alpha
   sweep was not run.
