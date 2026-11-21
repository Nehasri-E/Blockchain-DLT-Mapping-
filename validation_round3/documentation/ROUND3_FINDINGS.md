# Validation round 3: findings

**Bottom line.** Consensus LDA does **not** rescue the corpus. Pooling six seeds
yields exactly **one** theme that passes all criteria, because consensus cannot
create agreement that is not in the runs.

NMF, run on the identical corpus and vocabulary, is a different story: it
converges to the **same solution from six genuinely different random starts**
(matched cosine 1.000000 at K=4 and K=6) and yields **five of six** qualifying
topics. The corpus is not the problem. LDA's posterior is.

**Three themes have convergent validity across both methods.** Two are the
Round 2 themes; the third is new and was previously dismissed as a catch-all.

A six-topic descriptive map is defensible **under NMF**, not under LDA.

---

## 1. Does consensus LDA beat individual runs?

**Marginally, and not enough.**

Primary consensus is anchor-based Hungarian matching, which enforces seed
uniqueness by construction: six groups, six members each, one per seed.

| CT | Within cos | Min within | JS | Separation | Doc Jaccard | Prevalence | Term lift | Core | Anchor agreement | Qualifies |
|---|---|---|---|---|---|---|---|---|---|---|
| **1** | 0.850 | 0.667 | 0.497 | 0.657 | **0.361** | 0.133 | 4.16 | 16 | 0.800 | **yes** |
| 4 | 0.754 | 0.468 | 0.450 | 0.490 | 0.186 | 0.139 | 3.62 | 18 | 0.433 | no |
| 3 | 0.749 | 0.479 | 0.450 | 0.463 | 0.099 | 0.188 | 2.70 | 14 | 0.544 | no |
| 5 | 0.732 | 0.481 | 0.445 | 0.455 | 0.105 | 0.206 | 2.85 | 14 | 0.356 | no |
| 2 | 0.605 | 0.232 | 0.402 | 0.311 | 0.065 | 0.202 | **1.88** | 9 | 0.433 | no |
| 6 | 0.415 | 0.139 | 0.269 | 0.121 | 0.054 | 0.132 | 2.02 | 4 | 0.289 | no |

**One qualifying theme: CT1, cryptocurrency and token markets.**

Three things are worth reading carefully:

- **CT4 fails on one criterion by 0.032.** It clears the term core (18),
  document overlap (0.186 against a 0.15 floor) and term lift (3.62), and fails
  only `min_within_cosine` at 0.468 against a 0.50 floor. One of its six members
  is weak. Under the hierarchical route at thresholds 0.625 and above it
  qualifies outright, and under NMF it is the single strongest topic in the
  entire study. This is a borderline case and is reported as one.
- **CT2 is the catch-all, correctly caught.** Term lift 1.88 is below the
  generic-vocabulary floor: its core (industry, challenge, management,
  potential, business, process) is corpus-wide language. Its document overlap is
  0.065, so it claims almost entirely different papers each run. Both methods
  flag it (NMF's N1 fails on lift 1.61).
- **Anchor agreement is 0.476 overall.** The six anchor choices substantially
  disagree about the grouping. Only CT1 holds up at 0.800. This is itself
  evidence that there is little stable structure to pool.

**The seed-impure clusters were diagnosed, not dropped.** The hierarchical route
produced two seven-member clusters. The test was whether the duplicated seed's
two topics are at least as similar to each other as the cluster is internally.
Both failed decisively (0.311 vs 0.734; 0.269 vs 0.797), so both were
conflations rather than justified merges.

### The threshold was not shopped

The multi-seed cluster count is flat at 6 across 0.500-0.750. The headline was
fixed at **0.55**, at the conservative end of that plateau. The cluster-count
maximiser is **0.800**, giving 8 clusters and 3 qualifying themes; it was not
chosen.

Full disclosure of the dependence, because it is not invariant:

| Threshold | Qualifying themes |
|---|---|
| 0.500-0.600 | 1 (cryptocurrency) |
| 0.625-0.750 | 2 (+ pricing and consumer platforms) |
| 0.800 | 3 |

The headline gives the **fewest** themes available on the plateau. Raising it
within the plateau adds the pricing theme, which is consistent with every other
line of evidence below.

---

## 2. Does NMF independently recover the same themes?

**Yes, and it is dramatically more stable.**

| K | Matched cosine | ARI | Doc Jaccard | C_v | Distinctiveness |
|---|---|---|---|---|---|
| 4 | LDA 0.835 / **NMF 1.000** | LDA 0.459 / **NMF 0.873** | NMF 1.000 | 0.429 / **0.532** | 0.719 / 0.700 |
| 6 | LDA 0.728 / **NMF 0.942** | LDA 0.285 / **NMF 0.749** | NMF 0.805 | 0.438 / **0.521** | 0.811 / 0.729 |
| 8 | LDA 0.653 / **NMF 0.887** | LDA 0.265 / **NMF 0.700** | NMF 0.624 | 0.427 / **0.557** | 0.844 / 0.753 |

**This is not an initialisation artifact.** NMF was run with `init='random'`,
not the deterministic `nndsvd`, and the six starts were verified to genuinely
differ (max |H0| difference 0.078-0.096). They nonetheless converge to the same
solution direction: matched cosine **1.000000** at K=4 and K=6 (raw H differs in
magnitude only, which is NMF's scale indeterminacy). The TF-IDF Frobenius
objective is effectively globally identifiable on this corpus.

**Five of six NMF K=6 consensus topics qualify** (N2, N3, N4, N5, N6), against
one for LDA. N1 fails on term lift 1.61, the same catch-all LDA flags.

**Important caveat.** NMF's stability is a property of its objective, not
evidence that its themes are truer. LDA optimises a probabilistic likelihood
with a genuinely multimodal posterior; NMF optimises a Frobenius reconstruction
with a near-unique optimum. Stability and validity are different things.

What **is** evidence is convergent validity: when two methods with different
objectives, different representations and different failure modes find the same
theme, that theme is a property of the literature rather than of the algorithm.

---

## 3. Are the two Round 2 themes supported by both methods?

**Yes, both, and a third joins them.**

Convergent validity requires alignment cosine >= 0.50, term-core Jaccard >= 0.30
**and** representative-paper Jaccard >= 0.20. The paper criterion is what
separates a shared theme from shared vocabulary.

| LDA / NMF | Cosine | Term core | Papers | Convergent | Theme |
|---|---|---|---|---|---|
| **CT1 / N2** | 0.878 | 0.647 | **0.613** | **yes** | Cryptocurrency and token markets |
| **CT4 / N3** | 0.871 | 0.556 | 0.282 | **yes** | Pricing, channel strategy and consumer platforms |
| **CT3 / N4** | 0.864 | 0.474 | 0.282 | **yes** | Firm-level adoption and performance |
| CT5 / N6 | 0.791 | 0.556 | 0.190 | no | System architecture and security |
| CT2 / N1 | 0.856 | 0.353 | **0.042** | no | generic corpus vocabulary |
| CT6 / N5 | 0.596 | 0.059 | 0.087 | no | unstable |

CT2/N1 is the instructive failure: cosine 0.856 and a respectable term-core
overlap, but the two methods agree on only **2 of 25** representative papers.
Shared words, different literatures. Exactly the catch-all pattern, and it is
why the paper criterion was imposed.

### The third theme

**CT3/N4, firm-level adoption and performance** (core: adoption,
blockchain_technology, firm, performance, relationship, supply_chain, impact,
enhance, theory), shows convergent validity. Round 2 dismissed this region as a
catch-all at K=12. At K=6 with consensus and cross-method confirmation it holds
up: term lift 2.70, robust to journal exclusion (0.82-0.95), and it qualifies
under NMF.

It is the weakest of the three. Its LDA document overlap is 0.099, so the
specific papers shift between seeds even though the theme recurs. It should be
reported as a theme of the literature, not used to carry fine-grained claims
about which papers belong to it.

---

## 4. Sensitivity

**Journal exclusion** (drop top 1, 3, 5 journals; refit LDA K=6 x 6 seeds):

| CT | min cosine to reference | mean |
|---|---|---|
| **4** | **0.941** | **0.961** |
| **1** | 0.884 | 0.915 |
| 3 | 0.821 | 0.901 |
| 6 | 0.773 | 0.832 |
| 5 | 0.604 | 0.733 |
| 2 | 0.562 | 0.760 |

**CT4 is the most journal-robust theme in the study**, more robust than CT1.
Its apparent weakness in the LDA consensus is not a dependence on a few
dominant outlets.

**Corpus definition** (B250 vs BFull, uncapped conclusions): CT1 0.935, CT4
0.881, CT3 0.762, others 0.710-0.838. Both headline themes survive the
conclusion-cap change, consistent with Round 1's finding that the cap is not a
material decision.

**Supporting-seed requirement** (3 to 6 of 6): only CT1 qualifies at any level,
so the result is not an artifact of that threshold either.

---

## 5. Corrected trend inference

**The Round 2 p-values are withdrawn.** The journal-cluster bootstrap tail
proportion is not a p-value: the bootstrap distribution is centred on the
observed slope, not the null, so its tail mass describes precision rather than
evidence against H0. The Round 2 confidence intervals are unaffected and are
retained as descriptive uncertainty.

Replaced with document-level inference, 956 papers, 132 journal clusters, null
imposed.

| Procedure | Valid p? | Significant after FDR | Topics |
|---|---|---|---|
| OLS-7 (R1) | yes | 4 | T4, T7, T8, T9 |
| WLS-7 (R1) | yes | 1 | T8 |
| Theil-Sen (R1) | yes | 4 | T4, T7, T8, T9 |
| R2 journal bootstrap tail proportion | **no** | (1) | **withdrawn** |
| **Wild cluster bootstrap-t, null imposed** | **yes** | **0** | **none** |
| Within-journal year permutation | yes | 4 | T4, T5, T7, T8 |

**No topic survives the wild cluster bootstrap after FDR correction.** T4 has
raw p = 0.0098 but q = 0.118 across 12 tests; T8 has raw p = 0.043, q = 0.260.

Two CR1 confidence intervals exclude zero (T4, T8) while zero topics are
significant. As the brief anticipated, **CI exclusion of zero is not
significance**, and the two are reported in separate columns so they cannot be
conflated.

The wild bootstrap was validated before its null result was accepted: pure noise
p = 0.728, planted 0.02/yr trend p = 0.0005, planted 0.05/yr trend p = 0.0005,
purely between-journal signal p = 0.0005. It has power. The nulls are real.

**Why V2 and V3 disagree.** The within-journal permutation tests a narrower
null, that year is exchangeable *within* each journal, and therefore removes
between-journal variation from the reference distribution. The wild bootstrap
admits that variation. Much of the apparent temporal movement is between-journal
composition change over time, not within-journal change. The wild bootstrap is
the appropriate test for a corpus-wide claim and is the more conservative one.

**Remaining gap.** Journal clustering does not absorb author-level dependence.
Authors publish across journals and research groups produce correlated runs of
papers in several outlets. The `Authors` field exists but was never parsed into
disambiguated identities, so author clustering could not be implemented.

---

## 6. Final recommendation

### Answers to the five questions

1. **Does consensus LDA produce more defensible themes than individual runs?**
   Marginally. One theme qualifies against essentially one at K=12. Consensus
   cannot manufacture agreement that is not present in the runs. It is worth
   doing as a diagnostic, not as a rescue.
2. **Does NMF independently recover the same themes?** Yes, and with far greater
   stability (ARI 0.749 vs 0.285 at K=6; identical solutions from different
   random starts at K=4 and K=6).
3. **Are the two Round 2 themes supported by both methods?** Yes. Both show
   convergent validity on words and on papers. A third theme joins them.
4. **Is a six-topic descriptive map defensible?** **Under NMF, yes** (5 of 6
   topics qualify, ARI 0.749). **Under LDA, no** (1 of 6, ARI 0.285). If the
   project wants a six-topic map, it should be built with NMF and that choice
   stated.
5. **Two themes or more?** **Three** substantive themes are supported, with
   explicitly different evidential weight.

### Recommended reporting structure

| Theme | Evidence | Use |
|---|---|---|
| **Cryptocurrency and token markets** | LDA consensus (only qualifier), NMF (qualifies), convergent cosine 0.878, paper overlap 0.613, journal-robust 0.884-0.934, corpus-robust 0.935 | Full substantive claims |
| **Pricing, channel strategy and consumer platforms** | NMF strongest topic (within cosine 0.9998, doc Jaccard 1.000), convergent 0.871, **most journal-robust in the study** (0.941-0.970), qualifies in hierarchical LDA at thresholds >= 0.625; fails anchor-LDA on one criterion by 0.032 | Full substantive claims, noting the LDA near-miss |
| **Firm-level adoption and performance** | Convergent 0.864, NMF qualifies, journal-robust 0.821-0.950; LDA document overlap only 0.099 | Report as a theme; do not make paper-level claims |

**Primary model recommendation: NMF at K=6**, reported as the descriptive map,
with the LDA consensus presented as the stability diagnostic that motivated the
switch. This is a defensible methodological narrative, not a retreat: the paper
can state that LDA was the planned method, that it proved non-identifiable on
this corpus under systematic testing across 138 models, that NMF was adopted as
a stable alternative, and that the themes carried forward are those both methods
agree on.

**Temporal claims should be dropped or heavily qualified.** No trend survives
valid clustered inference with multiplicity correction. The honest statement is
that the corpus does not support claims about how topic attention has changed
over 2019-2025, given 7 annual observations, strong journal clustering and
unmodelled author dependence.

### What not to do

Do not report the LDA K=12 solution. Do not raise the similarity threshold to
0.80 to obtain three LDA consensus themes. Do not cite the Round 2
journal-bootstrap p-values. Do not claim T4's trend is significant because its
confidence interval excludes zero.

---

## 7. Limitations

1. **Six seeds / six initialisations.** Adequate for the descriptive picture;
   the anchor-agreement estimate (0.476) is coarse.
2. **NMF stability is an objective property, not a validity claim.** The
   argument rests on convergent validity, not on NMF being more stable.
3. **Author-level dependence unmodelled.** The main remaining inferential gap.
4. **Qualification thresholds** (core >= 4, doc Jaccard >= 0.15, lift >= 2.0,
   min cosine >= 0.50) are reasoned but not derived. All underlying values are in
   `C10` and `N3` so every verdict can be recomputed under different thresholds.
5. **S3 holds the dictionary fixed** when dropping journals, by design; a full
   refit would confound vocabulary change with journal removal.
6. **NMF K was not swept as finely as LDA's**, only 4, 6 and 8 as specified.
7. **Topic labels remain interpretation.** They are proposed from term cores and
   representative papers together, and an evaluator may reasonably disagree.
