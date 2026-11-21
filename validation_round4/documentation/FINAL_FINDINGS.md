# Final findings

**Primary model: NMF with K=6 on TF-IDF.** Three substantive themes are
defensible, two strongly and one with qualification. The remaining three NMF
components are reported as part of the descriptive map but not as validated
themes. **No temporal claim is supported** under the primary inference
procedure.

LDA, the originally planned method, is retained as an independent comparator. It
is not the primary model because it proved non-identifiable on this corpus.

---

## 1. Corrections to Round 3

Thirty-one numerical statements in `ROUND3_FINDINGS.md` were re-derived from
their source CSVs. **Twenty-eight were correct. Three were wrong, and all three
are the same error.**

| Original statement | Actual | Corrected |
|---|---|---|
| "NMF achieves matched cosine 1.000000 at both K=4 and K=6" | K=4: 1.0000 (15/15 pairs). K=6: **0.9415** mean, 10/15 pairs identical | NMF converges identically from all six initialisations at K=4. At K=6 it has a **dominant basin**, not a unique optimum: five of six initialisations (42, 7, 2024, 123, 31337) converge identically; seed 777 finds a different optimum |
| "raw H differs in magnitude only" | True for the one pair spot-checked (42 vs 7), both in the dominant basin | Generalising from one pair to all fifteen was the error |
| "the objective is effectively globally identifiable on this corpus" | K=4 yes; K=6 dominant basin; K=8 only 2/15 pairs identical | Globally identifiable at K=4, dominant basin at K=6, degrading with K |

The cause: I verified K=6 with a single seed pair, both of which happened to lie
in the majority basin, and wrote the result as if it held for all pairs. The CSV
and the figure were always right. Full table in
`FINAL_TABLES/Table3_round3_corrections.csv`; per-pair detail in
`outputs/F1_nmf_pairwise_detail.csv`.

**This does not change any conclusion.** The outlier initialisation was tested
directly: a consensus built from the five-seed basin alone reproduces the
six-seed consensus at cosine 0.989 to 1.000
(`outputs/F4_nmf_outlier_seed_sensitivity.csv`). Seed 777 lowers the reported
within-cluster statistics but does not alter the themes.

The corrected comparison still favours NMF decisively: at K=6, NMF ARI **0.749**
against LDA **0.285**.

A further check was built to stop this class of error recurring:
`FINAL_TABLES/Table7_claim_to_file.csv` re-derives all 69 numeric claims in this
document from their source files on every run. It caught three wrong
distinctiveness values in the first draft of this section, which are corrected
below. Two claims cannot be machine-checked and are marked as such.

---

## 2. The three validated themes

Primary model NMF K=6, six random initialisations. Full detail in
`FINAL_TABLES/Table2_final_themes.csv`.

### N2. Cryptocurrency and token markets: **Strongly supported**

- **Core terms**: cryptocurrency, market, bitcoin, investor, return, price, ico,
  financial, token, icos, asset, find
- **Prevalence**: 15.8% of document-topic mass
- **Distinctiveness** 0.722 JS; **term lift** 3.82, the highest in the model
- **Cross-method**: LDA CT1 at cosine **0.878**, term-core overlap **0.647**,
  representative-paper overlap **0.613** (the highest in the study)
- **Robustness**: journal exclusion min 0.884; uncapped-conclusion corpus 0.935
- **Representative papers**: "Initial coin offerings (ICOs): market cycles and
  relationship with bitcoin" (2020); "What factors drive returns on initial coin
  offerings?" (2020)
- The only theme that qualifies under every criterion in every analysis in all
  four rounds.

### N3. Pricing, channel strategy and consumer platforms: **Strongly supported**

- **Core terms**: manufacturer, consumer, product, retailer, profit, platform,
  cost, adoption, strategy, optimal, adopt, decision, supply_chain
- **Prevalence**: 16.3%
- **Distinctiveness** 0.753 JS, the highest of the six components; **term lift**
  2.71
- **Cross-method**: LDA CT4 at cosine **0.871**, term-core overlap 0.556,
  paper overlap 0.282
- **Robustness**: journal exclusion min **0.941** (the most journal-robust topic
  in the study); uncapped corpus 0.881
- Within NMF it is the single most stable topic: within-initialisation cosine
  **0.9998**, document overlap **1.000**
- **Important disclosure**: in the LDA consensus this theme was reported in
  Round 3 as a near-miss. It fails only `min_within_cosine` (0.468 against
  0.50), and **that criterion was added after the first consensus run had
  already produced results**. Under the three criteria actually pre-specified,
  it qualifies (`outputs/F10_criterion_sensitivity.csv`). Classified as strongly
  supported on the combined NMF, cross-method and robustness evidence.

### N4. Firm-level adoption and performance outcomes: **Supported with qualification**

- **Core terms**: firm, adoption, blockchain_technology, performance,
  relationship, capability, effect, supply_chain, impact, influence, enhance,
  innovation
- **Prevalence**: 17.4% (the largest of the three)
- **Distinctiveness** 0.691 JS; **term lift** 2.10, the lowest of the three and
  closest to the generic-vocabulary floor
- **Cross-method**: LDA CT3 at cosine 0.864, term-core overlap 0.474, paper
  overlap 0.282
- **Robustness**: journal exclusion min 0.821; uncapped corpus 0.762, the
  weakest of the three
- **Why qualified**: Round 2 dismissed this region as a catch-all at K=12. At
  K=6 with cross-method confirmation it holds up, but its LDA document overlap
  across seeds is only 0.099, so the specific papers shift between runs even
  though the theme recurs. **Report the theme; do not make paper-level claims
  about which papers belong to it.**

No theme is classified Exploratory. The three that would fall there are not
presented as themes at all.

---

## 3. The three components that are not themes

Retained in the descriptive map, not advanced as substantive findings.

| Component | Prevalence | Why not a theme |
|---|---|---|
| **N1 Generic operations and management vocabulary** | 17.8% | Term lift **1.61**, below the 2.0 generic-vocabulary floor. Its LDA counterpart shares only **2 of 25** representative papers with it. Shared words, different literatures. This is the catch-all both methods produce |
| **N5 Governance, digital platforms and NFTs** | 17.3% | Qualifies on NMF's own criteria but has no cross-method support: LDA counterpart aligns at cosine **0.596** with term-core overlap **0.059** |
| **N6 System architecture and security design** | 15.5% | Aligns with LDA CT5 at cosine 0.791, but paper overlap is **0.190** against the 0.20 convergent-validity floor. The methods agree on vocabulary but not sufficiently on which papers belong. **A threshold of 0.15 would admit it as a fourth theme**, and that threshold was set after the results were known |

N1 is the most prevalent component in the entire model at 17.8%. Reporting a
six-theme map would make corpus-generic vocabulary the single largest "finding".

---

## 4. Temporal inference

**The correct statement is narrower than Round 3's.** It is not true that every
valid procedure produced null results.

| Procedure | Null hypothesis | Significant after FDR |
|---|---|---|
| **Wild cluster bootstrap-t** (primary) | no corpus-wide association between year and topic share, allowing for arbitrary within-journal correlation **and** between-journal heterogeneity | **0 of 12** |
| Within-journal year permutation | no association between year and topic share **within** journals, holding each journal's composition and year profile fixed | **4 of 12** (T4, T5, T7, T8) |

These test different nulls and the difference is substantive. The permutation
test conditions on journal composition, so it removes between-journal variation
from the reference distribution and asks only whether topic attention shifts
*inside* journals. The wild bootstrap admits that variation and asks whether
there is a corpus-wide trend. Much of the apparent movement is between-journal
composition change over time: different journals publishing at different rates
across the window. The corpus grew from 38 papers in 2019 to 217 in 2025, and
that growth was not evenly distributed across outlets.

**Recommended conclusion:** there is **insufficient evidence for general
corpus-wide temporal trend claims under the primary clustered inference
procedure**. Evidence of within-journal shifts exists for four topics and may be
reported as such, explicitly scoped, in supplementary material. It is not a main
result.

Two CR1 confidence intervals exclude zero (T4, T8) while zero topics are
significant. **CI exclusion of zero is not significance** and the two are
reported in separate columns. T4's raw clustered p is 0.0098; its BH-adjusted q
across 12 tests is 0.118.

All previous trend analyses are retained in
`SUPPLEMENTARY_MATERIAL/` and in the Round 1 to 3 output trees.

---

## 5. Author-level dependence

**Assessed and found infeasible to model, with the gap quantified.**

- Author strings parse for **987 of 987 papers (100%)**, mean 3.53 authors per
  paper, 3,483 mentions, 2,299 distinct surname-plus-initial keys.
- **Normalisation**: lowercase, strip accents and punctuation, split on `;`,
  reduce to surname plus first given initial.
- **This is string matching, not disambiguation.** The dataset has no
  affiliation, ORCID, email or resolved author identifier.
- **Measured ambiguity**: 135 of 2,299 keys (5.9%) map to more than one distinct
  raw spelling, affecting **567 of 3,483 mentions (16.3%)**. `wang_y` alone
  covers 5 distinct raw spellings.
- **Scale of the unmodelled dependence**: 548 normalised authors appear on more
  than one in-window paper, touching **718 of 956 papers (75%)**.

Multiway clustering by journal and author was therefore **not performed**. It
would rest on identity claims the data cannot support, and the error direction
is not even predictable: name collisions inflate clusters while name variants
fragment true ones. This is the most serious remaining inferential gap, and it
makes the temporal nulls in section 4 *conservative in an unknown direction*
rather than safely conservative.

---

## 6. Final decisions

**1. What is the final primary model?**
NMF with K=6 on sublinear TF-IDF over the 3,335-term vocabulary, six random
initialisations, consensus by Hungarian matching. LDA K=6 consensus is the
independent comparator, not the primary model.

**2. How many research themes can we defensibly report?**
**Three.** Six NMF components are reported as the descriptive map; three carry
validated substantive themes.

**3. Which themes have strong versus qualified support?**
Strongly supported: **N2 Cryptocurrency and token markets** and **N3 Pricing,
channel strategy and consumer platforms**. Supported with qualification:
**N4 Firm-level adoption and performance outcomes**, which should carry
theme-level but not paper-level claims.

**4. Can we make temporal claims?**
**No corpus-wide ones.** Within-journal shifts for four topics may be reported
as explicitly scoped supplementary observations. Topic prevalence measures
research attention, never real-world blockchain adoption.

**5. What are the most important limitations?**
In order: (i) author-level dependence is unmodelled and touches 75% of in-window
papers; (ii) the primary model was switched from LDA to NMF after seeing
results; (iii) seven annual observations cannot support trend inference; (iv)
the corpus is blockchain in operations and supply chain management, not the
general blockchain and DLT literature; (v) several thresholds that determine
theme counts were set after results were visible. Full list in
`FINAL_LIMITATIONS.md`.

**6. Is the computational analysis ready for a research paper?**

**The computational analysis is sound and reproducible. The manuscript is not
yet ready, and passing QC is not the reason to think otherwise.**

What is ready: the pipeline is deterministic and re-runs bit-identically; every
number traces to a named output file; the methodology has survived four rounds
of adversarial self-audit including three defect fixes that changed conclusions
(LdaMulticore non-determinism, the eta no-op, the invalid bootstrap p-value);
and the three themes have convergent support across two methods with different
objectives.

What is not ready, and must be resolved before submission:

- **The post-hoc model switch must be narrated honestly in the manuscript**, not
  presented as though NMF were the plan. A reviewer who sees NMF without the LDA
  failure history will reasonably ask why a non-standard choice was made.
- **Author-level dependence is unresolved.** With 75% of papers sharing a
  string-matched author, a reviewer may regard the clustered inference as
  insufficient even for the null results.
- **Three thresholds that determine the headline theme count were set after
  results were visible** (the fourth consensus criterion, the similarity
  threshold, the convergent-validity paper floor). Moving the last from 0.20 to
  0.15 would yield four themes. This must be disclosed, and ideally the
  thresholds should be pre-registered and the analysis re-run.
- **The corpus is narrower than the framing implies.** Scope the title and
  claims to operations and supply chain management.
- **No external validity check.** Nothing confirms the three themes against an
  independent source such as expert coding of a paper sample, and that is the
  single most valuable addition remaining.

My assessment: **the analysis is defensible; the paper needs its framing and
disclosure built around what actually happened.** That is a writing and
pre-registration task, not a computational one.
