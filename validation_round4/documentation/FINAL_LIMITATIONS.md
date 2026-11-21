# Final limitations

Ordered by how much each one threatens the conclusions, not by how easy it is to
describe. The first five are the ones a reviewer will raise.

---

## 1. Author-level dependence is unmodelled and the exposure is large

**548 string-matched authors appear on more than one in-window paper, touching
718 of 956 papers (75%).** Papers by the same author are not independent
observations, and the inference in section 8 of `FINAL_METHODS.md` clusters only
by journal.

It cannot be fixed with this dataset. Author strings parse for all 987 papers,
but the dataset carries **no affiliation, ORCID, email or resolved author
identifier**, so only surname-plus-initial matching is available. Measured
ambiguity: 135 of 2,299 normalised keys (5.9%) map to more than one raw spelling,
affecting **567 of 3,483 mentions (16.3%)**; `wang_y` alone covers five distinct
spellings.

Multiway clustering was therefore not performed, and the direction of the error
it would introduce is not predictable: name collisions merge distinct people and
inflate clusters, while spelling variants fragment real ones. **The temporal
nulls are conservative in an unknown direction, not safely conservative.**

Resolution requires linking to an external author registry.

## 2. The primary model was chosen after seeing results

The planned analysis was LDA at K=12. The reported analysis is NMF at K=6. Both
changes followed evidence (LDA K=12 reproduced only 1 of 12 topics across seeds,
mean matched cosine 0.561, ARI 0.192) and both are defensible, but neither was
pre-specified.

This must be narrated in the manuscript as what it is. A reader who encounters
NMF without the LDA failure history has no way to judge whether the method was
chosen because it worked or because it was right.

K=6 is also not the most stable K available: NMF has ARI **0.873** at K=4 against
0.749 at K=6. K=6 was retained for interpretability, which is a legitimate reason
and a post-hoc one.

## 3. Three thresholds that determine the theme count were set after the fact

| Threshold | Value | Effect of changing it |
|---|---|---|
| `min_within_cosine` (consensus criterion 4) | 0.50 | Added after the first consensus run had produced results. It is the **only** criterion CT4 fails (0.468). Under the three genuinely pre-specified criteria, 2 LDA themes qualify rather than 1 |
| Convergent-validity paper-overlap floor | 0.20 | N6 (system architecture) fails only this, at 0.190. A floor of **0.15 would yield four themes instead of three** |
| Headline similarity threshold | 0.55 | Chosen with the threshold plateau already visible. It happens to give the fewest qualifying themes on that plateau, so the direction is conservative, but the choice was still informed |

The headline finding of three themes is **threshold-contingent at the margin**.
The right remedy is pre-registration followed by a re-run, not an argument that
the chosen values are reasonable.

## 4. Seven annual observations cannot support trend inference

The window is 2019 to 2025. Seven points, with the corpus growing from 38 papers
to 217 over that span. No regression on seven unevenly weighted annual
observations can distinguish a trend from the composition change that accompanied
the growth, which is exactly what the divergence between the two tests in
`FINAL_FINDINGS.md` section 4 shows.

The null results are the honest outcome here, and they should not be read as
evidence that topic attention is stable. They are evidence that this design
cannot detect change.

## 5. The corpus is narrower than "blockchain literature"

987 papers across 133 journals, concentrated in operations, supply chain and
management research. Computer science, cryptography, law, economics and
information systems venues are largely absent. The themes recovered are
**themes of blockchain research in operations and management**, and the title and
claims must say so.

The 2017 to 2026 range also has incomplete years at both ends, which is why the
trend window is restricted to 2019 to 2025.

---

## 6. Measurement limitations

- **Topic prevalence measures research attention, not adoption.** A rise in
  papers about cryptocurrency markets says nothing about cryptocurrency use.
  This is the single most common misreading of topic-model output.
- **Text source is title plus abstract plus a 250-word conclusion**, not full
  text. Methods sections, results and discussion are absent, so topics reflect
  how papers are framed rather than what they do. Round 1 showed the structure is
  insensitive to the cap from 250 words to uncapped, which addresses the cap but
  not the absence of body text.
- **Corpus-defining terms were removed** (`blockchain`, `technology`). This
  sharpens discrimination between topics and means prevalence figures are shares
  of the remaining vocabulary, not of all text.
- **The lemma repair map is hand-built.** Four cases were found and fixed. There
  is no guarantee the list is complete; other lemmatization splits may remain.

## 7. Model limitations

- **NMF at K=6 has a dominant basin, not a unique optimum.** Five of six
  initialisations agree; seed 777 does not. The consensus is robust to dropping
  it (cosine 0.989 to 1.000), but a single-seed NMF run by another researcher has
  roughly a one in six chance of landing in the other basin and reporting
  different components.
- **N4 is theme-stable but not paper-stable.** Its LDA document overlap across
  seeds is 0.099. The theme recurs; the specific papers assigned to it do not.
  No paper-level claim about N4 is supportable.
- **Three of six components are not themes.** N1 is corpus-generic vocabulary
  (lift 1.61), N5 has no cross-method support (cosine 0.596), N6 narrowly fails
  paper overlap. Reporting a six-theme map would present the catch-all component,
  the most prevalent in the model at 17.8%, as a finding.
- **Coherence was not used for final selection.** C_v and u_mass were computed
  across the K sweep but reproducibility was the selection criterion. Coherence
  rewards K values that are not stable across seeds, which is how the original
  K=12 choice arose.

## 8. Inference limitations

- Only 12 topics are tested, so Benjamini-Hochberg has little to work with.
  T4's raw clustered p is 0.0098 and its adjusted q is 0.118.
- Clusters are journals: 133 of them, but highly unequal in size. Wild cluster
  bootstrap-t is the appropriate procedure for few and unequal clusters, which is
  why it is primary, but it has low power in exactly this configuration.
- The within-journal permutation result (4 of 12 significant) is reported but is
  **not** a corpus-wide finding. It conditions away the between-journal variation
  that is plausibly where the real change lives.

## 9. What was not done

Stated explicitly so that no reader has to infer it from absence:

- **No external validity check.** Nothing confirms the three themes against an
  independent source. Expert coding of a sample of papers against the three
  themes is the single most valuable addition remaining, and it is cheap.
- **No held-out evaluation.** All models are fit on all 987 papers. There is no
  train/test split and no out-of-sample perplexity comparison.
- **No author-level or multiway clustering** (see section 1).
- **No pre-registration.** Nothing in this project was registered in advance of
  data access.
- **No hierarchical or dynamic topic model.** Structural topic models and dynamic
  LDA would address the temporal question directly and were not attempted.
- **No sensitivity analysis on the stopword list.** The removal of `blockchain`
  and `technology` was not varied.

## 10. Honest summary

The computational work is sound, reproducible and extensively self-audited. Four
rounds of validation found and corrected five substantive defects, three of
which changed conclusions. That is the strength of this project.

The weaknesses are not computational. They are that the model was chosen after
seeing results, several decisive thresholds were set after seeing results, the
dependence structure of the data cannot be modelled with the fields available,
and nothing outside the pipeline has ever checked whether the three themes are
real. Passing every internal check establishes internal consistency. It does not
establish validity, and the two should not be confused in the write-up.
