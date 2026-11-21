# LDA_Project: Blockchain and DLT Research Landscape Topic Model

Handover package for independent technical audit, reproduction and continuation.

Everything here was produced in a single analysis session. No result in this
package has been regenerated, recomputed or altered for the handover. The only
files created at packaging time are serialization exports read directly out of
the saved pickles (see `scripts/07_export_artifacts.py`) and the four
documentation files listed at the bottom of this README.

**Headline result:** 987 papers, Corpus B (title + abstract + capped conclusion),
K=12 topics, gensim `LdaModel`, alpha symmetric, eta auto, 20 passes, 200
iterations, `random_state=42`.

---

## 1. Integrity statement: read this first

### 1.1 The saved model provably reproduces the delivered outputs

`scripts/07_export_artifacts.py` loads `models/candidates.pkl`, recomputes every
document-topic distribution, and asserts them against the delivered CSVs. It was
run and passed:

```
topic prevalence matches topic_summary.csv : True
dominant-topic counts match                : True
per-document distributions match           : True
```

Re-run that script to re-verify. It will fail loudly if any artifact has drifted.

### 1.2 Determinism

The final models use `gensim.models.LdaModel` (single process). `LdaMulticore`
was tried first and **discarded**: it applies worker updates asynchronously, so
`random_state` does not pin the output. Two identical runs produced topic
prevalences differing by up to one percentage point. With `LdaModel`, two full
pipeline executions produced byte-identical `topic_summary.csv` and
`topic_trends.csv`.

`scripts/04_tune.py` still uses `LdaMulticore`. This is deliberate and
acceptable: its output is a seed-averaged metric table with reported standard
deviations, not a cited point estimate. An auditor re-running stage 4 should
expect C_v values to differ in the third decimal place. Stages 5 and 6 are exact.

### 1.3 Which model file reproduces the document-topic matrix exactly

This package ships the final model in two formats. **Their parameters are
bit-identical** (`expElogbeta` and `state.sstats` compare exactly equal), but
they do not behave identically when you re-run inference:

| Load path | Max deviation from `outputs/document_topic_distributions_k12.csv` |
|---|---|
| `models/candidates.pkl` (pickle) | 5.0e-07, which is just the CSV's 6-decimal rounding |
| `models/lda_corpus_b_k12.model` (gensim native) | 2.4e-02 on the worst single cell, mean 1.6e-05 |

The cause is in gensim, not in this package. `LdaModel.get_document_topics`
initializes its variational gamma from `self.random_state`, a **mutable**
`RandomState` object whose internal state advances on every call. Pickling
preserves that object's exact state; `LdaModel.save()`/`load()` does not.
Reseeding with `RandomState(42)` after loading does **not** fix it, because the
state when the delivered CSV was written was mid-stream, not freshly seeded.

Practical guidance for an auditor:

- **The delivered CSVs are authoritative.** `papers_with_topics.csv` and
  `document_topic_distributions_k12.csv` are the theta of record. Read them;
  do not recompute them and expect the last decimal to agree.
- **To reproduce theta exactly**, load `models/candidates.pkl`, not the `.model`
  file. `scripts/07_export_artifacts.py` does this and asserts agreement.
- **To inspect the model** (topic-word distributions, top terms, new unseen
  documents), either file is fine. Beta is deterministic and identical in both.
- The effect is a property of per-document variational inference, not of
  training. It does not affect the topic-word matrix, the topic labels, the
  prevalence figures to four decimal places, or any conclusion in the analysis.

### 1.4 Known vocabulary defect, found and fixed mid-analysis

spaCy lemmatized `data` to `datum`, which collided with a separately surviving
`data` token: 915 and 153 occurrences competing as two vocabulary entries for one
concept. Same for `cryptocurrency` (287) and `cryptocurrencie` (268). A lemma
repair map in `03_preprocess.py` fixes this.

The full 312-model tuning grid was re-run after the fix. The pre-fix grid is
preserved as `outputs/tuning_results_SUPERSEDED_pre_lemma_fix.csv` for audit
purposes only. **Do not use it.** `outputs/tuning_results.csv` is authoritative.

---

## 2. How to reproduce

### 2.1 Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2.2 Paths must be edited before running

The scripts are preserved **verbatim** as they ran, so they contain absolute
paths from the original environment. Change these six constants, then run:

| Script | Constant | Original value | Set to |
|---|---|---|---|
| `01_build_corpus.py` | `RAW` | `/root/.claude/uploads/.../d42072f9-EDA_INPUT.csv` | `<pkg>/data/raw/EDA_INPUT.csv` |
| `01_build_corpus.py` | `OUT` | `/home/claude/blockchain_lda/data` | `<pkg>/data/processed` |
| `02,03,04,05,06,07` | `DATA` | `/home/claude/blockchain_lda/data` | `<pkg>/data/processed` |
| `02_eda.py`, `05_select.py` | `FIG` | `/mnt/user-data/outputs/figures` | `<pkg>/figures` |
| `06_interpret.py` | `OUT` | `/mnt/user-data/outputs` | `<pkg>/outputs` |
| `07_export_artifacts.py` | `PKG` | `/home/claude/LDA_Project` | `<pkg>` |

Note that `01_build_corpus.py` writes `lda_master.*` and the QC files into its
`OUT` directory, whereas this package splits them across `data/interim/` and
`data/processed/`. Point `OUT` at one directory and let it write there; the split
in this package is purely for human navigation.

```bash
python scripts/01_build_corpus.py
python scripts/02_eda.py
python scripts/03_preprocess.py
python scripts/04_tune.py 600      # resumable; repeat until it reports 312/312
python scripts/05_select.py
python scripts/06_interpret.py
python scripts/07_export_artifacts.py
```

Approximate runtimes on 2 cores: stage 1 about 20s, stage 2 about 15s, stage 3
about 25s, **stage 4 about 26 minutes**, stage 5 about 3 minutes, stage 6 about
2 minutes.

### 2.3 Skipping the expensive stage

To inspect the final model without re-running anything:

```python
from gensim.models import LdaModel
from gensim.corpora import Dictionary, MmCorpus
m = LdaModel.load("models/lda_corpus_b_k12.model")
d = Dictionary.load("models/dictionary_corpus_b.dict")
c = MmCorpus("data/processed/corpus_b_bow.mm")
```

Row order in `corpus_b_bow.mm` maps to papers via
`data/processed/corpus_b_row_index.csv`.

---

## 3. Project structure

```
LDA_Project/
├── README.md                    this file
├── requirements.txt             pinned library versions
├── data/
│   ├── raw/                     original inputs, unmodified
│   ├── interim/                 stage 1 audit trail
│   └── processed/               modelling corpus, dictionaries, BoW, tokens
├── scripts/                     01-07, verbatim as executed
├── models/                      trained LDA models, dictionaries, config
├── outputs/                     all tabular results
├── figures/                     11 PNGs at 300 DPI, plus interactive pyLDAvis
└── documentation/               METHODS.md, FILE_MANIFEST.md, NOT_SAVED.md,
                                 original guide PDF
```

---

## 4. How the files connect

```
data/raw/EDA_INPUT.csv
    |
    | 01_build_corpus.py   decode, repair mojibake, dedupe, filter
    v
data/processed/lda_master.{csv,parquet}      987 analysis-ready papers
data/interim/quality_control.csv             per-record flags, all 1,000 records
data/interim/reconciliation.csv              1,003 -> 987 audit trail
    |
    | 02_eda.py            descriptive statistics
    v
figures/01-06, outputs/eda_summary.csv, length_percentiles.csv,
top_unigrams.csv, top_bigrams.csv
    |
    | 03_preprocess.py     cap, normalize, lemmatize, n-grams, vectorize
    v
data/processed/corpora.pkl                   tokens + Dictionary + BoW, both corpora
data/processed/lda_master_capped.parquet     adds Conclusion_Capped, Text_A, Text_B
models/dictionary_corpus_{a,b}.{dict,txt}    exported by 07
data/processed/corpus_{a,b}_bow.mm           exported by 07
data/processed/corpus_{a,b}_tokens.jsonl     exported by 07
    |
    | 04_tune.py           312 models, seed-averaged metrics
    v
outputs/tuning_results.csv
    |
    | 05_select.py         fit K=6 and K=12 candidates, distinctiveness
    v
models/candidates.pkl, figures/07_model_selection.png
models/lda_corpus_b_k{6,12}.model            exported by 07
    |
    | 06_interpret.py      label, assign, trend, journal, robustness
    v
outputs/topic_summary.csv, topic_trends.csv, topic_prevalence_by_year.csv,
representative_papers.csv, papers_with_topics.csv, corpus_robustness.csv
figures/08-11, figures/pyldavis_corpus_b.html
```

`Paper_ID` is the join key throughout. It is `Row_UID` from the source file where
present (870 `Scopus:*`, 129 `WoS:*`), with one generated `GEN:1`. Uniqueness is
asserted in `01_build_corpus.py`.

---

## 5. Analysis decisions and rationale

### 5.1 Which source file was used, and why the other was discarded

Two files were supplied. **`LDA_INPUT.csv` was discarded entirely.**

It is a 3-column file (Article Title, Abstract, Conclusion) with no join key. Its
rows are **offset by one** from the metadata file after row 314: positional
alignment holds for rows 0 to 313 and then fails, so `EDA_INPUT[317:]` equals
`LDA_INPUT[316:]`. Zipping the two files by position, which is the obvious thing
to do, would have attached the wrong metadata to 687 of 1,002 papers.

A title join showed all 1,001 non-empty titles match and the Abstract and
Conclusion text is byte identical in both files. `LDA_INPUT.csv` therefore
contributes no information and carries a misalignment risk. It is included in
`data/raw/` for completeness only; nothing reads it.

### 5.2 Columns used

| Column | Role |
|---|---|
| `Article Title (unified)` | text input (as `Title_Clean`) and identification |
| `Abstract` | text input (as `Abstract_Clean`) |
| `Conclusion` | text input (as `Conclusion_Clean`, capped at 250 words) |
| `Year` | longitudinal analysis |
| `Source title` | journal analysis (canonicalized to `Journal`) |
| `DOI_URL` | duplicate detection (as `DOI`) |
| `Row_UID` | `Paper_ID` |
| `Authors` | retained as metadata, not used in any analysis |
| `CABS_Rank`, `ABDC_Rank` | retained as metadata for future journal-quality stratification |

Only `Title_Clean`, `Abstract_Clean` and `Conclusion_Capped` enter the model.
Everything else is metadata used after the fact.

### 5.3 Columns excluded

`Unnamed: 10` and `Unnamed: 11` were dropped: both entirely empty apart from one
stray value belonging to the Excel spill described below.

**Important divergence from the guide PDF.** `documentation/LDA_Analysis_Guide.pdf`
audits a different, richer file (`Copy of papers_cleaned_v14.xlsx`, 1,010 rows,
38 columns) containing quality-control columns `Analysis_Ready`,
`Duplicate_Record`, `Needs_Manual_Review`, `Mojibake_In_Title`, `Abstract_Found`,
`Conclusion_Found`, `Cleaning_Action` and `Fetch_Audit`, plus a reported 18
repeated DOIs. **None of those columns exist in the supplied `EDA_INPUT.csv`.**
They did not survive the export. Every inclusion and exclusion decision in this
pipeline was therefore re-derived from the text itself rather than inherited from
prior manual QC. The guide's figures (839 ready, 117 abstract-only, 41 damaged,
6 no abstract, 7 duplicates, 997 final) do not describe this dataset and should
not be cited against these outputs.

### 5.4 Preprocessing, step by step

| Step | What | Why |
|---|---|---|
| Hybrid UTF-8 / CP1252 decode | Byte-level fallback decoder | The source is a mixed byte stream. Plain `pd.read_csv` raises `UnicodeDecodeError` at byte 254,190 (0x92, a CP1252 right quote). A naive latin-1 read "works" but corrupts 210 cells instead of 77 |
| `ftfy.fix_text`, up to 3 passes | Mojibake repair | 77 suspect cells to 0. Repeated passes handle double encoding (`consumersÃ¢â¬â¢`) |
| Drop 3 spill rows | `Year` not parseable as a number | One conclusion is 32,746 characters, hitting Excel's 32,767-character cell ceiling. It overflowed into three phantom rows, putting the text fragment "Case 49) Charging technology licensing fees..." in the title column and "as a platform" in `Year` |
| Drop 3 correction notices | Title starts with correction / corrigendum / erratum / retraction | These are metadata records, not research papers. One was a near-duplicate of its own original article |
| Drop 5 duplicates | By `DOI` then normalized title; longer-conclusion record kept | 5 repeated DOIs found, not the 18 the guide reports. One pair is the same paper ingested from both WoS and Scopus under different encodings |
| Drop 6 short abstracts | Fewer than 50 words after cleaning | Too little text for a meaningful topic posterior |
| Journal canonicalization | Most frequent casing per lowercase key | 155 spellings collapse to 133. IEEE TEM appeared as both "IEEE Transactions on Engineering Management" (99) and "IEEE TRANSACTIONS ON ENGINEERING MANAGEMENT" (23), which would have split the top journal in the bar chart |
| Conclusion cap at 250 words | Truncate, keep head | See 5.5 |
| Lowercase, strip URLs/DOIs/publisher boilerplate | Regex | Removes "(c) 2023 Elsevier B.V. All rights reserved" style trailers that otherwise form a spurious topic |
| Intra-word hyphen preservation | `proof-of-work` survives tokenization | Domain compounds would otherwise fragment |
| spaCy lemmatization, POS-filtered | Keep NOUN, PROPN, ADJ, VERB | POS-aware lemmatization is more accurate than NLTK WordNet without tags. Dropping other POS removes function words structurally rather than by list |
| Lemma repair map | `datum`→`data`, `cryptocurrencie`→`cryptocurrency`, and abbreviation expansion (`bct`→`blockchain_technology`, `dlt`, `scm`, `ai`) | See 1.3 |
| Stopword removal | Standard English + academic boilerplate + `blockchain` | `blockchain` appears in nearly every document and discriminates nothing. 12 domain terms are explicitly protected from the generic list: supply, chain, technology, distributed, ledger, data, system, information, network, platform, contract, token |
| Bigrams and trigrams | gensim `Phrases`, min_count 12 / threshold 12.0, then 8 / 10.0 | 719 phrases in Corpus B, including `supply_chain_management`, `smart_contract`, `dynamic_capability`, `agri_food_supply_chain` |
| `filter_extremes(no_below=5, no_above=0.50)` | Vocabulary 10,873 to 3,335 | Drops hapax noise and corpus-ubiquitous terms |
| Drop documents with fewer than 10 BoW entries | 0 dropped from Corpus B, 1 from Corpus A | Guards against degenerate documents |

Raw columns are never overwritten. Cleaned text lives in new columns alongside.

### 5.5 The conclusion cap, and why it is the weakest assumption here

Conclusions are a median 502 words against 189 for abstracts, and carry **77.3%**
of all Corpus B tokens uncapped. Because LDA's document-topic posterior is
length sensitive, this makes conclusion length a confound in every topic
prevalence number.

Measured effect of each candidate cap:

| Cap | Papers truncated | Conclusion token share | Median document |
|---|---|---|---|
| none | 0 (0%) | 77.3% | 715w |
| 250w | 800 (81%) | 55.0% | 442w |
| 500w | 494 (50%) | 67.4% | 649w |
| 1,000w | 176 (18%) | 74.1% | 715w |
| 1,275w (90th pct) | 97 (10%) | 75.3% | 715w |

The imbalance is structural, not driven by outliers, so percentile caps barely
move it. **250 words was chosen** to bring the share to roughly parity, on the
reasoning that conclusion sections lead with findings and the truncated tail is
mostly limitations and future-work boilerplate.

**This was an analyst assumption, not a user decision.** It is the single
parameter most worth re-testing. Change `CONCLUSION_CAP` in `03_preprocess.py`
and re-run stages 3 to 6.

### 5.6 How K was selected

312 models: K in {4,5,6,7,8,9,10,11,12,14,16,18,20} x 4 prior settings
(alpha symmetric/asymmetric x eta symmetric/auto) x 3 seeds {42, 7, 2024},
on both corpora. Metrics: C_v coherence, u_mass coherence, log perplexity.

**C_v does not identify an optimal K for this corpus.** Peak mean C_v is 0.4072
at K=4, but K values 4, 10, 11 and 12 are statistically indistinguishable at one
standard deviation, and the whole curve from K=4 to K=12 sits inside seed noise.
Log perplexity declines monotonically with K, as it almost always does, and is
not informative for selection.

K was therefore chosen on **topic distinctiveness and interpretability**:

| K | Mean pairwise Jensen-Shannon | Closest topic pair |
|---|---|---|
| 6 | 0.789 | 0.708 |
| 12 | 0.852 | 0.752 |

K=6 collapses distinguishable themes into broad, hard-to-name clusters. K=12
yields twelve topics that each map to a recognizable literature. K=12 also sits
inside the band of K values indistinguishable from the C_v peak, so nothing is
sacrificed on coherence.

**State this honestly in any write-up.** Do not present a coherence peak at K=12;
there is not one. The defensible claim is that coherence does not discriminate
and that distinctiveness and interpretability were used to break the tie.

### 5.7 Final hyperparameters

```
corpus        Corpus B, 987 documents, 3,335-term vocabulary
implementation gensim.models.LdaModel  (single process, deterministic)
num_topics    12
alpha         symmetric
eta           auto
passes        20
iterations    200
chunksize     200
random_state  42
eval_every    None
```

Also in machine-readable form at `models/model_config.json`.

### 5.8 How topics were labelled

Labels were assigned by hand from the top 15 keywords plus the 3 highest-scoring
papers per topic (`outputs/representative_papers.csv`). **They are
interpretation, not model output**, and an auditor is free to disagree. The raw
material for relabelling is `outputs/topic_top30_terms_k12.csv` and
`outputs/topic_word_distributions_k12.csv`. Labels live in a single `LABELS` dict
at the top of `06_interpret.py`.

| # | Label | Prevalence |
|---|---|---|
| 1 | Empirical Market and Disclosure Effects | 8.1% |
| 2 | Supply Chain Operations and Integration | 13.0% |
| 3 | FinTech, Payments and Financial Markets | 3.7% |
| 4 | Pricing and Channel Strategy | 14.2% |
| 5 | System Architecture and Security Design | 5.3% |
| 6 | Cryptocurrency Markets and Token Pricing | 8.6% |
| 7 | Smart Contract Protocols and Schemes | 6.5% |
| 8 | Sustainable Adoption and Firm Performance | 11.5% |
| 9 | Trust, Transparency and Sector Applications | 9.1% |
| 10 | Governance, Capabilities and Ecosystems | 9.5% |
| 11 | NFTs, Marketing and Research Agenda | 8.7% |
| 12 | Risk Management, Accounting and Credit | 2.0% |

### 5.9 Longitudinal analysis

Mean topic share per year, with OLS slope of share on year, restricted to
**2019 to 2025**. 2017 (6 papers), 2018 (7) and 2026 (18, a partial year) train
the model but are excluded from the trend window, where small-sample swings would
read as real movement. Significance is p < 0.05 and is marked on every panel of
`figures/09_topic_evolution.png`.

Only 4 of 12 topics move significantly: Pricing and Channel Strategy rising
(+1.87pp/yr, p=0.008), Sustainable Adoption and Firm Performance rising
(+1.10pp/yr, p=0.0001), Trust and Transparency declining (p=0.016), Smart
Contract Protocols declining (p=0.008). The other eight are flat.

Topic prevalence measures **research attention, not enterprise adoption**. Do not
conflate the two.

### 5.10 Robustness evaluation

A Corpus A model (title + abstract only, 986 documents) was fitted at the same
K=12 with the same priors and seed. Its topic-word distributions were padded to a
shared vocabulary and matched to Corpus B's by cosine similarity
(`outputs/corpus_robustness.csv`, `figures/11_corpus_robustness.png`).

**7 of 12 Corpus B topics have a clear abstract-only counterpart** (cosine >= 0.50;
mean best match 0.518). Five do not:

| Topic | Best match | Reading |
|---|---|---|
| 12 Risk Management, Accounting and Credit | 0.18 | conclusion-specific |
| 3 FinTech, Payments and Financial Markets | 0.35 | conclusion-specific |
| 1 Empirical Market and Disclosure Effects | 0.36 | conclusion-specific |
| 7 Smart Contract Protocols and Schemes | 0.40 | conclusion-specific |
| 11 NFTs, Marketing and Research Agenda | 0.48 | borderline |

This cuts both ways and should be reported as such. Conclusions genuinely carry
thematic content that abstracts do not. Equally, those five topics are partly an
artifact of conclusion-section conventions rather than of paper content. Both
readings belong in the limitations.

---

## 6. Limitations, assumptions and unresolved issues

1. **Corpus scope is narrower than "blockchain and DLT".** Top bigrams are
   `supply chain` (3,123) and `blockchain technology` (2,134); top journals are
   IEEE TEM, TFSC, IJPR, IJPE, Annals of OR and TRE. Consensus mechanisms, DeFi
   protocols and cryptography are thin. This is a **blockchain in operations and
   supply chain management** corpus. Scope the paper's title and claims to match.
2. **C_v does not discriminate.** K=12 rests on distinctiveness and
   interpretability, not on a coherence optimum. See 5.6.
3. **The 250-word conclusion cap is an analyst assumption.** See 5.5.
4. **Topic labels are subjective** and were assigned by one person. See 5.8.
5. **Five of twelve topics are conclusion-specific.** See 5.10.
6. **One conclusion was truncated at source** by Excel's 32,767-character cell
   limit. Flagged `Conclusion_Truncated` in `data/interim/quality_control.csv`.
   The original full text is unrecoverable from the supplied data.
7. **The guide PDF's quality-control columns are absent** from the supplied CSV,
   so prior manual QC could not be used. See 5.3.
8. **Stage 4's grid is not bit-reproducible** because it uses `LdaMulticore`. See 1.2.
9. **`Authors`, `CABS_Rank` and `ABDC_Rank` are carried but unanalyzed.** A
   journal-quality stratification of topic prevalence is the obvious next step
   and the data for it is already in `outputs/papers_with_topics.csv`.
10. **No train/test split or held-out perplexity.** Perplexity is reported
    in-sample. For a predictive claim this would need a holdout.

---

## 7. Documentation index

| File | Contents |
|---|---|
| `README.md` | this file: structure, reproduction, all analysis decisions |
| `documentation/METHODS.md` | condensed methods note written during the analysis |
| `documentation/FILE_MANIFEST.md` | every file, its size, producer and contents |
| `documentation/NOT_SAVED.md` | artifacts that were never written to disk, stated explicitly |
| `documentation/LDA_Analysis_Guide.pdf` | the original guide supplied as input; see 5.3 for where it diverges from the delivered data |
