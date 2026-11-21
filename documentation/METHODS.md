# Blockchain and DLT Research Landscape: LDA Pipeline

Reproducible pipeline. Run the scripts in numeric order. Stage 4 is resumable and
takes a time budget in seconds as its only argument.

```
python 01_build_corpus.py
python 02_eda.py
python 03_preprocess.py
python 04_tune.py 600      # repeat until it reports 312/312
python 05_select.py
python 06_interpret.py
```

Environment: Python 3.13, gensim 4.4.0, spaCy 3.8.18 with `en_core_web_sm`,
pyLDAvis 3.4.1, pandas, matplotlib, ftfy.

## Stage 1: corpus construction

Source file is `EDA_INPUT.csv`. The 3-column `LDA_INPUT.csv` was audited and
discarded: its rows are offset by one from the metadata file after row 314, and a
title join showed its Abstract and Conclusion text to be byte identical to the
metadata file's. It contributes nothing and introduces a misalignment risk.

Raw columns are never overwritten. Cleaned text is written to new columns.

| Step | Effect |
|---|---|
| Hybrid UTF-8 / CP1252 decode | Source is a mixed byte stream; plain `read_csv` raises `UnicodeDecodeError` at byte 254,190 |
| `ftfy` repair, up to 3 passes | Mojibake-suspect cells 77 to 0. Multiple passes handle double encoding |
| Drop Excel spill rows | 3 rows. One conclusion hit Excel's 32,767-character cell limit and overflowed into adjacent rows |
| Drop correction notices | 3 rows matching `correction to`, `corrigendum`, `erratum`, `retraction` |
| Drop duplicates | 5 rows, by DOI then normalized title. Longer-conclusion record is kept |
| Drop short abstracts | 6 remaining rows under 50 words |
| Journal canonicalization | 155 spellings to 133, by most frequent casing per lowercase key |
| Paper_ID | `Row_UID` where present, generated otherwise. Uniqueness asserted in code |

Reconciliation: 1,003 raw to **987 analysis-ready papers**.

## Stage 3: preprocessing

Lowercasing, URL and DOI stripping, publisher boilerplate removal, punctuation
stripping with intra-word hyphens preserved.

Lemmatization uses spaCy with POS filtering, keeping NOUN, PROPN, ADJ and VERB.
A lemma repair map corrects `datum` to `data` and `cryptocurrencie` to
`cryptocurrency`. This matters: before the fix, `data` (153) and `datum` (915)
were competing as separate vocabulary entries, splitting the weight of one of the
corpus's most important terms.

Stopwords are the moderate list: standard English plus academic boilerplate
(paper, study, findings, propose, approach, contribution and similar) plus
`blockchain` itself, which appears in nearly every document and discriminates
nothing. Domain terms carrying real signal are explicitly protected: supply,
chain, technology, distributed, ledger, data, system, information, network,
platform, contract, token.

Bigrams and trigrams via gensim `Phrases`. 719 phrases detected in Corpus B,
including `supply_chain_management`, `smart_contract`, `dynamic_capability`,
`agri_food_supply_chain`.

Vocabulary filtered with `no_below=5, no_above=0.50`: 10,873 to 3,335 terms.

**Conclusion cap: 250 words.** Conclusions carry 77.3 percent of Corpus B tokens
uncapped (median 502 words against 189 for abstracts). The cap brings that to
55 percent so document length does not confound topic prevalence. It truncates
800 of 987 papers. This was an assumption, not a user decision, and is the single
parameter most worth revisiting: change `CONCLUSION_CAP` in `03_preprocess.py`.

## Stage 4: model selection

312 models: K in {4,5,6,7,8,9,10,11,12,14,16,18,20}, 4 prior settings
(alpha symmetric/asymmetric by eta symmetric/auto), 3 random seeds, on both
corpora. Reported metrics are C_v coherence and log perplexity, each averaged
over seeds with standard deviations.

C_v does not identify an optimal K here. The curve is flat from K=4 to K=12 and
the differences are inside seed noise. K was therefore selected on topic
distinctiveness and interpretability, both of which point to K=12:

| K | Mean pairwise Jensen-Shannon | Closest topic pair |
|---|---|---|
| 6 | 0.789 | 0.708 |
| 12 | 0.852 | 0.752 |

Final model: Corpus B, K=12, alpha symmetric, eta auto, 20 passes, 200
iterations, seed 42.

**Reproducibility note.** The final models use gensim `LdaModel`, not
`LdaMulticore`. `LdaMulticore` applies worker updates asynchronously, so
`random_state` does not make it deterministic: two identical runs produced topic
prevalences differing by up to 1 percentage point. `LdaModel` is bit-identical
across runs, verified by diffing `topic_summary.csv` and `topic_trends.csv` over
two full pipeline executions. `04_tune.py` still uses `LdaMulticore` for speed,
which is acceptable because its output is an average over seeds with reported
variance, not a result anyone cites.

## Stage 6: interpretation and longitudinal analysis

Topic labels are assigned by hand from the top 15 keywords and the 3 most
representative papers per topic. They are interpretation, not model output.

Trends are OLS slopes of mean topic share on year, restricted to 2019 to 2025.
2017 (6 papers), 2018 (7) and 2026 (18, partial year) train the model but are
excluded from the trend window, where small-sample noise would read as real
movement. Significance is p < 0.05 and is marked on every trend panel.

Robustness: a Corpus A (title plus abstract) model at the same K is fitted and
its topic-word distributions matched to Corpus B's by cosine similarity.

## Outputs

| File | Contents |
|---|---|
| `papers_with_topics.csv` | Every paper with Paper_ID, metadata and all 12 topic proportions |
| `topic_summary.csv` | Labels, prevalence, dominant-paper counts, top keywords |
| `topic_trends.csv` | OLS slope, p-value, 2019 and 2025 share, direction, significance |
| `topic_prevalence_by_year.csv` | Mean topic share per year |
| `representative_papers.csv` | Top 3 papers per topic by topic share |
| `corpus_robustness.csv` | Corpus B to Corpus A topic similarity matrix |
| `quality_control.csv` | Per-record QC flags for all 1,000 parsed records |
| `reconciliation.csv` | Row counts at each exclusion step |
| `tuning_results.csv` | All 312 models with C_v, u_mass and log perplexity |
| `pyldavis_corpus_b.html` | Interactive topic map, self-contained, no network needed |
| `figures/01` to `figures/11` | Publication charts at 300 DPI |

## Known limitations

1. The corpus is blockchain in operations and supply chain management, not the
   general blockchain and DLT landscape. Top bigrams are `supply chain` (3,123)
   and `blockchain technology` (2,134); top journals are IEEE TEM, TFSC, IJPR,
   IJPE, Annals of OR and TRE. Consensus mechanisms, DeFi protocols and
   cryptography are thinly represented. Scope the paper's claims accordingly.
2. C_v does not discriminate between K values here, so K=12 rests on
   distinctiveness and interpretability. State this openly rather than presenting
   a coherence peak that does not exist.
3. The 250-word conclusion cap is an assumption.
4. Topic prevalence measures research attention, not enterprise adoption.
5. One paper's conclusion was truncated at source by Excel's cell limit and is
   flagged `Conclusion_Truncated` in `quality_control.csv`.
