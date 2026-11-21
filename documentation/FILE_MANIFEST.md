# File manifest

73 files. Every file in the package is listed. `Produced by` names
the script that wrote it, so each artifact can be traced to its source.

SHA-256 is the first 16 hex characters, enough to detect drift.

## Root

| File | Size | SHA-256 | Produced by | Contents |
|---|---|---|---|---|
| `README.md` | 22.3 KB | `7e2a2e8fed60bbee` | documentation | Full handover: structure, reproduction steps, every analysis decision and rationale |
| `requirements.txt` | 662 B | `b33bcc4856ae1bfc` | documentation | Pinned library versions and the spaCy model version |

## data/raw

| File | Size | SHA-256 | Produced by | Contents |
|---|---|---|---|---|
| `EDA_INPUT.csv` | 5.8 MB | `30c9e36aeefd3346` | supplied input | Original supplied metadata file. 1,003 rows x 12 columns. Mixed UTF-8/CP1252 byte stream; plain read_csv fails on it. THE source for the whole pipeline |
| `LDA_INPUT.csv` | 5.7 MB | `8f2580f76c377d1f` | supplied input | Original supplied 3-column file. DISCARDED: rows offset by one after row 314, no join key, text byte-identical to EDA_INPUT. Nothing reads it. Kept for completeness |

## data/interim

| File | Size | SHA-256 | Produced by | Contents |
|---|---|---|---|---|
| `all_records_audited.parquet` | 5.9 MB | `c4b0801173709308` | 01_build_corpus.py | All 1,000 parsed records with every derived flag and cleaned column, before the analysis-ready filter |
| `quality_control.csv` | 197.9 KB | `9161c560fa891b84` | 01_build_corpus.py | Per-record QC flags: Abstract_Valid, Conclusion_Valid, Duplicate_Record, Is_Correction, Mojibake_Residual, Conclusion_Truncated, Analysis_Ready |
| `reconciliation.csv` | 185 B | `a47b0079e3c9b997` | 01_build_corpus.py | Row-count audit trail, 1,003 raw down to 987 analysis-ready, one row per exclusion step |
| `stage1_log.txt` | 636 B | `198dada4a8aa65a9` | 01_build_corpus.py | Console log of stage 1 as it ran: decode, repair and exclusion counts |

## data/processed

| File | Size | SHA-256 | Produced by | Contents |
|---|---|---|---|---|
| `corpora.pkl` | 4.2 MB | `d7553aafbef82901` | 03_preprocess.py | Pickle holding both corpora: tokens, gensim Dictionary, BoW, paper_ids. The handoff object between stages 3, 4, 5 and 6 |
| `corpus_a_bow.mm` | 549.7 KB | `ac90b4abc3c9f3a0` | 07_export_artifacts.py | Corpus A (abstract-only) document-term matrix. 986 docs x 2,308 terms |
| `corpus_a_bow.mm.index` | 4.6 KB | `80fd10eb3488f55d` | 07_export_artifacts.py | Index sidecar for corpus_a_bow.mm |
| `corpus_a_row_index.csv` | 13.9 KB | `7cd161907cab79ac` | 07_export_artifacts.py | Maps corpus_a_bow.mm row order to Paper_ID |
| `corpus_a_tokens.jsonl` | 1.1 MB | `19f73b4e295c3b35` | 07_export_artifacts.py | Corpus A final token lists, keyed by Paper_ID |
| `corpus_b_bow.mm` | 997.7 KB | `e58fd86693a8b174` | 07_export_artifacts.py | Corpus B document-term matrix, sparse Matrix Market. 987 docs x 3,335 terms |
| `corpus_b_bow.mm.index` | 4.7 KB | `7af08dda38da12aa` | 07_export_artifacts.py | Index sidecar for corpus_b_bow.mm |
| `corpus_b_row_index.csv` | 13.9 KB | `b296cfa629d9e4bc` | 07_export_artifacts.py | Maps corpus_b_bow.mm row order to Paper_ID. Needed to join the matrix back to metadata |
| `corpus_b_tokens.jsonl` | 2.3 MB | `fbc48ee63006458c` | 07_export_artifacts.py | Corpus B final token lists, one JSON object per paper, keyed by Paper_ID |
| `lda_master.csv` | 12.7 MB | `21e1728288a53020` | 01_build_corpus.py | The 987-paper analysis corpus with cleaned text and metadata, CSV form |
| `lda_master.parquet` | 6.4 MB | `7f3e1047a501703f` | 01_build_corpus.py | Same as lda_master.csv, parquet form, read by stages 2 and 3 |
| `lda_master_capped.parquet` | 9.4 MB | `710e6466f7a26a20` | 03_preprocess.py | lda_master plus Conclusion_Capped (250w), Text_A and Text_B. Read by stage 6 |

## scripts

| File | Size | SHA-256 | Produced by | Contents |
|---|---|---|---|---|
| `01_build_corpus.py` | 7.0 KB | `71806afbf25b4d9c` | source code | Stage 1. Hybrid decode, ftfy repair, spill/correction/duplicate/short-abstract exclusion, journal canonicalization, Paper_ID assignment |
| `02_eda.py` | 9.1 KB | `d89d64bb58db43de` | source code | Stage 2. Summary stats, figures 01-06, length percentiles, raw n-gram counts |
| `03_preprocess.py` | 4.9 KB | `eb85a482e767b0a0` | source code | Stage 3. 250w cap, normalization, spaCy POS-filtered lemmatization, lemma repair map, stopwords, bigrams/trigrams, Dictionary and BoW |
| `04_tune.py` | 2.1 KB | `79efb73de93d14cd` | source code | Stage 4. Resumable 312-model grid. Takes a time budget in seconds as argv[1]. Uses LdaMulticore, NOT bit-reproducible |
| `05_select.py` | 4.2 KB | `943f9c4e5f48bc3a` | source code | Stage 5. Figure 07, fits K=6 and K=12 candidates with LdaModel, Jensen-Shannon distinctiveness |
| `06_interpret.py` | 11.8 KB | `2178f6d673ba4c01` | source code | Stage 6. Topic labels, assignment, prevalence, trends, journal heatmap, Corpus A robustness, pyLDAvis. Holds the LABELS dict |
| `07_export_artifacts.py` | 5.8 KB | `8de1b460b0731dce` | source code | Packaging only. Verifies the saved model against delivered outputs, then exports dictionaries, BoW, tokens, models and matrices. Creates no new analysis |
| `08_make_manifest.py` | 11.2 KB | `59e768e1b307a753` | source code | Packaging only. Generates documentation/FILE_MANIFEST.md with sizes and checksums. Creates no new analysis |

## models

| File | Size | SHA-256 | Produced by | Contents |
|---|---|---|---|---|
| `candidates.pkl` | 627.0 KB | `21e9996d41c4eb0c` | 05_select.py | Pickle of both fitted candidates as stage 5 wrote them. PROVENANCE ARTIFACT: the only load path that reproduces document_topic_distributions_k12.csv exactly, to 5e-07 (README 1.3) |
| `dictionary_corpus_a.dict` | 56.1 KB | `d864af744f9223fd` | 07_export_artifacts.py | Corpus A gensim Dictionary, 2,308 terms |
| `dictionary_corpus_a.txt` | 38.2 KB | `cb6aefa6846a89b3` | 07_export_artifacts.py | Corpus A vocabulary human-readable |
| `dictionary_corpus_b.dict` | 82.1 KB | `6edd5d00daea750c` | 07_export_artifacts.py | Corpus B gensim Dictionary, 3,335 terms. The vectorizer |
| `dictionary_corpus_b.txt` | 56.9 KB | `b9e59dfab0345660` | 07_export_artifacts.py | Corpus B vocabulary human-readable: id, term, document frequency |
| `lda_corpus_b_k12.model` | 19.6 KB | `ab348e0a538164ce` | 07_export_artifacts.py | THE FINAL MODEL in gensim native format. K=12, alpha symmetric, eta auto, 20 passes, 200 iterations, seed 42. Parameters bit-identical to candidates.pkl; use the pickle to reproduce theta exactly (README 1.3) |
| `lda_corpus_b_k12.model.expElogbeta.npy` | 156.5 KB | `fee0425d6d988316` | 07_export_artifacts.py | Final model: topic-word parameter array (gensim sidecar, required) |
| `lda_corpus_b_k12.model.id2word` | 106.8 KB | `29d64256c42456c7` | 07_export_artifacts.py | Final model: vocabulary mapping (gensim sidecar, required) |
| `lda_corpus_b_k12.model.state` | 170.0 KB | `77bf31eb0968f6f8` | 07_export_artifacts.py | Final model: sufficient statistics (gensim sidecar, required) |
| `lda_corpus_b_k6.model` | 19.6 KB | `8ae0c2a50edf5a71` | 07_export_artifacts.py | The rejected K=6 candidate, kept so the selection decision can be audited |
| `lda_corpus_b_k6.model.expElogbeta.npy` | 78.3 KB | `5da5761424f97916` | 07_export_artifacts.py | K=6 candidate sidecar |
| `lda_corpus_b_k6.model.id2word` | 106.8 KB | `29d64256c42456c7` | 07_export_artifacts.py | K=6 candidate sidecar |
| `lda_corpus_b_k6.model.state` | 91.9 KB | `61c5f749d2cdb427` | 07_export_artifacts.py | K=6 candidate sidecar |
| `model_config.json` | 2.5 KB | `31a81912c38597eb` | 07_export_artifacts.py | Machine-readable hyperparameters, seeds, preprocessing constants and selection rationale for all three model configurations |

## outputs

| File | Size | SHA-256 | Produced by | Contents |
|---|---|---|---|---|
| `corpus_robustness.csv` | 932 B | `d5f82a22ad5d8ad4` | 06_interpret.py | 12 x 12 cosine similarity, Corpus B topics against Corpus A topics |
| `document_topic_distributions_k12.csv` | 113.4 KB | `496de2f54fcf2682` | 07_export_artifacts.py | Full document-topic matrix, 987 x 12, keyed by Paper_ID. The model's complete theta |
| `eda_summary.csv` | 276 B | `905c4c395f072307` | 02_eda.py | Corpus-level descriptive statistics |
| `length_percentiles.csv` | 127 B | `d3dbb8f98cfd23f2` | 02_eda.py | Abstract and conclusion word counts at the 50th to 100th percentiles. The evidence behind the 250w cap |
| `papers_with_topics.csv` | 532.8 KB | `0cdc9e7d33740346` | 06_interpret.py | PRIMARY ANALYTICAL OUTPUT. Every paper with metadata, dominant topic, dominant share and all 12 topic proportions |
| `representative_papers.csv` | 6.5 KB | `b0b9265ea2773020` | 06_interpret.py | Top 3 papers per topic by topic share, with year and journal. Basis for the hand labels |
| `top_bigrams.csv` | 1.3 KB | `27a35569a826dbfb` | 02_eda.py | Top 60 raw bigrams before modelling stopwords were applied |
| `top_unigrams.csv` | 796 B | `697806cf8c4cb132` | 02_eda.py | Top 60 raw unigrams before modelling stopwords were applied |
| `topic_prevalence_by_year.csv` | 799 B | `4444b82dc84b2a37` | 06_interpret.py | Mean topic share per year, 2019-2025. The data behind figure 09 |
| `topic_summary.csv` | 1.8 KB | `6232107dd2e76a45` | 06_interpret.py | Per topic: label, prevalence, count of papers where dominant, top 10 keywords |
| `topic_top30_terms_k12.csv` | 8.1 KB | `8bf927103043d875` | 07_export_artifacts.py | Top 30 terms per topic with probabilities. Raw material for relabelling |
| `topic_trends.csv` | 1.1 KB | `d42a32459657e0b3` | 06_interpret.py | Per topic: OLS slope on year, p-value, 2019 and 2025 share, direction, significance flag |
| `topic_word_distributions_k12.csv` | 559.2 KB | `9b337525f7599e43` | 07_export_artifacts.py | Full topic-word matrix, 12 x 3,335. The model's complete beta |
| `tuning_results.csv` | 15.8 KB | `664a3cf47e45ac2e` | 04_tune.py | AUTHORITATIVE grid. 312 rows: corpus, k, alpha, eta, seed, c_v, u_mass, log_perplexity |
| `tuning_results_SUPERSEDED_pre_lemma_fix.csv` | 15.7 KB | `ef8610c437078c93` | 04_tune.py | SUPERSEDED. The same grid run before the data/datum lemma fix. Kept for audit only. DO NOT USE |

## figures

| File | Size | SHA-256 | Produced by | Contents |
|---|---|---|---|---|
| `01_papers_by_year.png` | 122.8 KB | `916c638b7ee57514` | 02_eda.py | Publication counts by year, trend window highlighted |
| `02_top_journals.png` | 279.2 KB | `624d6df90325ab09` | 02_eda.py | Top 15 source journals after canonicalization |
| `03_text_length_distributions.png` | 120.7 KB | `175d25955200fc82` | 02_eda.py | Abstract vs conclusion word-count histograms with medians |
| `04_text_availability.png` | 131.3 KB | `736a6034ca5d2afe` | 02_eda.py | Text availability and every exclusion reason |
| `05_top_unigrams.png` | 203.7 KB | `7570c69ff756ff5e` | 02_eda.py | Top 20 raw unigrams |
| `06_top_bigrams.png` | 271.1 KB | `91e154d76a01eaad` | 02_eda.py | Top 20 raw bigrams |
| `07_model_selection.png` | 305.5 KB | `1e260d073376ea4f` | 05_select.py | C_v and log perplexity vs K, both corpora, seed error bars |
| `08_topic_prevalence.png` | 292.8 KB | `fac89ee619d03c2c` | 06_interpret.py | Topic prevalence across the corpus |
| `09_topic_evolution.png` | 623.8 KB | `d7394b5165d3256a` | 06_interpret.py | Small multiples, topic share 2019-2025, OLS trend and significance per panel |
| `10_topic_by_journal.png` | 306.9 KB | `c3eb5fa3d47c6fae` | 06_interpret.py | Mean topic share by journal, top 12 journals |
| `11_corpus_robustness.png` | 273.4 KB | `747791a6f70f3307` | 06_interpret.py | Corpus B to Corpus A topic similarity heatmap |
| `pyldavis_corpus_b.html` | 417.6 KB | `93e7624172873fe5` | 06_interpret.py | Interactive topic map. Self-contained, d3 and ldavis inlined, works offline |

## documentation

| File | Size | SHA-256 | Produced by | Contents |
|---|---|---|---|---|
| `FILE_MANIFEST.md` | 12.0 KB | `b70dff58f4678735` | documentation | This file |
| `LDA_Analysis_Guide.pdf` | 1.3 MB | `c000b947d4669349` | supplied input | The original guide supplied as input. Audits a DIFFERENT file than the one delivered; see README section 5.3 |
| `METHODS.md` | 6.9 KB | `f00324be48c10630` | documentation | Condensed methods note written during the analysis |
| `NOT_SAVED.md` | 5.2 KB | `f7c49566876291a2` | documentation | Artifacts that were never written to disk, stated explicitly |

**Total: 73 files, 62.8 MB uncompressed.**
