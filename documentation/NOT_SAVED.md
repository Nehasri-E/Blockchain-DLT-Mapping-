# Artifacts that were never written to disk

Stated explicitly rather than reconstructed, so an auditor knows the difference
between "missing from this package" and "never existed".

Nothing in this list was fabricated or recreated for the handover.

---

## 1. The Corpus A (abstract-only) trained model

**Status: never serialized.**

`06_interpret.py` fits the Corpus A robustness model in memory, uses it to
compute the similarity matrix, and discards it. Only its derived outputs survive:
`outputs/corpus_robustness.csv` and `figures/11_corpus_robustness.png`.

This is recoverable exactly. The model uses `LdaModel` with `random_state=42`,
which is deterministic, so re-running `06_interpret.py` refits it bit-identically.
To persist it, add one line after the fit:

```python
model_a.save("models/lda_corpus_a_k12.model")
```

It was left out of this package rather than refitting, because refitting would
have meant re-running stage 6, and the instruction was to preserve the existing
outputs untouched. Say the word and I will add it.

## 2. The 312 individual models from the tuning grid

**Status: never serialized, by design.**

`04_tune.py` fits each model, records C_v, u_mass and log perplexity to
`outputs/tuning_results.csv`, and releases it. Keeping 312 gensim models would
have been roughly 10 GB for no analytical gain.

Not exactly recoverable: the grid uses `LdaMulticore`, which is not deterministic
(see README section 1.2). Re-running reproduces the metric table to within seed
noise, not bit-identically.

## 3. Jupyter notebooks

**Status: none ever existed.**

The entire analysis was written and run as plain `.py` scripts. There is no
notebook to hand over. If you want one, the scripts convert cleanly with
`jupytext --to notebook scripts/0*.py`.

## 4. scikit-learn vectorizer objects

**Status: none ever existed.**

The guide PDF suggested `sklearn.feature_extraction.text.CountVectorizer` plus
`sklearn.decomposition.LatentDirichletAllocation`. That path was **not taken**.
The pipeline is gensim end to end, so the vectorizer is a
`gensim.corpora.Dictionary`, not an sklearn object. There is no `.pkl` of a
fitted `CountVectorizer` or `TfidfVectorizer` because none was ever created.

The equivalents are supplied:

| sklearn concept | gensim equivalent in this package |
|---|---|
| fitted `CountVectorizer` | `models/dictionary_corpus_{a,b}.dict` (and `.txt`) |
| `X` document-term matrix | `data/processed/corpus_{a,b}_bow.mm` (sparse Matrix Market) |
| `vectorizer.get_feature_names_out()` | `models/dictionary_corpus_{a,b}.txt`, column 2 |
| `lda.transform(X)` | `outputs/document_topic_distributions_k12.csv` |
| `lda.components_` | `outputs/topic_word_distributions_k12.csv` |

No TF-IDF weighting was applied anywhere. LDA is a model of raw counts, and the
guide is correct on that point.

## 5. Intermediate text states inside stage 3

**Status: never persisted; they are in-memory locals.**

`03_preprocess.py` passes text through normalize, then lemmatize, then bigram,
then trigram. Only the final post-trigram token lists are saved (in
`corpora.pkl`, exported to `data/processed/corpus_{a,b}_tokens.jsonl`). Not
saved:

- normalized-but-not-lemmatized text
- lemmatized-but-pre-n-gram tokens
- the fitted `Phrases` / `Phraser` objects themselves
- spaCy `Doc` objects

All are deterministic functions of `lda_master_capped.parquet`, so re-running
stage 3 reproduces each exactly. To capture any of them, insert a write between
the relevant steps.

## 6. A dense document-term matrix

**Status: never created.**

987 x 3,335 as dense float64 is about 26 MB, which is tolerable, but nothing in
the pipeline needed it. The sparse Matrix Market file at
`data/processed/corpus_b_bow.mm` is the full equivalent. To densify:

```python
from gensim.matutils import corpus2dense
from gensim.corpora import MmCorpus, Dictionary
c = MmCorpus("data/processed/corpus_b_bow.mm")
d = Dictionary.load("models/dictionary_corpus_b.dict")
X = corpus2dense(c, num_terms=len(d)).T     # (987, 3335)
```

## 7. Verification screenshots

**Status: ephemeral, not part of the analysis.**

During the session a headless-browser screenshot of `pyldavis_corpus_b.html` was
taken to confirm the page renders offline. It was a throwaway check written to a
temp path, not an analysis output. The page itself is in
`figures/pyldavis_corpus_b.html` and is self-contained, so you can regenerate the
view by opening it.

## 8. The source Excel workbook

**Status: never supplied to the analysis.**

The guide PDF references `Copy of papers_cleaned_v14.xlsx` (1,010 rows, 38
columns, with quality-control flags). That workbook was **never provided**. The
analysis received only `EDA_INPUT.csv` (1,003 rows, 10 usable columns) and
`LDA_INPUT.csv`, both in `data/raw/`.

This is the single largest gap in the handover and it is an input gap, not an
output gap. The QC columns `Analysis_Ready`, `Duplicate_Record`,
`Needs_Manual_Review`, `Mojibake_In_Title`, `Abstract_Found`, `Conclusion_Found`,
`Cleaning_Action` and `Fetch_Audit` exist only in that workbook. If you can
supply it, the inclusion and exclusion logic in `01_build_corpus.py` could be
cross-checked against the prior manual QC instead of relying solely on
text-derived rules. See README section 5.3.
