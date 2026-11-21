import hashlib
from pathlib import Path

PKG = Path("/home/claude/LDA_Project")

DESC = {
    "README.md": ("documentation", "Full handover: structure, reproduction steps, every analysis decision and rationale"),
    "requirements.txt": ("documentation", "Pinned library versions and the spaCy model version"),
    "data/raw/EDA_INPUT.csv": ("input", "Original supplied metadata file. 1,003 rows x 12 columns. Mixed UTF-8/CP1252 byte stream; plain read_csv fails on it. THE source for the whole pipeline"),
    "data/raw/LDA_INPUT.csv": ("input", "Original supplied 3-column file. DISCARDED: rows offset by one after row 314, no join key, text byte-identical to EDA_INPUT. Nothing reads it. Kept for completeness"),
    "data/interim/all_records_audited.parquet": ("01", "All 1,000 parsed records with every derived flag and cleaned column, before the analysis-ready filter"),
    "data/interim/quality_control.csv": ("01", "Per-record QC flags: Abstract_Valid, Conclusion_Valid, Duplicate_Record, Is_Correction, Mojibake_Residual, Conclusion_Truncated, Analysis_Ready"),
    "data/interim/reconciliation.csv": ("01", "Row-count audit trail, 1,003 raw down to 987 analysis-ready, one row per exclusion step"),
    "data/interim/stage1_log.txt": ("01", "Console log of stage 1 as it ran: decode, repair and exclusion counts"),
    "data/processed/lda_master.csv": ("01", "The 987-paper analysis corpus with cleaned text and metadata, CSV form"),
    "data/processed/lda_master.parquet": ("01", "Same as lda_master.csv, parquet form, read by stages 2 and 3"),
    "data/processed/lda_master_capped.parquet": ("03", "lda_master plus Conclusion_Capped (250w), Text_A and Text_B. Read by stage 6"),
    "data/processed/corpora.pkl": ("03", "Pickle holding both corpora: tokens, gensim Dictionary, BoW, paper_ids. The handoff object between stages 3, 4, 5 and 6"),
    "data/processed/corpus_b_bow.mm": ("07", "Corpus B document-term matrix, sparse Matrix Market. 987 docs x 3,335 terms"),
    "data/processed/corpus_b_bow.mm.index": ("07", "Index sidecar for corpus_b_bow.mm"),
    "data/processed/corpus_b_tokens.jsonl": ("07", "Corpus B final token lists, one JSON object per paper, keyed by Paper_ID"),
    "data/processed/corpus_b_row_index.csv": ("07", "Maps corpus_b_bow.mm row order to Paper_ID. Needed to join the matrix back to metadata"),
    "data/processed/corpus_a_bow.mm": ("07", "Corpus A (abstract-only) document-term matrix. 986 docs x 2,308 terms"),
    "data/processed/corpus_a_bow.mm.index": ("07", "Index sidecar for corpus_a_bow.mm"),
    "data/processed/corpus_a_tokens.jsonl": ("07", "Corpus A final token lists, keyed by Paper_ID"),
    "data/processed/corpus_a_row_index.csv": ("07", "Maps corpus_a_bow.mm row order to Paper_ID"),
    "scripts/01_build_corpus.py": ("script", "Stage 1. Hybrid decode, ftfy repair, spill/correction/duplicate/short-abstract exclusion, journal canonicalization, Paper_ID assignment"),
    "scripts/02_eda.py": ("script", "Stage 2. Summary stats, figures 01-06, length percentiles, raw n-gram counts"),
    "scripts/03_preprocess.py": ("script", "Stage 3. 250w cap, normalization, spaCy POS-filtered lemmatization, lemma repair map, stopwords, bigrams/trigrams, Dictionary and BoW"),
    "scripts/04_tune.py": ("script", "Stage 4. Resumable 312-model grid. Takes a time budget in seconds as argv[1]. Uses LdaMulticore, NOT bit-reproducible"),
    "scripts/05_select.py": ("script", "Stage 5. Figure 07, fits K=6 and K=12 candidates with LdaModel, Jensen-Shannon distinctiveness"),
    "scripts/06_interpret.py": ("script", "Stage 6. Topic labels, assignment, prevalence, trends, journal heatmap, Corpus A robustness, pyLDAvis. Holds the LABELS dict"),
    "scripts/08_make_manifest.py": ("script", "Packaging only. Generates documentation/FILE_MANIFEST.md with sizes and checksums. Creates no new analysis"),
    "scripts/07_export_artifacts.py": ("script", "Packaging only. Verifies the saved model against delivered outputs, then exports dictionaries, BoW, tokens, models and matrices. Creates no new analysis"),
    "models/lda_corpus_b_k12.model": ("07", "THE FINAL MODEL in gensim native format. K=12, alpha symmetric, eta auto, 20 passes, 200 iterations, seed 42. Parameters bit-identical to candidates.pkl; use the pickle to reproduce theta exactly (README 1.3)"),
    "models/lda_corpus_b_k12.model.expElogbeta.npy": ("07", "Final model: topic-word parameter array (gensim sidecar, required)"),
    "models/lda_corpus_b_k12.model.id2word": ("07", "Final model: vocabulary mapping (gensim sidecar, required)"),
    "models/lda_corpus_b_k12.model.state": ("07", "Final model: sufficient statistics (gensim sidecar, required)"),
    "models/lda_corpus_b_k6.model": ("07", "The rejected K=6 candidate, kept so the selection decision can be audited"),
    "models/lda_corpus_b_k6.model.expElogbeta.npy": ("07", "K=6 candidate sidecar"),
    "models/lda_corpus_b_k6.model.id2word": ("07", "K=6 candidate sidecar"),
    "models/lda_corpus_b_k6.model.state": ("07", "K=6 candidate sidecar"),
    "models/candidates.pkl": ("05", "Pickle of both fitted candidates as stage 5 wrote them. PROVENANCE ARTIFACT: the only load path that reproduces document_topic_distributions_k12.csv exactly, to 5e-07 (README 1.3)"),
    "models/dictionary_corpus_b.dict": ("07", "Corpus B gensim Dictionary, 3,335 terms. The vectorizer"),
    "models/dictionary_corpus_b.txt": ("07", "Corpus B vocabulary human-readable: id, term, document frequency"),
    "models/dictionary_corpus_a.dict": ("07", "Corpus A gensim Dictionary, 2,308 terms"),
    "models/dictionary_corpus_a.txt": ("07", "Corpus A vocabulary human-readable"),
    "models/model_config.json": ("07", "Machine-readable hyperparameters, seeds, preprocessing constants and selection rationale for all three model configurations"),
    "outputs/tuning_results.csv": ("04", "AUTHORITATIVE grid. 312 rows: corpus, k, alpha, eta, seed, c_v, u_mass, log_perplexity"),
    "outputs/tuning_results_SUPERSEDED_pre_lemma_fix.csv": ("04", "SUPERSEDED. The same grid run before the data/datum lemma fix. Kept for audit only. DO NOT USE"),
    "outputs/topic_summary.csv": ("06", "Per topic: label, prevalence, count of papers where dominant, top 10 keywords"),
    "outputs/topic_top30_terms_k12.csv": ("07", "Top 30 terms per topic with probabilities. Raw material for relabelling"),
    "outputs/topic_word_distributions_k12.csv": ("07", "Full topic-word matrix, 12 x 3,335. The model's complete beta"),
    "outputs/document_topic_distributions_k12.csv": ("07", "Full document-topic matrix, 987 x 12, keyed by Paper_ID. The model's complete theta"),
    "outputs/papers_with_topics.csv": ("06", "PRIMARY ANALYTICAL OUTPUT. Every paper with metadata, dominant topic, dominant share and all 12 topic proportions"),
    "outputs/topic_trends.csv": ("06", "Per topic: OLS slope on year, p-value, 2019 and 2025 share, direction, significance flag"),
    "outputs/topic_prevalence_by_year.csv": ("06", "Mean topic share per year, 2019-2025. The data behind figure 09"),
    "outputs/representative_papers.csv": ("06", "Top 3 papers per topic by topic share, with year and journal. Basis for the hand labels"),
    "outputs/corpus_robustness.csv": ("06", "12 x 12 cosine similarity, Corpus B topics against Corpus A topics"),
    "outputs/eda_summary.csv": ("02", "Corpus-level descriptive statistics"),
    "outputs/length_percentiles.csv": ("02", "Abstract and conclusion word counts at the 50th to 100th percentiles. The evidence behind the 250w cap"),
    "outputs/top_unigrams.csv": ("02", "Top 60 raw unigrams before modelling stopwords were applied"),
    "outputs/top_bigrams.csv": ("02", "Top 60 raw bigrams before modelling stopwords were applied"),
    "figures/01_papers_by_year.png": ("02", "Publication counts by year, trend window highlighted"),
    "figures/02_top_journals.png": ("02", "Top 15 source journals after canonicalization"),
    "figures/03_text_length_distributions.png": ("02", "Abstract vs conclusion word-count histograms with medians"),
    "figures/04_text_availability.png": ("02", "Text availability and every exclusion reason"),
    "figures/05_top_unigrams.png": ("02", "Top 20 raw unigrams"),
    "figures/06_top_bigrams.png": ("02", "Top 20 raw bigrams"),
    "figures/07_model_selection.png": ("05", "C_v and log perplexity vs K, both corpora, seed error bars"),
    "figures/08_topic_prevalence.png": ("06", "Topic prevalence across the corpus"),
    "figures/09_topic_evolution.png": ("06", "Small multiples, topic share 2019-2025, OLS trend and significance per panel"),
    "figures/10_topic_by_journal.png": ("06", "Mean topic share by journal, top 12 journals"),
    "figures/11_corpus_robustness.png": ("06", "Corpus B to Corpus A topic similarity heatmap"),
    "figures/pyldavis_corpus_b.html": ("06", "Interactive topic map. Self-contained, d3 and ldavis inlined, works offline"),
    "documentation/METHODS.md": ("documentation", "Condensed methods note written during the analysis"),
    "documentation/FILE_MANIFEST.md": ("documentation", "This file"),
    "documentation/NOT_SAVED.md": ("documentation", "Artifacts that were never written to disk, stated explicitly"),
    "documentation/LDA_Analysis_Guide.pdf": ("input", "The original guide supplied as input. Audits a DIFFERENT file than the one delivered; see README section 5.3"),
}

PRODUCER = {
    "input": "supplied input",
    "script": "source code",
    "documentation": "documentation",
    "01": "01_build_corpus.py", "02": "02_eda.py", "03": "03_preprocess.py",
    "04": "04_tune.py", "05": "05_select.py", "06": "06_interpret.py",
    "07": "07_export_artifacts.py",
}


def human(n):
    for u in ["B", "KB", "MB"]:
        if n < 1024 or u == "MB":
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024


files = sorted(p for p in PKG.rglob("*") if p.is_file())
rel = [str(p.relative_to(PKG)) for p in files]

missing = [r for r in rel if r not in DESC]
undescribed = [k for k in DESC if k not in rel]
assert not missing, f"undocumented files: {missing}"
assert not undescribed, f"described but absent: {undescribed}"

lines = [
    "# File manifest",
    "",
    f"{len(files)} files. Every file in the package is listed. `Produced by` names",
    "the script that wrote it, so each artifact can be traced to its source.",
    "",
    "SHA-256 is the first 16 hex characters, enough to detect drift.",
    "",
]

for folder in ["", "data/raw", "data/interim", "data/processed", "scripts",
               "models", "outputs", "figures", "documentation"]:
    group = [r for r in rel if (str(Path(r).parent) == "." if folder == ""
                                else str(Path(r).parent) == folder)]
    if not group:
        continue
    lines += [f"## {folder or 'Root'}", "",
              "| File | Size | SHA-256 | Produced by | Contents |",
              "|---|---|---|---|---|"]
    for r in group:
        p = PKG / r
        h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        kind, desc = DESC[r]
        lines.append(f"| `{Path(r).name}` | {human(p.stat().st_size)} | `{h}` | "
                     f"{PRODUCER[kind]} | {desc} |")
    lines.append("")

total = sum(p.stat().st_size for p in files)
lines += [f"**Total: {len(files)} files, {human(total)} uncompressed.**", ""]

(PKG / "documentation" / "FILE_MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")
print(f"manifest written: {len(files)} files, {human(total)}")
