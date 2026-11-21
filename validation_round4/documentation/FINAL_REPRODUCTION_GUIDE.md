# Final reproduction guide

Everything here was run on the environment described in section 1. Every script
is deterministic given its seeds. Expected console output and expected output
files are listed per step so that a mismatch is detectable without reading the
code.

---

## 1. Environment

```
Python 3.13
gensim 4.3.x        spacy 3.7.x + en_core_web_sm
scikit-learn 1.5.x  scipy 1.14.x
pandas 2.2.x        numpy 1.26.x
pyarrow             ftfy            matplotlib
```

```bash
pip install --break-system-packages gensim spacy scikit-learn scipy pandas numpy pyarrow ftfy matplotlib
python -m spacy download en_core_web_sm
```

**Set `ROOT` once.** Every script hardcodes `ROOT = Path("/home/claude/LDA_Project")`.
If the archive is extracted elsewhere, edit that line, or create a symlink at
that path. There is no command-line argument for it; that is a limitation of the
scripts, not a design choice worth defending.

Approximate single-core runtimes: baseline pipeline 35 minutes (the 312-model
tuning grid dominates), Round 1 20 minutes, Round 2 **4 to 5 hours** (138
models), Round 3 40 minutes, Round 4 6 minutes.

## 2. Run order

Each stage depends on the previous one. Stages 1 to 3 can be skipped if the
shipped artifacts are used, since Round 4 reads from Round 3 outputs and the
baseline parquet files.

### Stage 1, baseline pipeline

```bash
cd /home/claude/LDA_Project
python scripts/01_build_corpus.py        # 1,003 -> 987; writes reconciliation.csv
python scripts/02_eda.py
python scripts/03_preprocess.py          # spaCy + phrases + dictionary, 3,335 terms
python scripts/04_tune.py 7200            # 312-model grid; argv[1] is a time
                                          # budget in seconds, resumable
python scripts/05_select.py               # K=6 and K=12 candidates
python scripts/06_interpret.py
python scripts/07_export_artifacts.py
python scripts/08_make_manifest.py       # 72 SHA-256 checksums
```

Expected after stage 1: `data/processed/corpora.pkl`, `lda_master.parquet`,
`lda_master_capped.parquet`, `models/candidates.pkl`, `outputs/papers_with_topics.csv`,
`documentation/FILE_MANIFEST.md`.

Checkpoint: `data/interim/reconciliation.csv` must end at **987** retained
papers, and the dictionary must have **3,335** terms. If either differs, the
spaCy model version differs and nothing downstream will match.

`04_tune.py` uses `LdaMulticore` and is therefore **not** bit-reproducible. It is
a tuning scan only; nothing in the final results depends on its exact numbers,
and every reported model is refit single-process in the validation rounds. This
is recorded rather than fixed because rerunning the grid single-process would
take days and change no conclusion.

### Stage 2, validation Round 1 (cap sensitivity, seed stability, FDR trends)

```bash
cd validation_round1/scripts
python v01_build_corpora.py              # A, B250, B500, B1000, BFull
python v02_train_grid.py
python v03_cap_sensitivity.py
python v04_seed_stability.py
python v05_corpus_a_model.py             # preserves the Corpus A model
python v06_trend_reanalysis.py           # BH FDR across topics
python v07_cap_vs_noise.py
python v08_figures.py
python v09_quality_control.py
```

Checkpoint: `C5_trend_headline_table.csv`, `D2_seed_stability_summary.csv` and
the 6 figures `V1` to `V6`. `v09` must report all checks PASS.

### Stage 3, validation Round 2 (why K=12 is unstable)

```bash
cd ../../validation_round2/scripts
python r01_convergence.py                # baseline, C1, C2, C3; 30 models
python r02_priors.py                     # 18 models
python r02b_eta_explicit.py              # 24 models; the eta no-op fix
python r02c_eta_tradeoff.py
python r03_k_sweep.py                    # K=4..16, 66 models; the slow step
python r04_recovery.py
python r05_cap_vs_seed_corrected.py
python r06_trend_bootstrap.py
python r07_figures.py
python r08_quality_control.py
```

Checkpoint: `T3_stability_by_config.csv` must show the baseline reaching
**0.99983** mean E-step convergence and C3 agreeing with the 200-pass anchor at
**0.9992**, and `T11_k_sweep_summary.csv` must show `recovery_above_chance` flat
across K (**0.341 to 0.418**). The flat profile is the central Round 2 finding:
it is what shows the apparent stability gain from lowering K to be mechanical. If
it is not flat, the chance baseline is being computed differently.

### Stage 4, validation Round 3 (consensus, NMF, corrected inference)

```bash
cd ../../validation_round3/scripts
python s01_consensus.py                  # LDA consensus, criteria 1-3
python s01b_consensus_seedpure.py        # adds criterion 4 (post-hoc)
python s01c_threshold_plateau.py
python s02_nmf.py                        # NMF K=4,6,8 x 6 seeds
python s03_sensitivity.py
python s04_trend_inference.py            # wild cluster bootstrap-t
python s05_lda_vs_nmf.py
python s06_figures.py
python s07_quality_control.py
```

Checkpoint: `N2_nmf_stability_by_k.csv` must report mean matched cosine
**1.0000 at K=4, 0.9415 at K=6, 0.8865 at K=8**. Those three numbers are the
ones Round 4 audits; if they differ, the NMF seeds or the TF-IDF vocabulary
differ.

`s04` prints its positive control before its results. Noise must give p ≈ 0.728
and a planted 0.02/yr slope must give p ≈ 0.0005. **If the positive control
fails, the null results are not interpretable and the run should be stopped.**

### Stage 5, validation Round 4 (this round)

```bash
cd ../../validation_round4/scripts
python f01_audit_round3.py               # 31 claims; expect 28 CORRECT, 3 to correct
python f02_final_themes.py               # Table2; outlier-seed check
python f03_authors.py                    # feasibility verdict: False
python f04_methodology_audit.py          # 30 decisions; 6 pre-specified
python f05_final_outputs.py              # 6 figures, 6 tables, 18 supplementary
python f06_quality_control.py            # FINAL_QC_CHECKLIST.csv + claim map
python f07_build_archive.py              # archive, checksums, clean-extract check
```

Checkpoints:

- `f01` prints `28 CORRECT / 3 REQUIRE CORRECTION`. All three corrections concern
  the NMF identifiability claim.
- `f02` prints the outlier-seed table; `cosine_full_vs_basin` must be 0.989 or
  higher on all six rows. It must also print prevalence values in the range
  0.155 to 0.178, **not** six values of 0.1667. Six identical values mean the W
  columns are being normalised before averaging, which is the bug this script
  fixes.
- `f03` must print `reliable disambiguation feasible = False`.
- `f06` must report **38 of 38 PASS** and a claim map in which no claim has
  status MISMATCH. It re-derives all 72 numeric claims in `FINAL_FINDINGS.md` and
  `FINAL_METHODS.md` from their source files, so a MISMATCH means the document
  and the data disagree and the document is wrong. It caught three such errors in
  the first draft of `FINAL_FINDINGS.md`.
- `f07` must report **11 of 11 PASS** on the clean extract, including that the
  baseline is unchanged against its own 73-file manifest.

## 3. Integrity verification

Whole archive, 1,229 files against `ARCHIVE_MANIFEST.csv`:

```bash
python validation_round4/scripts/verify_archive.py
```

Exit code 0 means every file matches its SHA-256. Its result for this archive is
recorded in `validation_round4/ARCHIVE_VERIFICATION.csv`, alongside nine further
checks run on a clean extract: the baseline unchanged against its own manifest,
load-bearing files reachable, the pickled LDA
models loading, the master parquet reading 987 rows, the six final figures
present, and the Round 4 QC checklist and claim map both clean when read from the
extract rather than from the working tree.

Round 4 alone, 57 files, against the manifest `f06` writes:

```bash
python - <<'EOF'
import hashlib, pandas as pd
from pathlib import Path
R4 = Path("/home/claude/LDA_Project/validation_round4")
m = pd.read_csv(R4 / "ROUND4_FILE_MANIFEST.csv")
bad = [(r.relative_path, "MISSING" if not (R4 / r.relative_path).exists() else "CHANGED")
       for r in m.itertuples()
       if not (R4 / r.relative_path).exists()
       or hashlib.sha256((R4 / r.relative_path).read_bytes()).hexdigest() != r.sha256]
print("OK" if not bad else bad)
EOF
```

`documentation/FILE_MANIFEST.md` holds the 72 baseline checksums from
`08_make_manifest.py`. That script writes the manifest; it has no verify flag, so
use `verify_archive.py`, which covers those files too.

## 4. Known reproduction caveats

**Recover theta from the pickle, not the `.model` file.**
`LdaModel.get_document_topics` seeds its gamma from a mutable `random_state`.
Pickling the model object preserves that state; `save()` followed by `load()`
does not. Document-topic distributions can therefore differ in the fourth decimal
place between the two routes. Model parameters are bit-identical either way.

**Never substitute `LdaMulticore`.** It is not reproducible even with
`random_state` fixed. Two runs with identical parameters differed by up to 1
percentage point in topic prevalence. All reported models use single-process
`LdaModel`.

**Set `eta` numerically, never `eta="auto"`.** `eta="auto"` is a no-op for topics
in gensim: `init_dir_prior` returns 1/K for both `"auto"` and `"symmetric"`, and
`update_eta` writes to `model.eta` rather than `model.state.eta`.

**spaCy version pins the vocabulary.** A different `en_core_web_sm` will lemmatize
differently and the 3,335-term dictionary will not reproduce. Check the
dictionary size first; everything downstream depends on it.

**`pyLDAvis` needs `mds="mmds"`.** The default PCoA produces complex eigenvalues
on this corpus and fails JSON serialisation. The shipped HTML has d3 and the
pyLDAvis JS inlined so it renders offline.

**NMF at K=6 has two basins.** A single-seed run has roughly a one in six chance
of reproducing seed 777's solution instead of the dominant one. Reproduce with
all six seeds, not one.

## 5. Where things are

```
LDA_Project/
  scripts/                  01 to 08, baseline pipeline
  data/interim/             reconciliation.csv, quality_control.csv
  data/processed/           corpora.pkl, lda_master*.parquet
  models/                   candidates.pkl, lda_corpus_b_k12.model
  outputs/                  papers_with_topics.csv, topic_summary.csv, ...
  documentation/            FILE_MANIFEST.md, pipeline methods
  validation_round1/        cap sensitivity, seed stability, FDR trends
  validation_round2/        convergence, priors, K sweep, bootstrap
  validation_round3/        consensus, NMF, corrected inference
  validation_round4/
    scripts/                f01 to f07, verify_archive.py
    outputs/                F1 to F10
    FINAL_TABLES/           Table1 to Table7
    FINAL_FIGURES/          Figure1 to Figure6
    SUPPLEMENTARY_MATERIAL/ 18 figures from Rounds 1 to 3
    documentation/          FINAL_METHODS, FINAL_FINDINGS,
                            FINAL_LIMITATIONS, this file
    FINAL_QC_CHECKLIST.csv       38 checks
    ARCHIVE_VERIFICATION.csv     11 clean-extract checks
    ROUND4_FILE_MANIFEST.csv
  ARCHIVE_MANIFEST.csv          1,229 files with SHA-256
  README_FINAL.md
```

**Read in this order**: `FINAL_FINDINGS.md`, then `FINAL_METHODS.md`, then
`FINAL_LIMITATIONS.md`. `FINAL_TABLES/Table7_claim_to_file.csv` maps every claim
in the findings to the file that supports it, so any number can be checked
without rerunning anything.
