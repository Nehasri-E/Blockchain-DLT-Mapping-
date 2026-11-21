# Blockchain literature topic modelling: final verified archive

1229 files, built after four rounds of validation.

## Read first

    validation_round4/documentation/FINAL_FINDINGS.md
    validation_round4/documentation/FINAL_METHODS.md
    validation_round4/documentation/FINAL_LIMITATIONS.md
    validation_round4/documentation/FINAL_REPRODUCTION_GUIDE.md

## Headline

Primary model is NMF K=6 on TF-IDF, not the originally planned LDA K=12.
Three themes are defensible: two strongly supported, one with qualification.
No corpus-wide temporal claim is supported under the primary inference procedure.

## Verifying this archive

    python validation_round4/scripts/verify_archive.py

Checks every file in ARCHIVE_MANIFEST.csv against its SHA-256 and exits non-zero
on any mismatch. Results for this build are in
validation_round4/ARCHIVE_VERIFICATION.csv.

## Structure

    scripts/                  baseline pipeline, 01 to 08
    data/                     interim and processed
    models/                   baseline LDA candidates
    outputs/                  baseline topic assignments and trends
    documentation/            baseline manifest and methods
    validation_round1/        cap sensitivity, seed stability, FDR trends
    validation_round2/        convergence, priors, K sweep, bootstrap
    validation_round3/        consensus, NMF, corrected inference
    validation_round4/        final consolidation
    ARCHIVE_MANIFEST.csv      1229 files with SHA-256

## Status

The computational analysis is reproducible and extensively self-audited.
The manuscript is not ready. FINAL_LIMITATIONS.md section 10 states why.
