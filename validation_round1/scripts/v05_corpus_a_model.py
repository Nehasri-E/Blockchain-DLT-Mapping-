"""Task B: fit, serialize and validate the Corpus A (abstract-only) K=12 model.

NOT_SAVED.md records that the original Corpus A model was fitted in memory by
scripts/06_interpret.py and discarded. This script recreates it under the exact
baseline configuration, persists it, and then asks the only question that
matters: does the refit reproduce the published robustness result?

Two similarity procedures are reported:

  baseline procedure   cosine on a union vocabulary, best match per row via
                       max(axis=1). This is what 06_interpret.py did. It is
                       many-to-one: several Corpus B topics may claim the same
                       Corpus A topic. Reproduced here only to validate the refit
                       against the published numbers.

  corrected procedure  the same cosine matrix, but one-to-one via the Hungarian
                       algorithm. This is the defensible measure and is reported
                       alongside, NOT substituted for the published figure.

Nothing in LDA_Project/models/ or LDA_Project/outputs/ is written or modified.
"""
import pickle
import json
import numpy as np
import pandas as pd
from gensim.models import LdaModel
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import valtools as vt

ROOT = Path("/home/claude/LDA_Project")
VR = ROOT / "validation_round1"
OUT = VR / "outputs"
MODELS = VR / "models"
K = 12
SEED = 42
HP = dict(num_topics=K, alpha="symmetric", eta="auto", passes=20,
          iterations=200, chunksize=200, eval_every=None)

LABELS = {int(k): v for k, v in
          json.loads((VR / "scripts" / "baseline_labels.json").read_text()).items()}

base_corpora = pickle.load(open(ROOT / "data/processed/corpora.pkl", "rb"))
val_corpora = pickle.load(open(MODELS / "validation_corpora.pkl", "rb"))

print("STEP 1  Does the rebuilt Corpus A match the baseline Corpus A?")
ba, va = base_corpora["A"], val_corpora["A"]
checks = {
    "document count": len(ba["bow"]) == len(va["bow"]),
    "paper_ids identical": ba["paper_ids"] == va["paper_ids"],
    "vocabulary size": len(ba["dictionary"]) == len(va["dictionary"]),
    "token lists identical": ba["tokens"] == va["tokens"],
    "BoW identical": ba["bow"] == va["bow"],
}
for k, v in checks.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")
assert all(checks.values()), "rebuilt Corpus A differs from baseline"

print("\nSTEP 2  Fit and serialize the Corpus A model")
model_a = LdaModel(corpus=va["bow"], id2word=va["dictionary"], random_state=SEED, **HP)
model_a.save(str(MODELS / "lda_corpus_a_k12.model"))
va["dictionary"].save(str(MODELS / "dictionary_corpus_a_validation.dict"))
va["dictionary"].save_as_text(str(MODELS / "dictionary_corpus_a_validation.txt"),
                              sort_by_word=False)
print(f"  saved lda_corpus_a_k12.model  ({len(va['bow'])} docs, "
      f"{len(va['dictionary'])} terms)")

grid_copy = LdaModel.load(str(MODELS / "grid" / f"lda_A_k{K}_seed{SEED}.model"))
print(f"  identical to grid model lda_A_k12_seed42: "
      f"{np.array_equal(model_a.expElogbeta, grid_copy.expElogbeta)}")

print("\nSTEP 3  Export distributions")
TA = model_a.get_topics()
vocab_a = [va["dictionary"][i] for i in range(len(va["dictionary"]))]
pd.DataFrame(TA, index=[f"Topic_{i+1}" for i in range(K)], columns=vocab_a).to_csv(
    OUT / "B1_corpusA_topic_word_distributions.csv")

theta_a = vt.doc_topic_matrix(model_a, va["bow"], K)
dta = pd.DataFrame(theta_a.round(6), columns=[f"Topic_{i+1}" for i in range(K)])
dta.insert(0, "Paper_ID", va["paper_ids"])
dta.to_csv(OUT / "B2_corpusA_document_topic_distributions.csv", index=False)

rows = []
for i in range(K):
    for rank, (w, p) in enumerate(model_a.show_topic(i, topn=30), 1):
        rows.append({"Topic": i + 1, "Rank": rank, "Term": w,
                     "Probability": round(float(p), 6)})
pd.DataFrame(rows).to_csv(OUT / "B3_corpusA_top30_terms.csv", index=False)
print(f"  topic-word {TA.shape}, document-topic {theta_a.shape}, top-30 terms")

print("\nSTEP 4  Reproduce the published robustness comparison")
model_b = pickle.load(open(ROOT / "models" / "candidates.pkl", "rb"))[K]
db = base_corpora["B"]["dictionary"]

ta = vt.topic_matrix(model_a, va["dictionary"])
tb = vt.topic_matrix(model_b, db)
Bm, Am, _ = vt.project(tb, ta, "union")
S = vt.cosine_matrix(Bm, Am)

published = pd.read_csv(ROOT / "outputs" / "corpus_robustness.csv", index_col=0)
dev = float(np.abs(S - published.values).max())
print(f"  max deviation from published corpus_robustness.csv : {dev:.2e}")
print(f"  reproduces within 1e-3                              : {dev < 1e-3}")

best = S.max(1)
pub_best = published.values.max(1)
print(f"  best-match vector reproduces (max dev)              : "
      f"{float(np.abs(best - pub_best).max()):.2e}")
print(f"  published claim: 7/12 topics at cosine >= 0.50      : "
      f"{int((pub_best >= 0.50).sum())}/12")
print(f"  refit          : {int((best >= 0.50).sum())}/12, "
      f"mean best match {best.mean():.4f}")

cols, hsim = vt.match(S)
print(f"\n  one-to-one (Hungarian) mean matched cosine          : {hsim.mean():.4f}")
print(f"  one-to-one topics at cosine >= 0.50                 : {int((hsim >= 0.50).sum())}/12")
collisions = len(cols) - len(set(S.argmax(1)))
print(f"  many-to-one collisions in the published procedure   : {collisions} "
      f"(Corpus A topics claimed by more than one Corpus B topic)")

pd.DataFrame(S.round(4), index=[f"B_T{i+1}" for i in range(K)],
             columns=[f"A_T{j+1}" for j in range(K)]).to_csv(
    OUT / "B4_corpusB_vs_corpusA_similarity_refit.csv")

comp = pd.DataFrame({
    "corpusB_topic": range(1, K + 1),
    "baseline_label": [LABELS[i + 1] for i in range(K)],
    "published_best_match_cosine": pub_best.round(4),
    "refit_best_match_cosine": best.round(4),
    "abs_difference": np.abs(best - pub_best).round(6),
    "refit_best_match_topic": S.argmax(1) + 1,
    "hungarian_matched_topic": cols + 1,
    "hungarian_cosine": hsim.round(4),
    "published_verdict_ge_0.50": pub_best >= 0.50,
    "hungarian_verdict_ge_0.50": hsim >= 0.50,
})
comp.to_csv(OUT / "B5_robustness_published_vs_refit.csv", index=False)

(MODELS / "corpus_a_model_config.json").write_text(json.dumps({
    "model_file": "lda_corpus_a_k12.model",
    "corpus": "A (title + abstract only)",
    "implementation": "gensim.models.LdaModel (single process, deterministic)",
    "hyperparameters": {**{k: (v if v is not None else "None") for k, v in HP.items()},
                        "random_state": SEED},
    "documents": len(va["bow"]), "vocabulary": len(va["dictionary"]),
    "rebuilt_corpus_matches_baseline": {k: bool(v) for k, v in checks.items()},
    "reproduces_published_robustness_matrix": bool(dev < 1e-3),
    "max_deviation_from_published": dev,
    "note": ("The published robustness figure used max(axis=1), which is "
             "many-to-one. The Hungarian one-to-one figure is reported alongside "
             "in B5 but does not replace the published number."),
}, indent=2), encoding="utf-8")

print(f"\n{comp.to_string(index=False, max_colwidth=40)}")
