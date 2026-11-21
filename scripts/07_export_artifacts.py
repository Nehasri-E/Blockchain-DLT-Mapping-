import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from gensim.corpora import MmCorpus

DATA = Path("/home/claude/blockchain_lda/data")
PKG = Path("/home/claude/LDA_Project")
K = 12

corpora = pickle.load(open(DATA / "corpora.pkl", "rb"))
models = pickle.load(open(DATA / "candidates.pkl", "rb"))
model = models[K]
cb, ca = corpora["B"], corpora["A"]

print("VERIFICATION: does the saved model reproduce the delivered outputs?")
dist = np.zeros((len(cb["bow"]), K))
for i, doc in enumerate(cb["bow"]):
    for t, p in model.get_document_topics(doc, minimum_probability=0.0):
        dist[i, t] = p

delivered = pd.read_csv("/mnt/user-data/outputs/topic_summary.csv")
recomputed = dist.mean(0).round(4)
match = np.allclose(delivered["Prevalence"].values, recomputed, atol=1e-4)
print(f"  topic prevalence matches topic_summary.csv : {match}")
for i in range(K):
    assert abs(delivered["Prevalence"][i] - recomputed[i]) < 1e-4, f"topic {i+1} mismatch"

dom = dist.argmax(1) + 1
dcounts = [int((dom == i).sum()) for i in range(1, K + 1)]
print(f"  dominant-topic counts match               : {dcounts == delivered['Papers_Dominant'].tolist()}")

pw = pd.read_csv("/mnt/user-data/outputs/papers_with_topics.csv")
tcols = [f"Topic_{i+1}" for i in range(K)]
pw_sorted = pw.set_index("Paper_ID").loc[cb["paper_ids"]][tcols].values
print(f"  per-document distributions match          : {np.allclose(pw_sorted, dist, atol=1e-5)}")
print()

for sub in ["data/raw", "data/interim", "data/processed", "scripts", "models",
            "outputs", "figures", "documentation"]:
    (PKG / sub).mkdir(parents=True, exist_ok=True)

for key, c, name in [("B", cb, "corpus_b"), ("A", ca, "corpus_a")]:
    d = c["dictionary"]
    d.save(str(PKG / "models" / f"dictionary_{name}.dict"))
    d.save_as_text(str(PKG / "models" / f"dictionary_{name}.txt"), sort_by_word=False)
    MmCorpus.serialize(str(PKG / "data/processed" / f"{name}_bow.mm"), c["bow"])
    with open(PKG / "data/processed" / f"{name}_tokens.jsonl", "w", encoding="utf-8") as f:
        for pid, toks in zip(c["paper_ids"], c["tokens"]):
            f.write(json.dumps({"Paper_ID": pid, "tokens": toks}) + "\n")
    pd.DataFrame({"Paper_ID": c["paper_ids"],
                  "row_index": range(len(c["paper_ids"]))}).to_csv(
        PKG / "data/processed" / f"{name}_row_index.csv", index=False)
    print(f"  exported {name}: dictionary ({len(d)} terms), BoW .mm, tokens .jsonl, row index")

for k, m in models.items():
    m.save(str(PKG / "models" / f"lda_corpus_b_k{k}.model"))
    print(f"  exported trained model: lda_corpus_b_k{k}.model")

T = model.get_topics()
vocab = [cb["dictionary"][i] for i in range(len(cb["dictionary"]))]
pd.DataFrame(T, index=[f"Topic_{i+1}" for i in range(K)], columns=vocab).to_csv(
    PKG / "outputs" / "topic_word_distributions_k12.csv")
print(f"  exported topic-word matrix: {T.shape[0]} topics x {T.shape[1]} terms")

rows = []
for i in range(K):
    for rank, (w, p) in enumerate(model.show_topic(i, topn=30), 1):
        rows.append({"Topic": i + 1, "Rank": rank, "Term": w, "Probability": round(float(p), 6)})
pd.DataFrame(rows).to_csv(PKG / "outputs" / "topic_top30_terms_k12.csv", index=False)

dt = pd.DataFrame(dist.round(6), columns=tcols)
dt.insert(0, "Paper_ID", cb["paper_ids"])
dt.to_csv(PKG / "outputs" / "document_topic_distributions_k12.csv", index=False)
print("  exported document-topic matrix and top-30 terms")

config = {
    "final_model": {
        "corpus": "B (title + abstract + conclusion capped at 250 words)",
        "implementation": "gensim.models.LdaModel (single-process, deterministic)",
        "num_topics": 12, "alpha": "symmetric", "eta": "auto",
        "passes": 20, "iterations": 200, "chunksize": 200,
        "random_state": 42, "eval_every": None,
        "documents": len(cb["bow"]), "vocabulary": len(cb["dictionary"]),
    },
    "robustness_model": {
        "corpus": "A (title + abstract)",
        "implementation": "gensim.models.LdaModel",
        "num_topics": 12, "alpha": "symmetric", "eta": "auto",
        "passes": 20, "iterations": 200, "chunksize": 200, "random_state": 42,
        "documents": len(ca["bow"]), "vocabulary": len(ca["dictionary"]),
        "saved_to_disk": False,
        "note": "fitted inside 06_interpret.py and never serialised; deterministic, so re-running that script reproduces it exactly",
    },
    "tuning_grid": {
        "implementation": "gensim.models.LdaMulticore (workers=2) - NOT deterministic, used only for seed-averaged metrics",
        "K": [4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20],
        "priors": [["symmetric", "symmetric"], ["asymmetric", "symmetric"],
                   ["symmetric", "auto"], ["asymmetric", "auto"]],
        "seeds": [42, 7, 2024], "passes": 10, "iterations": 100,
        "total_models": 312, "corpora": ["A", "B"],
    },
    "preprocessing": {
        "conclusion_cap_words": 250,
        "min_abstract_words": 50, "min_conclusion_words": 50,
        "spacy_model": "en_core_web_sm", "keep_pos": ["NOUN", "PROPN", "ADJ", "VERB"],
        "bigram": {"min_count": 12, "threshold": 12.0},
        "trigram": {"min_count": 8, "threshold": 10.0},
        "filter_extremes": {"no_below": 5, "no_above": 0.50},
        "min_bow_length_to_retain_doc": 10,
    },
    "selection": {
        "metric_used_for_K": "topic distinctiveness (mean pairwise Jensen-Shannon) and manual interpretability",
        "reason": "C_v is flat from K=4 to K=12; differences are inside seed noise and do not identify an optimum",
        "js_mean_k6": 0.789, "js_mean_k12": 0.852,
    },
    "trend_analysis": {"years_modelled": "2017-2026", "years_plotted": "2019-2025",
                       "method": "OLS slope of mean topic share on year", "alpha": 0.05},
}
(PKG / "models" / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
print("  wrote model_config.json")
