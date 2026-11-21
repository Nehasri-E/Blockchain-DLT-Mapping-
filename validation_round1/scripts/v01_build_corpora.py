"""Build the five cap-condition corpora for the conclusion-cap sensitivity analysis.

Preprocessing replicates scripts/03_preprocess.py exactly. The only thing that
varies between conditions is CONCLUSION_CAP.

Phrase models and dictionaries are REFITTED per corpus. They are fitted on the
token lists, which differ by cap, so reusing one corpus's phraser or vocabulary
on another would be a design error: phrase statistics (min_count, threshold) and
filter_extremes (no_below, no_above) are both corpus-relative quantities.

Baseline reproduction check: condition B250 must reproduce the baseline
data/processed/corpora.pkl Corpus B exactly. This is asserted, not assumed.
"""
import pandas as pd
import re
import spacy
import pickle
import json
import hashlib
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.corpora import Dictionary
from pathlib import Path

ROOT = Path("/home/claude/LDA_Project")
VR = ROOT / "validation_round1"
OUT = VR / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

CONDITIONS = {
    "A": None,
    "B250": 250,
    "B500": 500,
    "B1000": 1000,
    "BFull": None,
}
USES_CONCLUSION = {"A": False, "B250": True, "B500": True, "B1000": True, "BFull": True}

KEEP_POS = {"NOUN", "PROPN", "ADJ", "VERB"}

GENERIC = set("""paper study studies research researcher article literature review section
result results finding findings show shows shown demonstrate demonstrates indicate indicates
suggest suggests propose proposed proposes present presents introduce introduces
approach approaches method methods methodology framework analysis analyse analyze
examine examines investigate investigates explore explores identify identifies
contribution contributions implication implications conclusion conclude concludes
future further work works provide provides discuss discusses consider considers
use used using utilize apply applied application
obtain obtained develop developed developing
also however therefore thus hence moreover furthermore additionally
first second third finally overall respectively
different various several many much more most high low large small
new novel important significant key main major
case base based number level type kind way time year
blockchain blockchains""".split())

DOMAIN_KEEP = {"supply", "chain", "technology", "distributed", "ledger", "data",
               "system", "information", "network", "platform", "contract", "token"}
GENERIC -= DOMAIN_KEEP

LEMMA_FIX = {
    "datum": "data", "cryptocurrencie": "cryptocurrency", "identifie": "identify",
    "focusse": "focus", "capabilitie": "capability", "analysi": "analysis",
    "bct": "blockchain_technology", "dlt": "distributed_ledger_technology",
    "scm": "supply_chain_management", "ai": "artificial_intelligence",
}

url = re.compile(r"https?://\S+|www\.\S+|doi:\S+|10\.\d{4,}/\S+")
noise = re.compile(r"[^a-z\s\-]")
dash = re.compile(r"(?<=\w)-(?=\w)")


def normalize(text):
    t = text.lower()
    t = url.sub(" ", t)
    t = re.sub(r"\(c\)\s*\d{4}.*", " ", t)
    t = re.sub(r"\b(elsevier|springer|wiley|emerald|taylor francis|ieee|mdpi)\b.{0,40}"
               r"(all rights reserved|ltd|inc|b\.?v\.?)", " ", t)
    t = dash.sub("_", t)
    t = noise.sub(" ", t)
    t = t.replace("_", "-")
    return re.sub(r"\s+", " ", t).strip()


def cap_words(text, n):
    return text if n is None else " ".join(text.split()[:n])


df = pd.read_parquet(ROOT / "data/processed/lda_master.parquet")
print(f"Loaded baseline master table: {len(df)} papers\n")

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
nlp.max_length = 2_000_000


def lemmatize(texts):
    out = []
    for doc in nlp.pipe(texts, batch_size=64, n_process=1):
        out.append([
            LEMMA_FIX.get(t.lemma_.lower(), t.lemma_.lower()) for t in doc
            if t.pos_ in KEEP_POS
            and not t.is_stop
            and len(t.lemma_) > 2
            and t.lemma_.lower() not in GENERIC
            and t.lemma_.isascii()
        ])
    return out


corpora = {}
rows = []
for cond, cap in CONDITIONS.items():
    if USES_CONCLUSION[cond]:
        concl = df["Conclusion_Clean"].map(lambda t: cap_words(t, cap))
        text = (df["Title_Clean"] + " " + df["Abstract_Clean"] + " " +
                concl.where(df["Conclusion_Valid"], "")).str.strip()
        n_trunc = int((df["Conclusion_Words"] > cap).sum()) if cap else 0
        concl_tokens = concl.where(df["Conclusion_Valid"], "").str.split().str.len().sum()
    else:
        text = (df["Title_Clean"] + " " + df["Abstract_Clean"]).str.strip()
        n_trunc, concl_tokens = 0, 0

    abs_tokens = df["Abstract_Words"].sum()
    share = concl_tokens / (abs_tokens + concl_tokens) if concl_tokens else 0.0

    toks = lemmatize([normalize(t) for t in text])

    bigram = Phraser(Phrases(toks, min_count=12, threshold=12.0))
    trigram = Phraser(Phrases(bigram[toks], min_count=8, threshold=10.0))
    toks = [trigram[bigram[d]] for d in toks]
    n_phrases = len({t for d in toks for t in d if "_" in t})

    d = Dictionary(toks)
    vocab_pre = len(d)
    d.filter_extremes(no_below=5, no_above=0.50)
    d.compactify()
    bow = [d.doc2bow(t) for t in toks]
    keep = [i for i, b in enumerate(bow) if len(b) >= 10]

    corpora[cond] = {
        "tokens": [toks[i] for i in keep],
        "dictionary": d,
        "bow": [bow[i] for i in keep],
        "paper_ids": df["Paper_ID"].iloc[keep].tolist(),
        "cap": cap,
        "uses_conclusion": USES_CONCLUSION[cond],
    }

    med = int(pd.Series([len(t) for t in toks]).median())
    rows.append({
        "condition": cond,
        "conclusion_cap_words": "none" if cap is None else cap,
        "uses_conclusion": USES_CONCLUSION[cond],
        "papers_truncated": n_trunc,
        "pct_papers_truncated": round(100 * n_trunc / len(df), 1),
        "conclusion_token_share_raw": round(share, 4),
        "phrases_detected": n_phrases,
        "vocab_before_filter": vocab_pre,
        "vocab_after_filter": len(d),
        "documents_retained": len(keep),
        "documents_dropped": len(toks) - len(keep),
        "median_tokens_per_doc": med,
        "total_tokens": sum(len(t) for t in toks),
    })
    print(f"{cond:>6}  cap={str(cap):>4}  vocab {vocab_pre:>6} -> {len(d):>5}  "
          f"docs {len(keep)}/{len(toks)}  phrases {n_phrases:>4}  median tokens {med:>4}")

summary = pd.DataFrame(rows)
summary.to_csv(OUT / "A1_corpus_construction_summary.csv", index=False)

print("\n" + "=" * 78)
print("BASELINE REPRODUCTION CHECK: does rebuilt B250 equal the baseline Corpus B?")
base = pickle.load(open(ROOT / "data/processed/corpora.pkl", "rb"))["B"]
new = corpora["B250"]


def dhash(d):
    return hashlib.sha256(
        "\n".join(f"{i}\t{d[i]}" for i in range(len(d))).encode()).hexdigest()[:16]


checks = {
    "document count": len(base["bow"]) == len(new["bow"]),
    "paper_ids identical": base["paper_ids"] == new["paper_ids"],
    "vocabulary size": len(base["dictionary"]) == len(new["dictionary"]),
    "vocabulary hash": dhash(base["dictionary"]) == dhash(new["dictionary"]),
    "token lists identical": base["tokens"] == new["tokens"],
    "BoW identical": base["bow"] == new["bow"],
}
for k, v in checks.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")
assert all(checks.values()), "B250 does not reproduce the baseline Corpus B"
print("\nB250 reproduces the baseline exactly. The baseline K=12 seed-42 model is")
print("therefore a valid member of this grid and is reused rather than refitted.")

with open(VR / "models" / "validation_corpora.pkl", "wb") as f:
    pickle.dump(corpora, f)

(OUT / "A1_corpus_construction_summary.json").write_text(
    json.dumps({
        "preprocessing": "identical to scripts/03_preprocess.py except CONCLUSION_CAP",
        "refitted_per_corpus": ["gensim Phrases bigram (min_count=12, threshold=12.0)",
                                "gensim Phrases trigram (min_count=8, threshold=10.0)",
                                "gensim Dictionary + filter_extremes(no_below=5, no_above=0.50)",
                                "document retention filter (>=10 BoW entries)"],
        "reused_unchanged": ["spaCy en_core_web_sm 3.8.0 POS-filtered lemmatization",
                             "LEMMA_FIX map", "GENERIC stopword list", "DOMAIN_KEEP protections",
                             "normalize() regexes", "source table data/processed/lda_master.parquet"],
        "rationale_for_refit": ("Phrase statistics and filter_extremes thresholds are "
                                "corpus-relative. Holding them fixed across caps would make "
                                "the vocabulary a function of one arbitrary cap choice and "
                                "would confound the sensitivity test."),
        "baseline_reproduction_check": {k: bool(v) for k, v in checks.items()},
        "conditions": rows,
    }, indent=2), encoding="utf-8")

print(f"\nSaved validation_corpora.pkl with {len(corpora)} conditions")
print(summary.to_string(index=False))
