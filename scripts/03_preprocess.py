import pandas as pd
import re
import spacy
import pickle
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.corpora import Dictionary
from pathlib import Path

DATA = Path("/home/claude/blockchain_lda/data")
CONCLUSION_CAP = 250
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

df = pd.read_parquet(DATA / "lda_master.parquet")
print(f"Loaded {len(df)} papers")


def cap_words(text, n):
    w = text.split()
    return " ".join(w[:n])


df["Conclusion_Capped"] = df["Conclusion_Clean"].map(lambda t: cap_words(t, CONCLUSION_CAP))
df["Text_B"] = (df["Title_Clean"] + " " + df["Abstract_Clean"] + " " +
                df["Conclusion_Capped"].where(df["Conclusion_Valid"], "")).str.strip()
df["Text_A"] = (df["Title_Clean"] + " " + df["Abstract_Clean"]).str.strip()

trunc = int((df["Conclusion_Words"] > CONCLUSION_CAP).sum())
print(f"Conclusions capped at {CONCLUSION_CAP} words: {trunc} papers ({trunc / len(df):.0%})")

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


nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
nlp.max_length = 2_000_000


def lemmatize(texts, label):
    out = []
    for i, doc in enumerate(nlp.pipe(texts, batch_size=64, n_process=1)):
        toks = [
            LEMMA_FIX.get(t.lemma_.lower(), t.lemma_.lower()) for t in doc
            if t.pos_ in KEEP_POS
            and not t.is_stop
            and len(t.lemma_) > 2
            and t.lemma_.lower() not in GENERIC
            and t.lemma_.isascii()
        ]
        out.append(toks)
        if (i + 1) % 250 == 0:
            print(f"  {label}: {i + 1}/{len(texts)}")
    return out


corpora = {}
for key, col in [("B", "Text_B"), ("A", "Text_A")]:
    print(f"\nLemmatizing Corpus {key}")
    texts = [normalize(t) for t in df[col]]
    toks = lemmatize(texts, f"Corpus {key}")

    bigram = Phraser(Phrases(toks, min_count=12, threshold=12.0))
    trigram = Phraser(Phrases(bigram[toks], min_count=8, threshold=10.0))
    toks = [trigram[bigram[d]] for d in toks]

    phrases = sorted({t for d in toks for t in d if "_" in t})
    print(f"  phrases detected: {len(phrases)}")
    print(f"  sample: {', '.join(phrases[:18])}")

    d = Dictionary(toks)
    before = len(d)
    d.filter_extremes(no_below=5, no_above=0.50)
    d.compactify()
    bow = [d.doc2bow(t) for t in toks]

    keep = [i for i, b in enumerate(bow) if len(b) >= 10]
    print(f"  vocabulary: {before} -> {len(d)} after filter_extremes(no_below=5, no_above=0.50)")
    print(f"  documents retained: {len(keep)}/{len(toks)}")
    print(f"  median tokens per document: {int(pd.Series([len(t) for t in toks]).median())}")

    corpora[key] = {
        "tokens": [toks[i] for i in keep],
        "dictionary": d,
        "bow": [bow[i] for i in keep],
        "paper_ids": df["Paper_ID"].iloc[keep].tolist(),
    }

with open(DATA / "corpora.pkl", "wb") as f:
    pickle.dump(corpora, f)

df.to_parquet(DATA / "lda_master_capped.parquet", index=False)
print(f"\nSaved corpora.pkl  (A: {len(corpora['A']['bow'])} docs, B: {len(corpora['B']['bow'])} docs)")
