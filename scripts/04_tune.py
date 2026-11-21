import pickle
import sys
import time
import pandas as pd
from gensim.models import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from pathlib import Path

DATA = Path("/home/claude/blockchain_lda/data")
RESULTS = DATA / "tuning_results.csv"
SEEDS = [42, 7, 2024]
K_SWEEP = [4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20]
PRIORS = [("symmetric", "symmetric"), ("asymmetric", "symmetric"),
          ("symmetric", "auto"), ("asymmetric", "auto")]

corpora = pickle.load(open(DATA / "corpora.pkl", "rb"))
budget = float(sys.argv[1]) if len(sys.argv) > 1 else 500.0

jobs = []
for corpus in ["B", "A"]:
    for a, e in PRIORS:
        for k in K_SWEEP:
            for s in SEEDS:
                jobs.append((corpus, k, a, e, s))

done = set()
if RESULTS.exists():
    prev = pd.read_csv(RESULTS)
    done = set(map(tuple, prev[["corpus", "k", "alpha", "eta", "seed"]].values))

todo = [j for j in jobs if j not in done]
print(f"{len(done)} models already fitted, {len(todo)} remaining")

start = time.time()
rows = []
for corpus, k, a, e, s in todo:
    if time.time() - start > budget:
        break
    c = corpora[corpus]
    m = LdaMulticore(corpus=c["bow"], id2word=c["dictionary"], num_topics=k,
                     alpha=a, eta=e, passes=10, iterations=100,
                     random_state=s, workers=2, chunksize=200)
    cv = CoherenceModel(model=m, texts=c["tokens"], dictionary=c["dictionary"],
                        coherence="c_v", processes=1).get_coherence()
    umass = CoherenceModel(model=m, corpus=c["bow"], dictionary=c["dictionary"],
                           coherence="u_mass", processes=1).get_coherence()
    rows.append({"corpus": corpus, "k": k, "alpha": a, "eta": e, "seed": s,
                 "c_v": round(cv, 5), "u_mass": round(umass, 5),
                 "log_perplexity": round(m.log_perplexity(c["bow"]), 5)})

if rows:
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)
    print(f"Wrote {len(rows)} results in {time.time() - start:.0f}s")

total = len(pd.read_csv(RESULTS)) if RESULTS.exists() else 0
print(f"Progress: {total}/{len(jobs)} models")
