"""Task 2 addendum: does the qualifying-theme conclusion depend on the threshold?

The headline threshold of 0.55 sits inside a plateau: the number of multi-seed
hierarchical clusters is constant at 6 for every threshold from 0.500 to 0.750.
It is NOT the cluster-count maximiser; that is 0.800, which yields 8 multi-seed
clusters. Choosing 0.80 would have produced more "consensus topics", which is
precisely the temptation the brief warns against.

This script re-runs the qualification test at every threshold on the plateau and
at the maximiser, and reports how many themes qualify at each. If the answer is
invariant, the headline choice carries no weight in the conclusion.

Note that the PRIMARY consensus in s01b uses anchor-based Hungarian matching,
which has no similarity threshold at all. The threshold affects only the
secondary hierarchical view.
"""
import sys
import itertools
from collections import Counter
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).parent))
from s00_common import (R3, ROOT, SEEDS, load_corpus, load_k6_models, theta,
                        cosine_matrix, term_lift)

OUT = R3 / "outputs"
K, TOPN, TOPDOCS = 6, 20, 25
THRESHOLDS = [0.500, 0.525, 0.550, 0.575, 0.600, 0.625, 0.650, 0.675,
              0.700, 0.725, 0.750, 0.800]

c = load_corpus("B250")
bow, dic, pids = c["bow"], c["dictionary"], c["paper_ids"]
vocab = [dic[i] for i in range(len(dic))]
cfs = np.array([dic.cfs[i] for i in range(len(dic))], float)
cfs_norm = cfs / cfs.sum()

models = load_k6_models()
TW = np.vstack([models[s].get_topics() for s in SEEDS])
TH = {s: theta(models[s], bow, K) for s in SEEDS}
labels = [(s, t) for s in SEEDS for t in range(K)]
S = cosine_matrix(TW, TW)
np.fill_diagonal(S, 1.0)
D = np.clip(1.0 - S, 0, None)
np.fill_diagonal(D, 0.0)
Z = linkage(squareform(D, checks=False), method="average")

rows = []
for t in THRESHOLDS:
    cl = fcluster(Z, t=1 - t, criterion="distance")
    qual, details = 0, []
    for cid in set(cl):
        mem = [i for i in range(len(labels)) if cl[i] == cid]
        seeds_in = [labels[i][0] for i in mem]
        if len(set(seeds_in)) < 4 or len(set(seeds_in)) != len(seeds_in):
            continue
        w = TW[mem].mean(0)
        w = w / w.sum()
        terms = [vocab[i] for i in np.argsort(-w)[:TOPN]]
        cnt = Counter()
        for i in mem:
            cnt.update([vocab[j] for j in np.argsort(-TW[i])[:TOPN]])
        core = [x for x in terms if cnt[x] >= 4]
        th_cols = [TH[labels[i][0]][:, labels[i][1]] for i in mem]
        td = [set(np.argsort(-tc)[:TOPDOCS]) for tc in th_cols]
        jac = float(np.mean([len(a & b) / len(a | b)
                             for a, b in itertools.combinations(td, 2)]))
        lift = term_lift(w, vocab, cfs_norm, TOPN)
        if len(core) >= 4 and jac >= 0.15 and lift >= 2.0:
            qual += 1
            details.append(", ".join(core[:5]))
    rows.append({"threshold": t, "n_qualifying_themes": qual,
                 "qualifying_cores": " || ".join(details)})
    print(f"  threshold {t:.3f}: {qual} qualifying theme(s)")

df = pd.DataFrame(rows)
df.to_csv(OUT / "C13_qualification_across_plateau.csv", index=False)

print("\n" + "=" * 110)
print("QUALIFYING THEMES ACROSS THE THRESHOLD PLATEAU")
print(df.to_string(index=False, max_colwidth=74))
inv = df[df.threshold <= 0.750]["n_qualifying_themes"].nunique() == 1
print(f"\nConclusion invariant across the 0.50-0.75 plateau: {inv}")
print(f"At the cluster-count maximiser (0.800): "
      f"{int(df[df.threshold == 0.800]['n_qualifying_themes'].iloc[0])} qualifying")
