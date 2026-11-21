import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gensim.models import LdaModel
from scipy.spatial.distance import jensenshannon
from pathlib import Path

DATA = Path("/home/claude/blockchain_lda/data")
FIG = Path("/mnt/user-data/outputs/figures")

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11.5,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8, "xtick.color": MUTED,
    "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "grid.color": GRID, "grid.linewidth": 0.7, "legend.frameon": False, "legend.fontsize": 9,
})


def strip(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


res = pd.read_csv(DATA / "tuning_results.csv")
corpora = pickle.load(open(DATA / "corpora.pkl", "rb"))

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
for ax, metric, label in [(axes[0], "c_v", "C_v coherence (higher is better)"),
                          (axes[1], "log_perplexity", "Log perplexity (higher is better)")]:
    for corpus, color, name in [("B", BLUE, "Corpus B (abstract + conclusion)"),
                                ("A", ORANGE, "Corpus A (abstract only)")]:
        g = res[res.corpus == corpus].groupby("k")[metric].agg(["mean", "std"]).reset_index()
        ax.errorbar(g["k"], g["mean"], yerr=g["std"], color=color, linewidth=2,
                    marker="o", markersize=5, capsize=3, elinewidth=1,
                    markeredgecolor=SURFACE, markeredgewidth=1, label=name, zorder=3)
    ax.set_xlabel("Number of topics (K)")
    ax.set_ylabel(label)
    ax.set_xticks([4, 6, 8, 10, 12, 14, 16, 18, 20])
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    strip(ax)
lo, hi = axes[0].get_ylim()
axes[0].axvspan(10, 13, color=AQUA, alpha=0.12, zorder=0)
axes[0].annotate("K = 12 selected\n(see distinctiveness\nand interpretability)",
                 xy=(12, 0.3875), xytext=(15.2, 0.405), fontsize=8.5, color=INK2,
                 ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=1))
axes[0].set_ylim(lo, hi)
axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, 0.02))
fig.suptitle("LDA model selection: 312 models, 4 prior settings, 3 random seeds per configuration",
             fontsize=12.5, fontweight="bold", color=INK, y=1.02)
fig.savefig(FIG / "07_model_selection.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  wrote 07_model_selection.png")

b = res[res.corpus == "B"]
band = b.groupby("k")["c_v"].agg(["mean", "std"])
peak = band["mean"].max()
within = band[band["mean"] + band["std"] >= peak - band.loc[band["mean"].idxmax(), "std"]]
print(f"\nCorpus B peak mean C_v = {peak:.4f} at K={band['mean'].idxmax()}")
print(f"K values statistically indistinguishable from the peak (1 SD overlap): {list(within.index)}")

candidates = {}
for k, a, e in [(6, "asymmetric", "symmetric"), (12, "symmetric", "auto")]:
    c = corpora["B"]
    m = LdaModel(corpus=c["bow"], id2word=c["dictionary"], num_topics=k, alpha=a,
                 eta=e, passes=20, iterations=200, random_state=42, chunksize=200,
                 eval_every=None)
    T = m.get_topics()
    js = [jensenshannon(T[i], T[j], base=2) for i in range(k) for j in range(i + 1, k)]
    candidates[k] = {"model": m, "js_mean": float(np.mean(js)), "js_min": float(np.min(js))}
    print(f"\n{'=' * 78}\nK={k}  alpha={a}  eta={e}   "
          f"topic distinctiveness: mean JS {np.mean(js):.3f}, closest pair {np.min(js):.3f}")
    for i in range(k):
        words = [w for w, _ in m.show_topic(i, topn=10)]
        print(f"  T{i + 1:>2}: {', '.join(words)}")

with open(DATA / "candidates.pkl", "wb") as f:
    pickle.dump({k: v["model"] for k, v in candidates.items()}, f)

print(f"\n{'=' * 78}")
for k, v in candidates.items():
    print(f"K={k}: mean pairwise Jensen-Shannon {v['js_mean']:.3f}, "
          f"closest topic pair {v['js_min']:.3f}")
