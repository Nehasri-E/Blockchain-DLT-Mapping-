import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim.models import LdaModel
from scipy.stats import linregress
from scipy.spatial.distance import cosine
from pathlib import Path

DATA = Path("/home/claude/blockchain_lda/data")
OUT = Path("/mnt/user-data/outputs")
FIG = OUT / "figures"
K = 12
TREND_YEARS = list(range(2019, 2026))

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
GOOD, CRIT = "#0ca30c", "#d03b3b"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8, "xtick.color": MUTED,
    "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "grid.color": GRID, "grid.linewidth": 0.7, "legend.frameon": False, "legend.fontsize": 9,
})

LABELS = {
    1: "Empirical Market and Disclosure Effects",
    2: "Supply Chain Operations and Integration",
    3: "FinTech, Payments and Financial Markets",
    4: "Pricing and Channel Strategy",
    5: "System Architecture and Security Design",
    6: "Cryptocurrency Markets and Token Pricing",
    7: "Smart Contract Protocols and Schemes",
    8: "Sustainable Adoption and Firm Performance",
    9: "Trust, Transparency and Sector Applications",
    10: "Governance, Capabilities and Ecosystems",
    11: "NFTs, Marketing and Research Agenda",
    12: "Risk Management, Accounting and Credit",
}


def strip(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


corpora = pickle.load(open(DATA / "corpora.pkl", "rb"))
model = pickle.load(open(DATA / "candidates.pkl", "rb"))[K]
meta = pd.read_parquet(DATA / "lda_master_capped.parquet")
cb = corpora["B"]

dist = np.zeros((len(cb["bow"]), K))
for i, doc in enumerate(cb["bow"]):
    for t, p in model.get_document_topics(doc, minimum_probability=0.0):
        dist[i, t] = p

topics = pd.DataFrame(dist, columns=[f"Topic_{i + 1}" for i in range(K)])
topics.insert(0, "Paper_ID", cb["paper_ids"])
df = meta.merge(topics, on="Paper_ID", how="inner", validate="one_to_one")
assert len(df) == len(topics), "topic-metadata join lost rows"
print(f"Joined topic distributions to metadata: {len(df)} papers, no rows lost")

tcols = [f"Topic_{i + 1}" for i in range(K)]
df["Dominant_Topic"] = df[tcols].values.argmax(1) + 1
df["Dominant_Share"] = df[tcols].values.max(1)
df["Dominant_Label"] = df["Dominant_Topic"].map(LABELS)

keywords = {i + 1: [w for w, _ in model.show_topic(i, topn=15)] for i in range(K)}
prevalence = df[tcols].mean().values

summary = pd.DataFrame({
    "Topic": range(1, K + 1),
    "Label": [LABELS[i] for i in range(1, K + 1)],
    "Prevalence": prevalence.round(4),
    "Papers_Dominant": [int((df["Dominant_Topic"] == i).sum()) for i in range(1, K + 1)],
    "Top_Keywords": [", ".join(keywords[i][:10]) for i in range(1, K + 1)],
})

rows = []
for i in range(1, K + 1):
    top = df.nlargest(3, f"Topic_{i}")
    for _, r in top.iterrows():
        rows.append({"Topic": i, "Label": LABELS[i], "Share": round(r[f"Topic_{i}"], 3),
                     "Year": r["Year"], "Title": r["Title_Clean"], "Journal": r["Journal"]})
representative = pd.DataFrame(rows)

yearly = df[df["Year"].isin(TREND_YEARS)].groupby("Year")[tcols].mean()
counts = df[df["Year"].isin(TREND_YEARS)].groupby("Year").size()

trend = []
for i, c in enumerate(tcols, 1):
    s = linregress(yearly.index.values, yearly[c].values)
    trend.append({"Topic": i, "Label": LABELS[i],
                  "Slope_per_year": round(s.slope, 5), "p_value": round(s.pvalue, 4),
                  "Start_2019": round(yearly[c].iloc[0], 4),
                  "End_2025": round(yearly[c].iloc[-1], 4),
                  "Direction": "rising" if s.slope > 0 else "declining",
                  "Significant": s.pvalue < 0.05})
trend = pd.DataFrame(trend).sort_values("Slope_per_year", ascending=False)

summary.to_csv(OUT / "topic_summary.csv", index=False)
representative.to_csv(OUT / "representative_papers.csv", index=False)
trend.to_csv(OUT / "topic_trends.csv", index=False)
yearly.round(5).to_csv(OUT / "topic_prevalence_by_year.csv")
df[["Paper_ID", "Title_Clean", "Year", "Journal", "DOI", "Authors", "CABS_Rank",
    "ABDC_Rank", "Dominant_Topic", "Dominant_Label", "Dominant_Share"] + tcols].to_csv(
    OUT / "papers_with_topics.csv", index=False)

print("\n" + "=" * 92)
print(summary.to_string(index=False, max_colwidth=62))
print("\n" + "=" * 92)
print("TOPIC TRENDS 2019-2025")
print(trend.to_string(index=False, max_colwidth=42))

order = np.argsort(prevalence)
fig, ax = plt.subplots(figsize=(9.5, 5.6))
ax.barh(range(K), prevalence[order] * 100, color=BLUE, height=0.66, zorder=3)
for i, o in enumerate(order):
    ax.text(prevalence[o] * 100 + 0.12, i, f"{prevalence[o] * 100:.1f}%",
            va="center", fontsize=8.5, color=INK2)
ax.set_yticks(range(K))
ax.set_yticklabels([f"T{o + 1}  {LABELS[o + 1]}" for o in order], fontsize=9)
ax.set_title(f"Topic prevalence across {len(df)} blockchain and DLT papers (K={K})")
ax.set_xlabel("Mean share of corpus (%)")
ax.xaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.set_xlim(0, prevalence.max() * 100 * 1.14)
strip(ax)
fig.savefig(FIG / "08_topic_prevalence.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("\n  wrote 08_topic_prevalence.png")

fig, axes = plt.subplots(3, 4, figsize=(14, 8.4), sharex=True, sharey=True)
ymax = yearly.values.max() * 1.18
for i, ax in enumerate(axes.ravel(), 1):
    y = yearly[f"Topic_{i}"].values
    s = linregress(yearly.index.values, y)
    col = GOOD if (s.slope > 0 and s.pvalue < 0.05) else CRIT if (s.slope < 0 and s.pvalue < 0.05) else MUTED
    ax.plot(yearly.index, y, color=BLUE, linewidth=2, marker="o", markersize=4.5,
            markeredgecolor=SURFACE, markeredgewidth=1, zorder=3)
    ax.plot(yearly.index, s.intercept + s.slope * yearly.index.values, color=col,
            linewidth=1.6, linestyle="--", zorder=2)
    tag = "rising" if s.slope > 0 else "declining"
    sig = "" if s.pvalue < 0.05 else " (n.s.)"
    ax.set_title(f"T{i}  {LABELS[i]}", fontsize=9, loc="left")
    ax.text(0.03, 0.90, f"{tag}{sig}", transform=ax.transAxes, fontsize=8,
            color=col, fontweight="bold", va="top")
    ax.set_ylim(0, ymax)
    ax.set_xticks(TREND_YEARS[::2])
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    strip(ax)
for ax in axes[:, 0]:
    ax.set_ylabel("Mean topic share")
for ax in axes[-1, :]:
    ax.set_xlabel("Year")
fig.suptitle("Topic evolution 2019 to 2025   (dashed line = OLS trend; green rising, red declining, grey not significant at p<0.05)",
             fontsize=12, fontweight="bold", color=INK, y=1.00)
fig.tight_layout()
fig.savefig(FIG / "09_topic_evolution.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  wrote 09_topic_evolution.png")

topj = df["Journal"].value_counts().head(12).index
mat = df[df["Journal"].isin(topj)].groupby("Journal")[tcols].mean().loc[topj]
fig, ax = plt.subplots(figsize=(12, 5.6))
im = ax.imshow(mat.values, cmap="Blues", aspect="auto", vmin=0)
ax.set_xticks(range(K))
ax.set_xticklabels([f"T{i + 1}" for i in range(K)], fontsize=9)
ax.set_yticks(range(len(topj)))
ax.set_yticklabels([j if len(j) <= 46 else j[:43] + "..." for j in topj], fontsize=8.5)
for r in range(len(topj)):
    for c in range(K):
        v = mat.values[r, c]
        if v >= 0.10:
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if v > mat.values.max() * 0.6 else INK)
ax.set_title("Mean topic share by journal (top 12 journals by paper count)")
cb_ = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.015)
cb_.set_label("Mean topic share", color=INK2, fontsize=9)
cb_.outline.set_visible(False)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
fig.savefig(FIG / "10_topic_by_journal.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  wrote 10_topic_by_journal.png")

ca = corpora["A"]
model_a = LdaModel(corpus=ca["bow"], id2word=ca["dictionary"], num_topics=K,
                   alpha="symmetric", eta="auto", passes=20, iterations=200,
                   random_state=42, chunksize=200, eval_every=None)

vb = {w: i for i, w in enumerate(set(cb["dictionary"].values()) | set(ca["dictionary"].values()))}


def dense(m, d):
    T = m.get_topics()
    out = np.zeros((K, len(vb)))
    for t in range(K):
        for tid, p in enumerate(T[t]):
            out[t, vb[d[tid]]] = p
    return out


A, B = dense(model_a, ca["dictionary"]), dense(model, cb["dictionary"])
sim = np.array([[1 - cosine(B[i], A[j]) for j in range(K)] for i in range(K)])
best = sim.max(1)
print(f"\nCorpus A robustness: {int((best >= 0.5).sum())}/{K} Corpus B topics have a "
      f"Corpus A counterpart at cosine >= 0.50 (mean best match {best.mean():.3f})")

fig, ax = plt.subplots(figsize=(8.6, 6.4))
im = ax.imshow(sim, cmap="Blues", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(K))
ax.set_xticklabels([f"A{i + 1}" for i in range(K)], fontsize=9)
ax.set_yticks(range(K))
ax.set_yticklabels([f"B{i + 1}  {LABELS[i + 1][:30]}" for i in range(K)], fontsize=8)
for r in range(K):
    c = int(sim[r].argmax())
    ax.text(c, r, f"{sim[r, c]:.2f}", ha="center", va="center", fontsize=7.5,
            color="#ffffff" if sim[r, c] > 0.6 else INK, fontweight="bold")
ax.set_title("Robustness check: Corpus B topics matched to Corpus A (abstract-only) topics")
ax.set_xlabel("Corpus A topic")
cb_ = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cb_.set_label("Cosine similarity of topic-word distributions", color=INK2, fontsize=9)
cb_.outline.set_visible(False)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
fig.savefig(FIG / "11_corpus_robustness.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  wrote 11_corpus_robustness.png")

pd.DataFrame(sim.round(3), index=[f"B{i + 1}" for i in range(K)],
             columns=[f"A{i + 1}" for i in range(K)]).to_csv(OUT / "corpus_robustness.csv")

vis = gensimvis.prepare(model, cb["bow"], cb["dictionary"], sort_topics=False, mds="mmds")
raw = pyLDAvis.prepared_data_to_html(vis)

js = Path(pyLDAvis.__file__).parent / "js"
el = raw.split('<div id="')[1].split('"')[0]
body = raw.split("var " + el + "_data = ")[1].rsplit("function LDAvis_load_lib", 1)[0].strip()
data = body.rsplit(";", 1)[0] if body.endswith(";") else body.split("\n")[0].rstrip(";")

html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blockchain and DLT topic model (K={K})</title>
<style>{(js / 'ldavis.v1.0.0.css').read_text()}
body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
        background:{SURFACE}; color:{INK}; margin:0; padding:24px 16px; }}
h1 {{ font-size:18px; margin:0 0 4px; }}
p.sub {{ color:{INK2}; font-size:13px; margin:0 0 20px; max-width:900px; }}
#vis {{ background:#fff; }}</style></head>
<body>
<h1>Blockchain and DLT research landscape: LDA topic model</h1>
<p class="sub">Corpus B, {len(df)} papers, K={K}, alpha=symmetric, eta=auto, seed 42. Circle area is
topic prevalence; distance is inter-topic similarity (multidimensional scaling). Drag the relevance
slider toward 0 to see terms distinctive to a topic, toward 1 for the most frequent terms.</p>
<div id="vis"></div>
<script>{(js / 'd3.v5.min.js').read_text()}</script>
<script>{(js / 'ldavis.v3.0.0.js').read_text()}</script>
<script>new LDAvis("#vis", {data});</script>
</body></html>"""

(OUT / "pyldavis_corpus_b.html").write_text(html, encoding="utf-8")
print(f"  wrote pyldavis_corpus_b.html (self-contained, {len(html) / 1e6:.1f} MB)")
