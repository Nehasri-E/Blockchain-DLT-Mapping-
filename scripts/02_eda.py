import pandas as pd
import numpy as np
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import Counter
from pathlib import Path

DATA = Path("/home/claude/blockchain_lda/data")
FIG = Path("/mnt/user-data/outputs/figures")
FIG.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
PLOT_YEARS = (2019, 2025)

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12.5,
    "axes.titleweight": "semibold",
    "axes.titlecolor": INK,
    "axes.labelcolor": INK2,
    "axes.labelsize": 10,
    "axes.edgecolor": BASE,
    "axes.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "legend.frameon": False,
    "legend.fontsize": 9,
})


def strip(ax, keep=("left", "bottom")):
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(side in keep)
    ax.tick_params(length=0)


def save(fig, name):
    fig.savefig(FIG / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


df = pd.read_parquet(DATA / "lda_master.parquet")
qc = pd.read_csv(DATA / "quality_control.csv")

print(f"Analysis corpus: {len(df)} papers, {df['Year'].min()} to {df['Year'].max()}")
print(f"Journals: {df['Journal'].nunique()}   Unique DOIs: {df['DOI'].ne('').sum()}")
print()

summary = pd.DataFrame({
    "Metric": [
        "Papers in corpus", "Distinct journals", "Publication years",
        "Median abstract words", "Median conclusion words",
        "Median Corpus A document words", "Median Corpus B document words",
        "Papers with valid conclusion", "Papers abstract-only in Corpus B",
    ],
    "Value": [
        len(df), df["Journal"].nunique(), f"{df['Year'].min()}-{df['Year'].max()}",
        int(df["Abstract_Words"].median()), int(df["Conclusion_Words"].median()),
        int(df["Doc_Words_A"].median()), int(df["Doc_Words_B"].median()),
        int(df["Conclusion_Valid"].sum()), int((~df["Conclusion_Valid"]).sum()),
    ],
})
summary.to_csv(FIG.parent / "eda_summary.csv", index=False)
print(summary.to_string(index=False))
print()

counts = df["Year"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(8.5, 4.2))
cols = [BLUE if PLOT_YEARS[0] <= y <= PLOT_YEARS[1] else MUTED for y in counts.index]
bars = ax.bar(counts.index, counts.values, color=cols, width=0.68, zorder=3)
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 4, str(v), ha="center",
            va="bottom", fontsize=8.5, color=INK2)
ax.set_title("Blockchain and DLT papers by publication year")
ax.set_xlabel("Publication year")
ax.set_ylabel("Papers")
ax.set_xticks(counts.index)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.set_ylim(0, counts.max() * 1.14)
strip(ax)
ax.text(0.5, 0.97, f"Grey years fall outside the {PLOT_YEARS[0]}-{PLOT_YEARS[1]} trend window "
        f"(n < 30 or partial year); all years still train the model.",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=MUTED)
save(fig, "01_papers_by_year.png")

top = df["Journal"].value_counts().head(15).sort_values()
fig, ax = plt.subplots(figsize=(8.5, 5.6))
bars = ax.barh(range(len(top)), top.values, color=BLUE, height=0.66, zorder=3)
for i, v in enumerate(top.values):
    ax.text(v + 1.5, i, str(v), va="center", fontsize=8.5, color=INK2)
ax.set_yticks(range(len(top)))
ax.set_yticklabels([t if len(t) <= 52 else t[:49] + "..." for t in top.index], fontsize=8.5)
ax.set_title(f"Top 15 source journals ({df['Journal'].nunique()} journals in corpus)")
ax.set_xlabel("Papers")
ax.xaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.set_xlim(0, top.max() * 1.12)
strip(ax)
save(fig, "02_top_journals.png")

fig, ax = plt.subplots(figsize=(8.5, 4.4))
bins = np.arange(0, 1600, 40)
ax.hist(df["Abstract_Words"].clip(upper=1580), bins=bins, color=BLUE,
        alpha=0.85, label="Abstract", zorder=3)
ax.hist(df["Conclusion_Words"].clip(upper=1580), bins=bins, color=ORANGE,
        alpha=0.72, label="Conclusion", zorder=3)
ymax = ax.get_ylim()[1]
for v, c, lab, frac in [(df["Abstract_Words"].median(), BLUE, "Abstract", 0.97),
                        (df["Conclusion_Words"].median(), ORANGE, "Conclusion", 0.86)]:
    ax.axvline(v, color=c, linewidth=1.6, linestyle="--", zorder=4)
    ax.annotate(f"{lab} median {int(v)}w", xy=(v, ymax * frac), xytext=(v + 30, ymax * frac),
                fontsize=8.5, color=c, fontweight="bold", va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none"))
ax.set_title("Abstract and conclusion length distributions")
ax.set_xlabel("Words (clipped at 1,580 for display)")
ax.set_ylabel("Papers")
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.80))
strip(ax)
save(fig, "03_text_length_distributions.png")

qs = [0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
lentab = pd.DataFrame({
    "Percentile": [f"{int(q * 100)}th" for q in qs],
    "Abstract words": [int(df["Abstract_Words"].quantile(q)) for q in qs],
    "Conclusion words": [int(df["Conclusion_Words"].quantile(q)) for q in qs],
})
lentab.to_csv(FIG.parent / "length_percentiles.csv", index=False)
print(lentab.to_string(index=False))
print()

share = df["Conclusion_Words"].sum() / (df["Abstract_Words"] + df["Conclusion_Words"] + 0.0).sum()
print(f"Conclusion share of Corpus B tokens: {share:.1%}")
over = [(c, int((df['Conclusion_Words'] > c).sum())) for c in [250, 400, 500, 750, 1000, 2000]]
print("Papers with conclusion longer than cap:", ", ".join(f"{c}w: {n}" for c, n in over))
print()

avail = pd.Series({
    "Title, abstract and conclusion": int(df["Conclusion_Valid"].sum()),
    "Title and abstract only": int((~df["Conclusion_Valid"]).sum()),
    "Excluded: duplicate record": int(qc["Duplicate_Record"].sum()),
    "Excluded: correction notice": int(qc["Is_Correction"].sum()),
    "Excluded: abstract under 50 words": int((~qc["Abstract_Valid"]).sum()),
    "Excluded: Excel spill row": 3,
}).sort_values()
fig, ax = plt.subplots(figsize=(8.5, 4.0))
cols = [AQUA if "Excluded" not in i else MUTED for i in avail.index]
bars = ax.barh(range(len(avail)), avail.values, color=cols, height=0.62, zorder=3)
for i, v in enumerate(avail.values):
    ax.text(v + 8, i, str(v), va="center", fontsize=8.5, color=INK2)
ax.set_yticks(range(len(avail)))
ax.set_yticklabels(avail.index, fontsize=9)
ax.set_title("Text availability and exclusion reasons (1,003 raw records)")
ax.set_xlabel("Records")
ax.xaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.set_xlim(0, avail.max() * 1.15)
strip(ax)
save(fig, "04_text_availability.png")

STOP = set("""a an the and or but if of to in on for with by from as at is are was were be been being
this that these those it its their there which who whom whose what when where how we our us they them
he she his her i you your can could will would shall should may might must not no nor do does did done
has have had having than then so such also more most other some any each both between into through
during before after above below up down out off over under again further once here only own same too
very s t don now d ll m o re ve y ain aren couldn didn doesn hadn hasn haven isn ma mightn mustn needn
shan shouldn wasn weren won wouldn use used using based new paper study research results result find
found show shows shown propose proposed approach method methods analysis data model models""".split())

tok = re.compile(r"[a-z][a-z\-]{2,}")
uni, bi = Counter(), Counter()
for t in df["Document_Text_B"]:
    ws = [w for w in tok.findall(t.lower()) if w not in STOP]
    uni.update(ws)
    bi.update(f"{a} {b}" for a, b in zip(ws, ws[1:]))

for data, name, title, color in [
    (uni, "05_top_unigrams.png", "Top 20 unigrams in raw Corpus B text", BLUE),
    (bi, "06_top_bigrams.png", "Top 20 bigrams in raw Corpus B text", ORANGE),
]:
    top20 = pd.Series(dict(data.most_common(20))).sort_values()
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.barh(range(len(top20)), top20.values, color=color, height=0.66, zorder=3)
    for i, v in enumerate(top20.values):
        ax.text(v + top20.max() * 0.012, i, f"{v:,}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20.index, fontsize=9)
    ax.set_title(title + "  (generic academic stopwords already removed)")
    ax.set_xlabel("Occurrences")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(0, top20.max() * 1.14)
    strip(ax)
    save(fig, name)

pd.DataFrame(uni.most_common(60), columns=["Term", "Count"]).to_csv(
    FIG.parent / "top_unigrams.csv", index=False)
pd.DataFrame(bi.most_common(60), columns=["Bigram", "Count"]).to_csv(
    FIG.parent / "top_bigrams.csv", index=False)

print("Top 25 unigrams:")
print(", ".join(f"{w} ({c})" for w, c in uni.most_common(25)))
print()
print("Top 20 bigrams:")
print(", ".join(f"{w} ({c})" for w, c in bi.most_common(20)))
