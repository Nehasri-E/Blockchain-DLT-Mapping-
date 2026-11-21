"""Task 6: final publication figures and tables.

Six main figures. Everything else is supplementary. Temporal trends are NOT a
main figure, because the primary inference does not support them.

All figures are generated from result CSVs, never from model objects.
"""
import json
import shutil
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

ROOT = Path("/home/claude/LDA_Project")
R1, R2, R3, R4 = (ROOT / f"validation_round{i}" for i in (1, 2, 3, 4))
OUT, TAB, FIG, SUP = R4 / "outputs", R4 / "FINAL_TABLES", R4 / "FINAL_FIGURES", R4 / "SUPPLEMENTARY_MATERIAL"
for d in (TAB, FIG, SUP):
    d.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
GOOD, CRIT, WARN = "#0ca30c", "#d03b3b", "#fab219"

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


def save(fig, n):
    fig.savefig(FIG / n, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {n}")


themes = pd.read_csv(TAB / "Table2_final_themes.csv")
papers = pd.read_csv(ROOT / "outputs" / "papers_with_topics.csv")
recon = pd.read_csv(ROOT / "data" / "interim" / "reconciliation.csv")

# ----------------------------------------------------------------- Figure 1
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2),
                         gridspec_kw={"width_ratios": [1.0, 1.35, 1.0]})
fig.subplots_adjust(wspace=0.60)
yc = papers["Year"].value_counts().sort_index()
ax = axes[0]
cols = [BLUE if 2019 <= y <= 2025 else MUTED for y in yc.index]
ax.bar(yc.index, yc.values, color=cols, width=0.68, zorder=3)
for y, v in yc.items():
    ax.text(y, v + 4, str(v), ha="center", fontsize=8, color=INK2)
ax.set_title("Papers by publication year", fontsize=10.5)
ax.set_xlabel("Year")
ax.set_ylabel("Papers")
ax.set_xticks(yc.index)
ax.tick_params(axis="x", rotation=45)
ax.set_ylim(0, yc.max() * 1.18)

ax = axes[1]
jc = papers["Journal"].value_counts().head(8).sort_values()
ax.barh(range(len(jc)), jc.values, color=BLUE, height=0.66, zorder=3)
for i, v in enumerate(jc.values):
    ax.text(v + 1.5, i, str(v), va="center", fontsize=8, color=INK2)
ax.set_yticks(range(len(jc)))
ABBR = {"IEEE Transactions on Engineering Management": "IEEE Trans. Eng. Management",
        "International Journal of Production Research": "Int. J. Production Research",
        "Technological Forecasting and Social Change": "Technol. Forecast. Soc. Change",
        "International Journal of Production Economics": "Int. J. Production Economics",
        "Transportation Research Part E: Logistics and Transportation Review":
            "Transp. Research Part E",
        "Business Strategy and the Environment": "Business Strategy & Environment"}
ax.set_yticklabels([ABBR.get(j, j if len(j) <= 30 else j[:28] + "...") for j in jc.index],
                   fontsize=8)
ax.set_title(f"Top 8 of {papers['Journal'].nunique()} journals", fontsize=10.5)
ax.set_xlabel("Papers")
ax.set_xlim(0, jc.max() * 1.15)

ax = axes[2]
steps = recon["Stage"].tolist()
vals = recon["Rows"].tolist()
ax.plot(range(len(vals)), vals, color=ORANGE, marker="o", linewidth=2, markersize=7,
        markeredgecolor=SURFACE, markeredgewidth=1.3, zorder=3)
for i, v in enumerate(vals):
    ax.text(i, v + 4, str(v), ha="center", fontsize=8.5, color=INK2)
ax.set_xticks(range(len(steps)))
ax.set_xticklabels([s.replace("Less ", "- ").replace("Raw rows in EDA_INPUT.csv", "raw")
                    .replace("Final analysis corpus", "final") for s in steps],
                   rotation=38, ha="right", fontsize=7.5)
ax.set_title("Inclusion cascade: 1,003 to 987", fontsize=10.5)
ax.set_ylabel("Papers")
ax.set_ylim(min(vals) - 12, max(vals) + 14)
for a in axes:
    a.yaxis.grid(True, zorder=0)
    a.set_axisbelow(True)
    strip(a)
fig.suptitle("Figure 1. Research-paper corpus overview (987 papers, blockchain in operations and supply chain management)",
             fontsize=12, fontweight="bold", color=INK, y=1.04)
save(fig, "Figure1_corpus_overview.png")

# ----------------------------------------------------------------- Figure 2
t = themes.sort_values("prevalence_share_of_W", ascending=True)
fig, ax = plt.subplots(figsize=(11.5, 5.4))
colmap = {"Strongly supported": GOOD, "Supported with qualification": WARN,
          "Not presented as a validated theme": MUTED}
cols = [colmap[s] for s in t["evidence_strength"]]
y = np.arange(len(t))
ax.barh(y, t["prevalence_share_of_W"] * 100, color=cols, height=0.62, zorder=3)
for i, r in enumerate(t.itertuples()):
    ax.text(r.prevalence_share_of_W * 100 + 0.25, i,
            f"{r.prevalence_share_of_W*100:.1f}%", va="center", fontsize=9, color=INK2)
    core = ", ".join(str(r.stable_term_core).split(", ")[:7])
    ax.text(0.4, i - 0.27, core, fontsize=7.8, color=INK2, style="italic", va="top")
ax.set_yticks(y)
ax.set_yticklabels([f"N{int(r.nmf_topic)}  {r.final_label}" for r in t.itertuples()],
                   fontsize=9)
ax.set_xlabel("Share of total document-topic mass (%)")
ax.set_xlim(0, 22)
ax.set_title("Figure 2. NMF six-topic descriptive map of the corpus\n"
             "Three topics carry validated substantive themes; three are retained as components only",
             fontsize=11.5)
ax.legend(handles=[Line2D([], [], color=c, linewidth=9, label=l) for l, c in colmap.items()],
          loc="lower right", bbox_to_anchor=(1.0, -0.26), ncol=3, fontsize=8.5)
ax.xaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
strip(ax)
save(fig, "Figure2_nmf_topic_map.png")

# ----------------------------------------------------------------- Figure 3
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
ax = axes[0]
tt = themes.sort_values("prevalence_share_of_W", ascending=False)
cols = [colmap[s] for s in tt["evidence_strength"]]
ax.bar(range(len(tt)), tt["prevalence_share_of_W"] * 100, color=cols, width=0.62, zorder=3)
ax.axhline(100 / 6, color=INK2, linestyle="--", linewidth=1.2, zorder=2)
ax.text(5.3, 100 / 6 + 0.25, "equal share (16.7%)", fontsize=8, color=INK2, ha="right")
for i, r in enumerate(tt.itertuples()):
    ax.text(i, r.prevalence_share_of_W * 100 + 0.2, f"{r.prevalence_share_of_W*100:.1f}",
            ha="center", fontsize=8.5, color=INK2)
ax.set_xticks(range(len(tt)))
ax.set_xticklabels([f"N{int(r.nmf_topic)}" for r in tt.itertuples()])
ax.set_ylabel("Share of document-topic mass (%)")
ax.set_ylim(0, 20)
ax.set_title("Topic prevalence is near-uniform", fontsize=10.5)

ax = axes[1]
ax.scatter(themes["distinctiveness_js"], themes["term_lift_vs_corpus"],
           s=themes["prevalence_share_of_W"] * 1400,
           color=[colmap[s] for s in themes["evidence_strength"]],
           zorder=3, edgecolor=SURFACE, linewidth=1.5)
for r in themes.itertuples():
    ax.annotate(f"N{int(r.nmf_topic)}", (r.distinctiveness_js, r.term_lift_vs_corpus),
                textcoords="offset points", xytext=(11, 5), fontsize=9.5, color=INK2)
ax.axhline(2.0, color=INK2, linestyle="--", linewidth=1.1, zorder=1)
ax.text(ax.get_xlim()[0], 2.06, "generic-vocabulary floor", fontsize=8, color=INK2)
ax.set_xlabel("Distinctiveness (mean JS distance to other topics)")
ax.set_ylabel("Term lift over corpus frequency")
ax.set_title("Specificity vs distinctiveness\n(bubble area = prevalence)", fontsize=10.5)
ax.margins(0.18)
for a in axes:
    a.yaxis.grid(True, zorder=0)
    a.set_axisbelow(True)
    strip(a)
fig.suptitle("Figure 3. Topic prevalence and specificity", fontsize=12,
             fontweight="bold", color=INK, y=1.03)
save(fig, "Figure3_prevalence_distribution.png")

# ----------------------------------------------------------------- Figure 4
val = themes[themes.presented_as_theme].sort_values("lda_nmf_cosine", ascending=False)
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
fig.subplots_adjust(wspace=0.30)
for ax, r in zip(axes, val.itertuples()):
    terms = str(r.top20_terms).split(", ")[:12]
    core = set(str(r.stable_term_core).split(", "))
    yy = np.arange(len(terms))[::-1]
    cols = [GOOD if tm in core else MUTED for tm in terms]
    ax.barh(yy, np.linspace(1.0, 0.42, len(terms)), color=cols, height=0.66, zorder=3)
    for i, tm in enumerate(terms):
        ax.text(0.03, yy[i], tm, va="center", fontsize=9,
                color="#ffffff" if tm in core else INK, fontweight="bold" if tm in core else "normal")
    ax.set_xlim(0, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    lab = r.final_label if len(r.final_label) <= 38 else r.final_label[:36] + "..."
    ax.set_title(f"N{int(r.nmf_topic)}  {lab}\n"
                 f"{r.evidence_strength} · {r.prevalence_share_of_W*100:.1f}% of mass",
                 fontsize=9.5)
    for s in ax.spines.values():
        s.set_visible(False)
fig.legend(handles=[Line2D([], [], color=GOOD, linewidth=9, label="in the stable term core (present in >=4 of 6 initialisations)"),
                    Line2D([], [], color=MUTED, linewidth=9, label="top-20 term, not in the stable core")],
           loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.07))
fig.suptitle("Figure 4. Representative keywords for the three validated themes",
             fontsize=12, fontweight="bold", color=INK, y=1.03)
save(fig, "Figure4_theme_keywords.png")

# ----------------------------------------------------------------- Figure 5
x2 = pd.read_csv(R3 / "outputs" / "X2_lda_vs_nmf_crosswalk.csv")
n2 = pd.read_csv(R3 / "outputs" / "N2_nmf_stability_by_k.csv")
lk = pd.read_csv(R2 / "outputs" / "T11_k_sweep_summary.csv")
lk = lk[lk.k.isin([4, 6, 8])]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axes[0]
w = 0.34
xk = np.arange(3)
ax.bar(xk - w / 2, lk["mean_ari"].values, width=w, color=BLUE, label="LDA", zorder=3)
ax.bar(xk + w / 2, n2["mean_ari"].values, width=w, color=ORANGE, label="NMF", zorder=3)
for i in range(3):
    ax.text(i - w / 2, lk["mean_ari"].values[i] + 0.015, f"{lk['mean_ari'].values[i]:.2f}",
            ha="center", fontsize=8.5, color=INK2)
    ax.text(i + w / 2, n2["mean_ari"].values[i] + 0.015, f"{n2['mean_ari'].values[i]:.2f}",
            ha="center", fontsize=8.5, color=INK2)
ax.set_xticks(xk)
ax.set_xticklabels([f"K={k}" for k in n2["k"]])
ax.set_ylabel("Adjusted Rand Index across 6 initialisations")
ax.set_ylim(0, 1.0)
ax.set_title("Document-assignment reproducibility\n(identical corpus and vocabulary)", fontsize=10.5)
ax.legend(loc="upper right")

ax = axes[1]
xo = x2.sort_values("top25_paper_jaccard")
yy = np.arange(len(xo))
ax.barh(yy + 0.19, xo["term_core_jaccard"], height=0.36, color=BLUE,
        label="term-core overlap", zorder=3)
ax.barh(yy - 0.19, xo["top25_paper_jaccard"], height=0.36, color=AQUA,
        label="representative-paper overlap", zorder=3)
ax.axvline(0.30, color=BLUE, linestyle="--", linewidth=1.1, zorder=1)
ax.axvline(0.20, color=AQUA, linestyle=":", linewidth=1.3, zorder=1)
ax.set_yticks(yy)
ax.set_yticklabels([f"CT{int(r.lda_consensus_topic)} / N{int(r.nmf_consensus_topic)}"
                    for r in xo.itertuples()], fontsize=9)
for i, r in enumerate(xo.itertuples()):
    if r.convergent_validity:
        ax.text(0.80, i, "CONVERGENT", va="center", fontsize=8, color=GOOD, fontweight="bold")
ax.set_xlim(0, 1.1)
ax.set_xlabel("Jaccard overlap between LDA and NMF")
ax.set_title("Cross-method agreement on words AND papers", fontsize=10.5, pad=24)
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
for a in axes:
    a.grid(True, axis="both", zorder=0)
    a.set_axisbelow(True)
    strip(a)
fig.suptitle("Figure 5. LDA versus NMF cross-method validation", fontsize=12,
             fontweight="bold", color=INK, y=1.08)
save(fig, "Figure5_cross_method_validation.png")

# ----------------------------------------------------------------- Figure 6
s3 = pd.read_csv(R3 / "outputs" / "S3_journal_exclusion_sensitivity.csv")
s4 = pd.read_csv(R3 / "outputs" / "S4_bfull_corpus_sensitivity.csv").set_index("ref_consensus_topic")
fig, ax = plt.subplots(figsize=(11, 4.8))
order = themes.sort_values(["presented_as_theme", "journal_sensitivity_min_cosine"],
                           ascending=[False, False])
xk = np.arange(len(order))
w = 0.2
piv = s3.pivot(index="ref_consensus_topic", columns="dropped_journals",
               values="cosine_to_reference")
for i, dn in enumerate([1, 3, 5]):
    vals = [piv.loc[int(str(r.lda_counterpart).replace("CT", "")), dn] for r in order.itertuples()]
    ax.bar(xk + (i - 1.5) * w, vals, width=w, color=[BLUE, AQUA, ORANGE][i], zorder=3,
           label=f"drop top {dn} journal(s)")
bvals = [s4.loc[int(str(r.lda_counterpart).replace("CT", "")), "cosine_B250_vs_BFull"]
         for r in order.itertuples()]
ax.bar(xk + 1.5 * w, bvals, width=w, color=MUTED, zorder=3, label="uncapped-conclusion corpus")
ax.axhline(0.70, color=INK2, linestyle="--", linewidth=1.2, zorder=1)
ax.text(-0.45, 0.715, "0.70", fontsize=8.5, color=INK2)
ax.set_xticks(xk)
ax.set_xticklabels([f"N{int(r.nmf_topic)}\n{'THEME' if r.presented_as_theme else 'component'}"
                    for r in order.itertuples()], fontsize=8.5)
ax.set_ylabel("Cosine to the reference topic")
ax.set_ylim(0, 1.08)
ax.set_title("Figure 6. Topic robustness to influential journals and to the corpus definition\n"
             "(measured on the LDA counterpart, where refitting is tractable)", fontsize=11)
ax.legend(loc="lower right", ncol=2, fontsize=8.5)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
strip(ax)
save(fig, "Figure6_robustness.png")

# ----------------------------------------------------------------- supplementary
sup_map = [
    (R1 / "figures", "S1_round1_"), (R2 / "figures", "S2_round2_"), (R3 / "figures", "S3_round3_"),
]
n = 0
for src, pre in sup_map:
    for f in sorted(src.glob("*.png")):
        shutil.copy(f, SUP / f"{pre}{f.name}")
        n += 1
for f in [R1 / "figures" / "pyldavis_corpus_b.html"]:
    if f.exists():
        shutil.copy(f, SUP / f"S1_round1_{f.name}")
        n += 1
print(f"  copied {n} supplementary figures")

# ----------------------------------------------------------------- tables
t1 = pd.DataFrame({
    "Property": ["Papers analysed", "Source records", "Journals", "Year range",
                 "Trend window", "Vocabulary terms", "Median tokens per document",
                 "Primary model", "Initialisations", "Validated themes"],
    "Value": [len(papers), 1003, papers["Journal"].nunique(),
              f"{papers['Year'].min()}-{papers['Year'].max()}", "2019-2025", 3335, 191,
              "NMF K=6 on TF-IDF", 6, 3],
})
t1.to_csv(TAB / "Table1_corpus_summary.csv", index=False)

pd.read_csv(OUT / "F3_round3_discrepancy_table.csv").to_csv(
    TAB / "Table3_round3_corrections.csv", index=False)
v1 = pd.read_csv(R3 / "outputs" / "V1_trend_valid_inference.csv")
v1[["topic", "label", "doc_slope", "cr1_ci_low", "cr1_ci_high", "cr1_ci_excludes_zero",
    "p_wild_cluster", "p_wild_cluster_fdr", "p_wild_cluster_sig_fdr",
    "p_within_journal_perm", "p_within_journal_perm_fdr",
    "p_within_journal_perm_sig_fdr"]].to_csv(TAB / "Table4_temporal_inference.csv", index=False)
pd.read_csv(OUT / "F6_author_parsing_diagnostics.csv").to_csv(
    TAB / "Table6_author_feasibility.csv", index=False)

print(f"\n{len(list(FIG.glob('*.png')))} main figures, "
      f"{len(list(TAB.glob('*.csv')))} final tables, "
      f"{len(list(SUP.glob('*')))} supplementary items")
