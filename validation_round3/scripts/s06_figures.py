"""Round 3 figures. Generated from result CSVs only, never from model objects."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

R3 = Path("/home/claude/LDA_Project/validation_round3")
OUT, FIG = R3 / "outputs", R3 / "figures"
FIG.mkdir(parents=True, exist_ok=True)

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


# ------------------------------------------------------------------ G1
cons = pd.read_csv(OUT / "C10_consensus_topics_seedpure.csv").sort_values("mean_within_cosine")
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax = axes[0]
y = np.arange(len(cons))
ax.barh(y + 0.19, cons["mean_within_cosine"], height=0.36, color=BLUE,
        label="within-cluster cosine", zorder=3)
ax.barh(y - 0.19, cons["mean_between_cosine"], height=0.36, color=MUTED,
        label="between-cluster cosine", zorder=3)
for i, r in enumerate(cons.itertuples()):
    ax.text(r.mean_within_cosine + 0.012, i + 0.19, f"{r.mean_within_cosine:.2f}",
            va="center", fontsize=8.5, color=INK2)
ax.set_yticks(y)
ax.set_yticklabels([f"CT{int(r.consensus_topic)}" + ("  *" if r.qualifies else "")
                    for r in cons.itertuples()])
ax.set_xlabel("Cosine")
ax.set_title("Consensus cluster cohesion and separation\n(* = qualifies on all four criteria)",
             fontsize=10.5)
ax.legend(loc="lower right")
ax.set_xlim(0, 1.0)

ax = axes[1]
cols = [GOOD if r.qualifies else CRIT for r in cons.itertuples()]
ax.scatter(cons["mean_top25_doc_jaccard"], cons["term_lift_vs_corpus"],
           s=190 * cons["consensus_prevalence"] / cons["consensus_prevalence"].max() + 55,
           color=cols, zorder=3, edgecolor=SURFACE, linewidth=1.4)
for r in cons.itertuples():
    ax.annotate(f"CT{int(r.consensus_topic)}", (r.mean_top25_doc_jaccard, r.term_lift_vs_corpus),
                textcoords="offset points", xytext=(9, 4), fontsize=9, color=INK2)
ax.axvline(0.15, color=INK2, linestyle="--", linewidth=1.1, zorder=1)
ax.axhline(2.0, color=INK2, linestyle="--", linewidth=1.1, zorder=1)
ax.text(0.155, 1.72, "doc-overlap floor 0.15", fontsize=8, color=INK2)
ax.text(0.012, 2.06, "generic-vocabulary floor (lift 2.0)", fontsize=8, color=INK2)
ax.set_xlabel("Representative-paper overlap across seeds (top-25 Jaccard)")
ax.set_ylabel("Term lift over corpus frequency")
ax.set_title("A theme must recur in BOTH words and papers,\nand not be generic vocabulary",
             fontsize=10.5)
ax.margins(0.16)
for a in axes:
    a.grid(True, axis="x", zorder=0)
    a.set_axisbelow(True)
    strip(a)
save(fig, "G1_consensus_quality.png")

# ------------------------------------------------------------------ G2
nmf = pd.read_csv(OUT / "N2_nmf_stability_by_k.csv")
lda_k = pd.read_csv(Path("/home/claude/LDA_Project/validation_round2/outputs/T11_k_sweep_summary.csv"))
lda_k = lda_k[lda_k.k.isin([4, 6, 8])]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
for ax, lcol, ncol, ttl in [
    (axes[0], "mean_matched_cosine", "mean_matched_cosine", "Topic-word stability\n(matched cosine)"),
    (axes[1], "mean_ari", "mean_ari", "Document assignment stability\n(Adjusted Rand Index)"),
    (axes[2], "mean_c_v", "mean_c_v", "Coherence (C_v)")]:
    w = 0.34
    xk = np.arange(len(nmf))
    ax.bar(xk - w / 2, lda_k[lcol].values, width=w, color=BLUE, label="LDA", zorder=3)
    ax.bar(xk + w / 2, nmf[ncol].values, width=w, color=ORANGE, label="NMF", zorder=3)
    for i in range(len(nmf)):
        ax.text(i - w / 2, lda_k[lcol].values[i] + 0.015, f"{lda_k[lcol].values[i]:.2f}",
                ha="center", fontsize=8.5, color=INK2)
        ax.text(i + w / 2, nmf[ncol].values[i] + 0.015, f"{nmf[ncol].values[i]:.2f}",
                ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(xk)
    ax.set_xticklabels([f"K={k}" for k in nmf["k"]])
    ax.set_title(ttl, fontsize=10.5)
    ax.set_ylim(0, max(lda_k[lcol].max(), nmf[ncol].max()) * 1.2)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    strip(ax)
axes[0].legend(loc="upper right")
fig.suptitle("LDA and NMF on the identical corpus and vocabulary, six initialisations each",
             fontsize=12.5, fontweight="bold", color=INK, y=1.03)
save(fig, "G2_lda_vs_nmf_stability.png")

# ------------------------------------------------------------------ G3
Sx = pd.read_csv(OUT / "X1_lda_vs_nmf_similarity.csv", index_col=0)
xw = pd.read_csv(OUT / "X2_lda_vs_nmf_crosswalk.csv")
fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8),
                         gridspec_kw={"width_ratios": [1.0, 1.3]})
fig.subplots_adjust(wspace=0.52)
ax = axes[0]
im = ax.imshow(Sx.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(6))
ax.set_xticklabels([c.replace("NMF_", "") for c in Sx.columns], fontsize=9)
ax.set_yticks(range(6))
ax.set_yticklabels([i.replace("LDA_", "") for i in Sx.index], fontsize=9)
for r in range(6):
    for cc in range(6):
        v = Sx.values[r, cc]
        if v >= 0.3:
            ax.text(cc, r, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if v > 0.6 else INK)
ax.set_title("LDA consensus vs NMF consensus\n(cosine, shared vocabulary)", fontsize=10.5)
ax.set_xlabel("NMF consensus topic")
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02).outline.set_visible(False)

ax = axes[1]
xw2 = xw.sort_values("top25_paper_jaccard")
yy = np.arange(len(xw2))
cols = [GOOD if r.convergent_validity else MUTED for r in xw2.itertuples()]
ax.barh(yy + 0.19, xw2["term_core_jaccard"], height=0.36, color=BLUE,
        label="term-core overlap", zorder=3)
ax.barh(yy - 0.19, xw2["top25_paper_jaccard"], height=0.36, color=AQUA,
        label="representative-paper overlap", zorder=3)
ax.axvline(0.30, color=BLUE, linestyle="--", linewidth=1.1, zorder=1)
ax.axvline(0.20, color=AQUA, linestyle=":", linewidth=1.3, zorder=1)
ax.set_yticks(yy)
ax.set_yticklabels([f"CT{int(r.lda_consensus_topic)} / N{int(r.nmf_consensus_topic)}"
                    for r in xw2.itertuples()], fontsize=9.5)
for i, r in enumerate(xw2.itertuples()):
    ax.text(0.70, i, f"cos {r.cosine:.2f}", va="center", fontsize=8.5,
            color=INK if r.convergent_validity else MUTED,
            fontweight="bold" if r.convergent_validity else "normal")
    if r.convergent_validity:
        ax.text(0.86, i, "CONVERGENT", va="center", fontsize=8,
                color=GOOD, fontweight="bold")
ax.set_xlim(0, 1.18)
ax.set_xlabel("Jaccard overlap between the two methods")
ax.set_title("Convergent validity requires agreement on\nwords AND on papers", fontsize=10.5, pad=26)
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
ax.xaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
strip(ax)
save(fig, "G3_lda_nmf_crosswalk.png")

# ------------------------------------------------------------------ G4
sw = pd.read_csv(OUT / "C4_threshold_sensitivity.csv")
fig, ax = plt.subplots(figsize=(9.5, 4.4))
ax.plot(sw["threshold"], sw["hier_n_clusters"], color=BLUE, marker="o", linewidth=2,
        markersize=5, label="hierarchical: total clusters", zorder=3)
ax.plot(sw["threshold"], sw["hier_clusters_multiseed"], color=AQUA, marker="D",
        linewidth=2, markersize=5, label="hierarchical: multi-seed clusters", zorder=3)
ax.plot(sw["threshold"], sw["cc_components_multiseed"], color=ORANGE, marker="s",
        linewidth=2, markersize=5, label="connected components: multi-seed", zorder=3)
ax.axvline(0.55, color=INK2, linestyle="--", linewidth=1.3, zorder=1)
ax.text(0.558, ax.get_ylim()[1] * 0.93, "headline threshold 0.55", fontsize=8.5, color=INK2)
ax.axvspan(0.30, 0.3452, color=CRIT, alpha=0.10, zorder=0)
ax.text(0.305, 1.2, "below chance\n(0.345)", fontsize=8, color=CRIT)
ax.set_xlabel("Cosine similarity threshold")
ax.set_ylabel("Number of clusters")
ax.set_title("Threshold sensitivity. The multi-seed cluster count is flat at 6 from 0.50 to 0.80,\n"
             "so the headline threshold was not chosen to maximise it.", fontsize=10.5)
ax.legend(loc="upper left")
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
strip(ax)
save(fig, "G4_threshold_sensitivity.png")

# ------------------------------------------------------------------ G5
s3 = pd.read_csv(OUT / "S3_journal_exclusion_sensitivity.csv")
s4 = pd.read_csv(OUT / "S4_bfull_corpus_sensitivity.csv")
piv = s3.pivot(index="ref_consensus_topic", columns="dropped_journals",
               values="cosine_to_reference")
fig, ax = plt.subplots(figsize=(9.5, 4.6))
xk = np.arange(len(piv))
w = 0.2
for i, dn in enumerate(piv.columns):
    ax.bar(xk + (i - 1) * w, piv[dn].values, width=w, zorder=3,
           color=[BLUE, AQUA, ORANGE][i], label=f"drop top {dn} journal(s)")
ax.bar(xk + 2 * w, s4.set_index("ref_consensus_topic")["cosine_B250_vs_BFull"].reindex(piv.index).values,
       width=w, color=MUTED, zorder=3, label="B250 vs BFull corpus")
ax.axhline(0.70, color=INK2, linestyle="--", linewidth=1.2, zorder=1)
ax.text(-0.42, 0.715, "0.70", fontsize=8.5, color=INK2)
ax.set_xticks(xk + 0.5 * w)
ax.set_xticklabels([f"CT{i}" for i in piv.index])
ax.set_ylabel("Cosine to the reference consensus topic")
ax.set_ylim(0, 1.05)
ax.set_title("Sensitivity of each consensus theme to dropping influential journals\n"
             "and to the conclusion-cap definition", fontsize=10.5)
ax.legend(loc="lower right", ncol=2)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
strip(ax)
save(fig, "G5_sensitivity.png")

# ------------------------------------------------------------------ G6
tv = pd.read_csv(OUT / "V2_trend_headline_corrected.csv").sort_values("doc_slope")
fig, ax = plt.subplots(figsize=(10.5, 6))
for i, r in enumerate(tv.itertuples()):
    sig = r.p_wild_cluster_sig_fdr
    col = GOOD if (r.doc_slope > 0 and sig) else CRIT if (r.doc_slope < 0 and sig) else MUTED
    ax.hlines(i, r.cr1_ci_low, r.cr1_ci_high, color=col, linewidth=2.2, zorder=3)
    ax.scatter([r.doc_slope], [i], s=62, color=col, zorder=4,
               edgecolor=SURFACE, linewidth=1.1)
    ax.text(0.0206, i, f"raw p {r.p_wild_cluster:.3f}   q {r.p_wild_cluster_fdr:.3f}",
            va="center", fontsize=8,
            color=INK if r.p_wild_cluster < 0.05 else MUTED,
            fontweight="bold" if r.p_wild_cluster < 0.05 else "normal")
ax.axvline(0, color=INK2, linewidth=1.1, zorder=1)
ax.set_yticks(range(len(tv)))
ax.set_yticklabels([f"T{int(r.topic)}  {r.label[:34]}" for r in tv.itertuples()], fontsize=8.5)
ax.set_xlim(-0.023, 0.040)
ax.set_xlabel("Document-level slope per year, 2019-2025 (bars = CR1 journal-clustered 95% CI)")
ax.set_title("Corrected trend inference: wild cluster bootstrap-t with the null imposed.\n"
             "No topic survives FDR correction. CI exclusion of zero is NOT significance.",
             fontsize=11)
ax.xaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
strip(ax)
save(fig, "G6_corrected_trend_inference.png")

print(f"\n{len(list(FIG.glob('*.png')))} figures written")
