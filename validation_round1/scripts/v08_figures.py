"""Publication-quality figures for validation round 1.

Every figure is generated from the result CSVs written by v03-v07, never from
in-memory model objects, so the figures and the tables cannot disagree.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

VR = Path("/home/claude/LDA_Project/validation_round1")
OUT, FIG = VR / "outputs", VR / "figures"
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


def save(fig, name):
    fig.savefig(FIG / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


# ---------------------------------------------------------------- V1
raw = pd.read_csv(OUT / "A10_cap_effect_vs_seed_noise_raw.csv")
summ = pd.read_csv(OUT / "A10_cap_effect_vs_seed_noise_summary.csv")
noise = raw[raw.contrast.str.startswith("noise")]["mean_matched_cosine"].values

fig, ax = plt.subplots(figsize=(9.5, 4.6))
lo, hi, mu = noise.min(), noise.max(), noise.mean()
ax.axhspan(mu - noise.std(ddof=1), mu + noise.std(ddof=1), color=MUTED, alpha=0.16, zorder=0)
ax.axhline(mu, color=MUTED, linewidth=1.4, linestyle="--", zorder=1)
groups = ["noise floor\n(B250 vs B250,\nseed varies)"] + [
    f"B250\nvs {c}" for c in ["A", "B500", "B1000", "BFull"]]
data = [noise] + [raw[raw.contrast.str.contains(f"vs {c},")]["mean_matched_cosine"].values
                  for c in ["A", "B500", "B1000", "BFull"]]
rng = np.random.default_rng(1)
for i, (g, d) in enumerate(zip(groups, data)):
    col = MUTED if i == 0 else BLUE
    ax.scatter(np.full(len(d), i) + rng.uniform(-0.08, 0.08, len(d)), d,
               s=46, color=col, alpha=0.8, edgecolor=SURFACE, linewidth=1, zorder=3)
    ax.scatter([i], [d.mean()], s=150, marker="_", color=ORANGE if i else INK2,
               linewidth=2.6, zorder=4)
for i, r in enumerate(summ.itertuples(), start=1):
    ax.text(i, 0.455, f"{r.difference_in_noise_sds:+.2f} SD",
            ha="center", fontsize=8.5, color=INK2, fontweight="bold")
ax.text(0, 0.455, "reference", ha="center", fontsize=8.5, color=MUTED, style="italic")
ax.set_xticks(range(len(groups)))
ax.set_xticklabels(groups, fontsize=8.5)
ax.set_ylabel("Mean matched topic cosine")
ax.set_ylim(0.44, 0.65)
ax.set_title("Conclusion-cap effect measured against the seed-noise floor")
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.text(0.99, 0.97, "Grey band: ±1 SD of the noise floor.\nPoints inside it are indistinguishable "
        "from\nsimply re-running with a different seed.",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK2)
strip(ax)
save(fig, "V1_cap_effect_vs_seed_noise.png")

# ---------------------------------------------------------------- V2
pt = pd.read_csv(OUT / "D3_per_topic_seed_stability.csv")
b = pt[pt.condition == "B250"].sort_values("mean_cosine")
fig, ax = plt.subplots(figsize=(10, 5.8))
cols = [GOOD if v == "reproducible" else WARN if v == "partially reproducible" else CRIT
        for v in b["verdict"]]
ax.hlines(range(len(b)), b["min_cosine"], b["max_cosine"], color=BASE, linewidth=2.4, zorder=2)
ax.scatter(b["mean_cosine"], range(len(b)), s=78, color=cols, zorder=3,
           edgecolor=SURFACE, linewidth=1.2)
for i, (_, r) in enumerate(b.iterrows()):
    ax.text(r["max_cosine"] + 0.018, i, f"{r['mean_cosine']:.2f}", va="center",
            fontsize=8.5, color=INK2)
ax.axvline(0.70, color=INK2, linewidth=1.2, linestyle="--", zorder=1)
ax.text(0.715, len(b) - 0.4, "0.70 reproducibility threshold", fontsize=8.5, color=INK2)
ax.set_yticks(range(len(b)))
ax.set_yticklabels([f"T{int(r.anchor_topic)}  {r.baseline_label}" for _, r in b.iterrows()],
                   fontsize=9)
ax.set_xlabel("Matched topic cosine vs the seed-42 baseline (bar = min to max over 5 other seeds)")
ax.set_xlim(0, 1.05)
ax.set_title("Topic reproducibility across six random seeds, Corpus B250")
ax.xaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", color=c, linestyle="", markersize=8, label=l)
                   for c, l in [(GOOD, "reproducible"), (WARN, "partially reproducible"),
                                (CRIT, "not reproducible")]], loc="lower right")
strip(ax)
save(fig, "V2_topic_seed_stability.png")

# ---------------------------------------------------------------- V3
st = pd.read_csv(OUT / "A9_topic_stability_across_caps.csv")
m = pd.read_csv(OUT / "A5_matched_topics_seed42.csv")
m = m[m.projection == "union"]
piv = m.pivot_table(index="ref_topic", columns="condition", values="cosine")
piv = piv[["A", "B500", "B1000", "BFull"]]
order = st.sort_values("mean_cosine_across_B_caps", ascending=False)["ref_topic"].tolist()
piv = piv.loc[order]
labels = dict(zip(st["ref_topic"], st["ref_label"]))
fig, ax = plt.subplots(figsize=(8.6, 6.0))
im = ax.imshow(piv.values, cmap="Blues", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(piv.shape[1]))
ax.set_xticklabels(piv.columns, fontsize=9.5)
ax.set_yticks(range(len(piv)))
ax.set_yticklabels([f"T{t}  {labels[t][:34]}" for t in piv.index], fontsize=8.5)
for r in range(piv.shape[0]):
    for c in range(piv.shape[1]):
        v = piv.values[r, c]
        ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                color="#ffffff" if v > 0.55 else INK)
ax.set_title("Baseline topic recovery under each alternative corpus (seed 42)")
ax.set_xlabel("Corpus condition compared against the B250 baseline")
cb = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
cb.set_label("Matched cosine", color=INK2, fontsize=9)
cb.outline.set_visible(False)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
save(fig, "V3_topic_recovery_by_cap.png")

# ---------------------------------------------------------------- V4
tr = pd.read_csv(OUT / "C2_trend_reanalysis_full.csv").sort_values("ols7_slope")
fig, ax = plt.subplots(figsize=(10, 5.8))
for i, (_, r) in enumerate(tr.iterrows()):
    sig = r["ols7_sig_fdr"]
    col = (GOOD if r["ols7_slope"] > 0 else CRIT) if sig else MUTED
    ax.hlines(i, r["ols7_ci_low"], r["ols7_ci_high"], color=col, linewidth=2.2, zorder=2)
    ax.scatter([r["ols7_slope"]], [i], s=70, color=col, zorder=3,
               edgecolor=SURFACE, linewidth=1.2)
    ax.scatter([r["theilsen_slope"]], [i], s=46, marker="D", color=col, alpha=0.5, zorder=3)
    ax.text(0.0295, i, f"q={r['ols7_p_fdr']:.3f}", va="center", fontsize=8,
            color=INK if sig else MUTED, fontweight="bold" if sig else "normal")
ax.axvline(0, color=INK2, linewidth=1.1, zorder=1)
ax.set_yticks(range(len(tr)))
ax.set_yticklabels([f"T{int(r.topic)}  {r.label[:38]}" for _, r in tr.iterrows()], fontsize=8.5)
ax.set_xlabel("Change in mean topic share per year, 2019-2025 (95% CI)")
ax.set_xlim(-0.028, 0.040)
ax.set_title("Topic trends with Benjamini-Hochberg FDR correction across 12 tests")
ax.xaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.legend(handles=[
    Line2D([], [], marker="o", color=GOOD, linestyle="", markersize=8, label="rising, significant after FDR"),
    Line2D([], [], marker="o", color=CRIT, linestyle="", markersize=8, label="declining, significant after FDR"),
    Line2D([], [], marker="o", color=MUTED, linestyle="", markersize=8, label="not significant after FDR"),
    Line2D([], [], marker="D", color=INK2, linestyle="", markersize=6, alpha=0.5, label="Theil-Sen slope")],
    loc="center left", bbox_to_anchor=(0.0, 0.42))
strip(ax)
save(fig, "V4_trends_fdr_corrected.png")

# ---------------------------------------------------------------- V5
cnt = pd.read_csv(OUT / "C1_papers_per_year.csv")
diag = pd.read_csv(OUT / "C4_year_level_diagnostics.csv")
infl = diag[diag.influential].groupby("year").size().reindex(cnt["Year"], fill_value=0)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
axes[0].bar(cnt["Year"], cnt["Papers"], color=BLUE, width=0.64, zorder=3)
for _, r in cnt.iterrows():
    axes[0].text(r["Year"], r["Papers"] + 4, int(r["Papers"]), ha="center",
                 fontsize=8.5, color=INK2)
axes[0].set_title("Papers per year in the 2019-2025 trend window")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Papers")
axes[0].set_ylim(0, cnt["Papers"].max() * 1.16)
axes[1].bar(infl.index, infl.values, color=CRIT, width=0.64, zorder=3)
for y, v in infl.items():
    axes[1].text(y, v + 0.12, int(v), ha="center", fontsize=8.5, color=INK2)
axes[1].set_title("Topics for which that year is an influential observation")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Topics (of 12)")
axes[1].set_ylim(0, max(infl.max() * 1.25, 1))
for ax in axes:
    ax.set_xticks(cnt["Year"])
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    strip(ax)
fig.suptitle("Why seven annual observations is a weak basis for trend inference "
             "(Cook's distance > 4/n)", fontsize=11.5, fontweight="bold", color=INK, y=1.03)
save(fig, "V5_trend_window_diagnostics.png")

# ---------------------------------------------------------------- V6
grid = pd.read_csv(OUT / "A2_model_grid.csv")
fig, ax = plt.subplots(figsize=(8.6, 4.4))
conds = ["A", "B250", "B500", "B1000", "BFull"]
for i, c in enumerate(conds):
    d = grid[grid.condition == c]["c_v"].values
    ax.scatter(np.full(len(d), i) + rng.uniform(-0.07, 0.07, len(d)), d, s=48,
               color=ORANGE if c == "B250" else BLUE, alpha=0.82,
               edgecolor=SURFACE, linewidth=1, zorder=3)
    ax.scatter([i], [d.mean()], s=165, marker="_",
               color=INK if c == "B250" else INK2, linewidth=2.6, zorder=4)
    ax.text(i, d.mean() + 0.013, f"{d.mean():.3f}", ha="center", fontsize=8.5,
            color=INK, fontweight="bold")
ax.set_xticks(range(len(conds)))
ax.set_xticklabels(conds)
ax.set_ylabel("C_v coherence")
ax.set_xlabel("Corpus condition (orange = published baseline, 6 seeds each)")
ax.set_title("Coherence by conclusion cap. NOT directly comparable across conditions:\n"
             "each C_v is computed on its own corpus and vocabulary.",
             fontsize=10.5)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
strip(ax)
save(fig, "V6_coherence_by_cap_condition.png")

print(f"\n{len(list(FIG.glob('*.png')))} figures written")
