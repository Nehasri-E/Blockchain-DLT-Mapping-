"""Round 2 figures. Generated from result CSVs only, never from model objects."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

R2 = Path("/home/claude/LDA_Project/validation_round2")
OUT, FIG = R2 / "outputs", R2 / "figures"
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


# ---------------------------------------------------------------- F1 convergence
st = pd.read_csv(OUT / "T3_stability_by_config.csv")
inv = pd.read_csv(OUT / "T2_solution_invariance_vs_anchor.csv")
order = ["baseline", "C1", "C2", "C3", "C4_anchor"]
st = st.set_index("config").loc[order].reset_index()
lab = [f"{r.config}\n{r.passes}p/{r.iterations}i" for r in st.itertuples()]
x = np.arange(len(st))

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
fig.subplots_adjust(wspace=0.32)
ax = axes[0]
iv = inv.groupby("config")["cosine_vs_C4_anchor"].mean().reindex(order[:-1])
ax.bar(x[:-1], iv.values, color=BLUE, width=0.6, zorder=3)
ax.bar([x[-1]], [1.0], color=MUTED, width=0.6, zorder=3)
for i, v in enumerate(list(iv.values) + [1.0]):
    ax.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=8.5, color=INK2)
ax.axhline(0.99, color=GOOD, linestyle="--", linewidth=1.3, zorder=2)
ax.text(0.05, 0.9915, "0.99 convergence criterion", fontsize=8, color=GOOD)
ax.set_ylim(0.85, 1.02)
ax.set_title("E3  Solution invariance\n(cosine vs 200-pass anchor, same seed)", fontsize=10.5)
ax.set_ylabel("Mean matched cosine")

ax = axes[1]
ax.errorbar(x, st["mean_bound_per_word"], yerr=st["sd_bound_per_word"], color=ORANGE,
            marker="o", markersize=6, linewidth=2, capsize=3, markeredgecolor=SURFACE,
            markeredgewidth=1, zorder=3)
ax.set_title("E2  Variational bound per word\n(higher is better)", fontsize=10.5)
ax.set_ylabel("Bound per word")

ax = axes[2]
ax.errorbar(x, st["mean_matched_cosine"], yerr=st["sd_matched_cosine"], color=CRIT,
            marker="o", markersize=6, linewidth=2, capsize=3, markeredgecolor=SURFACE,
            markeredgewidth=1, zorder=3, label="matched cosine")
ax.plot(x, st["mean_ari"], color=INK2, marker="s", markersize=5, linewidth=1.8,
        linestyle="--", zorder=3, label="ARI")
ax.set_ylim(0, 0.75)
ax.set_title("Seed stability does NOT improve\nwith more training", fontsize=10.5)
ax.set_ylabel("Stability")
ax.legend(loc="upper right")

for a in axes:
    a.set_xticks(x)
    a.set_xticklabels(lab, fontsize=8.5)
    a.yaxis.grid(True, zorder=0)
    a.set_axisbelow(True)
    strip(a)
fig.suptitle("Task 1: the baseline is converged, and convergence is not the cause of seed instability",
             fontsize=12.5, fontweight="bold", color=INK, y=1.04)
save(fig, "F1_convergence_diagnostics.png")

# ---------------------------------------------------------------- F2 eta trade-off
e = pd.read_csv(OUT / "T23_eta_chance_corrected.csv")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
ax = axes[0]
ax.plot(e["eta_value"], e["mean_matched_cosine"], color=BLUE, marker="o", linewidth=2,
        markersize=7, markeredgecolor=SURFACE, markeredgewidth=1, label="matched cosine", zorder=3)
ax.plot(e["eta_value"], e["random_pairing_cosine"], color=MUTED, marker="s", linewidth=1.8,
        linestyle=":", markersize=6, label="chance (random pairing)", zorder=3)
ax.plot(e["eta_value"], e["recovery_above_chance"], color=AQUA, marker="D", linewidth=2,
        markersize=6, label="recovery above chance", zorder=3)
ax.plot(e["eta_value"], e["mean_ari"], color=CRIT, marker="^", linewidth=2, markersize=7,
        label="ARI (cannot be inflated)", zorder=3)
ax.set_xscale("log")
ax.set_xticks(e["eta_value"])
ax.set_xticklabels([f"{v:g}" for v in e["eta_value"]])
ax.set_xlabel("eta (topic-word prior)")
ax.set_ylabel("Stability measure")
ax.set_title("Stability rises with eta, but most of the\ncosine gain is the chance baseline rising", fontsize=10.5)
ax.legend(loc="upper left", fontsize=8.5)

ax = axes[1]
ax.plot(e["mean_distinctiveness"], e["mean_ari"], color=INK2, linewidth=1.6,
        linestyle="--", zorder=2)
ax.scatter(e["mean_distinctiveness"], e["mean_ari"], s=130, color=[BLUE, ORANGE, AQUA, CRIT],
           zorder=3, edgecolor=SURFACE, linewidth=1.5)
for _, r in e.iterrows():
    ax.annotate(f"eta={r['eta_value']:.3g}", (r["mean_distinctiveness"], r["mean_ari"]),
                textcoords="offset points", xytext=(-10, 12), fontsize=9, color=INK2,
                ha="right")
ax.margins(x=0.14, y=0.12)
ax.set_xlabel("Topic distinctiveness (mean pairwise JS distance)")
ax.set_ylabel("Document-level stability (ARI)")
ax.set_title("The trade-off: reproducibility is bought\nwith topic distinctiveness", fontsize=10.5)
ax.invert_xaxis()
for a in axes:
    a.yaxis.grid(True, zorder=0)
    a.set_axisbelow(True)
    strip(a)
save(fig, "F2_eta_tradeoff.png")

# ---------------------------------------------------------------- F3/F4 K sweep
k = pd.read_csv(OUT / "T11_k_sweep_summary.csv")
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
ax = axes[0]
ax.plot(k["k"], k["mean_matched_cosine"], color=BLUE, marker="o", linewidth=2,
        markersize=6, markeredgecolor=SURFACE, markeredgewidth=1, label="matched cosine", zorder=3)
ax.fill_between(k["k"], k["mean_matched_cosine"] - k["sd_matched_cosine"],
                k["mean_matched_cosine"] + k["sd_matched_cosine"], color=BLUE, alpha=0.15, zorder=1)
ax.plot(k["k"], k["random_pairing_cosine"], color=MUTED, marker="s", linestyle=":",
        linewidth=1.8, markersize=5, label="chance (random pairing)", zorder=3)
ax.plot(k["k"], k["recovery_above_chance"], color=AQUA, marker="D", linewidth=2,
        markersize=6, label="recovery above chance", zorder=3)
ax.plot(k["k"], k["mean_ari"], color=CRIT, marker="^", linewidth=2, markersize=6,
        label="ARI", zorder=3)
ax.axvline(12, color=INK2, linestyle="--", linewidth=1.2, zorder=1)
ax.text(12.2, ax.get_ylim()[1] * 0.96, "baseline K=12", fontsize=8.5, color=INK2, va="top")
ax.set_xlabel("Number of topics (K)")
ax.set_ylabel("Stability measure")
ax.set_title("Stability vs K, with the chance baseline that\nrises mechanically as K falls", fontsize=10.5)
ax.legend(loc="upper right", fontsize=8.5, bbox_to_anchor=(1.0, 0.80))
ax.set_xticks(k["k"])

ax = axes[1]
ax.plot(k["k"], k["pct_reproducible_topics"], color=BLUE, marker="o", linewidth=2,
        markersize=6, label="% topics reproducible (cos>=0.70)", zorder=3)
ax.set_ylabel("% of topics reproducible", color=BLUE)
ax.set_xlabel("Number of topics (K)")
ax.set_xticks(k["k"])
ax.axvline(12, color=INK2, linestyle="--", linewidth=1.2, zorder=1)
ax.set_title("Reproducible share falls as K rises", fontsize=10.5)
for a in axes:
    a.yaxis.grid(True, zorder=0)
    a.set_axisbelow(True)
    strip(a)
save(fig, "F3_k_stability.png")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, col, ttl, col_c in [
    (axes[0], "mean_c_v", "C_v coherence", BLUE),
    (axes[1], "mean_distinctiveness", "Topic distinctiveness (JS)", AQUA),
    (axes[2], "mean_n_small_topics", "Topics below 2% prevalence", CRIT)]:
    ax.plot(k["k"], k[col], color=col_c, marker="o", linewidth=2, markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1, zorder=3)
    if col == "mean_c_v":
        ax.fill_between(k["k"], k[col] - k["sd_c_v"], k[col] + k["sd_c_v"],
                        color=col_c, alpha=0.15, zorder=1)
    ax.axvline(12, color=INK2, linestyle="--", linewidth=1.2, zorder=1)
    ax.set_title(ttl, fontsize=10.5)
    ax.set_xlabel("K")
    ax.set_xticks(k["k"])
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    strip(ax)
fig.suptitle("Task 3: granularity, coherence and topic size across K (dashed line = baseline K=12)",
             fontsize=12, fontweight="bold", color=INK, y=1.03)
save(fig, "F4_k_quality.png")

# ---------------------------------------------------------------- F5 corrected cap
cc = pd.read_csv(OUT / "T12_cap_vs_seed_corrected.csv")
sl = pd.read_csv(OUT / "T13_cap_vs_seed_seedlevel.csv")
fig, ax = plt.subplots(figsize=(9.5, 4.6))
y = np.arange(len(cc))
rng = np.random.default_rng(3)
for i, r in enumerate(cc.itertuples()):
    cond = r.comparison.split("vs ")[1]
    d = sl[sl.condition == cond]["paired_difference"].values
    ax.scatter(d, np.full(len(d), i) + rng.uniform(-0.09, 0.09, len(d)), s=42,
               color=MUTED, alpha=0.7, zorder=2, edgecolor=SURFACE, linewidth=0.8)
    col = GOOD if r.ci_excludes_zero else MUTED
    ax.hlines(i, r.boot_ci_low, r.boot_ci_high, color=col, linewidth=2.6, zorder=3)
    ax.scatter([r.mean_paired_difference], [i], s=95, color=col, zorder=4,
               edgecolor=SURFACE, linewidth=1.3)
    ax.text(0.105, i, f"p={r.exact_signflip_p_one_sided:.4f}  dz={r.cohens_dz:+.2f}",
            va="center", fontsize=8.5, color=INK2)
ax.axvline(0, color=INK2, linewidth=1.2, zorder=1)
ax.set_yticks(y)
ax.set_yticklabels(cc["comparison"])
ax.set_xlim(-0.06, 0.155)
ax.set_xlabel("Seed-level paired difference: noise floor minus cap contrast (95% bootstrap CI over 6 seeds)")
ax.set_title("Task 5: corrected cap-versus-seed comparison\n"
             "(grey dots = the 6 seed-level paired differences; exact sign-flip floor is p=0.0156)",
             fontsize=11)
ax.xaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
strip(ax)
save(fig, "F5_cap_vs_seed_corrected.png")

# ---------------------------------------------------------------- F6 trends
tb = pd.read_csv(OUT / "T16_trend_bootstrap_full.csv").sort_values("ols7_slope")
fig, ax = plt.subplots(figsize=(10.5, 6))
for i, r in enumerate(tb.itertuples()):
    ax.hlines(i + 0.18, r.B1_iid_ci_low, r.B1_iid_ci_high, color=BLUE, linewidth=2.1, zorder=3)
    ax.scatter([r.ols7_slope], [i + 0.18], s=46, color=BLUE, zorder=4,
               edgecolor=SURFACE, linewidth=1)
    ax.hlines(i - 0.18, r.B2_journal_ci_low, r.B2_journal_ci_high, color=ORANGE,
              linewidth=2.1, zorder=3)
    ax.scatter([r.ols7_slope], [i - 0.18], s=46, color=ORANGE, zorder=4,
               edgecolor=SURFACE, linewidth=1)
    marks = ("B1 " + ("yes" if r.B1_iid_sig_fdr else "no") + "   B2 " +
             ("yes" if r.B2_journal_sig_fdr else "no"))
    ax.text(0.0295, i, marks, va="center", fontsize=8,
            color=INK if (r.B1_iid_sig_fdr or r.B2_journal_sig_fdr) else MUTED,
            fontweight="bold" if r.B2_journal_sig_fdr else "normal")
ax.axvline(0, color=INK2, linewidth=1.1, zorder=1)
ax.set_yticks(range(len(tb)))
ax.set_yticklabels([f"T{int(r.topic)}  {r.label[:36]}" for r in tb.itertuples()], fontsize=8.5)
ax.set_xlim(-0.028, 0.042)
ax.set_xlabel("Change in mean topic share per year, 2019-2025")
ax.set_title("Task 6: trend uncertainty under document and journal-cluster bootstraps\n"
             "(significance after BH FDR shown at right)", fontsize=11)
ax.legend(handles=[Line2D([], [], color=BLUE, linewidth=2.4, label="B1 iid document bootstrap"),
                   Line2D([], [], color=ORANGE, linewidth=2.4, label="B2 journal cluster bootstrap")],
          loc="lower right")
ax.xaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
strip(ax)
save(fig, "F6_trend_bootstrap.png")

print(f"\n{len(list(FIG.glob('*.png')))} figures written")
