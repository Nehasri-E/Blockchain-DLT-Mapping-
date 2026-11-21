"""Task C: re-evaluation of topic prevalence trends.

Starts from the baseline artifacts (outputs/papers_with_topics.csv and
outputs/topic_prevalence_by_year.csv). The baseline model is not refitted and
the baseline window (2019-2025) is not changed.

Four estimators are reported for every topic:

  OLS-7        the baseline procedure: OLS of the annual mean topic share on
               year, 7 annual observations. Reproduced first, for verification.
  WLS-7        the same 7 points weighted by papers per year. The annual means
               are estimated from between 38 and 217 papers, so they are not
               equally precise; unweighted OLS implicitly assumes they are.
  Theil-Sen    median of pairwise slopes. Robust to a single influential year,
               which matters at n=7, and assumes only monotonic association.
  OLS-doc      document-level OLS of each paper's topic share on year, 987
               observations. Uses all the data rather than 7 summary points.
               Assumes papers are independent, which ignores journal and author
               clustering, so its very small p-values should be read as an upper
               bound on evidence, not as the headline.

Multiplicity: 12 topics are tested simultaneously, so Benjamini-Hochberg FDR
control at q=0.05 is applied within each estimator family.
"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.stattools import durbin_watson
from pathlib import Path

ROOT = Path("/home/claude/LDA_Project")
VR = ROOT / "validation_round1"
OUT = VR / "outputs"

K = 12
YEARS = list(range(2019, 2026))
ALPHA = 0.05
RNG = np.random.default_rng(20261113)
N_PERM = 20000

LABELS = {int(k): v for k, v in
          json.loads((VR / "scripts" / "baseline_labels.json").read_text()).items()}

papers = pd.read_csv(ROOT / "outputs" / "papers_with_topics.csv")
baseline_trends = pd.read_csv(ROOT / "outputs" / "topic_trends.csv")
baseline_yearly = pd.read_csv(ROOT / "outputs" / "topic_prevalence_by_year.csv", index_col=0)
tcols = [f"Topic_{i+1}" for i in range(K)]

sub = papers[papers["Year"].isin(YEARS)]
yearly = sub.groupby("Year")[tcols].mean()
counts = sub.groupby("Year").size()

print("STEP 1  Reproduce the baseline yearly prevalence table")
dev = float(np.abs(yearly.values - baseline_yearly.loc[YEARS, tcols].values).max())
print(f"  max deviation from topic_prevalence_by_year.csv : {dev:.2e}")
assert dev < 1e-4, "cannot reproduce the baseline yearly table"

print("\nSTEP 2  Papers per year in the trend window")
cnt = pd.DataFrame({"Year": counts.index, "Papers": counts.values})
cnt["Share_of_corpus"] = (cnt["Papers"] / len(papers)).round(4)
cnt.to_csv(OUT / "C1_papers_per_year.csv", index=False)
print(cnt.to_string(index=False))
print(f"  total in window {counts.sum()} of {len(papers)} papers "
      f"({counts.sum()/len(papers):.1%}); excluded: "
      f"{sorted(set(papers['Year']) - set(YEARS))}")

x = np.array(YEARS, dtype=float)
xc = x - x.mean()
rows, diag_rows = [], []

for i, c in enumerate(tcols, 1):
    y = yearly[c].values
    w = counts.values.astype(float)

    ols = sm.OLS(y, sm.add_constant(xc)).fit()
    slope, p_ols = ols.params[1], ols.pvalues[1]
    ci = ols.conf_int(alpha=0.05)[1]

    wls = sm.WLS(y, sm.add_constant(xc), weights=w).fit()

    ts = stats.theilslopes(y, x, 0.95)
    tau, p_tau = stats.kendalltau(x, y)

    obs = abs(slope)
    perm = sum(abs(np.polyfit(RNG.permutation(xc), y, 1)[0]) >= obs - 1e-15
               for _ in range(N_PERM))
    p_perm = (perm + 1) / (N_PERM + 1)

    dv = papers[papers["Year"].isin(YEARS)]
    doc = sm.OLS(dv[c].values, sm.add_constant(dv["Year"].values - np.mean(YEARS))).fit()

    base = baseline_trends[baseline_trends["Topic"] == i].iloc[0]
    rows.append({
        "topic": i, "label": LABELS[i], "n_years": len(YEARS),
        "n_papers": int(counts.sum()),
        "baseline_slope": base["Slope_per_year"], "baseline_p": base["p_value"],
        "ols7_slope": round(float(slope), 6),
        "ols7_ci_low": round(float(ci[0]), 6), "ols7_ci_high": round(float(ci[1]), 6),
        "ols7_p": round(float(p_ols), 5), "ols7_r2": round(float(ols.rsquared), 4),
        "wls7_slope": round(float(wls.params[1]), 6), "wls7_p": round(float(wls.pvalues[1]), 5),
        "theilsen_slope": round(float(ts[0]), 6),
        "theilsen_ci_low": round(float(ts[2]), 6), "theilsen_ci_high": round(float(ts[3]), 6),
        "kendall_tau": round(float(tau), 4), "kendall_p": round(float(p_tau), 5),
        "permutation_p": round(float(p_perm), 5),
        "docOLS_slope": round(float(doc.params[1]), 6),
        "docOLS_p": float(f"{doc.pvalues[1]:.3e}"),
        "docOLS_ci_low": round(float(doc.conf_int()[1][0]), 6),
        "docOLS_ci_high": round(float(doc.conf_int()[1][1]), 6),
    })

    infl = ols.get_influence()
    cooks = infl.cooks_distance[0]
    thresh = 4 / len(YEARS)
    for j, yr in enumerate(YEARS):
        diag_rows.append({"topic": i, "label": LABELS[i], "year": yr,
                          "papers": int(counts.values[j]),
                          "mean_share": round(float(y[j]), 5),
                          "residual": round(float(ols.resid[j]), 5),
                          "cooks_distance": round(float(cooks[j]), 4),
                          "leverage": round(float(infl.hat_matrix_diag[j]), 4),
                          "influential": bool(cooks[j] > thresh)})

    diag_rows[-1]["_dw"] = round(float(durbin_watson(ols.resid)), 3)
    diag_rows[-1]["_shapiro_p"] = round(float(stats.shapiro(ols.resid).pvalue), 4)

res = pd.DataFrame(rows)

for fam, col in [("ols7", "ols7_p"), ("wls7", "wls7_p"),
                 ("permutation", "permutation_p"), ("docOLS", "docOLS_p")]:
    rej, adj, _, _ = multipletests(res[col].values, alpha=ALPHA, method="fdr_bh")
    res[f"{fam}_p_fdr"] = np.round(adj, 5)
    res[f"{fam}_sig_fdr"] = rej

res["significant_before_fdr_ols7"] = res["ols7_p"] < ALPHA
res = res.sort_values("ols7_slope", ascending=False)
res.to_csv(OUT / "C2_trend_reanalysis_full.csv", index=False)

diag = pd.DataFrame(diag_rows)
assumptions = diag.groupby(["topic", "label"]).agg(
    durbin_watson=("_dw", "max"), shapiro_p=("_shapiro_p", "max"),
    n_influential_years=("influential", "sum"),
    max_cooks=("cooks_distance", "max")).reset_index()
assumptions["influential_years"] = [
    ", ".join(str(y) for y in diag[(diag.topic == t) & diag.influential]["year"]) or "none"
    for t in assumptions["topic"]]
assumptions["residual_normality_ok"] = assumptions["shapiro_p"] > 0.05
assumptions["autocorrelation_flag"] = (assumptions["durbin_watson"] < 1.0) | \
                                      (assumptions["durbin_watson"] > 3.0)
assumptions.round(4).to_csv(OUT / "C3_regression_assumptions.csv", index=False)
diag.drop(columns=["_dw", "_shapiro_p"]).to_csv(OUT / "C4_year_level_diagnostics.csv", index=False)

head = res[["topic", "label", "ols7_slope", "ols7_ci_low", "ols7_ci_high", "ols7_p",
            "ols7_p_fdr", "ols7_sig_fdr", "theilsen_slope", "permutation_p",
            "docOLS_p_fdr", "docOLS_sig_fdr"]].copy()
head.to_csv(OUT / "C5_trend_headline_table.csv", index=False)

print("\nSTEP 3  Baseline reproduction of the 12 slopes")
d = float(np.abs(res["ols7_slope"].values - res["baseline_slope"].values).max())
print(f"  max slope deviation vs topic_trends.csv : {d:.2e}  (reproduced: {d < 1e-4})")

print("\n" + "=" * 110)
print("C5  TREND RESULTS, ORIGINAL vs FDR-ADJUSTED")
print(head.to_string(index=False, max_colwidth=42))

print("\n" + "=" * 110)
print("SIGNIFICANCE COUNTS (12 simultaneous tests)")
for fam in ["ols7", "wls7", "permutation", "docOLS"]:
    raw = int((res[f"{fam}_p" if fam != "docOLS" else "docOLS_p"] < ALPHA).sum())
    print(f"  {fam:<12} raw p<0.05: {raw:>2}/12   after BH FDR q=0.05: "
          f"{int(res[f'{fam}_sig_fdr'].sum()):>2}/12")

print("\n" + "=" * 110)
print("C3  REGRESSION ASSUMPTIONS (n=7 per topic)")
print(assumptions[["topic", "label", "durbin_watson", "shapiro_p",
                   "n_influential_years", "influential_years"]].to_string(
    index=False, max_colwidth=42))
