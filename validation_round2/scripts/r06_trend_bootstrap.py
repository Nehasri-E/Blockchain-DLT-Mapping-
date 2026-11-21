"""Task 6: supplementary trend uncertainty via resampling.

The 2019-2025 window is retained unchanged. The baseline topic model is held
fixed; only the sampling of documents is resampled, so these intervals describe
sampling uncertainty in the corpus, NOT uncertainty in the topic model itself.
Given Task 1's finding that the topic solution is seed-unstable, model
uncertainty is the larger unquantified term and is stated as a limitation
rather than silently ignored.

Three resampling schemes, in increasing conservatism:

  B1  iid document bootstrap. Resample the 956 in-window papers with
      replacement, recompute the 7 annual means, refit OLS-7. Propagates
      document sampling error into the annual means, which OLS-7 on fixed
      means ignores entirely. Assumes papers are independent.

  B2  cluster bootstrap by journal. Resample the 133 journals with
      replacement, take every paper in each drawn journal, then recompute.
      Papers in one journal share editorial scope and often authors, so they
      are not independent draws. This is the scheme that addresses the
      dependence the brief flags in document-level OLS.

  B3  cluster bootstrap by journal with a stratified year constraint is NOT
      used. It was considered and rejected: resampling journals already
      perturbs the year composition, and constraining it would reintroduce the
      assumption being tested.

Bootstrap p-values are two-sided and computed by inverting the interval
(proportion of resamples on the opposite side of zero, doubled), which is
appropriate for a slope with no natural null distribution to permute.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from r00_common import R2, ROOT

OUT = R2 / "outputs"
K = 12
YEARS = list(range(2019, 2026))
NBOOT = 4000
ALPHA = 0.05
rng = np.random.default_rng(20261115)

LABELS = {int(k): v for k, v in json.loads(
    (ROOT / "validation_round1/scripts/baseline_labels.json").read_text()).items()}

papers = pd.read_csv(ROOT / "outputs" / "papers_with_topics.csv")
tcols = [f"Topic_{i+1}" for i in range(K)]
sub = papers[papers["Year"].isin(YEARS)].reset_index(drop=True)
print(f"In-window papers: {len(sub)}; journals: {sub['Journal'].nunique()}")

x = np.array(YEARS, dtype=float)
xc = x - x.mean()


def slope_from_frame(df):
    g = df.groupby("Year")[tcols].mean()
    if len(g) != len(YEARS):
        return None
    y = g.loc[YEARS].values
    xm = xc - xc.mean()
    return (xm @ (y - y.mean(0))) / (xm @ xm)


point = slope_from_frame(sub)

jidx = {j: np.where(sub["Journal"].values == j)[0] for j in sub["Journal"].unique()}
journals = list(jidx)

boot = {"B1_iid": np.full((NBOOT, K), np.nan), "B2_journal": np.full((NBOOT, K), np.nan)}
n = len(sub)
for b in range(NBOOT):
    s1 = slope_from_frame(sub.iloc[rng.integers(0, n, n)])
    if s1 is not None:
        boot["B1_iid"][b] = s1
    pick = rng.integers(0, len(journals), len(journals))
    rows = np.concatenate([jidx[journals[i]] for i in pick])
    s2 = slope_from_frame(sub.iloc[rows])
    if s2 is not None:
        boot["B2_journal"][b] = s2

r1 = pd.read_csv(ROOT / "validation_round1" / "outputs" / "C2_trend_reanalysis_full.csv")

rows = []
for i in range(K):
    rec = {"topic": i + 1, "label": LABELS[i + 1],
           "point_slope": round(float(point[i]), 6)}
    base = r1[r1.topic == i + 1].iloc[0]
    rec["ols7_slope"] = base["ols7_slope"]
    rec["ols7_p"] = base["ols7_p"]
    rec["ols7_p_fdr"] = base["ols7_p_fdr"]
    rec["wls7_p"] = base["wls7_p"]
    rec["theilsen_slope"] = base["theilsen_slope"]
    for name, arr in boot.items():
        v = arr[:, i]
        v = v[~np.isnan(v)]
        lo, hi = np.percentile(v, [2.5, 97.5])
        frac = float((v <= 0).mean()) if point[i] > 0 else float((v >= 0).mean())
        rec[f"{name}_ci_low"] = round(float(lo), 6)
        rec[f"{name}_ci_high"] = round(float(hi), 6)
        rec[f"{name}_se"] = round(float(v.std(ddof=1)), 6)
        rec[f"{name}_p"] = round(min(1.0, 2 * frac), 5)
        rec[f"{name}_excludes_zero"] = bool(lo > 0 or hi < 0)
        rec[f"{name}_n_valid"] = len(v)
    rows.append(rec)

res = pd.DataFrame(rows)
for name in boot:
    rej, adj, _, _ = multipletests(res[f"{name}_p"].values, alpha=ALPHA, method="fdr_bh")
    res[f"{name}_p_fdr"] = np.round(adj, 5)
    res[f"{name}_sig_fdr"] = rej

res = res.sort_values("point_slope", ascending=False)
res.to_csv(OUT / "T16_trend_bootstrap_full.csv", index=False)

head = res[["topic", "label", "ols7_slope", "ols7_p_fdr",
            "B1_iid_ci_low", "B1_iid_ci_high", "B1_iid_p_fdr", "B1_iid_sig_fdr",
            "B2_journal_ci_low", "B2_journal_ci_high", "B2_journal_p_fdr",
            "B2_journal_sig_fdr", "theilsen_slope", "wls7_p"]]
head.to_csv(OUT / "T17_trend_bootstrap_headline.csv", index=False)

jc = sub.groupby("Journal").size().sort_values(ascending=False)
pd.DataFrame({"journal": jc.index, "papers": jc.values,
              "share": (jc.values / len(sub)).round(4)}).to_csv(
    OUT / "T18_journal_concentration.csv", index=False)

surv = pd.DataFrame({
    "estimator": ["OLS-7 (Round 1)", "WLS-7 (Round 1)", "Theil-Sen (Round 1)",
                  "B1 iid document bootstrap", "B2 journal cluster bootstrap"],
    "n_significant_after_fdr": [
        int((r1["ols7_p_fdr"] < ALPHA).sum()),
        int((r1["wls7_p_fdr"] < ALPHA).sum()),
        int((r1["permutation_p_fdr"] < ALPHA).sum()),
        int(res["B1_iid_sig_fdr"].sum()),
        int(res["B2_journal_sig_fdr"].sum())],
    "topics": [
        ", ".join(f"T{t}" for t in sorted(r1[r1.ols7_p_fdr < ALPHA].topic)),
        ", ".join(f"T{t}" for t in sorted(r1[r1.wls7_p_fdr < ALPHA].topic)),
        ", ".join(f"T{t}" for t in sorted(r1[r1.permutation_p_fdr < ALPHA].topic)),
        ", ".join(f"T{t}" for t in sorted(res[res.B1_iid_sig_fdr].topic)),
        ", ".join(f"T{t}" for t in sorted(res[res.B2_journal_sig_fdr].topic))],
})
surv.to_csv(OUT / "T19_estimator_survival.csv", index=False)

print("\n" + "=" * 128)
print(f"BOOTSTRAP TREND UNCERTAINTY ({NBOOT} resamples, 2019-2025 window unchanged)")
print(head.to_string(index=False, max_colwidth=36))
print("\n" + "=" * 128)
print("HOW MANY TRENDS SURVIVE EACH ESTIMATOR (BH FDR q=0.05)")
print(surv.to_string(index=False))
print("\n" + "=" * 128)
print(f"Journal concentration: top journal {jc.iloc[0]} papers "
      f"({jc.iloc[0]/len(sub):.1%}), top 10 journals {jc.head(10).sum()/len(sub):.1%} of window")
print(f"Effective clusters for B2: {len(journals)} journals vs {len(sub)} documents")
