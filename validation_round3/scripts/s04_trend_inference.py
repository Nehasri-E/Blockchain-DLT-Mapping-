"""Task 6: corrected temporal trend inference.

WHAT WAS WRONG IN ROUND 2
=========================
Round 2 computed, for each topic, the proportion of journal-cluster bootstrap
resamples whose slope fell on the opposite side of zero from the point estimate,
doubled it, and called that a p-value:

    frac = (v <= 0).mean() if point > 0 else (v >= 0).mean()
    p    = 2 * frac

That quantity is not a p-value. The bootstrap distribution is centred on the
OBSERVED slope, not on the null. Its tail mass measures how much of the sampling
distribution around the estimate lies beyond zero, which is a statement about
the estimate's precision, not about the probability of data this extreme under
H0: slope = 0. Using it as a p-value conflates a confidence statement with a
test, and it is anti-conservative in exactly the regime that matters here, few
clusters and a heavy-tailed cluster-size distribution.

The Round 2 confidence intervals are unaffected by this error and are retained
below as descriptive uncertainty. Only the p-values are withdrawn.

WHAT IS DONE INSTEAD
====================
Inference is moved to the document level, where the clustering actually lives,
and the null is imposed explicitly.

  Model      share_ij = a + b * (year_ij - mean_year) + e_ij
             over the 956 in-window papers, i indexed by paper, j by journal.

  V1  Cluster-robust (CR1) standard errors by journal, giving a descriptive
      95% CI and a t-statistic. Reported but not used alone for inference,
      because CR1 is known to be anti-conservative with unbalanced clusters.

  V2  WILD CLUSTER BOOTSTRAP-t, Rademacher weights, null imposed
      (Cameron, Gelbach and Miller 2008). The restricted model with b = 0 is
      fitted, residuals are resampled at the JOURNAL level with +/-1 weights,
      a null distribution of the t-statistic is built, and the observed t is
      compared against it. This is a genuine null-centred procedure and is the
      primary p-value.

  V3  WITHIN-JOURNAL YEAR PERMUTATION. Year labels are permuted among papers
      inside each journal, which destroys any year-share association while
      preserving both journal composition and each journal's year profile.
      Distribution-free, and a useful cross-check on V2's parametric form.

BH correction is applied only to V2 and V3, which are valid p-values. It is NOT
applied to the Round 2 bootstrap tail proportions, which are withdrawn.

LIMITATIONS OF JOURNAL CLUSTERING
=================================
Journal is a proxy for the real dependence structure, not the structure itself.
Authors publish across journals, so author-level correlation leaks across
cluster boundaries and is not absorbed. Research groups produce correlated
series of papers on one theme in several outlets. Special issues create
within-journal, within-year clumping that journal clustering treats as ordinary.
The author field is present in the data but not parsed into disambiguated author
identities, so author-level clustering could not be implemented here; it is
named as the main remaining gap rather than silently omitted.
"""
import json
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from s00_common import R3, ROOT

OUT = R3 / "outputs"
K = 12
YEARS = list(range(2019, 2026))
NBOOT = 9999
ALPHA = 0.05
rng = np.random.default_rng(20261116)

LABELS = {int(k): v for k, v in json.loads(
    (ROOT / "validation_round1/scripts/baseline_labels.json").read_text()).items()}

papers = pd.read_csv(ROOT / "outputs" / "papers_with_topics.csv")
sub = papers[papers["Year"].isin(YEARS)].reset_index(drop=True)
tcols = [f"Topic_{i+1}" for i in range(K)]
x = (sub["Year"].values - np.mean(YEARS)).astype(float)
Xd = sm.add_constant(x)
jid = pd.factorize(sub["Journal"])[0]
clusters = [np.where(jid == g)[0] for g in range(jid.max() + 1)]
print(f"{len(sub)} documents, {len(clusters)} journal clusters")
print(f"cluster sizes: min {min(len(c) for c in clusters)}, "
      f"median {int(np.median([len(c) for c in clusters]))}, "
      f"max {max(len(c) for c in clusters)}")

r2 = pd.read_csv(ROOT / "validation_round2" / "outputs" / "T16_trend_bootstrap_full.csv")
r1 = pd.read_csv(ROOT / "validation_round1" / "outputs" / "C2_trend_reanalysis_full.csv")

rows = []
for i, col in enumerate(tcols, 1):
    y = sub[col].values.astype(float)

    fit = sm.OLS(y, Xd).fit(cov_type="cluster",
                            cov_kwds={"groups": jid, "use_correction": True})
    b, se = float(fit.params[1]), float(fit.bse[1])
    tobs = b / se
    ci = fit.conf_int(alpha=0.05)[1]

    # V2 wild cluster bootstrap-t, null imposed
    r_fit = sm.OLS(y, sm.add_constant(np.zeros_like(x))).fit()
    yhat0, resid0 = r_fit.fittedvalues, r_fit.resid
    tstars = np.empty(NBOOT)
    for bidx in range(NBOOT):
        w = rng.choice([-1.0, 1.0], size=len(clusters))
        ystar = yhat0.copy()
        for gi, idxs in enumerate(clusters):
            ystar[idxs] = yhat0[idxs] + w[gi] * resid0[idxs]
        f = sm.OLS(ystar, Xd).fit(cov_type="cluster",
                                  cov_kwds={"groups": jid, "use_correction": True})
        tstars[bidx] = f.params[1] / f.bse[1]
    p_wild = float((np.abs(tstars) >= abs(tobs)).sum() + 1) / (NBOOT + 1)

    # V3 within-journal year permutation
    bstars = np.empty(NBOOT)
    for bidx in range(NBOOT):
        xp = x.copy()
        for idxs in clusters:
            if len(idxs) > 1:
                xp[idxs] = rng.permutation(x[idxs])
        xc = xp - xp.mean()
        bstars[bidx] = (xc @ (y - y.mean())) / (xc @ xc)
    p_perm = float((np.abs(bstars) >= abs(b)).sum() + 1) / (NBOOT + 1)

    old = r2[r2.topic == i].iloc[0]
    base = r1[r1.topic == i].iloc[0]
    rows.append({
        "topic": i, "label": LABELS[i],
        "doc_slope": round(b, 6),
        "cluster_robust_se": round(se, 6),
        "cr1_ci_low": round(float(ci[0]), 6), "cr1_ci_high": round(float(ci[1]), 6),
        "t_observed": round(tobs, 3),
        "p_wild_cluster": round(p_wild, 5),
        "p_within_journal_perm": round(p_perm, 5),
        "ols7_slope": base["ols7_slope"], "ols7_p_fdr": base["ols7_p_fdr"],
        "wls7_p": base["wls7_p"], "theilsen_slope": base["theilsen_slope"],
        "r2_B2_ci_low": old["B2_journal_ci_low"], "r2_B2_ci_high": old["B2_journal_ci_high"],
        "r2_B2_tailprop_WITHDRAWN": old["B2_journal_p"],
    })
    print(f"  T{i:<3} b={b:+.5f} t={tobs:+6.2f}  wild p={p_wild:.4f}  perm p={p_perm:.4f}")

res = pd.DataFrame(rows)
for fam in ["p_wild_cluster", "p_within_journal_perm"]:
    rej, adj, _, _ = multipletests(res[fam].values, alpha=ALPHA, method="fdr_bh")
    res[f"{fam}_fdr"] = np.round(adj, 5)
    res[f"{fam}_sig_fdr"] = rej
res["cr1_ci_excludes_zero"] = (res.cr1_ci_low > 0) | (res.cr1_ci_high < 0)
res = res.sort_values("doc_slope", ascending=False)
res.to_csv(OUT / "V1_trend_valid_inference.csv", index=False)

head = res[["topic", "label", "doc_slope", "cr1_ci_low", "cr1_ci_high",
            "cr1_ci_excludes_zero", "p_wild_cluster", "p_wild_cluster_fdr",
            "p_wild_cluster_sig_fdr", "p_within_journal_perm_fdr",
            "p_within_journal_perm_sig_fdr", "r2_B2_tailprop_WITHDRAWN"]]
head.to_csv(OUT / "V2_trend_headline_corrected.csv", index=False)

surv = pd.DataFrame({
    "procedure": ["OLS-7 (R1)", "WLS-7 (R1)", "Theil-Sen (R1)",
                  "R2 journal bootstrap tail proportion (WITHDRAWN, not a p-value)",
                  "Wild cluster bootstrap-t, null imposed (VALID)",
                  "Within-journal year permutation (VALID)"],
    "valid_p_value": [True, True, True, False, True, True],
    "n_significant_after_fdr": [
        int((r1["ols7_p_fdr"] < ALPHA).sum()),
        int((r1["wls7_p_fdr"] < ALPHA).sum()),
        int((r1["permutation_p_fdr"] < ALPHA).sum()),
        int((r2["B2_journal_p_fdr"] < ALPHA).sum()),
        int(res["p_wild_cluster_sig_fdr"].sum()),
        int(res["p_within_journal_perm_sig_fdr"].sum())],
    "topics": [
        ", ".join(f"T{t}" for t in sorted(r1[r1.ols7_p_fdr < ALPHA].topic)),
        ", ".join(f"T{t}" for t in sorted(r1[r1.wls7_p_fdr < ALPHA].topic)),
        ", ".join(f"T{t}" for t in sorted(r1[r1.permutation_p_fdr < ALPHA].topic)),
        ", ".join(f"T{t}" for t in sorted(r2[r2.B2_journal_p_fdr < ALPHA].topic)),
        ", ".join(f"T{t}" for t in sorted(res[res.p_wild_cluster_sig_fdr].topic)),
        ", ".join(f"T{t}" for t in sorted(res[res.p_within_journal_perm_sig_fdr].topic))],
})
surv.to_csv(OUT / "V3_procedure_comparison.csv", index=False)

print("\n" + "=" * 134)
print("CORRECTED TREND INFERENCE (document level, journal clusters, null imposed)")
print(head.to_string(index=False, max_colwidth=34))
print("\n" + "=" * 134)
print("PROCEDURE COMPARISON")
print(surv.to_string(index=False, max_colwidth=62))
