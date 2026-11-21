"""Task 5: corrected cap-versus-seed comparison.

WHAT WAS WRONG IN ROUND 1
=========================
Round 1 ran a Mann-Whitney U test with the 15 within-B250 seed-pair
similarities as one sample and 6 seed-matched cap contrasts as the other.

Mann-Whitney assumes the observations within each sample are independent. The
15 seed pairs are not: they are all pairs from only 6 underlying seeds, so each
seed appears in 5 of the 15 pairs. The effective sample size is closer to 6 than
to 15, and the test treats shared-seed pairs as independent replicates, which
understates the variance and overstates the significance. The reported p-values
(0.007 to 0.047) should not be read as inferential evidence.

A second, subtler dependence: the noise and cap samples are not independent of
each other either. Both are computed from the SAME six B250 models, so a seed
that happens to produce an atypical B250 solution shifts both samples together.

WHAT IS DONE INSTEAD
====================
1. Descriptive distributions are preserved unchanged (Round 1 CSVs are not
   touched; they are re-read and re-reported here).

2. The comparison is restructured as a SEED-LEVEL PAIRED contrast. For each
   seed s:
       noise_s = mean over the 5 pairs containing s of cos(B250(s), B250(s'))
       cap_s   = cos(B250(s), X(s))
       d_s     = noise_s - cap_s
   The six d_s are one observation per seed, which removes the repeated-seed
   dependence. They are still not fully independent (noise_s shares pairs with
   noise_s'), which is stated rather than assumed away.

3. Inference by EXACT paired sign-flip permutation over the 6 seeds. All 2^6 =
   64 sign assignments are enumerated, so there is no simulation error. The
   smallest attainable two-sided p-value is 2/64 = 0.031, and with the
   one-sided convention used here 1/64 = 0.0156. No p-value below that is
   achievable with six seeds, whatever the effect size. This is a hard floor
   and is reported as such.

4. Uncertainty by seed-level bootstrap (resampling the 6 seeds with
   replacement, 20,000 draws), reporting a percentile CI on the mean paired
   difference. With 6 clusters the CI is wide and its coverage is approximate;
   both facts are reported.

5. Effect sizes: paired Cohen's dz and the rank-based matched-pairs proportion.

No claim of statistical significance is made from six seeds. The honest output
is an effect size with an interval, plus the observation that the interval is
too wide for six seeds to settle the question.
"""
import json
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from gensim.models import LdaModel
import sys

sys.path.insert(0, str(Path(__file__).parent))
from r00_common import R2, ROOT, SEEDS, compare

import pickle

OUT = R2 / "outputs"
R1 = ROOT / "validation_round1"
GRID = R1 / "models" / "grid"
K = 12
CONDITIONS = ["A", "B500", "B1000", "BFull"]
REF = "B250"
NBOOT = 20000
rng = np.random.default_rng(20261114)

corpora = pickle.load(open(R1 / "models" / "validation_corpora.pkl", "rb"))
models = {(c, s): LdaModel.load(str(GRID / f"lda_{c}_k{K}_seed{s}.model"))
          for c in CONDITIONS + [REF] for s in SEEDS}
dref = corpora[REF]["dictionary"]

pair_cos = {}
for s1, s2 in itertools.combinations(SEEDS, 2):
    _, _, sims = compare(models[(REF, s1)], models[(REF, s2)], dref)
    pair_cos[(s1, s2)] = float(sims.mean())

noise_s = {s: float(np.mean([v for (a, b), v in pair_cos.items() if s in (a, b)]))
           for s in SEEDS}

cap_s = {}
for cond in CONDITIONS:
    dc = corpora[cond]["dictionary"]
    for s in SEEDS:
        A, B = (models[(REF, s)], models[(cond, s)])
        from r00_common import project_union, topic_dicts, cosine_matrix, hungarian
        Am, Bm = project_union(topic_dicts(A, dref), topic_dicts(B, dc))
        _, sims = hungarian(cosine_matrix(Am, Bm))
        cap_s[(cond, s)] = float(sims.mean())


def exact_signflip(d):
    obs = float(np.mean(d))
    n = len(d)
    means = [float(np.mean(d * np.array(sgn))) for sgn in itertools.product([1, -1], repeat=n)]
    means = np.array(means)
    p_two = float((np.abs(means) >= abs(obs) - 1e-15).mean())
    p_one = float((means >= obs - 1e-15).mean())
    return obs, p_one, p_two, len(means)


rows, seedrows = [], []
for cond in CONDITIONS:
    d = np.array([noise_s[s] - cap_s[(cond, s)] for s in SEEDS])
    obs, p1, p2, nperm = exact_signflip(d)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(NBOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    dz = obs / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan
    for s in SEEDS:
        seedrows.append({"condition": cond, "seed": s,
                         "noise_s": round(noise_s[s], 4),
                         "cap_s": round(cap_s[(cond, s)], 4),
                         "paired_difference": round(noise_s[s] - cap_s[(cond, s)], 4)})
    rows.append({
        "comparison": f"B250 vs {cond}",
        "n_seeds": len(SEEDS),
        "mean_noise_s": round(float(np.mean(list(noise_s.values()))), 4),
        "mean_cap_s": round(float(np.mean([cap_s[(cond, s)] for s in SEEDS])), 4),
        "mean_paired_difference": round(obs, 4),
        "boot_ci_low": round(float(lo), 4), "boot_ci_high": round(float(hi), 4),
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "cohens_dz": round(float(dz), 3),
        "n_seeds_favouring_noise": int((d > 0).sum()),
        "exact_signflip_p_one_sided": round(p1, 5),
        "exact_signflip_p_two_sided": round(p2, 5),
        "n_permutations": nperm,
        "min_attainable_p_one_sided": round(1 / 2 ** len(SEEDS), 5),
    })

res = pd.DataFrame(rows)
res.to_csv(OUT / "T12_cap_vs_seed_corrected.csv", index=False)
pd.DataFrame(seedrows).to_csv(OUT / "T13_cap_vs_seed_seedlevel.csv", index=False)

desc = pd.DataFrame([{"statistic": "within-B250 seed pair (15 pairs, DEPENDENT)",
                      "mean": round(float(np.mean(list(pair_cos.values()))), 4),
                      "sd": round(float(np.std(list(pair_cos.values()), ddof=1)), 4),
                      "n": len(pair_cos)}] +
                    [{"statistic": f"seed-level noise_s ({REF})",
                      "mean": round(float(np.mean(list(noise_s.values()))), 4),
                      "sd": round(float(np.std(list(noise_s.values()), ddof=1)), 4),
                      "n": len(noise_s)}] +
                    [{"statistic": f"cap contrast B250 vs {c}",
                      "mean": round(float(np.mean([cap_s[(c, s)] for s in SEEDS])), 4),
                      "sd": round(float(np.std([cap_s[(c, s)] for s in SEEDS], ddof=1)), 4),
                      "n": len(SEEDS)} for c in CONDITIONS])
desc.to_csv(OUT / "T14_cap_vs_seed_descriptives.csv", index=False)

r1 = pd.read_csv(R1 / "outputs" / "A10_cap_effect_vs_seed_noise_summary.csv")
comp = r1[["comparison", "difference", "difference_in_noise_sds", "mannwhitney_p"]].merge(
    res[["comparison", "mean_paired_difference", "boot_ci_low", "boot_ci_high",
         "exact_signflip_p_one_sided", "ci_excludes_zero"]], on="comparison")
comp = comp.rename(columns={"difference": "round1_difference",
                            "mannwhitney_p": "round1_mannwhitney_p_INVALID"})
comp.to_csv(OUT / "T15_round1_vs_corrected.csv", index=False)

print("=" * 120)
print("DESCRIPTIVE DISTRIBUTIONS (preserved from Round 1)")
print(desc.to_string(index=False))
print("\n" + "=" * 120)
print("CORRECTED SEED-LEVEL PAIRED ANALYSIS (n=6 seeds)")
print(res[["comparison", "mean_paired_difference", "boot_ci_low", "boot_ci_high",
           "ci_excludes_zero", "cohens_dz", "n_seeds_favouring_noise",
           "exact_signflip_p_one_sided"]].to_string(index=False))
print(f"\nExact permutation floor with 6 seeds: p >= {1/64:.5f} one-sided. "
      f"No result can go below it.")
print("\n" + "=" * 120)
print("ROUND 1 vs CORRECTED")
print(comp.to_string(index=False))
