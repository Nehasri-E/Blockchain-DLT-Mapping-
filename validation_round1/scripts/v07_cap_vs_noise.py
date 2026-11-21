"""Task A, controlling analysis: is the conclusion-cap effect larger than the
seed-noise floor?

Task D establishes that two models trained on the SAME corpus with different
seeds already disagree substantially. Any cross-cap difference must therefore be
judged against that floor, not against zero.

Two matched contrasts:

  noise floor   B250(seed i) vs B250(seed j) for all 15 unordered seed pairs.
                Corpus fixed, seed varies. This is pure estimation noise.

  cap effect    B250(seed s) vs X(seed s) for each of the 6 seeds.
                Seed held at the same value, cap varies. Note that the two
                models still have different random initialisations because the
                corpora differ in size and vocabulary, so this contrast is
                "cap + residual init noise", never cap alone. It is therefore a
                conservative (upper) estimate of the cap effect.

If the cap effect distribution is not separated from the noise floor
distribution, the cap is not the dominant driver of topic structure and the
choice of 250 words cannot be defended or attacked on topic-stability grounds.
"""
import pickle
import itertools
import numpy as np
import pandas as pd
from gensim.models import LdaModel
from scipy.stats import mannwhitneyu
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import valtools as vt

ROOT = Path("/home/claude/LDA_Project")
VR = ROOT / "validation_round1"
GRID = VR / "models" / "grid"
OUT = VR / "outputs"

K = 12
SEEDS = [42, 7, 2024, 123, 777, 31337]
CONDITIONS = ["A", "B500", "B1000", "BFull"]
REF = "B250"

corpora = pickle.load(open(VR / "models" / "validation_corpora.pkl", "rb"))
models = {(c, s): LdaModel.load(str(GRID / f"lda_{c}_k{K}_seed{s}.model"))
          for c in CONDITIONS + [REF] for s in SEEDS}


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


noise = []
dref = corpora[REF]["dictionary"]
for s1, s2 in itertools.combinations(SEEDS, 2):
    a = vt.align(models[(REF, s1)], dref, models[(REF, s2)], dref,
                 mode="union", metric="cosine")
    noise.append(float(a["matched_similarity"].mean()))

rows = [{"contrast": "noise floor (B250 vs B250, seed varies)", "pair": f"{s1}-{s2}",
         "mean_matched_cosine": round(v, 4)}
        for (s1, s2), v in zip(itertools.combinations(SEEDS, 2), noise)]

summary = []
for cond in CONDITIONS:
    dc = corpora[cond]["dictionary"]
    vals = []
    for s in SEEDS:
        a = vt.align(models[(REF, s)], dref, models[(cond, s)], dc,
                     mode="union", metric="cosine")
        v = float(a["matched_similarity"].mean())
        vals.append(v)
        rows.append({"contrast": f"cap effect (B250 vs {cond}, seed matched)",
                     "pair": f"seed {s}", "mean_matched_cosine": round(v, 4)})
    u, p = mannwhitneyu(noise, vals, alternative="greater")
    summary.append({
        "comparison": f"B250 vs {cond}",
        "noise_floor_mean": round(float(np.mean(noise)), 4),
        "noise_floor_sd": round(float(np.std(noise, ddof=1)), 4),
        "cap_contrast_mean": round(float(np.mean(vals)), 4),
        "cap_contrast_sd": round(float(np.std(vals, ddof=1)), 4),
        "difference": round(float(np.mean(noise) - np.mean(vals)), 4),
        "difference_in_noise_sds": round(
            float((np.mean(noise) - np.mean(vals)) / np.std(noise, ddof=1)), 2),
        "mannwhitney_p": round(float(p), 5),
        "cliffs_delta": round(cliffs_delta(noise, vals), 3),
        "separated_from_noise": bool(p < 0.05),
    })

pd.DataFrame(rows).to_csv(OUT / "A10_cap_effect_vs_seed_noise_raw.csv", index=False)
summ = pd.DataFrame(summary)
summ.to_csv(OUT / "A10_cap_effect_vs_seed_noise_summary.csv", index=False)

print("=" * 100)
print("A10  IS THE CAP EFFECT LARGER THAN THE SEED-NOISE FLOOR?")
print(f"\nNoise floor (B250 vs B250 across 15 seed pairs): "
      f"mean {np.mean(noise):.4f}, sd {np.std(noise, ddof=1):.4f}, "
      f"range {min(noise):.4f}-{max(noise):.4f}\n")
print(summ.to_string(index=False))
print("\nInterpretation guide: 'difference_in_noise_sds' below about 2 means the")
print("cap moves the topic solution no further than simply changing the seed does.")
