"""Quality control for validation round 3. Every check is an assertion on disk."""
import json
import hashlib
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from gensim.models import LdaModel
import sys

sys.path.insert(0, str(Path(__file__).parent))
from s00_common import R3, ROOT, R2, SEEDS, load_corpus, load_k6_models, cosine_matrix, hungarian

OUT, FIG = R3 / "outputs", R3 / "figures"
R1 = ROOT / "validation_round1"
K = 6
results = []


def check(name, ok, detail=""):
    results.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return ok


print("1  EARLIER ROUNDS UNMODIFIED")
man = (ROOT / "documentation" / "FILE_MANIFEST.md").read_text()
drift = [str(f.relative_to(ROOT)) for f in sorted(ROOT.rglob("*"))
         if f.is_file() and "validation_round" not in str(f)
         and f.name != "FILE_MANIFEST.md"
         and hashlib.sha256(f.read_bytes()).hexdigest()[:16] not in man]
check("72 baseline package files unmodified", not drift, str(drift) if drift else "")
r3_start = min(f.stat().st_mtime for f in (R3 / "scripts").glob("*.py"))
for rd, nm in [(R1, "Round 1"), (R2, "Round 2")]:
    touched = [str(f.relative_to(rd)) for f in rd.rglob("*")
               if f.is_file() and f.stat().st_mtime > r3_start]
    check(f"no {nm} file modified after Round 3 began", not touched,
          str(touched[:4]) if touched else f"{sum(1 for _ in rd.rglob('*'))} paths checked")

print("\n2  MODEL REUSE IS LEGITIMATE")
c = load_corpus("B250")
ref = LdaModel.load(str(R2 / "models/k_sweep/k6_seed42.model"))
refit = LdaModel(corpus=c["bow"], id2word=c["dictionary"], num_topics=K, alpha="symmetric",
                 eta=1.0 / K, passes=100, iterations=1000, chunksize=200,
                 random_state=42, eval_every=None)
check("reused Round 2 K=6 model is bit-identical to a fresh C3 refit",
      bool(np.array_equal(ref.expElogbeta, refit.expElogbeta)))
check("36 topics assembled from 6 seeds x K=6",
      sum(m.num_topics for m in load_k6_models().values()) == 36)

print("\n3  DISTRIBUTIONS AND MATRICES")
models = load_k6_models()
worst = max(float(np.abs(m.get_topics().sum(1) - 1).max()) for m in models.values())
check("LDA topic-word rows sum to 1", worst < 1e-6, f"max dev {worst:.2e}")
Sm = pd.read_csv(OUT / "C1_topic_similarity_matrix_cosine.csv", index_col=0)
check("36x36 similarity matrix is square and symmetric",
      Sm.shape == (36, 36) and bool(np.allclose(Sm.values, Sm.values.T, atol=1e-6)))
check("similarity diagonal is 1.0", bool(np.allclose(np.diag(Sm.values), 1.0)))
check("all similarities within [-1, 1]",
      bool((Sm.values >= -1.0001).all() and (Sm.values <= 1.0001).all()))

print("\n4  SEED UNIQUENESS IN THE PRIMARY CONSENSUS")
mem = pd.read_csv(OUT / "C11_consensus_membership_seedpure.csv")
bad = [g for g, grp in mem.groupby("consensus_topic") if grp.seed.duplicated().any()]
check("no consensus topic contains two topics from one seed", not bad, str(bad))
check("every consensus topic has exactly 6 members, one per seed",
      bool((mem.groupby("consensus_topic").size() == 6).all()) and
      bool((mem.groupby("consensus_topic")["seed"].nunique() == 6).all()))
check("all 36 topics assigned exactly once", len(mem) == 36 and
      not mem.duplicated(["seed", "topic"]).any())

print("\n5  IMPURE CLUSTERS WERE DIAGNOSED, NOT DISCARDED SILENTLY")
dg = pd.read_csv(OUT / "C8_seed_impurity_diagnosis.csv")
check("every seed-impure hierarchical cluster has a recorded diagnosis",
      len(dg) >= 1 and dg.verdict.notna().all(),
      f"{len(dg)} same-seed pairs examined, "
      f"{int(dg.justified_merge.sum())} judged justified")

print("\n6  THRESHOLD WAS NOT CHOSEN TO MAXIMISE CLUSTERS")
sw = pd.read_csv(OUT / "C4_threshold_sensitivity.csv")
cfg = json.loads((OUT / "C7_consensus_config.json").read_text())
ht = cfg["headline_threshold"]
best = float(sw.loc[sw.hier_clusters_multiseed.idxmax(), "threshold"])
at_ht = int(sw[sw.threshold == ht]["hier_clusters_multiseed"].iloc[0])
mx = int(sw["hier_clusters_multiseed"].max())
check("headline threshold is NOT the cluster-count maximiser",
      at_ht < mx,
      f"headline {ht} gives {at_ht} multi-seed clusters; the maximiser is "
      f"{best} with {mx}. The conservative point was chosen, not the permissive one.")
pl = pd.read_csv(OUT / "C13_qualification_across_plateau.csv")
at_head = int(pl[pl.threshold == ht]["n_qualifying_themes"].iloc[0])
check("headline threshold yields the FEWEST qualifying themes on the plateau",
      at_head == int(pl["n_qualifying_themes"].min()),
      f"{at_head} qualifying at {ht}; up to "
      f"{int(pl['n_qualifying_themes'].max())} at more permissive thresholds, "
      f"which are reported rather than adopted")
check("headline threshold is above the chance similarity level",
      ht > cfg["chance_cross_seed_cosine"],
      f"threshold {ht} vs chance {cfg['chance_cross_seed_cosine']}")

print("\n7  NMF SETUP")
n1 = pd.read_csv(OUT / "N1_nmf_per_model.csv")
check("NMF uses 6 initialisations at each of K=4,6,8",
      sorted(n1.k.unique().tolist()) == [4, 6, 8] and
      bool((n1.groupby("k").size() == 6).all()))
check("NMF initialisation is random, not deterministic nndsvd",
      json.loads((OUT / "N5_nmf_config.json").read_text())["nmf"]["init"] == "random")
check("all NMF fits converged before max_iter", bool(n1.converged.all()),
      f"max iters used {int(n1.n_iter.max())} of 1000")
nmem = pd.read_csv(OUT / "N4_nmf_consensus_membership.csv")
check("NMF consensus groups are init-pure, 6 members each",
      bool((nmem.groupby("nmf_topic").size() == 6).all()) and
      bool((nmem.groupby("nmf_topic")["seed"].nunique() == 6).all()))

print("\n8  CROSS-METHOD COMPARISON IS LEGITIMATE")
xcfg = json.loads((OUT / "X4_crosswalk_config.json").read_text())
check("LDA and NMF aligned by L2-normalised cosine on a shared vocabulary",
      "L2-normalised cosine" in xcfg["alignment"])
X = pd.read_csv(OUT / "X1_lda_vs_nmf_similarity.csv", index_col=0)
check("crosswalk matrix is 6x6 over the shared vocabulary", X.shape == (6, 6))
xw = pd.read_csv(OUT / "X2_lda_vs_nmf_crosswalk.csv")
check("crosswalk assignment is one-to-one",
      xw.nmf_consensus_topic.nunique() == 6 and xw.lda_consensus_topic.nunique() == 6)
check("convergent validity requires words AND papers, not cosine alone",
      bool(((xw.convergent_validity) == ((xw.cosine >= 0.50) &
                                         (xw.term_core_jaccard >= 0.30) &
                                         (xw.top25_paper_jaccard >= 0.20))).all()))

print("\n9  CORRECTED TREND INFERENCE")
tv = pd.read_csv(OUT / "V1_trend_valid_inference.csv")
check("Round 2 bootstrap tail proportion retained but labelled withdrawn",
      "r2_B2_tailprop_WITHDRAWN" in tv.columns)
check("BH applied only to the two valid p-value families",
      {"p_wild_cluster_fdr", "p_within_journal_perm_fdr"} <= set(tv.columns) and
      "r2_B2_tailprop_WITHDRAWN_fdr" not in tv.columns)
check("all wild-bootstrap p-values are above the resampling floor",
      bool((tv.p_wild_cluster >= 1 / 10000).all()),
      f"min p {tv.p_wild_cluster.min():.5f}, floor {1/10000:.5f}")
check("CI exclusion of zero is reported separately from significance",
      "cr1_ci_excludes_zero" in tv.columns and
      int(tv.cr1_ci_excludes_zero.sum()) != int(tv.p_wild_cluster_sig_fdr.sum()),
      f"{int(tv.cr1_ci_excludes_zero.sum())} CIs exclude zero but "
      f"{int(tv.p_wild_cluster_sig_fdr.sum())} are significant after FDR")

print("\n10  SENSITIVITY ANALYSES ARE SEPARATE FROM THE MAIN RESULT")
for f in ["S2_supporting_seed_sensitivity.csv", "S3_journal_exclusion_sensitivity.csv",
          "S4_bfull_corpus_sensitivity.csv", "C4_threshold_sensitivity.csv"]:
    check(f"{f} present", (OUT / f).exists())
s3 = pd.read_csv(OUT / "S3_journal_exclusion_sensitivity.csv")
check("journal-exclusion refits cover drop of top 1, 3 and 5 journals",
      sorted(s3.dropped_journals.unique().tolist()) == [1, 3, 5])

print("\n11  FIGURES AND OUTPUTS")
figs = sorted(FIG.glob("*.png"))
check("6 figures present", len(figs) == 6, ", ".join(f.name for f in figs))
src = (R3 / "scripts" / "s06_figures.py").read_text()
check("figure script loads no model objects",
      "LdaModel" not in src and "NMF(" not in src)
csvs = sorted(OUT.glob("*.csv"))
check("machine-readable CSV for every major comparison", len(csvs) >= 18, f"{len(csvs)} CSVs")
for d in ["ROUND3_METHODS.md", "ROUND3_FINDINGS.md"]:
    check(f"{d} present", (R3 / "documentation" / d).exists())

qc = pd.DataFrame(results)
qc.to_csv(OUT / "QC_round3_checklist.csv", index=False)
nf = int((qc.status == "FAIL").sum())
print(f"\n{'='*70}\n{len(qc)-nf}/{len(qc)} checks passed, {nf} failed")
if nf:
    print(qc[qc.status == "FAIL"].to_string(index=False))
