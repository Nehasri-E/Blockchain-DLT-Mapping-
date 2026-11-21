"""Shared machinery for validation round 2."""
import logging
import re
import pickle
import itertools
import numpy as np
from pathlib import Path
from gensim.models import LdaModel
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import adjusted_rand_score

ROOT = Path("/home/claude/LDA_Project")
R2 = ROOT / "validation_round2"
SEEDS = [42, 7, 2024, 123, 777, 31337]
BASE_HP = dict(alpha="symmetric", eta="auto", chunksize=200, eval_every=None)


def load_corpus(name="B250"):
    c = pickle.load(open(ROOT / "validation_round1/models/validation_corpora.pkl", "rb"))
    return c[name]


class ConvergenceLog(logging.Handler):
    """Captures gensim's per-chunk 'documents converged within N iterations' lines.

    These are emitted at DEBUG level from LdaModel.inference and are the only
    direct evidence gensim exposes about whether the per-document variational
    E-step reached its own convergence criterion before hitting the iteration
    cap. A model whose documents all converge has not been truncated by
    `iterations`; one where they do not is under-iterated by construction.
    """

    PAT = re.compile(r"(\d+)/(\d+) documents converged within (\d+) iterations")

    def __init__(self):
        super().__init__()
        self.chunks = []

    def emit(self, record):
        m = self.PAT.match(record.getMessage())
        if m:
            self.chunks.append((int(m.group(1)), int(m.group(2))))


def train(bow, dictionary, k, seed, passes, iterations, alpha="symmetric",
          eta="auto", capture=True):
    """Train one model, optionally capturing convergence diagnostics."""
    import time
    grab = ConvergenceLog() if capture else None
    lg = logging.getLogger("gensim.models.ldamodel")
    if capture:
        prev = lg.level
        lg.setLevel(logging.DEBUG)
        lg.addHandler(grab)
    t = time.time()
    hp = dict(BASE_HP)
    hp["alpha"], hp["eta"] = alpha, eta
    m = LdaModel(corpus=bow, id2word=dictionary, num_topics=k, passes=passes,
                 iterations=iterations, random_state=seed, **hp)
    elapsed = time.time() - t
    diag = {"seconds": round(elapsed, 1)}
    if capture:
        lg.removeHandler(grab)
        lg.setLevel(prev)
        n_chunks = int(np.ceil(len(bow) / BASE_HP["chunksize"]))
        tail = grab.chunks[-n_chunks:] if grab.chunks else []
        conv = sum(x[0] for x in tail)
        tot = sum(x[1] for x in tail)
        diag["final_pass_docs_converged"] = conv
        diag["final_pass_docs_total"] = tot
        diag["final_pass_convergence_rate"] = round(conv / tot, 5) if tot else None
        allc = sum(x[0] for x in grab.chunks)
        allt = sum(x[1] for x in grab.chunks)
        diag["overall_convergence_rate"] = round(allc / allt, 5) if allt else None
    return m, diag


def topic_dicts(model, dictionary):
    return [{dictionary[i]: float(p) for i, p in enumerate(row)}
            for row in model.get_topics()]


def project_union(ta, tb):
    vocab = sorted(set().union(*[set(t) for t in ta]) |
                   set().union(*[set(t) for t in tb]))
    idx = {w: i for i, w in enumerate(vocab)}
    A = np.zeros((len(ta), len(vocab)))
    B = np.zeros((len(tb), len(vocab)))
    for M, ts in ((A, ta), (B, tb)):
        for r, t in enumerate(ts):
            for w, p in t.items():
                M[r, idx[w]] = p
    return A, B


def cosine_matrix(A, B):
    An = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
    Bn = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-12, None)
    return An @ Bn.T


def js_matrix(A, B):
    An = A / np.clip(A.sum(1, keepdims=True), 1e-12, None)
    Bn = B / np.clip(B.sum(1, keepdims=True), 1e-12, None)
    return np.array([[1.0 - jensenshannon(a, b, base=2) for b in Bn] for a in An])


def hungarian(S):
    r, c = linear_sum_assignment(-S)
    assert len(set(c)) == len(c), "assignment not one-to-one"
    return c, S[r, c]


def compare(model_a, model_b, dictionary, metric="cosine"):
    A, B = project_union(topic_dicts(model_a, dictionary),
                         topic_dicts(model_b, dictionary))
    S = cosine_matrix(A, B) if metric == "cosine" else js_matrix(A, B)
    cols, sims = hungarian(S)
    return S, cols, sims


def theta(model, bow, k):
    out = np.zeros((len(bow), k))
    for i, doc in enumerate(bow):
        for t, p in model.get_document_topics(doc, minimum_probability=0.0):
            out[i, t] = p
    return out


def dominant_agreement(th_a, th_b, cols):
    a = th_a.argmax(1)
    back = {int(c): int(i) for i, c in enumerate(cols)}
    b = np.array([back[int(x)] for x in th_b.argmax(1)])
    return float((a == b).mean()), float(adjusted_rand_score(th_a.argmax(1),
                                                             th_b.argmax(1)))


def distinctiveness(model):
    """Mean pairwise Jensen-Shannon distance between a model's own topics."""
    T = model.get_topics()
    k = T.shape[0]
    return float(np.mean([jensenshannon(T[i], T[j], base=2)
                          for i, j in itertools.combinations(range(k), 2)]))


def stability_block(models, bow, dictionary, k, seeds=SEEDS, repro=0.70):
    """Full pairwise stability summary for a set of same-configuration models."""
    th = {s: theta(models[s], bow, k) for s in seeds}
    cos_means, n_repro, agrees, aris, js_means = [], [], [], [], []
    for s1, s2 in itertools.combinations(seeds, 2):
        _, cols, sims = compare(models[s1], models[s2], dictionary, "cosine")
        cos_means.append(float(sims.mean()))
        n_repro.append(int((sims >= repro).sum()))
        _, _, jsims = compare(models[s1], models[s2], dictionary, "js")
        js_means.append(float(jsims.mean()))
        ag, ari = dominant_agreement(th[s1], th[s2], cols)
        agrees.append(ag)
        aris.append(ari)
    return {
        "n_pairs": len(cos_means),
        "mean_matched_cosine": round(float(np.mean(cos_means)), 4),
        "sd_matched_cosine": round(float(np.std(cos_means, ddof=1)), 4),
        "min_matched_cosine": round(float(np.min(cos_means)), 4),
        "mean_matched_js": round(float(np.mean(js_means)), 4),
        "mean_reproducible_topics": round(float(np.mean(n_repro)), 2),
        "mean_dominant_agreement": round(float(np.mean(agrees)), 4),
        "sd_dominant_agreement": round(float(np.std(agrees, ddof=1)), 4),
        "mean_ari": round(float(np.mean(aris)), 4),
        "sd_ari": round(float(np.std(aris, ddof=1)), 4),
    }
