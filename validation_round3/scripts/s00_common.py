"""Shared machinery for validation round 3."""
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
R3 = ROOT / "validation_round3"
SEEDS = [42, 7, 2024, 123, 777, 31337]
C3 = dict(passes=100, iterations=1000, alpha="symmetric", chunksize=200, eval_every=None)


def load_corpus(name="B250"):
    return pickle.load(open(ROOT / "validation_round1/models/validation_corpora.pkl", "rb"))[name]


def load_k6_models():
    """Round 2 K=6 C3 models, reused read-only.

    Verified bit-identical to a fresh refit under
    K=6, passes=100, iterations=1000, alpha=symmetric, eta=1/6, chunksize=200.
    """
    return {s: LdaModel.load(str(R2 / "models/k_sweep" / f"k6_seed{s}.model")) for s in SEEDS}


def theta(model, bow, k):
    out = np.zeros((len(bow), k))
    for i, doc in enumerate(bow):
        for t, p in model.get_document_topics(doc, minimum_probability=0.0):
            out[i, t] = p
    return out


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
    assert len(set(c)) == len(c)
    return c, S[r, c]


def align_union(ta, tb, metric="cosine"):
    """Align two term-keyed topic sets over their union vocabulary.

    Used only for LDA-vs-NMF, where the two methods have different native
    weight scales. Both sides are L2-normalised by cosine_matrix, so the
    comparison is of direction in term space, not of raw weight magnitude.
    """
    vocab = sorted(set().union(*[set(t) for t in ta]) | set().union(*[set(t) for t in tb]))
    idx = {w: i for i, w in enumerate(vocab)}
    A = np.zeros((len(ta), len(vocab)))
    B = np.zeros((len(tb), len(vocab)))
    for M, ts in ((A, ta), (B, tb)):
        for r, t in enumerate(ts):
            for w, p in t.items():
                M[r, idx[w]] = p
    S = cosine_matrix(A, B) if metric == "cosine" else js_matrix(A, B)
    cols, sims = hungarian(S)
    return S, cols, sims


def top_terms(weights, vocab, n=20):
    idx = np.argsort(-weights)[:n]
    return [vocab[i] for i in idx]


def doc_jaccard(th_a, col_a, th_b, col_b, n=25):
    a = set(np.argsort(-th_a[:, col_a])[:n])
    b = set(np.argsort(-th_b[:, col_b])[:n])
    return len(a & b) / len(a | b)


def term_lift(weights, vocab, cfs_norm, n=20):
    idx = np.argsort(-weights)[:n]
    w = weights / max(weights.sum(), 1e-12)
    return float(np.mean([w[i] / max(cfs_norm[i], 1e-12) for i in idx]))


def ari(a, b):
    return float(adjusted_rand_score(a, b))
