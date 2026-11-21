"""Shared utilities for validation round 1: topic alignment and agreement measures.

Topic alignment across models trained on DIFFERENT vocabularies is the central
methodological problem in Task A. Two complementary projections are implemented
because neither alone is sufficient:

  union        Pad both topic-word distributions to the union vocabulary with
               zeros. Honest about disagreement (terms absent from one model
               contribute nothing to the similarity) but biases similarity
               DOWNWARD as vocabularies diverge, because mass sitting on
               out-of-vocabulary terms can never be matched.

  intersection Restrict both to the shared vocabulary and renormalise each topic
               to sum to 1. Removes the vocabulary-size confound and answers
               "do these topics agree about the words they can both express?",
               but is OPTIMISTIC: it discards exactly the terms that make the
               corpora different.

The truth is bracketed by the two. Both are reported everywhere.

Matching is one-to-one via the Hungarian algorithm (scipy
linear_sum_assignment), maximising total similarity. Greedy nearest-neighbour
matching is NOT used: it can assign two source topics to one target and
overstates agreement.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon


def topic_matrix(model, dictionary):
    """Return {term: probability} per topic, keyed by term string not term id."""
    T = model.get_topics()
    return [{dictionary[i]: float(p) for i, p in enumerate(row)} for row in T]


def project(ta, tb, mode):
    """Project two term-keyed topic sets onto a common vocabulary.

    mode='union'        -> zero-padded to the union vocabulary (rows may sum < 1)
    mode='intersection' -> restricted to shared terms, each row renormalised to 1
    """
    va = set().union(*[set(t) for t in ta])
    vb = set().union(*[set(t) for t in tb])
    vocab = sorted(va | vb) if mode == "union" else sorted(va & vb)
    idx = {w: i for i, w in enumerate(vocab)}
    A = np.zeros((len(ta), len(vocab)))
    B = np.zeros((len(tb), len(vocab)))
    for M, ts in ((A, ta), (B, tb)):
        for r, t in enumerate(ts):
            for w, p in t.items():
                j = idx.get(w)
                if j is not None:
                    M[r, j] = p
    if mode == "intersection":
        A = A / np.clip(A.sum(1, keepdims=True), 1e-12, None)
        B = B / np.clip(B.sum(1, keepdims=True), 1e-12, None)
    return A, B, vocab


def cosine_matrix(A, B):
    An = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
    Bn = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-12, None)
    return An @ Bn.T


def js_similarity_matrix(A, B):
    """1 - Jensen-Shannon distance (base 2). Requires rows to be distributions,
    so each row is renormalised first. Range [0, 1], 1 = identical."""
    An = A / np.clip(A.sum(1, keepdims=True), 1e-12, None)
    Bn = B / np.clip(B.sum(1, keepdims=True), 1e-12, None)
    S = np.zeros((len(An), len(Bn)))
    for i in range(len(An)):
        for j in range(len(Bn)):
            S[i, j] = 1.0 - jensenshannon(An[i], Bn[j], base=2)
    return S


def match(S):
    """One-to-one assignment maximising total similarity.

    Returns (col_for_each_row, similarity_of_each_matched_pair)."""
    r, c = linear_sum_assignment(-S)
    assert np.array_equal(r, np.arange(S.shape[0])), "row assignment not complete"
    assert len(set(c)) == len(c), "assignment is not one-to-one"
    return c, S[r, c]


def align(model_a, dict_a, model_b, dict_b, mode="union", metric="cosine"):
    """Full alignment of two models. Returns a dict of matrices and matches."""
    ta, tb = topic_matrix(model_a, dict_a), topic_matrix(model_b, dict_b)
    A, B, vocab = project(ta, tb, mode)
    S = cosine_matrix(A, B) if metric == "cosine" else js_similarity_matrix(A, B)
    cols, sims = match(S)
    return {"similarity": S, "match": cols, "matched_similarity": sims,
            "vocab_size": len(vocab), "mode": mode, "metric": metric}


def doc_topic_matrix(model, bow, k):
    """Document-topic matrix. Note: gensim seeds variational gamma from the
    model's mutable random_state, so values depend on RNG state at call time.
    Always computed in one pass immediately after loading, never incrementally."""
    out = np.zeros((len(bow), k))
    for i, doc in enumerate(bow):
        for t, p in model.get_document_topics(doc, minimum_probability=0.0):
            out[i, t] = p
    return out


def adjusted_rand(a, b):
    """Adjusted Rand Index. Permutation invariant, so it needs no topic matching
    and is the primary document-level agreement measure here."""
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(a, b))


def dominant_agreement(theta_ref, theta_cmp, match_cols):
    """Raw dominant-topic agreement after applying a one-to-one topic matching.

    match_cols[i] = index in the COMPARISON model matched to reference topic i,
    so the comparison labels are mapped back into the reference numbering."""
    ref = theta_ref.argmax(1)
    cmp_raw = theta_cmp.argmax(1)
    back = {int(c): int(i) for i, c in enumerate(match_cols)}
    cmp_mapped = np.array([back[int(x)] for x in cmp_raw])
    return float((ref == cmp_mapped).mean()), ref, cmp_mapped


def cohens_kappa(a, b, k):
    n = len(a)
    po = (a == b).mean()
    pa = np.bincount(a, minlength=k) / n
    pb = np.bincount(b, minlength=k) / n
    pe = float((pa * pb).sum())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")
