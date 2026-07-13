"""Text2CAD-Bench CadQuery operation-sequence edit distance.

Deterministic sequence-level metric for Text2CAD-Bench (Wang et al.,
"Text2CAD-Bench"). The benchmark ground truth is executable CadQuery whose
construction is a *sequence of API calls* (Appendix A reports mean "API Calls"
of 10.8 / 15.0 / 26.8 for L1/L2/L3). This module measures how close a model's
operation sequence is to the ground-truth construction sequence via Levenshtein
edit distance over operation tokens, normalised by the longer sequence.

Distinct from ``bench/edit_metrics`` (ranked B-rep edit retention / pass@k) and
``bench/sketch_sequence_metrics``: here the tokens are CadQuery API operation
names (normalised via ``t2cadbench_taxonomy.normalize_operation``), and we
report a length-normalised distance plus a similarity in [0, 1] tailored to the
benchmark's construction-sequence comparison. Sequences are injected.

No wall clock, no randomness.
"""

from __future__ import annotations

from harnesscad.eval.bench.data.difficulty_tiers import normalize_operation


def _tokenize(seq):
    toks = [normalize_operation(t) for t in seq]
    return [t for t in toks if t]


def levenshtein(a, b):
    """Unit-cost Levenshtein edit distance between two token lists."""
    a = list(a)
    b = list(b)
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1,        # deletion
                           cur[j - 1] + 1,     # insertion
                           prev[j - 1] + cost))  # substitution
        prev = cur
    return prev[-1]


def sequence_edit_distance(predicted, truth):
    """Edit distance / normalised distance / similarity of two op sequences.

    predicted, truth : iterables of CadQuery operation tokens.

    Returns a dict:
      distance          : raw Levenshtein token edit distance.
      max_len           : max(len(predicted), len(truth)).
      normalized_distance : distance / max_len (0.0 for two empty sequences).
      similarity        : 1 - normalized_distance.
      pred_len, truth_len, api_call_delta : length bookkeeping (paper "API
                          Calls" proxy; delta = pred_len - truth_len).
    """
    p = _tokenize(predicted)
    t = _tokenize(truth)
    dist = levenshtein(p, t)
    max_len = max(len(p), len(t))
    norm = dist / max_len if max_len else 0.0
    return {
        "distance": dist,
        "max_len": max_len,
        "normalized_distance": norm,
        "similarity": 1.0 - norm,
        "pred_len": len(p),
        "truth_len": len(t),
        "api_call_delta": len(p) - len(t),
    }


def mean_sequence_similarity(examples):
    """Mean normalised similarity across (predicted, truth) sequence pairs.

    Returns a dict: n, mean_similarity, mean_normalized_distance,
    exact_match_rate (fraction of pairs with distance 0).
    """
    rows = [sequence_edit_distance(p, t) for p, t in examples]
    n = len(rows)
    if n == 0:
        raise ValueError("no examples")
    return {
        "n": n,
        "mean_similarity": sum(r["similarity"] for r in rows) / n,
        "mean_normalized_distance":
            sum(r["normalized_distance"] for r in rows) / n,
        "exact_match_rate": sum(1 for r in rows if r["distance"] == 0) / n,
    }
