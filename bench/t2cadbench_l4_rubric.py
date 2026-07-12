"""Text2CAD-Bench L4 real-world VLM-scoring rubric aggregation.

Deterministic re-encoding of the Text2CAD-Bench L4 evaluation protocol (Wang et
al., "Text2CAD-Bench", Sections 3.3.3 / 4.3, Table 2). L4 targets real-world
application scenarios whose quality exceeds pure geometric fidelity, so each
generated model is rendered to 8 multi-view images and scored by a VLM judge on
five 0-10 questions plus an Overall geometric-similarity score:

  Q1 Overall Features, Q2 External Features, Q3 Functional Features,
  Q4 Extended Features, Q5 Detail Features, Overall Geometric Similarity.

Protocol rules implemented here:
  * Q1-Q5 and Overall are averaged over *valid (successfully executed) samples
    only*; Invalidity Rate is reported separately over all samples.
  * Three largely independent capability dimensions (Section 4.3): code
    executability (from IR), geometric similarity (from Overall), and
    feature-level design understanding (mean of Q1-Q5). This module computes the
    three and identifies the per-dimension leaders, quantifying the "capability
    decoupling" finding (no single model dominates all three).

The VLM judge itself is external -- per-sample question scores are injected.
This is DISTINCT from ``bench/engdesign_qa_scoring`` (VLM QA tasks, paper 85)
and ``bench/muse_*``: it is the specific 5-question + Overall L4 rubric with
valid-only averaging and three-way decoupling. No wall clock, no randomness.
"""

from __future__ import annotations

QUESTIONS = ("q1", "q2", "q3", "q4", "q5")
QUESTION_LABELS = {
    "q1": "Overall Features",
    "q2": "External Features",
    "q3": "Functional Features",
    "q4": "Extended Features",
    "q5": "Detail Features",
    "overall": "Geometric Similarity",
}
SCORE_MIN, SCORE_MAX = 0.0, 10.0


def _clamp_score(name, value):
    v = float(value)
    if v < SCORE_MIN or v > SCORE_MAX:
        raise ValueError("%s must be in [0, 10], got %r" % (name, value))
    return v


def l4_model_scorecard(samples):
    """Aggregate one model's L4 samples into IR + valid-only rubric means.

    samples : iterable of records. An invalid sample sets {"valid": False}. A
        valid sample provides {"valid": True, "q1".."q5": 0-10, "overall": 0-10}.

    Returns a dict:
      n_total, n_valid,
      ir            : invalidity rate in percent over all samples,
      q1..q5, overall : means over valid samples (None if none valid),
      feature_mean  : mean of Q1-Q5 means (None if none valid).
    """
    samples = list(samples)
    n_total = len(samples)
    if n_total == 0:
        raise ValueError("no samples")
    valid = [s for s in samples if s.get("valid")]
    n_valid = len(valid)
    ir = 100.0 * (n_total - n_valid) / n_total
    out = {"n_total": n_total, "n_valid": n_valid, "ir": ir}
    if n_valid == 0:
        for q in QUESTIONS + ("overall",):
            out[q] = None
        out["feature_mean"] = None
        return out
    for q in QUESTIONS + ("overall",):
        out[q] = sum(_clamp_score(q, s[q]) for s in valid) / n_valid
    out["feature_mean"] = sum(out[q] for q in QUESTIONS) / len(QUESTIONS)
    return out


def capability_dimensions(scorecard):
    """The three largely-independent L4 capability dimensions for one model.

    Returns {executability, geometric_similarity, feature_design}:
      executability        = 100 - IR (higher better; percent of valid code),
      geometric_similarity = Overall score (0-10, or None),
      feature_design       = mean of Q1-Q5 (0-10, or None).
    """
    return {
        "executability": 100.0 - scorecard["ir"],
        "geometric_similarity": scorecard.get("overall"),
        "feature_design": scorecard.get("feature_mean"),
    }


def decoupling_leaders(model_scorecards):
    """Per-dimension leaders across models, and whether they decouple.

    model_scorecards : mapping model-name -> scorecard (from
        ``l4_model_scorecard``).

    Returns a dict:
      leaders : dimension -> leading model name (None if all values missing),
      values  : dimension -> {model: value},
      decoupled : True iff the three leaders are not all the same model
        (the paper's capability-decoupling finding).
    """
    dims = ("executability", "geometric_similarity", "feature_design")
    values = {d: {} for d in dims}
    for model, sc in model_scorecards.items():
        cd = capability_dimensions(sc)
        for d in dims:
            values[d][model] = cd[d]
    leaders = {}
    for d in dims:
        present = {m: v for m, v in values[d].items() if v is not None}
        if not present:
            leaders[d] = None
        else:
            # highest value wins; stable model-name tiebreak.
            leaders[d] = max(sorted(present), key=lambda m: present[m])
    non_null = [leaders[d] for d in dims if leaders[d] is not None]
    decoupled = len(set(non_null)) > 1
    return {"leaders": leaders, "values": values, "decoupled": decoupled}
