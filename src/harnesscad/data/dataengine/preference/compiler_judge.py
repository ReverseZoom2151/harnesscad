"""Compiler-as-a-Judge (CJM) binary labelling rule (CAD-Judge, Zhou et al.,
2025, "Toward Efficient Morphological Grading and Verification for Text-to-CAD
Generation").

CAD-Judge replaces costly VLM-rendered pairwise ranking with a fast, rule-based
*per-sample binary* signal (Figure 2). A predicted sketch-and-extrude sequence
is compiled and, if it compiles, the Chamfer Distance (CD) to the ground-truth
point cloud is measured. The labelling rule is::

    desirable   <-  compiled AND cd <= threshold
    undesirable <-  failed to compile  OR  cd > threshold

These labels feed the KTO-style (prospect-theory) alignment objective, which
needs only a binary desirable/undesirable signal -- no pairwise construction.

This module implements that *decision rule* (the part missing from
``binary_preferences.py``, which stores an already-decided flag). It is
deterministic: given a compile flag and a CD it returns the label, plus batch
helpers for the label balance and the invalidity ratio the paper also reports.
"""

from __future__ import annotations


def cjm_label(compiled: bool, chamfer_distance, threshold: float):
    """Binary CJM label for one candidate.

    ``compiled`` -- did the CAD compiler successfully build a solid?
    ``chamfer_distance`` -- CD to ground truth (ignored / may be None when the
    candidate failed to compile). ``threshold`` -- CD acceptance threshold.

    Returns True (desirable) iff the candidate compiled and its CD is within the
    threshold, else False. Raises if a compiled candidate has no CD, or if the
    threshold is negative.
    """
    if threshold < 0:
        raise ValueError("threshold must be >= 0")
    if not compiled:
        return False
    if chamfer_distance is None:
        raise ValueError("compiled candidate requires a chamfer_distance")
    return float(chamfer_distance) <= float(threshold)


def label_batch(records, threshold: float):
    """Label a batch of candidates and summarise.

    ``records`` is an iterable of mappings each with ``compiled`` (bool) and,
    when compiled, ``cd`` (numeric). Returns a dict:
      labels          : tuple of per-record bool desirability.
      desirable       : count of desirable labels.
      undesirable     : count of undesirable labels.
      invalidity_ratio: fraction of records that failed to compile.
      total           : number of records.
    """
    rows = list(records)
    labels = []
    failed = 0
    for r in rows:
        compiled = bool(r.get("compiled", False))
        if not compiled:
            failed += 1
        labels.append(cjm_label(compiled, r.get("cd"), threshold))
    total = len(rows)
    desirable = sum(labels)
    return {
        "labels": tuple(labels),
        "desirable": desirable,
        "undesirable": total - desirable,
        "invalidity_ratio": failed / total if total else 0.0,
        "total": total,
    }
