"""PLLM candidate -> pseudo-label selection (Chamfer best-of-k with gating).

Implements the deterministic selection algorithm from PLLM ("Pseudo-Labeling
Large Language Models for CAD Program Synthesis", Section 3.1). For each
unlabeled target shape the pre-trained LLM samples k candidate programs
(k = 10). Each candidate is executed and compared to the input shape using
*Chamfer Distance* (lower is better; surface alignment). PLLM then:

  1. discards candidates that fail to execute (execution-validity gate);
  2. selects the candidate with the LOWEST Chamfer Distance as the
     representative program z*;
  3. when several candidates yield "nearly identical reconstructions"
     (Chamfer difference < 1e-4), prefers the SHORTER program to promote
     concise representations;
  4. only accepts z* as a pseudo-label when its Chamfer Distance is within a
     confidence/agreement threshold (geometric-agreement gate).

This is deliberately distinct from ``dataengine.gift_bootstrap_loop`` /
``gift_threshold_selection`` (which uses an IoU CDF band that keeps BOTH a
high-agreement SRS band and a near-miss FDA band). PLLM keeps a *single*
best-of-k winner per shape, uses Chamfer (not IoU), and gates on an absolute
confidence threshold with a shorter-program tie-break. It is also distinct
from cadrille hard-example mining (which surfaces the *worst* cases for extra
supervision); here low-fidelity shapes are simply rejected.

The LLM and executor are external. A candidate is represented by its already
computed Chamfer Distance, an executability flag, and a program length (token
count). Deterministic, stdlib-only.
"""

from __future__ import annotations

from collections import namedtuple

# A candidate program proposed by the (external) LLM for one target shape.
#   program:    opaque program identity (str / tuple) used for the label.
#   chamfer:    Chamfer Distance to the target shape (>= 0, lower = better).
#               Ignored when ``executable`` is False.
#   length:     program length in tokens (>= 0); the tie-break key.
#   executable: True iff the black-box executor produced valid geometry.
Candidate = namedtuple("Candidate", ["program", "chamfer", "length", "executable"])

# Near-identical-reconstruction tolerance (paper: difference < 1e-4).
NEAR_TIE = 1e-4


def _as_candidate(c):
    return c if isinstance(c, Candidate) else Candidate(*c)


def executable_candidates(candidates):
    """Return only the candidates that executed to valid geometry."""
    return [_as_candidate(c) for c in candidates if _as_candidate(c).executable]


def select_representative(candidates, near_tie=NEAR_TIE):
    """Best-of-k winner z* for one shape (lowest Chamfer, shorter on a tie).

    Filters non-executable candidates, then picks the minimum-Chamfer program.
    Any candidate whose Chamfer is within ``near_tie`` of the best is treated as
    a tie and the *shortest* such program wins (paper: prefer shorter programs
    when reconstructions are nearly identical); a further length tie is broken
    by lower Chamfer then by program identity for determinism. Returns the
    winning ``Candidate`` or None when nothing executed.
    """
    if near_tie < 0:
        raise ValueError("near_tie must be >= 0")
    execs = executable_candidates(candidates)
    if not execs:
        return None
    best_cd = min(c.chamfer for c in execs)
    tied = [c for c in execs if c.chamfer - best_cd <= near_tie]
    tied.sort(key=lambda c: (c.length, c.chamfer, str(c.program)))
    return tied[0]


def accept_pseudo_label(candidates, cd_threshold, near_tie=NEAR_TIE):
    """Select and gate a single pseudo-label for one shape.

    Runs :func:`select_representative`, then applies the geometric-agreement
    gate: the winner is accepted only when its Chamfer Distance is <=
    ``cd_threshold``. Returns a dict describing the outcome::

        {"accepted": bool, "program": program|None, "chamfer": float|None,
         "length": int|None, "reason": str}

    ``reason`` is one of ``"accepted"``, ``"no_executable"`` (nothing ran), or
    ``"below_confidence"`` (best CD exceeded the threshold).
    """
    if cd_threshold < 0:
        raise ValueError("cd_threshold must be >= 0")
    rep = select_representative(candidates, near_tie)
    if rep is None:
        return {"accepted": False, "program": None, "chamfer": None,
                "length": None, "reason": "no_executable"}
    if rep.chamfer > cd_threshold:
        return {"accepted": False, "program": rep.program, "chamfer": rep.chamfer,
                "length": rep.length, "reason": "below_confidence"}
    return {"accepted": True, "program": rep.program, "chamfer": rep.chamfer,
            "length": rep.length, "reason": "accepted"}


def select_dataset(shape_candidates, cd_threshold, near_tie=NEAR_TIE):
    """Select pseudo-labels across a batch of unlabeled shapes.

    ``shape_candidates`` maps shape_id -> iterable[Candidate]. Returns a dict
    with the accepted (shape_id, program, chamfer, length) records, the list of
    rejected shape ids with their reason, and summary counts including the
    yield (accepted / total shapes) which feeds the label-efficiency metric.
    """
    accepted, rejected = [], []
    for shape_id, cands in shape_candidates.items():
        out = accept_pseudo_label(cands, cd_threshold, near_tie)
        if out["accepted"]:
            accepted.append({"shape_id": shape_id, "program": out["program"],
                             "chamfer": out["chamfer"], "length": out["length"]})
        else:
            rejected.append({"shape_id": shape_id, "reason": out["reason"]})
    total = len(shape_candidates)
    return {
        "accepted": accepted,
        "rejected": rejected,
        "counts": {"total": total, "accepted": len(accepted),
                   "rejected": len(rejected)},
        "yield": (len(accepted) / total) if total else 0.0,
    }
