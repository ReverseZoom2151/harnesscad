"""The ONLY module allowed to import the hard corpus's held-out split.

Every experiment, notebook and debugging session reaches the held-out briefs
through here or not at all, and ``tests/eval/hardcorpus/test_holdout_isolation.py``
fails the suite if that stops being true. This mirrors
``eval/corpus/score.py`` -- same shape, same reason.

WHAT IT WILL AND WILL NOT HAND BACK
-----------------------------------
:func:`score` takes a SOLVER -- a callable from a brief's TEXT to an op stream --
runs it on every held-out brief, and returns NUMBERS. It grades on BOTH oracles,
always: the field's weak metrics (:mod:`~harnesscad.eval.hardcorpus.weak`) and the
measured oracle (:mod:`~harnesscad.eval.hardcorpus.oracle`). The caller sees the two
scores and the gap between them, and never the answer key.

:func:`reference_score` is the self-test, not a result: it scores every held-out
brief against its own reference stream, and if a reference does not pass its own
oracle the brief is broken and every score taken against it is meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from harnesscad.core.cisp.ops import Op
from harnesscad.eval.corpus.spec import Split
from harnesscad.eval.hardcorpus import discriminative as _disc
from harnesscad.eval.hardcorpus import heldout as _heldout   # THE ONLY IMPORT OF IT
from harnesscad.eval.hardcorpus import oracle as _oracle
from harnesscad.eval.hardcorpus import weak as _weak

__all__ = ["HeldOutReport", "score", "reference_score", "size", "near_miss_audit"]

#: A solver: brief text in, op stream out. It sees nothing else -- not the bbox, not
#: the volume, not the probes, not the reference.
Solver = Callable[[str], Sequence[Op]]


@dataclass
class HeldOutReport:
    """Numbers only. No briefs, no op streams, no closed forms."""

    n: int = 0
    built: int = 0
    #: the measured oracle's verdict (bbox+volume+genus+probes).
    oracle_solved: int = 0
    #: the field's grader (valid + IoU + Chamfer) on the same answers.
    weak_passed: int = 0
    #: answers the field PASSES but the oracle FAILS -- the corpus's reason to exist.
    field_fooled: int = 0
    mean_iou: Optional[float] = None
    #: brief id -> why the oracle failed it. Ids and reasons; never the answer.
    failed: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def oracle_rate(self) -> float:
        return self.oracle_solved / float(self.n) if self.n else 0.0

    @property
    def weak_rate(self) -> float:
        return self.weak_passed / float(self.n) if self.n else 0.0

    def to_dict(self) -> dict:
        return {"split": Split.HELDOUT, "n": self.n, "built": self.built,
                "oracle_solved": self.oracle_solved, "oracle_rate": self.oracle_rate,
                "weak_passed": self.weak_passed, "weak_rate": self.weak_rate,
                "field_fooled": self.field_fooled, "mean_iou": self.mean_iou,
                "failed": dict(self.failed)}


def size() -> int:
    """How many held-out briefs there are. A count is not a leak."""
    return len(_heldout.BRIEFS)


def score(solver: Solver) -> HeldOutReport:
    """Run ``solver`` on every held-out brief; report BOTH oracles and the gap.

    The solver is given the brief's TEXT and nothing else, so it cannot be handed
    the answer key through its own input.
    """
    r = HeldOutReport(n=len(_heldout.BRIEFS))
    ious: List[float] = []
    for brief in _heldout.BRIEFS:
        try:
            ops = list(solver(brief.text))
        except Exception as exc:                              # noqa: BLE001
            r.failed[brief.id] = ["the solver raised %s: %s"
                                  % (type(exc).__name__, exc)]
            continue
        osc = _oracle.grade(brief, ops)
        wsc = _weak.score_weak(ops, brief.reference)
        if osc.built:
            r.built += 1
        if wsc.iou is not None:
            ious.append(float(wsc.iou))
        if osc.solved:
            r.oracle_solved += 1
        if wsc.passes:
            r.weak_passed += 1
        if wsc.passes and not osc.solved:
            r.field_fooled += 1
        if not osc.solved:
            r.failed[brief.id] = list(osc.reasons)
    if ious:
        r.mean_iou = sum(ious) / len(ious)
    return r


def reference_score() -> HeldOutReport:
    """Score the held-out split against its own reference solutions.

    The corpus's self-test. If a reference does not pass its own oracle the brief is
    broken. The pressure corpus failed exactly this on two shell briefs and shipped
    anyway because nobody ran it; this is run in the test suite.
    """
    by_text = {b.text: b.reference for b in _heldout.BRIEFS}
    return score(lambda text: by_text[text])


@dataclass
class NearMissAudit:
    """Held-out proof that the field's grader is fooled and ours is not."""

    n: int = 0
    #: near-misses the field passed (should be all of them, by construction).
    weak_passed_near: int = 0
    #: near-misses the oracle failed (should be all of them).
    oracle_failed_near: int = 0
    #: near-misses where BOTH controls hold and the gap is demonstrated.
    gaps: int = 0

    def to_dict(self) -> dict:
        return {"n": self.n, "weak_passed_near": self.weak_passed_near,
                "oracle_failed_near": self.oracle_failed_near, "gaps": self.gaps}


def near_miss_audit() -> NearMissAudit:
    """Grade the held-out near-misses and count how many fool the field but not us."""
    a = NearMissAudit(n=len(_heldout.NEAR_MISSES))
    for nm in _heldout.NEAR_MISSES:
        v = _disc.grade_case(nm)
        if v.weak_near.get("valid"):
            a.weak_passed_near += 1
        if not v.oracle_near.get("solved"):
            a.oracle_failed_near += 1
        if v.controls_hold and (v.demonstrates_gap or v.defeats_geometric_family):
            a.gaps += 1
    return a
