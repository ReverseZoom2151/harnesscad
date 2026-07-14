"""The ONLY module allowed to import the held-out split.

Everything else in the repository -- every experiment, every notebook, every
debugging session, every other test -- reaches the held-out briefs through here or
not at all, and ``tests/eval/corpus/test_holdout_isolation.py`` fails the suite if
that stops being true.

WHAT THIS MODULE DELIBERATELY DOES NOT GIVE YOU
-----------------------------------------------
It does not hand back the briefs. :func:`score` takes a SOLVER -- a callable from
the brief's TEXT to an op stream -- runs it, and returns numbers. A caller
therefore sees the score and never the answer key, which is what makes the split
held-out rather than merely stored in a different file.

:func:`failures` exists because a scoreboard nobody can act on is a scoreboard
nobody will keep. It returns brief IDS AND REASONS -- not the reference op streams,
not the probe coordinates, not the closed forms. That is enough to know something
regressed and to go reproduce it on the dev split, and not enough to fix the code
against a brief you can only see here. If a held-out failure cannot be reproduced
on dev, THE DEV SPLIT IS TOO WEAK AND THAT IS THE BUG: add a dev brief.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from harnesscad.core.cisp.ops import Op
from harnesscad.eval.corpus import heldout as _heldout   # THE ONLY IMPORT OF IT
from harnesscad.eval.corpus.grade import Score, grade
from harnesscad.eval.corpus.spec import Split

__all__ = ["HeldOutReport", "score", "failures", "size", "reference_score"]

#: A solver: brief text in, op stream out. It never sees anything else.
Solver = Callable[[str], Sequence[Op]]


@dataclass
class HeldOutReport:
    """Numbers only. No briefs, no op streams, no closed forms."""

    n: int = 0
    measurable: int = 0                    # the denominator. See grade.resolvable.
    solved: int = 0                        # the ENVELOPE verdict
    solved_shape: int = 0                  # envelope AND shape
    built: int = 0
    backend: str = "frep"
    mean_iou: Optional[float] = None
    #: brief id -> why it failed. IDs and reasons; never the answer.
    failed: Dict[str, List[str]] = field(default_factory=dict)
    #: brief id -> why the ENGINE could not measure it. Not a failure of the model.
    unmeasurable: Dict[str, str] = field(default_factory=dict)

    @property
    def rate(self) -> float:
        return self.solved / float(self.measurable) if self.measurable else 0.0

    @property
    def shape_rate(self) -> float:
        return self.solved_shape / float(self.measurable) if self.measurable else 0.0

    def to_dict(self) -> dict:
        return {"split": Split.HELDOUT, "n": self.n, "backend": self.backend,
                "measurable": self.measurable,
                "unmeasurable": self.unmeasurable,
                "built": self.built, "solved": self.solved,
                "solved_shape": self.solved_shape,
                "rate": self.rate, "shape_rate": self.shape_rate,
                "mean_iou": self.mean_iou, "failed": self.failed}


def size() -> int:
    """How many briefs are held out. A number is not a leak."""
    return len(_heldout.BRIEFS)


def score(solver: Solver, backend: str = "frep") -> HeldOutReport:
    """Run ``solver`` on every held-out brief and report the score.

    The solver is given the brief's TEXT and nothing else -- not the bbox, not the
    volume, not the probes, not the reference stream. It cannot be handed the
    answer key through its own input, which is the same discipline the pressure
    experiment applied to its arms and the only part of that design that survived
    contact with the audit.
    """
    from harnesscad.eval.corpus.grade import resolvable

    r = HeldOutReport(n=len(_heldout.BRIEFS), backend=backend)
    ious: List[float] = []
    for brief in _heldout.BRIEFS:
        why = resolvable(brief, backend)
        if why is not None:
            # The ENGINE cannot resolve this part. Scoring the model on it would
            # bill the model for the grader's physics.
            r.unmeasurable[brief.id] = why
            continue
        r.measurable += 1
        try:
            ops = list(solver(brief.text))
        except Exception as exc:                              # noqa: BLE001
            r.failed[brief.id] = ["the solver raised %s: %s"
                                  % (type(exc).__name__, exc)]
            continue
        s: Score = grade(brief, ops, backend=backend)
        if s.built:
            r.built += 1
        if s.iou is not None:
            ious.append(float(s.iou))
        if s.solved:
            r.solved += 1
        if s.solved_shape:
            r.solved_shape += 1
        if not s.solved:
            r.failed[brief.id] = list(s.reasons)
    if ious:
        r.mean_iou = sum(ious) / len(ious)
    return r


def reference_score(backend: str = "frep") -> HeldOutReport:
    """Score the held-out split against its own reference solutions.

    This is the corpus's self-test, not a result: if a brief's own hand-written
    answer does not pass its own grader, the brief is broken and every score taken
    against it is meaningless. The pressure corpus failed exactly this check on
    two of its shell briefs and shipped anyway, because nobody ever ran it.
    """
    by_text = {b.text: b.reference for b in _heldout.BRIEFS}
    return score(lambda text: by_text[text], backend=backend)


def failures(report: HeldOutReport) -> List[Tuple[str, List[str]]]:
    """(brief id, reasons) for everything that failed. Sorted, deterministic.

    Take these to the dev split and reproduce them there. Do not fix the code
    against a brief you can only see here -- that is what turns a held-out set back
    into a training set, quietly, in one commit.
    """
    return sorted(report.failed.items())
