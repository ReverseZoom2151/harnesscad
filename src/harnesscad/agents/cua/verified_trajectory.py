"""verified_trajectory — Fara-7B's GUI-trajectory pipeline, with a FREE exact oracle.

Fara-7B is trained on GUI trajectories, and its central problem is one HarnessCAD
does NOT have: it cannot cheaply tell whether a trajectory actually succeeded, so it
must *train a verifier model* (a fallible judge) to label trajectories, and it
publishes CUAVerifierBench to measure how good that judge is. The trajectory is
represented as a sequence of ``(observation, action)`` steps; the verifier looks at
the whole thing and guesses a verdict.

The asymmetry this module makes concrete
----------------------------------------
For CAD we own an EXACT oracle (``agents/cua/grade.grade_ops`` + ``io/gate``): given
the ops a trajectory built, we know — to 4.5e-16 — whether the part is right. So the
Fara schema, ported here, gains a field Fara can only *predict*: a per-step and
final :class:`OracleVerdict` that our gate LABELS for free. This turns a stream of
GUI steps into *supervised* data with no human and no learned verifier — exactly the
data Fara spends a model to approximate.

Two things are built:

* :class:`VerifiedTrajectory` — the (observation, action, oracle-verdict) schema and
  its labelling flow (:func:`label_trajectory`), the CAD-native version of Fara's
  trajectory representation.
* :class:`CUAVerifierBench` — Fara's benchmark structure, reproduced as "how wrong
  is the judge": each example pairs the ORACLE verdict (ground truth, exact, ours
  for free) with a fallible JUDGE verdict, and :meth:`metrics` reports the judge's
  accuracy and — the number that matters for a CUA verifier — its FALSE-POSITIVE
  rate (trajectories the judge passed that the oracle failed).

Pure stdlib, import-safe, JSON round-trips. No model, no app, no network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


# Verdict labels. "verified"/"rejected" are DECIDED (by an oracle or a judge);
# "unlabeled" is a step no verdict has been assigned to yet.
VERIFIED = "verified"
REJECTED = "rejected"
UNLABELED = "unlabeled"


@dataclass(frozen=True)
class OracleVerdict:
    """A verdict on a step or a whole trajectory.

    ``ok`` is the boolean call; ``label`` is one of :data:`VERIFIED` /
    :data:`REJECTED` / :data:`UNLABELED`; ``source`` records WHO decided —
    ``"cad_oracle"`` (exact, ours) vs ``"judge"`` (a fallible model). The source is
    load-bearing: the whole point of :class:`CUAVerifierBench` is to compare the two.
    """

    ok: bool = False
    label: str = UNLABELED
    detail: str = ""
    source: str = "cad_oracle"

    @classmethod
    def unlabeled(cls) -> "OracleVerdict":
        return cls(ok=False, label=UNLABELED, detail="", source="")

    @classmethod
    def verified(cls, detail: str = "", source: str = "cad_oracle") -> "OracleVerdict":
        return cls(ok=True, label=VERIFIED, detail=detail, source=source)

    @classmethod
    def rejected(cls, detail: str = "", source: str = "cad_oracle") -> "OracleVerdict":
        return cls(ok=False, label=REJECTED, detail=detail, source=source)

    @property
    def is_labeled(self) -> bool:
        return self.label != UNLABELED

    def to_dict(self) -> dict:
        return {"ok": self.ok, "label": self.label, "detail": self.detail,
                "source": self.source}

    @classmethod
    def from_dict(cls, d: dict) -> "OracleVerdict":
        return cls(ok=bool(d.get("ok", False)), label=d.get("label", UNLABELED),
                   detail=d.get("detail", ""), source=d.get("source", ""))


@dataclass(frozen=True)
class TrajectoryStep:
    """One (observation, action, verdict) step — Fara's unit, plus our oracle field.

    ``observation`` is what the frame showed (e.g. a SetOfMarks element list or a
    measured-metrics dict); ``action`` is the step taken (a verb + target + params).
    Both are opaque JSON here — the schema pins their PLACE, not their contents.
    ``verdict`` starts UNLABELED and is filled by :func:`label_trajectory`.
    """

    index: int
    observation: Dict[str, Any] = field(default_factory=dict)
    action: Dict[str, Any] = field(default_factory=dict)
    verdict: OracleVerdict = field(default_factory=OracleVerdict.unlabeled)

    def with_verdict(self, verdict: OracleVerdict) -> "TrajectoryStep":
        return TrajectoryStep(index=self.index, observation=dict(self.observation),
                              action=dict(self.action), verdict=verdict)

    def to_dict(self) -> dict:
        return {"index": self.index, "observation": dict(self.observation),
                "action": dict(self.action), "verdict": self.verdict.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "TrajectoryStep":
        return cls(index=int(d["index"]), observation=dict(d.get("observation", {})),
                   action=dict(d.get("action", {})),
                   verdict=OracleVerdict.from_dict(d.get("verdict", {})))


@dataclass
class VerifiedTrajectory:
    """A GUI trajectory as Fara represents it, plus an oracle-labelled verdict track.

    ``final_verdict`` is the trajectory-level call (did the finished part grade out);
    per-step verdicts are the finer signal. :meth:`labeled_fraction` and
    :meth:`is_fully_verified` report how much of the trajectory the oracle has signed
    off — the coverage a training pipeline reads.
    """

    brief: str
    steps: List[TrajectoryStep] = field(default_factory=list)
    final_verdict: OracleVerdict = field(default_factory=OracleVerdict.unlabeled)
    trajectory_id: str = ""

    def labeled_fraction(self) -> float:
        if not self.steps:
            return 0.0
        return sum(1 for s in self.steps if s.verdict.is_labeled) / len(self.steps)

    def is_fully_verified(self) -> bool:
        """Every step verified AND the final part verified — the gold-label case."""
        return (bool(self.steps)
                and self.final_verdict.label == VERIFIED
                and all(s.verdict.label == VERIFIED for s in self.steps))

    def to_dict(self) -> dict:
        return {"brief": self.brief, "trajectory_id": self.trajectory_id,
                "steps": [s.to_dict() for s in self.steps],
                "final_verdict": self.final_verdict.to_dict(),
                "labeled_fraction": round(self.labeled_fraction(), 4)}

    @classmethod
    def from_dict(cls, d: dict) -> "VerifiedTrajectory":
        return cls(brief=d["brief"], trajectory_id=d.get("trajectory_id", ""),
                   steps=[TrajectoryStep.from_dict(s) for s in d.get("steps", [])],
                   final_verdict=OracleVerdict.from_dict(d.get("final_verdict", {})))


# A step-labeller maps a step to a verdict — in HarnessCAD this wraps the gate/grade.
StepLabeller = Callable[[TrajectoryStep], OracleVerdict]


def label_trajectory(trajectory: VerifiedTrajectory, step_labeller: StepLabeller,
                     final_verdict: OracleVerdict) -> VerifiedTrajectory:
    """Assign oracle verdicts to every step + the trajectory, returning a new one.

    ``step_labeller`` is the exact per-step oracle (in HarnessCAD, a function that
    checks the step's built op against the gate); ``final_verdict`` is the whole-part
    grade. This is the labelling Fara must approximate with a trained verifier and we
    get analytically — the concrete payoff of owning the oracle.
    """
    labelled = [s.with_verdict(step_labeller(s)) for s in trajectory.steps]
    return VerifiedTrajectory(brief=trajectory.brief,
                              trajectory_id=trajectory.trajectory_id,
                              steps=labelled, final_verdict=final_verdict)


# --- CUAVerifierBench: "how wrong is the judge?" ----------------------------

@dataclass(frozen=True)
class VerifierExample:
    """One benchmark row: the ORACLE truth vs a fallible JUDGE's guess.

    ``oracle_ok`` is ground truth (our exact gate); ``judge_ok`` is what a candidate
    verifier model claimed. The bench exists to measure the GAP between them — the
    thing Fara's CUAVerifierBench measures, made trivially labelable by our oracle.
    """

    trajectory_id: str
    oracle_ok: bool
    judge_ok: bool

    @property
    def agrees(self) -> bool:
        return self.oracle_ok == self.judge_ok

    @property
    def false_positive(self) -> bool:
        """Judge passed a trajectory the oracle FAILED — the dangerous error: it
        would admit a wrong part into training data or ship it as a success."""
        return self.judge_ok and not self.oracle_ok

    @property
    def false_negative(self) -> bool:
        """Judge failed a trajectory the oracle PASSED — wasteful but safe."""
        return (not self.judge_ok) and self.oracle_ok

    def to_dict(self) -> dict:
        return {"trajectory_id": self.trajectory_id, "oracle_ok": self.oracle_ok,
                "judge_ok": self.judge_ok, "agrees": self.agrees,
                "false_positive": self.false_positive,
                "false_negative": self.false_negative}


class CUAVerifierBench:
    """Fara's CUAVerifierBench, reproduced as a judge-error benchmark.

    Add ``(oracle_ok, judge_ok)`` examples; :meth:`metrics` reports how wrong the
    judge is against the exact oracle: accuracy, agreement, and — the number that
    actually matters for a CUA verifier — the FALSE-POSITIVE rate over oracle-failed
    trajectories (how often it waves a bad part through). Deterministic; pure counts.
    """

    def __init__(self, examples: Optional[Sequence[VerifierExample]] = None) -> None:
        self.examples: List[VerifierExample] = list(examples or [])

    def add(self, trajectory_id: str, oracle_ok: bool, judge_ok: bool) -> VerifierExample:
        ex = VerifierExample(trajectory_id=trajectory_id, oracle_ok=bool(oracle_ok),
                             judge_ok=bool(judge_ok))
        self.examples.append(ex)
        return ex

    def add_from_verdicts(self, trajectory_id: str, oracle: OracleVerdict,
                          judge: OracleVerdict) -> VerifierExample:
        return self.add(trajectory_id, oracle.ok, judge.ok)

    def metrics(self) -> Dict[str, Any]:
        n = len(self.examples)
        if n == 0:
            return {"n": 0, "accuracy": 0.0, "agreement": 0.0,
                    "false_positive_rate": 0.0, "false_negative_rate": 0.0,
                    "false_positives": 0, "false_negatives": 0,
                    "oracle_positives": 0, "oracle_negatives": 0}
        agree = sum(1 for e in self.examples if e.agrees)
        fps = [e for e in self.examples if e.false_positive]
        fns = [e for e in self.examples if e.false_negative]
        oracle_pos = sum(1 for e in self.examples if e.oracle_ok)
        oracle_neg = n - oracle_pos
        return {
            "n": n,
            "accuracy": agree / n,
            "agreement": agree / n,
            # FP rate is over trajectories the ORACLE rejected (where a pass is wrong).
            "false_positive_rate": (len(fps) / oracle_neg) if oracle_neg else 0.0,
            # FN rate is over trajectories the ORACLE accepted (where a fail is wrong).
            "false_negative_rate": (len(fns) / oracle_pos) if oracle_pos else 0.0,
            "false_positives": len(fps),
            "false_negatives": len(fns),
            "oracle_positives": oracle_pos,
            "oracle_negatives": oracle_neg,
        }

    def to_dict(self) -> dict:
        return {"examples": [e.to_dict() for e in self.examples],
                "metrics": self.metrics()}

    def __len__(self) -> int:
        return len(self.examples)
