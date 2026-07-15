"""The PDD lens on the hard corpus: grade a submission against a per-part CONTRACT.

The hard corpus grades a text-to-CAD submission on two field-weak oracles side by
side -- the metrics the published benchmarks use (valid + IoU + Chamfer, in
:mod:`~harnesscad.eval.hardcorpus.weak`) and the measured oracle
(:mod:`~harnesscad.eval.hardcorpus.oracle`). This module adds the third lens the
PDD synthesis (``audit/pdd_synthesis.md``) describes: the **Measured Geometric
Contract (MGC)** as the answer key. For each corpus case it derives an MGC from the
case's OWN known-good measurements -- the closed-form volume, the stated bounding
box, the exact genus that ``eval/corpus/spec.Brief`` already carries as independent
ground truth -- and then grades a submission's measured geometry through
:func:`harnesscad.domain.spec.contract.check`. A submission is thereby judged
against a per-part *measured contract*, not only against the field's weak metrics.

WHERE EACH CASE'S GROUND-TRUTH MEASUREMENTS COME FROM
-----------------------------------------------------
A corpus case is a :class:`~harnesscad.eval.corpus.spec.Brief` (or any object /
mapping exposing the same fields). Its ground truth is DECLARED, non-us and exact:

* ``volume``  the closed-form volume in mm3 (``Source.ANALYTIC``: arithmetic, no
  code here gets a vote). It becomes a ``volume_mm3`` predicate with the same
  relative tolerance the measured oracle uses.
* ``bbox``    the exact ``(dx, dy, dz)`` envelope -- REQUIRED on every Brief, the
  check the pressure corpus lacked. It becomes three per-axis ``bbox_{x,y,z}_mm``
  predicates with the oracle's absolute tolerance, so a dilated shell fails here.
* ``genus``   the exact topology when known in closed form (N through-holes =>
  genus N). It becomes an EXACT ``genus`` predicate. ``None`` => an unbound
  ``[NEEDS CLARIFICATION]`` marker (the anti-guess rule: topology is never guessed).

So the contract is compiled from the case's answer key, NOT from the submission --
the whole point of a contract. We build predicates directly rather than call
:func:`~harnesscad.domain.spec.contract.compile_contract`, because that entry point
reads a part brief's stated ``width/depth/height`` fields, whereas a corpus Brief
carries the already-reduced ``bbox``/``volume``/``genus`` truth; building predicates
straight from those measured quantities is the faithful path and needs no guessing.

WHERE THE SUBMISSION'S MEASURED GEOMETRY COMES FROM
---------------------------------------------------
A submission that already exists is measured, never generated, in one of two ways:

* the caller hands in a plain ``measurement`` mapping (keyed by contract keys), or
* the caller hands in an :class:`~harnesscad.eval.hardcorpus.oracle.OracleScore`
  (from :func:`harnesscad.eval.hardcorpus.oracle.grade` on the submission's op
  stream); :func:`measurement_from_oracle_score` reads its measured volume, bbox
  and genus. Refuse-with-taint is preserved: a ``None`` measurement flows through
  as ``MISSING`` and never reads as a pass.

:func:`grade_ops` is the convenience that measures a submission's op stream on the
exact kernel via the oracle and then applies the contract. It runs NO model -- it
grades geometry that already exists.

THE HONEST RESIDUAL (surfaced on every report, per the doc's discipline)
------------------------------------------------------------------------
The MGC is necessary, not sufficient. volume + bbox + genus do NOT pin a unique
part: a hole bored in the wrong place, or a box shelled on the wrong face, has the
same volume, box and topology as the correct part, so it SATISFIES this contract
while the measured oracle's point probes still fail it. That divergence -- contract
satisfied, oracle unsolved -- is not a bug in the grader; it is the many-to-one
residual the PDD synthesis names, made visible. :attr:`CaseContractGrade.residual_gap`
flags exactly those cases, and :data:`RESIDUAL_NOTE` states it on every report.

Absolute imports under ``harnesscad.``; the contract module is pure and imported at
top level, but the oracle / corpus scorer (which pull the OCCT kernel) are imported
lazily so this module imports cleanly with no kernel present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from harnesscad.domain.spec import contract as _contract

__all__ = [
    "RESIDUAL_NOTE",
    "VOLUME_REL",
    "BBOX_ABS",
    "VOLUME_FLOOR",
    "CaseContractGrade",
    "CorpusContractReport",
    "contract_for_case",
    "measurement_from_oracle_score",
    "grade",
    "grade_ops",
    "grade_discriminative",
    "render",
    "main",
]

#: The honest residual, printed on every report. Mirrors ``audit/pdd_synthesis.md``:
#: the MGC narrows the space of correct parts, it does not close it.
RESIDUAL_NOTE = (
    "The Measured Geometric Contract is necessary, not sufficient: volume + bbox "
    "+ genus do not pin a unique part. A submission that satisfies the contract is "
    "one that satisfies it -- not a proof it matches intent. The measured oracle's "
    "point probes catch what the envelope cannot; a 'residual_gap' row is a "
    "submission the contract passes and only the probes fail."
)

#: Tolerances, mirrored from :mod:`harnesscad.eval.hardcorpus.oracle` (VOLUME_REL,
#: BBOX_ABS) so the contract grades on the same numeric guards as the oracle. They
#: are copied, not imported, because importing the oracle pulls the OCCT kernel.
VOLUME_REL = 1e-3
BBOX_ABS = 1e-2
VOLUME_FLOOR = 1e-3

_AXES = ("x", "y", "z")


# --------------------------------------------------------------------------- #
# Per-case grade and the corpus-level report.
# --------------------------------------------------------------------------- #
@dataclass
class CaseContractGrade:
    """One submission judged against one case's Measured Geometric Contract.

    ``satisfied`` is the contract verdict (every bound MEASURED predicate PASSed).
    ``oracle_solved`` is the existing measured-oracle verdict, carried alongside so
    the two lenses are always read together -- never the contract alone.
    """

    case_id: str
    contract_digest: str = ""
    satisfied: bool = False
    #: predicate key -> (target, measured, delta) for each FAIL.
    failed: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    #: predicate keys that measured MISSING (absent or None -- refuse-with-taint).
    missing: List[str] = field(default_factory=list)
    #: unbound predicate keys -- the outstanding ``[NEEDS CLARIFICATION]`` markers.
    clarifications: List[str] = field(default_factory=list)
    #: the measured oracle's verdict on the same submission, when available.
    oracle_solved: Optional[bool] = None
    oracle_reasons: List[str] = field(default_factory=list)
    #: an optional tag for the submission (e.g. "correct" / "near-miss").
    label: str = ""
    residual_note: str = RESIDUAL_NOTE

    @property
    def residual_gap(self) -> bool:
        """The contract passes but the measured oracle does not.

        This is the many-to-one residual made visible: the envelope (volume, bbox,
        genus) cannot distinguish this submission from the correct part, yet a point
        probe can. Only meaningful when an oracle verdict is present.
        """
        return bool(self.satisfied and self.oracle_solved is False)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "label": self.label,
            "contract_digest": self.contract_digest,
            "satisfied": self.satisfied,
            "failed": {k: dict(v) for k, v in self.failed.items()},
            "missing": list(self.missing),
            "clarifications": list(self.clarifications),
            "oracle_solved": self.oracle_solved,
            "oracle_reasons": list(self.oracle_reasons),
            "residual_gap": self.residual_gap,
            "residual_note": self.residual_note,
        }


@dataclass
class CorpusContractReport:
    """A run of contract grades over several corpus cases."""

    grades: Tuple[CaseContractGrade, ...] = ()
    note: str = ""

    @property
    def n(self) -> int:
        return len(self.grades)

    @property
    def satisfied(self) -> int:
        return sum(1 for g in self.grades if g.satisfied)

    @property
    def residual_gaps(self) -> int:
        """Submissions the contract passes and the measured oracle fails."""
        return sum(1 for g in self.grades if g.residual_gap)

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "satisfied": self.satisfied,
            "residual_gaps": self.residual_gaps,
            "residual_note": RESIDUAL_NOTE,
            "note": self.note,
            "grades": [g.to_dict() for g in self.grades],
        }


# --------------------------------------------------------------------------- #
# Building the contract from a case's known-good measurements.
# --------------------------------------------------------------------------- #
def contract_for_case(case: Any) -> "_contract.MeasuredGeometricContract":
    """Derive a Measured Geometric Contract from a case's answer key.

    ``case`` may be a :class:`~harnesscad.eval.corpus.spec.Brief`, a mapping, or any
    object exposing ``id``, ``bbox``, ``volume`` and (optionally) ``genus``. Every
    measurable the case STATES becomes a bound predicate; anything it does not state
    becomes an ``unbound`` ``[NEEDS CLARIFICATION]`` marker whose magnitude is never
    guessed.
    """
    truth = _case_truth(case)
    predicates: List["_contract.Predicate"] = []

    bbox = truth["bbox"]
    if _is_triplet(bbox):
        for axis, want in zip(_AXES, bbox):
            predicates.append(
                _contract.Predicate(
                    key="bbox_%s_mm" % axis,
                    target=float(want),
                    tolerance=BBOX_ABS,
                    kind=_contract.PredicateKind.MEASURED,
                    note="stated envelope extent (%s axis)" % axis,
                )
            )
    else:
        predicates.append(_unbound_predicate("bbox_x_mm", "case states no envelope"))
        predicates.append(_unbound_predicate("bbox_y_mm", "case states no envelope"))
        predicates.append(_unbound_predicate("bbox_z_mm", "case states no envelope"))

    volume = _as_float(truth["volume"])
    if volume is not None and volume > 0.0:
        predicates.append(
            _contract.Predicate(
                key="volume_mm3",
                target=volume,
                tolerance=max(VOLUME_FLOOR, volume * VOLUME_REL),
                kind=_contract.PredicateKind.MEASURED,
                note="closed-form volume from the case's answer key",
            )
        )
    else:
        predicates.append(
            _unbound_predicate("volume_mm3", "case states no closed-form volume")
        )

    genus = _as_int(truth["genus"])
    if genus is not None:
        predicates.append(
            _contract.Predicate(
                key="genus",
                target=int(genus),
                kind=_contract.PredicateKind.MEASURED,
                note="exact topology (N through-holes => genus N)",
            )
        )
    else:
        # The anti-guess rule: an unstated topology is a clarification, not a zero.
        predicates.append(
            _unbound_predicate("genus", "case does not state a closed-form genus")
        )

    return _contract.MeasuredGeometricContract(
        part_id=_coerce_str(truth["id"]) or "case",
        predicates=tuple(predicates),
        intent=_coerce_str(truth["text"]),
    )


def measurement_from_oracle_score(score: Any) -> Dict[str, Any]:
    """Read a submission's measured geometry off an ``OracleScore`` into contract keys.

    ``score.measured`` carries ``{"volume": ..., "bbox": [dx, dy, dz], "genus": ...}``
    as measured off the exact kernel. A ``None`` value is passed through unchanged so
    the contract's refuse-with-taint resolves it to ``MISSING`` rather than a pass.
    """
    measured = getattr(score, "measured", None)
    if not isinstance(measured, Mapping):
        measured = {}
    out: Dict[str, Any] = {"volume_mm3": measured.get("volume")}
    bbox = measured.get("bbox")
    if _is_triplet(bbox):
        for axis, value in zip(_AXES, bbox):
            out["bbox_%s_mm" % axis] = _as_float(value)
    if "genus" in measured:
        out["genus"] = measured.get("genus")
    return out


# --------------------------------------------------------------------------- #
# Grading a submission against a case's contract.
# --------------------------------------------------------------------------- #
def grade(
    case: Any,
    measurement: Optional[Mapping[str, Any]] = None,
    *,
    oracle_score: Any = None,
    oracle_solved: Optional[bool] = None,
    oracle_reasons: Optional[Sequence[str]] = None,
    label: str = "",
) -> CaseContractGrade:
    """Grade one submission's measured geometry against one case's contract.

    ``measurement`` is a mapping keyed by contract keys (``volume_mm3``,
    ``bbox_{x,y,z}_mm``, ``genus``). If it is ``None`` and an ``oracle_score`` is
    given, the measurement is read from that score. The existing oracle verdict
    (``oracle_score.solved`` / ``oracle_solved``) is carried alongside so the two
    lenses are always reported together.
    """
    mgc = contract_for_case(case)

    if measurement is None:
        if oracle_score is not None:
            measurement = measurement_from_oracle_score(oracle_score)
        else:
            measurement = {}

    report = _contract.check(mgc, measurement)

    o_solved = oracle_solved
    o_reasons: List[str] = list(oracle_reasons or [])
    if oracle_score is not None:
        if hasattr(oracle_score, "solved"):
            o_solved = bool(oracle_score.solved)
        if getattr(oracle_score, "reasons", None):
            o_reasons = list(oracle_score.reasons)

    failed: Dict[str, Dict[str, Any]] = {}
    for r in report.failures():
        failed[r.predicate.key] = {
            "target": r.predicate.target,
            "measured": r.measured_value,
            "delta": r.delta,
            "note": r.predicate.note,
        }

    return CaseContractGrade(
        case_id=_coerce_str(_case_truth(case)["id"]) or "case",
        contract_digest=mgc.digest(),
        satisfied=report.satisfied,
        failed=failed,
        missing=[r.predicate.key for r in report.missing()],
        clarifications=[r.predicate.key for r in report.clarifications()],
        oracle_solved=o_solved,
        oracle_reasons=o_reasons,
        label=label,
    )


def grade_ops(case: Any, ops: Sequence[Any], *, label: str = "") -> CaseContractGrade:
    """Measure an existing submission's op stream on the exact kernel, then grade it.

    This imports the measured oracle lazily (it pulls the OCCT kernel). ``case`` must
    be a :class:`~harnesscad.eval.corpus.spec.Brief`, since the oracle needs the
    brief's probes and envelope to measure against. No model is run: the op stream is
    a submission that already exists.
    """
    from harnesscad.eval.hardcorpus import oracle as _oracle  # lazy: pulls kernel

    score = _oracle.grade(case, list(ops))
    return grade(case, oracle_score=score, label=label)


def grade_discriminative(split: Optional[str] = None) -> CorpusContractReport:
    """Grade every discriminative near-miss case's correct and wrong twin by contract.

    For each matched pair (a correct op stream and its plausible wrong twin) this
    measures both on the exact kernel via the oracle and grades each through the
    case's contract. It is the clearest demonstration of the residual: for the
    ``pos_hole`` and ``shell_face`` families the wrong twin has the SAME volume, bbox
    and genus, so it SATISFIES the contract -- and the oracle's probes still fail it
    (``residual_gap``). For ``dia_hole`` the volumes differ, so the contract itself
    catches the wrong twin.

    Degrades gracefully: if the discriminative module or its kernel is unavailable,
    an empty report with a ``note`` is returned rather than raising on import.
    """
    try:
        from harnesscad.eval.hardcorpus import discriminative as _disc
        from harnesscad.eval.hardcorpus import oracle as _oracle
    except Exception as exc:  # noqa: BLE001
        return CorpusContractReport(
            grades=(), note="discriminative/oracle unavailable: %s: %s"
            % (type(exc).__name__, exc)
        )

    try:
        cases = list(_disc.CASES if split is None else _disc.cases(split))
    except Exception as exc:  # noqa: BLE001
        return CorpusContractReport(
            grades=(), note="could not build cases: %s: %s"
            % (type(exc).__name__, exc)
        )

    grades: List[CaseContractGrade] = []
    for nm in cases:
        brief = nm.brief
        for label, ops in (("correct", nm.correct), ("near-miss", nm.near)):
            try:
                score = _oracle.grade(brief, list(ops))
                grades.append(grade(brief, oracle_score=score, label=label))
            except Exception as exc:  # noqa: BLE001
                grades.append(
                    CaseContractGrade(
                        case_id=getattr(brief, "id", "case"),
                        label=label,
                        oracle_reasons=["grading raised %s: %s"
                                        % (type(exc).__name__, exc)],
                    )
                )
    return CorpusContractReport(grades=tuple(grades))


# --------------------------------------------------------------------------- #
# Rendering.
# --------------------------------------------------------------------------- #
def render(report: CorpusContractReport) -> str:
    """The contract lens as text: contract verdict beside the measured oracle."""
    lines: List[str] = []
    lines.append("CONTRACT GRADER -- the PDD answer key applied to the hard corpus")
    lines.append("=" * 78)
    lines.append("Each submission is graded against a per-part Measured Geometric")
    lines.append("Contract derived from the case's own volume + bbox + genus, beside")
    lines.append("the measured oracle's verdict on the same submission.")
    lines.append("")
    if report.note:
        lines.append("note: " + report.note)
        lines.append("")
    lines.append("%-16s %-10s %-10s %-8s %-8s"
                 % ("case", "label", "contract", "oracle", "residual"))
    lines.append("-" * 78)
    if not report.grades:
        lines.append("     (no submission graded -- kernel/corpus absent in this env)")
    for g in report.grades:
        lines.append(
            "%-16s %-10s %-10s %-8s %-8s"
            % (g.case_id[:16], g.label[:10],
               "SATISFIED" if g.satisfied else "FAILED",
               "n/a" if g.oracle_solved is None
               else ("solved" if g.oracle_solved else "UNSOLVED"),
               "GAP" if g.residual_gap else "-"))
        if g.failed:
            lines.append("     contract fails: " + ", ".join(sorted(g.failed)))
    lines.append("-" * 78)
    lines.append("A 'residual' GAP is a submission the contract passes and the")
    lines.append("measured oracle fails -- the many-to-one residual, made visible.")
    lines.append("")
    lines.append(RESIDUAL_NOTE)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Helpers (pure, defensive).
# --------------------------------------------------------------------------- #
def _case_truth(case: Any) -> Dict[str, Any]:
    """Extract ``id``, ``bbox``, ``volume``, ``genus``, ``text`` from any case shape."""
    def get(name: str) -> Any:
        if isinstance(case, Mapping):
            return case.get(name)
        return getattr(case, name, None)

    text = get("text")
    if not text:
        text = get("intent")
    return {
        "id": get("id"),
        "bbox": get("bbox"),
        "volume": get("volume"),
        "genus": get("genus"),
        "text": text,
    }


def _unbound_predicate(key: str, note: str) -> "_contract.Predicate":
    """A ``[NEEDS CLARIFICATION]`` marker: the measurable is present, magnitude unknown."""
    return _contract.Predicate(
        key=key,
        target=None,
        tolerance=0.0,
        kind=_contract.PredicateKind.MEASURED,
        unbound=True,
        note=note,
    )


def _is_triplet(value: Any) -> bool:
    return isinstance(value, (tuple, list)) and len(value) == 3


def _as_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


# --------------------------------------------------------------------------- #
# Self-check: a synthetic case and synthetic submissions, no kernel, no model.
# --------------------------------------------------------------------------- #
def _selfcheck() -> int:
    """Grade synthetic submissions against a synthetic case's contract.

    Uses a hand-built case (a 60 x 40 x 12 mm solid plate, genus 0) and four
    synthetic measurements. Touches no kernel and runs no model -- it exercises the
    bridge end to end: contract compilation, satisfaction, float/exact failures, and
    the refuse-with-taint MISSING path.
    """
    from types import SimpleNamespace

    a, b, c = 60.0, 40.0, 12.0
    case = SimpleNamespace(
        id="selfcheck_plate",
        bbox=(a, b, c),
        volume=a * b * c,          # 28800.0 mm3, closed form
        genus=0,
        text="synthetic 60 x 40 x 12 mm solid plate",
    )

    good = {"volume_mm3": a * b * c, "bbox_x_mm": a, "bbox_y_mm": b,
            "bbox_z_mm": c, "genus": 0}
    wrong_volume = dict(good, volume_mm3=a * b * c * 1.5)
    wrong_genus = dict(good, genus=1)
    missing_volume = dict(good, volume_mm3=None)     # refuse-with-taint

    checks: List[Tuple[str, Dict[str, Any], Optional[bool], bool, str]] = [
        # (label, measurement, oracle_solved, expect_satisfied, expect_key_in)
        ("good", good, True, True, ""),
        ("wrong_volume", wrong_volume, True, False, "volume_mm3"),
        ("wrong_genus", wrong_genus, True, False, "genus"),
        ("missing_volume", missing_volume, False, False, "volume_mm3"),
    ]

    grades: List[CaseContractGrade] = []
    ok = True
    for label, meas, o_solved, expect_sat, expect_key in checks:
        g = grade(case, meas, oracle_solved=o_solved, label=label)
        grades.append(g)
        if g.satisfied != expect_sat:
            ok = False
            print("selfcheck FAIL: %s expected satisfied=%s, got %s"
                  % (label, expect_sat, g.satisfied))
        if expect_key and label == "missing_volume":
            if expect_key not in g.missing:
                ok = False
                print("selfcheck FAIL: %s expected %r in missing, got %s"
                      % (label, expect_key, g.missing))
        elif expect_key:
            if expect_key not in g.failed:
                ok = False
                print("selfcheck FAIL: %s expected %r in failed, got %s"
                      % (label, expect_key, sorted(g.failed)))

    # The contract digest must be stable across recompilation of the same case.
    if contract_for_case(case).digest() != contract_for_case(case).digest():
        ok = False
        print("selfcheck FAIL: contract digest is not deterministic")

    print(render(CorpusContractReport(grades=tuple(grades),
                                      note="synthetic self-check (no kernel, no model)")))
    print("")
    print("selfcheck: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Grade hard-corpus submissions against a per-part Measured "
                    "Geometric Contract (the PDD answer key), beside the oracle.")
    ap.add_argument("--selfcheck", action="store_true",
                    help="run the synthetic fixture (no kernel, no model) and exit.")
    ap.add_argument("--discriminative", action="store_true",
                    help="grade the dev discriminative near-misses by contract "
                         "(needs the exact kernel; degrades to an empty report).")
    ap.add_argument("--json", action="store_true",
                    help="emit JSON instead of the text table.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.discriminative:
        report = grade_discriminative()
        if args.json:
            print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            print(render(report))
        return 0

    # Default action is the self-check: this module grades submissions that already
    # exist, so with no input there is nothing else to run.
    return _selfcheck()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
