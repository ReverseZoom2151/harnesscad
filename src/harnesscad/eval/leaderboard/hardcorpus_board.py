"""Hard-corpus leaderboard -- the discriminative table, made rankable.

:mod:`harnesscad.eval.hardcorpus` grades a text-to-CAD solver TWICE: on the weak
metrics the field publishes (a valid solid + IoU + Chamfer, in their strongest,
exact form) and on the measured oracle (does each feature land where the brief put
it, checked by exact point membership). The novel result is the GAP between the two
-- parts the field's grader passes and only measurement catches. This module turns
that gap into a scoreboard: every submission gets a row with BOTH columns and the
gap between them, so a reader sees at once what the field would have believed and
what is actually true.

WHY BOTH COLUMNS, ALWAYS
------------------------
A board that ranked on the oracle alone would be as unfalsifiable as the field's
own boards that rank on IoU alone. So each row carries:

* ``weak_rate``   -- the fraction the field's grader (valid + IoU + Chamfer) passes.
  This is the number a Text2CAD-Bench-style leaderboard would print, and print
  alone.
* ``oracle_rate`` -- the fraction the measured oracle solves (bbox + volume + genus
  + point probes). This is the number that is true.
* ``fooled``      -- the count the field PASSED and the oracle FAILED. A high
  ``weak_rate`` next to a low ``oracle_rate`` is not a good submission; it is a
  submission the field would have overrated by exactly this many parts.

THE DISCRIMINATIVE STANDING
---------------------------
Ranking is on ``oracle_rate`` first, because that is the real result; ties break
toward the submission the field fooled LEAST (fewer parts wrongly passed), then by
name for determinism. The board also carries the near-miss audit -- the matched
correct/wrong pairs built so the wrong twin defeats the field's grader on purpose
-- as a fixed proof that sits beside the ranking: it does not depend on any
submission, it is the reason the two columns can diverge at all.

NO MODEL IS RUN HERE
--------------------
A submission is a finished :class:`~harnesscad.eval.hardcorpus.score.HeldOutReport`
(or its JSON). :func:`from_solvers` is a convenience that grades callables -- a
brief's text to an op stream -- by delegating to
:func:`harnesscad.eval.hardcorpus.score.score`. It runs whatever solver it is
handed; it never invokes a model of its own. The frontier run that populates this
board is pending.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Standing", "Board", "ranking", "from_reports", "from_solvers",
    "load_standings", "near_miss_proof", "render", "main",
]


@dataclass
class Standing:
    """One submission's row: the two columns, the gap, and the failure ids.

    Built from a :class:`~harnesscad.eval.hardcorpus.score.HeldOutReport` or its
    ``to_dict`` form. ``failed`` is brief-id -> reasons; it names WHICH briefs the
    oracle failed and WHY, never the answer key.
    """

    name: str
    n: int = 0
    built: int = 0
    oracle_solved: int = 0
    weak_passed: int = 0
    field_fooled: int = 0
    mean_iou: Optional[float] = None
    failed: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def oracle_rate(self) -> float:
        return self.oracle_solved / float(self.n) if self.n else 0.0

    @property
    def weak_rate(self) -> float:
        return self.weak_passed / float(self.n) if self.n else 0.0

    @property
    def fooled_rate(self) -> float:
        """The share of the corpus the field overrated: passed by them, failed by us."""
        return self.field_fooled / float(self.n) if self.n else 0.0

    @property
    def gap(self) -> float:
        """How far the field's headline number overstates the truth, in rate points."""
        return self.weak_rate - self.oracle_rate

    @classmethod
    def from_report(cls, name: str, report: Any) -> "Standing":
        d = report.to_dict() if hasattr(report, "to_dict") else dict(report)
        return cls._from_dict_body(name, d)

    @classmethod
    def from_dict(cls, d: Dict[str, Any], name: str = "") -> "Standing":
        return cls._from_dict_body(name or d.get("name", "") or "submission", d)

    @classmethod
    def _from_dict_body(cls, name: str, d: Dict[str, Any]) -> "Standing":
        return cls(
            name=name or "submission",
            n=int(d.get("n", 0) or 0),
            built=int(d.get("built", 0) or 0),
            oracle_solved=int(d.get("oracle_solved", 0) or 0),
            weak_passed=int(d.get("weak_passed", 0) or 0),
            field_fooled=int(d.get("field_fooled", 0) or 0),
            mean_iou=d.get("mean_iou"),
            failed={k: list(v) for k, v in (d.get("failed") or {}).items()})

    def to_dict(self) -> dict:
        return {"name": self.name, "n": self.n, "built": self.built,
                "oracle_solved": self.oracle_solved, "oracle_rate": self.oracle_rate,
                "weak_passed": self.weak_passed, "weak_rate": self.weak_rate,
                "field_fooled": self.field_fooled, "fooled_rate": self.fooled_rate,
                "gap": self.gap, "mean_iou": self.mean_iou,
                "failed": {k: list(v) for k, v in self.failed.items()}}


def ranking(standings: Sequence[Standing]) -> List[Standing]:
    """Rank on the measured oracle first, then fewest parts the field was fooled on.

    Deterministic: higher ``oracle_rate`` wins; ties go to the submission the field
    overrated least (lower ``field_fooled``); then alphabetical by name.
    """
    return sorted(standings,
                  key=lambda s: (-s.oracle_rate, s.field_fooled, s.name))


@dataclass
class Board:
    """The whole scoreboard: the ranked submissions and the standing near-miss proof."""

    standings: List[Standing] = field(default_factory=list)

    def add(self, standing: Standing) -> "Board":
        self.standings.append(standing)
        return self

    def ranked(self) -> List[Standing]:
        return ranking(self.standings)

    def to_dict(self) -> dict:
        return {"ranking": [s.to_dict() for s in self.ranked()],
                "near_miss_proof": near_miss_proof()}

    def render(self) -> str:
        return render(self.standings)


def from_reports(reports: Sequence[Tuple[str, Any]]) -> Board:
    """Build a board from ``(name, HeldOutReport)`` pairs already scored."""
    board = Board()
    for name, report in reports:
        board.add(Standing.from_report(name, report))
    return board


def from_solvers(solvers: Sequence[Tuple[str, Callable[[str], Any]]]) -> Board:
    """Score named solvers on the held-out split and rank them.

    A convenience over :func:`harnesscad.eval.hardcorpus.score.score`. Each solver
    maps a brief's TEXT to an op stream and sees nothing else -- not the bbox, not
    the probes, not the reference -- so it cannot be handed the answer key through
    its own input. This runs whatever callables it is given; it starts no model.
    """
    from harnesscad.eval.hardcorpus import score as hc_score

    board = Board()
    for name, solver in solvers:
        board.add(Standing.from_report(name, hc_score.score(solver)))
    return board


def load_standings(*paths: str) -> List[Standing]:
    """Load submissions from JSON ``HeldOutReport`` files (one or a list per file)."""
    standings: List[Standing] = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        items = payload if isinstance(payload, list) else [payload]
        stem = os.path.splitext(os.path.basename(path))[0]
        for item in items:
            standings.append(Standing.from_dict(item, name=item.get("name") or stem))
    return standings


def near_miss_proof() -> List[dict]:
    """The fixed, submission-independent proof the two columns can diverge.

    Grades the discriminative matched pairs (a correct part and its plausible wrong
    twin) on both graders and returns, per case, what the field says about the
    wrong twin (it should PASS) beside what the oracle says (it should FAIL). This
    is the headline of the hard corpus, carried on the board so a reader sees WHY a
    high ``weak_rate`` next to a low ``oracle_rate`` is possible at all.

    Grading touches the exact kernel; if that backend is absent this returns an
    empty list rather than failing to import.
    """
    try:
        from harnesscad.eval.hardcorpus import discriminative as disc
    except Exception:                                          # noqa: BLE001
        return []
    proof: List[dict] = []
    for nm in disc.CASES:
        try:
            v = disc.grade_case(nm)
        except Exception as exc:                               # noqa: BLE001
            proof.append({"id": nm.id, "level": nm.level,
                          "error": "%s: %s" % (type(exc).__name__, exc)})
            continue
        wn, on = v.weak_near, v.oracle_near
        proof.append({
            "id": v.id, "level": v.level, "defeats": v.defeats,
            "near_text": v.near_text,
            "field_passes_wrong_twin": bool(wn.get("passes")),
            "field_valid_wrong_twin": bool(wn.get("valid")),
            "oracle_fails_wrong_twin": not bool(on.get("solved")),
            "iou_wrong_twin": wn.get("iou"),
            "controls_hold": v.controls_hold,
            "scored": v.scored})
    return proof


def _fmt_rate(value: float) -> str:
    return "%6.3f" % value


def render(standings: Sequence[Standing]) -> str:
    """The full scoreboard as text: the two-column ranking, then the near-miss proof."""
    lines: List[str] = []
    lines.append("HARD-CORPUS LEADERBOARD -- the field's grader vs the measured oracle")
    lines.append("=" * 78)
    lines.append("Every submission is scored on BOTH the weak metrics the field uses")
    lines.append("(valid + IoU + Chamfer) and the measured oracle (point membership).")
    lines.append("'fooled' is the count the field PASSED and the oracle FAILED -- the gap.")
    lines.append("")
    lines.append("%-4s %-22s %6s %8s %8s %7s %7s"
                 % ("#", "submission", "n", "weak", "oracle", "gap", "fooled"))
    lines.append("-" * 78)
    ranked = ranking(standings)
    if not ranked:
        lines.append("     (no submission yet -- the frontier run is pending)")
    for i, s in enumerate(ranked):
        lines.append("%-4d %-22s %6d %8s %8s %7s %7d"
                     % (i + 1, s.name[:22], s.n, _fmt_rate(s.weak_rate),
                        _fmt_rate(s.oracle_rate), _fmt_rate(s.gap), s.field_fooled))
    lines.append("-" * 78)
    lines.append("Ranked on the measured oracle; ties to the submission the field")
    lines.append("overrated least. A wide 'gap' is a submission the field would have")
    lines.append("believed and the geometry does not.")
    lines.append("")

    proof = near_miss_proof()
    lines.append("NEAR-MISS PROOF -- why the two columns can diverge (no submission needed)")
    lines.append("-" * 78)
    if not proof:
        lines.append("     (the exact kernel is absent; proof unavailable in this env)")
    else:
        lines.append("%-13s %-5s %-6s %-6s %-8s  %s"
                     % ("case", "level", "field", "oracle", "iou", "defeats"))
        for row in proof:
            if "error" in row:
                lines.append("%-13s %-5s  error: %s"
                             % (row["id"], row.get("level", ""), row["error"]))
                continue
            iou = row.get("iou_wrong_twin")
            lines.append("%-13s %-5s %-6s %-6s %-8s  %s"
                         % (row["id"], row["level"],
                            "PASS" if row["field_passes_wrong_twin"]
                            or row["field_valid_wrong_twin"] else "fail",
                            "FAIL" if row["oracle_fails_wrong_twin"] else "pass",
                            "%.3f" % iou if iou is not None else "n/a",
                            row.get("defeats", "")))
        lines.append("")
        lines.append("Each row is a WRONG part the field's grader scores correct and the")
        lines.append("measured oracle catches on a single point. That is the result this")
        lines.append("board exists to rank submissions against.")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Rank hard-corpus submissions (HeldOutReport JSON files).")
    ap.add_argument("reports", nargs="*",
                    help="HeldOutReport.to_dict JSON files, one or more per run.")
    ap.add_argument("--json", action="store_true",
                    help="emit the board as JSON instead of a text table.")
    ap.add_argument("--proof", action="store_true",
                    help="print only the near-miss proof (needs the exact kernel).")
    args = ap.parse_args(list(argv) if argv is not None else None)
    if args.proof:
        print(json.dumps(near_miss_proof(), indent=2, sort_keys=True))
        return 0
    standings = load_standings(*args.reports) if args.reports else []
    board = Board(standings=list(standings))
    if args.json:
        print(json.dumps(board.to_dict(), indent=2, sort_keys=True))
    else:
        print(board.render())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
