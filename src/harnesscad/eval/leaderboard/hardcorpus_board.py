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
    "load_standings", "near_miss_proof", "contract_residual_proof", "render",
    "main",
]


@dataclass
class Standing:
    """One submission's row: the two columns, the gap, and the failure ids.

    Built from a :class:`~harnesscad.eval.hardcorpus.score.HeldOutReport` or its
    ``to_dict`` form. ``failed`` is brief-id -> reasons; it names WHICH briefs the
    oracle failed and WHY, never the answer key.

    THE THIRD LENS -- THE MEASURED GEOMETRIC CONTRACT
    -------------------------------------------------
    Beside the field's weak metric and the measured oracle, a submission MAY carry
    the PDD contract lens (``audit/pdd_synthesis.md``): the count its per-part
    Measured Geometric Contract SATISFIES (``contract_satisfied``) and the count it
    satisfies AND the oracle also solves (``contract_and_oracle``). A base
    ``HeldOutReport`` does not carry these -- the board stays runnable on the
    reports that already exist -- so they are optional: ``None`` when absent, and
    the row prints them as ``-``. When present the board ranks on the STRICTER
    ``contract_and_oracle`` rate, so the many-to-one residual (contract passes,
    oracle fails) is visible in its own column rather than hidden inside a headline.
    """

    name: str
    n: int = 0
    built: int = 0
    oracle_solved: int = 0
    weak_passed: int = 0
    field_fooled: int = 0
    mean_iou: Optional[float] = None
    failed: Dict[str, List[str]] = field(default_factory=dict)
    #: submissions whose per-part MGC is satisfied (optional; None when unscored).
    contract_satisfied: Optional[int] = None
    #: submissions the MGC satisfies AND the oracle solves (optional; the strict
    #: "both" count the board ranks on when it is present).
    contract_and_oracle: Optional[int] = None

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

    @property
    def has_contract(self) -> bool:
        """Whether this submission carries the MGC lens at all."""
        return self.contract_satisfied is not None or self.contract_and_oracle is not None

    @property
    def contract_rate(self) -> Optional[float]:
        """The share of the corpus the per-part MGC satisfies, or None if unscored."""
        if self.contract_satisfied is None or not self.n:
            return None
        return self.contract_satisfied / float(self.n)

    @property
    def combined_rate(self) -> Optional[float]:
        """The share the MGC satisfies AND the oracle solves -- the honest 'both'."""
        if self.contract_and_oracle is None or not self.n:
            return None
        return self.contract_and_oracle / float(self.n)

    @property
    def contract_residual(self) -> Optional[int]:
        """Count the MGC passes but the oracle FAILS -- the many-to-one residual.

        ``contract_satisfied - contract_and_oracle``: submissions whose envelope
        (volume + bbox + genus) cannot be told from the correct part yet a point
        probe still fails them. ``None`` when the contract lens is absent.
        """
        if self.contract_satisfied is None or self.contract_and_oracle is None:
            return None
        return max(0, self.contract_satisfied - self.contract_and_oracle)

    @property
    def rank_rate(self) -> float:
        """The rate the board ranks on: the strict 'both' when the MGC is present,
        else the measured oracle -- so an unscored submission ranks exactly as
        before and a contract-scored one is held to the stricter bar."""
        both = self.combined_rate
        return both if both is not None else self.oracle_rate

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
            failed={k: list(v) for k, v in (d.get("failed") or {}).items()},
            contract_satisfied=_opt_int(d.get("contract_satisfied")),
            contract_and_oracle=_opt_int(d.get("contract_and_oracle")))

    def to_dict(self) -> dict:
        return {"name": self.name, "n": self.n, "built": self.built,
                "oracle_solved": self.oracle_solved, "oracle_rate": self.oracle_rate,
                "weak_passed": self.weak_passed, "weak_rate": self.weak_rate,
                "field_fooled": self.field_fooled, "fooled_rate": self.fooled_rate,
                "gap": self.gap, "mean_iou": self.mean_iou,
                "contract_satisfied": self.contract_satisfied,
                "contract_rate": self.contract_rate,
                "contract_and_oracle": self.contract_and_oracle,
                "combined_rate": self.combined_rate,
                "contract_residual": self.contract_residual,
                "failed": {k: list(v) for k, v in self.failed.items()}}


def ranking(standings: Sequence[Standing]) -> List[Standing]:
    """Rank on the strict contract-AND-oracle rate, then fewest residual/fooled.

    Deterministic. The primary key is :attr:`Standing.rank_rate` -- the
    contract-satisfied-AND-oracle-solved rate where the submission carries the MGC
    lens, and the plain oracle rate where it does not (so an unscored submission
    ranks exactly as it did before the contract lens was added). Ties then go to
    the submission with the smaller many-to-one residual (contract passes, oracle
    fails), then the one the field overrated least (lower ``field_fooled``), then
    alphabetically by name.
    """
    return sorted(
        standings,
        key=lambda s: (-s.rank_rate,
                       s.contract_residual if s.contract_residual is not None else 0,
                       s.field_fooled, s.name))


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
                "near_miss_proof": near_miss_proof(),
                "contract_residual_proof": contract_residual_proof()}

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


def contract_residual_proof() -> List[dict]:
    """The fixed, submission-independent contract-vs-oracle residual.

    Grades the discriminative near-miss twins against each case's Measured
    Geometric Contract (via :mod:`harnesscad.eval.hardcorpus.contract_grader`,
    imported lazily) beside the measured oracle. A ``residual_gap`` row is a WRONG
    twin the CONTRACT passes -- same volume, bbox and genus as the correct part --
    that only the oracle's point probes catch: the many-to-one residual the PDD
    synthesis (``audit/pdd_synthesis.md``) names, made visible. This is why a high
    contract rate beside a low oracle rate is possible at all, and it is the exact
    thing the board's ``contract_and_oracle`` ranking bar defends against.

    Grading touches the exact kernel; if the contract grader, the discriminative
    corpus, or that backend is absent this returns an empty list rather than
    failing to import -- the board stays runnable with no kernel and no model.
    """
    try:
        from harnesscad.eval.hardcorpus import contract_grader as _cg
    except Exception:                                          # noqa: BLE001
        return []
    try:
        report = _cg.grade_discriminative()
    except Exception:                                          # noqa: BLE001
        return []
    grades = getattr(report, "grades", ()) or ()
    return [g.to_dict() for g in grades]


def _opt_int(value: Any) -> Optional[int]:
    """Coerce an optional count from a report dict; None stays None."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _fmt_rate(value: float) -> str:
    return "%6.3f" % value


def _fmt_opt_rate(value: Optional[float]) -> str:
    return "   -  " if value is None else "%6.3f" % value


def _fmt_opt_int(value: Optional[int]) -> str:
    return "  - " if value is None else "%4d" % value


def render(standings: Sequence[Standing]) -> str:
    """The full scoreboard: the three-lens ranking, then the two standing proofs."""
    lines: List[str] = []
    lines.append("HARD-CORPUS LEADERBOARD -- field grader vs oracle vs measured contract")
    lines.append("=" * 78)
    lines.append("Every submission is scored on the weak metric the field uses (valid +")
    lines.append("IoU + Chamfer), the measured oracle (point membership), and -- when it")
    lines.append("carries the lens -- the per-part Measured Geometric Contract (MGC).")
    lines.append("'ctrct' is the MGC-satisfied rate; 'both' is MGC-AND-oracle (the bar the")
    lines.append("board ranks on); 'resid' is the count the MGC passes and the oracle")
    lines.append("FAILS -- the many-to-one residual. A '-' means the lens was not scored.")
    lines.append("")
    lines.append("%-4s %-18s %5s %7s %7s %7s %7s %5s"
                 % ("#", "submission", "n", "weak", "oracle", "ctrct", "both", "resid"))
    lines.append("-" * 78)
    ranked = ranking(standings)
    if not ranked:
        lines.append("     (no submission yet -- the frontier run is pending)")
    for i, s in enumerate(ranked):
        lines.append("%-4d %-18s %5d %7s %7s %7s %7s %5s"
                     % (i + 1, s.name[:18], s.n, _fmt_rate(s.weak_rate),
                        _fmt_rate(s.oracle_rate), _fmt_opt_rate(s.contract_rate),
                        _fmt_opt_rate(s.combined_rate), _fmt_opt_int(s.contract_residual)))
    lines.append("-" * 78)
    lines.append("Ranked on the contract-AND-oracle 'both' rate where present, else the")
    lines.append("oracle; ties to the smaller residual, then the submission the field")
    lines.append("overrated least. A submission with a high 'ctrct' but a low 'both' hit")
    lines.append("the envelope and missed the geometry -- the residual made a column.")
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

    lines.append("")
    lines.append("CONTRACT RESIDUAL -- the MGC passes, the oracle fails (no submission needed)")
    lines.append("-" * 78)
    cres = contract_residual_proof()
    if not cres:
        lines.append("     (the contract grader/kernel is absent; residual unavailable here)")
    else:
        lines.append("%-16s %-10s %-10s %-8s %-8s"
                     % ("case", "label", "contract", "oracle", "residual"))
        for row in cres:
            lines.append(
                "%-16s %-10s %-10s %-8s %-8s"
                % (str(row.get("case_id", ""))[:16], str(row.get("label", ""))[:10],
                   "SATISFIED" if row.get("satisfied") else "FAILED",
                   "n/a" if row.get("oracle_solved") is None
                   else ("solved" if row.get("oracle_solved") else "UNSOLVED"),
                   "GAP" if row.get("residual_gap") else "-"))
        lines.append("")
        lines.append("A 'residual' GAP is a WRONG twin the per-part Measured Geometric")
        lines.append("Contract passes (same volume + bbox + genus as the correct part) and")
        lines.append("only the oracle's point probes fail -- the many-to-one residual the")
        lines.append("'both' ranking column exists to defend against.")
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
