"""THE JUDGE GATE — an uncalibrated evaluator may not instruct.

The washer bug, one layer up
----------------------------
`assets/pressure/report.md` is the story of an unaudited evaluator inside a
loop: 23 verifiers, each tested for "does it FIRE on bad input", none tested for
"does it STAY SILENT on good input". A false verdict is an instruction, and a
capable model obeys it.

`eval/verifiers/vlm_judge.py` ships a VLM judge. It has **never been calibrated
against a human or an oracle**. That is the same class of bug, waiting. And the
calibration machinery was already in the repository, orphaned:

  * `eval/bench/judges/judge_calibration.py` — threshold sweep computing
    precision / recall / F1 / acceptance-rate at every cut point, and
    `select_threshold` picking the best. Its only importer was its own test.
  * `eval/bench/judges/judge_human_agreement.py` — Pearson / Spearman /
    Kendall tau-b with a seeded bootstrap CI, at Item / Cell / System grain.
    Its only importer was its own test.

This module wires both to a policy and makes CI enforce it.

The policy
----------
A judge is one of two things and never a third:

  ADVISORY   — it may emit INFO. It may not emit WARNING or ERROR, it may not be
               written into a model's retry prompt, and nothing may gate on it.
               This is where an uncalibrated judge lives.

  CALIBRATED — it has a committed record in `judge_calibration.json` giving its
               operating threshold and its measured agreement with the ORACLE
               (`selftest.golden` / `selftest.differential`), meeting the
               reliability bar below. Only then may it raise severity.

The reliability bar (Hitchhiker's Guide, H2 s14.7, lines 10380-10381):

    "A judge is considered reliable if it achieves kappa > 0.6 and agreement
     rate > 80% with human annotators on a representative sample."

We substitute the oracle for the human, because the oracle is exact and the human
is not available; the bar is unchanged. `MIN_AGREEMENT = 0.80`, `MIN_KENDALL =
0.60`, and — because a judge inside a correction loop destroys work when it is
wrong in the positive direction — `MIN_PRECISION = 0.90`.

Run the gate (CI runs this):

    python -m harnesscad.eval.gates.judge_gate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from harnesscad.eval.bench.judges.judge_calibration import (
    calibrate_threshold,
    select_threshold,
)
from harnesscad.eval.bench.judges.judge_human_agreement import agreement_report

__all__ = [
    "CALIBRATION_PATH",
    "MIN_PRECISION",
    "MIN_AGREEMENT",
    "MIN_KENDALL",
    "JUDGES",
    "UncalibratedJudge",
    "Calibration",
    "calibrate",
    "load_calibrations",
    "calibration_for",
    "is_calibrated",
    "require_calibrated",
    "check",
    "format_text",
    "main",
]

CALIBRATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "judge_calibration.json")

#: A judge whose positive verdicts are wrong 1 time in 10 is not allowed to
#: raise severity: in a correction loop a false positive destroys correct work.
MIN_PRECISION = 0.90
#: The book's bar, with the oracle standing in for the human annotator.
MIN_AGREEMENT = 0.80
MIN_KENDALL = 0.60

#: Every judge the product ships. `model_facing` declares the INTENT; the gate
#: decides whether the intent is licensed by a calibration record.
JUDGES: Dict[str, dict] = {
    "vlm-judge": {
        "module": "harnesscad.eval.verifiers.vlm_judge",
        "model_facing": False,
        "note": ("Subjective design score from a vision model. NOT calibrated "
                 "against golden/differential. Advisory only: INFO, never above, "
                 "never into a retry prompt. Calibrate it or stop shipping it."),
    },
}


class UncalibratedJudge(RuntimeError):
    """Raised when a judge is asked to do something its calibration does not license."""


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Calibration:
    """One judge's committed operating point and its measured agreement."""

    judge: str
    threshold: float
    precision: float
    recall: float
    f1: float
    agreement: float          # fraction of cases where judge and oracle agree
    kendall: float            # tau-b against the oracle score
    n: int
    oracle: str = "selftest.golden"

    @property
    def reliable(self) -> bool:
        return (self.precision >= MIN_PRECISION
                and self.agreement >= MIN_AGREEMENT
                and self.kendall >= MIN_KENDALL)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["reliable"] = self.reliable
        return d


def calibrate(judge: str, records: Sequence[Dict[str, Any]],
              thresholds: Optional[Sequence[float]] = None,
              oracle: str = "selftest.golden") -> Calibration:
    """Calibrate a judge against ORACLE labels.

    ``records`` are ``{"distance": float, "accepted": bool}`` — the judge's
    (lower-is-better) distance / dissimilarity for a part, and whether the
    ORACLE accepted that part. This is exactly the input
    `judge_calibration.calibrate_threshold` already takes; it was never given
    one. The threshold sweep picks the operating point, and
    `judge_human_agreement.agreement_report` measures rank agreement with the
    oracle at the same time.
    """
    if not records:
        raise ValueError("cannot calibrate a judge on zero records")
    grid = list(thresholds) if thresholds is not None else sorted(
        {float(r["distance"]) for r in records})
    rows = calibrate_threshold(records, grid)
    best = select_threshold(rows)
    t = float(best["threshold"])

    # Agreement: the oracle's 0/1 label vs the judge's binary call at `t`, plus
    # rank agreement between the oracle label and the judge's raw score
    # (negated distance, so both are higher-is-better).
    calls = [(1.0 if bool(r["accepted"]) else 0.0,
              1.0 if float(r["distance"]) <= t else 0.0) for r in records]
    agree = sum(1 for o, j in calls if o == j) / len(calls)
    ranked = [(1.0 if bool(r["accepted"]) else 0.0, -float(r["distance"]))
              for r in records]
    rep = agreement_report(ranked, [(o, j, "all") for o, j in ranked])
    kendall = rep["Item"]["kendall"][0]
    if kendall != kendall:  # NaN -> undefined -> not reliable
        kendall = 0.0

    return Calibration(
        judge=judge, threshold=t,
        precision=float(best["precision"]), recall=float(best["recall"]),
        f1=float(best["f1"]), agreement=agree, kendall=float(kendall),
        n=len(records), oracle=oracle,
    )


def load_calibrations(path: str = CALIBRATION_PATH) -> Dict[str, Calibration]:
    """Committed calibration records, keyed by judge. Missing file -> {}."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    out: Dict[str, Calibration] = {}
    for name, d in (doc.get("judges") or {}).items():
        out[name] = Calibration(
            judge=name,
            threshold=float(d["threshold"]),
            precision=float(d["precision"]), recall=float(d["recall"]),
            f1=float(d["f1"]), agreement=float(d["agreement"]),
            kendall=float(d["kendall"]), n=int(d["n"]),
            oracle=d.get("oracle", "selftest.golden"),
        )
    return out


def calibration_for(judge: str, path: str = CALIBRATION_PATH) -> Optional[Calibration]:
    return load_calibrations(path).get(judge)


def is_calibrated(judge: str, path: str = CALIBRATION_PATH) -> bool:
    """True only when a committed record exists AND meets the reliability bar."""
    cal = calibration_for(judge, path)
    return bool(cal and cal.reliable)


def require_calibrated(judge: str, what: str = "raise severity",
                       path: str = CALIBRATION_PATH) -> Calibration:
    """Return the calibration, or refuse. This is the seam a judge calls.

    An uncalibrated judge is not silenced — it is capped at INFO. It is refused
    only when it tries to do the thing that can destroy work.
    """
    cal = calibration_for(judge, path)
    if cal is None:
        raise UncalibratedJudge(
            "judge %r has no committed calibration record and may not %s. "
            "An unaudited evaluator inside a loop is the bug that lost "
            "assets/pressure/report.md. Calibrate it against selftest.golden "
            "(harnesscad.eval.gates.judge_gate.calibrate) or leave it advisory."
            % (judge, what))
    if not cal.reliable:
        raise UncalibratedJudge(
            "judge %r is calibrated and BELOW the reliability bar "
            "(precision %.3f < %.2f or agreement %.3f < %.2f or kendall %.3f < "
            "%.2f) and may not %s."
            % (judge, cal.precision, MIN_PRECISION, cal.agreement, MIN_AGREEMENT,
               cal.kendall, MIN_KENDALL, what))
    return cal


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

@dataclass
class JudgeGateReport:
    violations: List[str] = field(default_factory=list)
    rows: List[Tuple[str, bool, bool]] = field(default_factory=list)  # name, facing, calibrated

    @property
    def ok(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "violations": self.violations,
            "judges": [{"judge": n, "model_facing": f, "calibrated": c}
                       for n, f, c in self.rows],
        }


def check(judges: Optional[Dict[str, dict]] = None,
          path: str = CALIBRATION_PATH) -> JudgeGateReport:
    """Every shipped judge is either calibrated or provably advisory-only."""
    reg = judges if judges is not None else JUDGES
    out = JudgeGateReport()
    for name, decl in sorted(reg.items()):
        facing = bool(decl.get("model_facing"))
        cal = is_calibrated(name, path)
        out.rows.append((name, facing, cal))
        if facing and not cal:
            out.violations.append(
                "judge %r is declared MODEL-FACING and has no reliable committed "
                "calibration. It may not enter the feedback channel." % name)
    return out


def format_text(report: JudgeGateReport) -> str:
    lines = ["JUDGE CALIBRATION GATE", "=" * 72, ""]
    lines.append("%-14s %-13s %s" % ("judge", "channel", "calibrated"))
    lines.append("-" * 72)
    for name, facing, cal in report.rows:
        lines.append("%-14s %-13s %s" % (
            name, "MODEL-FACING" if facing else "advisory",
            "yes" if cal else "NO"))
    lines.append("")
    if report.ok:
        lines.append("PASS: no uncalibrated judge is on the model-facing channel.")
        lines.append("NOTE: vlm-judge remains UNCALIBRATED and is therefore capped "
                     "at INFO by VLMJudgeCheck. It is shipped, it is not trusted, "
                     "and the code enforces the difference.")
    else:
        lines.append("FAIL:")
        for v in report.violations:
            lines.append("  " + v)
    return "\n".join(lines)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", dest="as_json")


def run(args: argparse.Namespace) -> int:
    rep = check()
    if getattr(args, "as_json", False):
        print(json.dumps(rep.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_text(rep))
    return 0 if rep.ok else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="judge_gate",
        description="Refuse an uncalibrated judge the model-facing channel.")
    add_arguments(parser)
    return run(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
