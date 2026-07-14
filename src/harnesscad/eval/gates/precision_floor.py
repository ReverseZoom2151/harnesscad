"""THE PRECISION FLOOR — the gate that would have prevented the -8.3.

What happened
-------------
Twenty-three verifiers were written, each unit-tested for "does it FIRE on bad
input". Not one was tested for "does it STAY SILENT on good input". Recall was
optimised; precision was never measured. `assets/pressure/report.md`: the typed
loop lost to blind resampling by 8.3 points and lost hardest on the strongest
model, because a false diagnostic is an INSTRUCTION and instructions get obeyed.

`eval/selftest/fleet_audit.py` then built the missing instrument: precision,
recall and F1 PER VERIFIER over a known-good and a known-bad corpus. It is
correct and complete. **And nothing enforced it.** A rule with precision 0.4
could be merged tomorrow.

This module is the enforcement.

The policy
----------
1.  **Every verifier must be declared in the baseline.** A verifier that appears
    in the fleet and not in `precision_baseline.json` fails the gate. "We never
    thought about it" is the failure mode that cost eight briefs; a new rule now
    cannot enter the fleet without a human writing down what its precision is
    and why that is acceptable.

2.  **A MODEL-FACING verifier must have precision 1.0.** Model-facing means the
    verifier can emit a PROVEN or MEASURED code (`verifiers.soundness`), i.e.
    something the planner is allowed to speak to the model. A false positive on
    that channel destroys correct work. The floor is 1.0 and there is no
    negotiating it. A model-facing verifier that never fires on the corpus has
    an undefined precision and passes — it cannot have lied yet — but a single
    false positive fails the build.

3.  **A HEURISTIC verifier is gated on NON-REGRESSION.** Its diagnostics never
    reach the model (the planner's soundness gate drops them), so a heuristic
    false positive costs a human's attention, not a correct part. It is still
    committed at its measured value, so it cannot silently get worse, and a new
    heuristic rule with precision 0.4 must be argued for in a diff.

4.  **The model-facing false-positive rate must be 0.** This is the book's
    over-refusal metric (H1 s1.17.5) transposed to geometry, restricted to the
    channel that can do damage.

The floors live in `precision_baseline.json` next to this file. Regenerate them
deliberately, never automatically:

    python -m harnesscad.eval.gates.precision_floor --update

Run the gate (this is what CI runs):

    python -m harnesscad.eval.gates.precision_floor
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from harnesscad.eval.verifiers.soundness import MODEL_FACING_TIERS, SOUNDNESS

__all__ = [
    "BASELINE_PATH",
    "MODEL_FACING_FLOOR",
    "is_model_facing",
    "model_facing_codes",
    "Violation",
    "GateReport",
    "check",
    "measure",
    "baseline",
    "write_baseline",
    "format_text",
    "main",
]

BASELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "precision_baseline.json")

#: A verifier that may instruct the model may not ever be wrong. No negotiation.
MODEL_FACING_FLOOR = 1.0


# ---------------------------------------------------------------------------
# Soundness projection: WHICH verifiers can reach the model at all
# ---------------------------------------------------------------------------

def model_facing_codes(name: str) -> List[str]:
    """The diagnostic codes of verifier ``name`` that the planner may speak.

    A verifier whose default tier is PROVEN/MEASURED is model-facing for every
    code it emits, and we cannot enumerate those statically, so we return
    ``["*"]``. A verifier whose default is HEURISTIC is model-facing only for the
    codes explicitly promoted in its ``by_code`` map (e.g. kernel-preflight's two
    theorems).
    """
    s = SOUNDNESS.get(name)
    if s is None:
        return []
    if s.default in MODEL_FACING_TIERS:
        return ["*"]
    return sorted(c for c, tier in s.by_code.items() if tier in MODEL_FACING_TIERS)


def is_model_facing(name: str) -> bool:
    """True when any diagnostic of this verifier can be written into a retry prompt."""
    return bool(model_facing_codes(name))


def _fp_is_model_facing(name: str, fp_codes: Dict[str, int]) -> bool:
    """Did this verifier's FALSE POSITIVES land on the model-facing channel?

    A heuristic verifier with two promoted codes (kernel-preflight) is only
    dangerous when the codes it got WRONG are the promoted ones. If every FP came
    in under a heuristic code, the planner dropped it and no model ever saw it.
    """
    facing = model_facing_codes(name)
    if not facing:
        return False
    if facing == ["*"]:
        return bool(fp_codes)
    return any(code in facing for code in fp_codes)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Violation:
    """One reason the build must fail."""

    verifier: str
    kind: str        # "unregistered" | "below-floor" | "model-facing-fp"
    message: str

    def to_dict(self) -> dict:
        return {"verifier": self.verifier, "kind": self.kind, "message": self.message}


@dataclass
class GateReport:
    """The verdict plus everything needed to understand it."""

    violations: List[Violation] = field(default_factory=list)
    measured: Dict[str, Optional[float]] = field(default_factory=dict)
    floors: Dict[str, Optional[float]] = field(default_factory=dict)
    model_facing: List[str] = field(default_factory=list)
    model_facing_false_positives: Dict[str, List[str]] = field(default_factory=dict)
    fleet_false_positive_rate: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def model_facing_false_positive_rate(self) -> float:
        """Fraction of known-good parts falsely rejected ON THE MODEL-FACING CHANNEL."""
        bad = set()
        for parts in self.model_facing_false_positives.values():
            bad.update(parts)
        n_good = self.known_good or 1
        return len(bad) / n_good

    known_good: int = 0

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "violations": [v.to_dict() for v in self.violations],
            "measured_precision": self.measured,
            "floors": self.floors,
            "model_facing": self.model_facing,
            "model_facing_false_positives": self.model_facing_false_positives,
            "model_facing_false_positive_rate": self.model_facing_false_positive_rate,
            "fleet_false_positive_rate": self.fleet_false_positive_rate,
            "known_good": self.known_good,
        }


def baseline(path: str = BASELINE_PATH) -> dict:
    """Load the committed floors. A missing baseline is a hard error, not a default."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def check(report: Any, floors: Optional[dict] = None) -> GateReport:
    """Score a `fleet_audit.FleetReport` (or its ``to_dict()``) against the floors.

    Pure: takes a measurement, returns a verdict. No I/O beyond loading the
    baseline, no geometry. This is the function the unit test drives.
    """
    data = report.to_dict() if hasattr(report, "to_dict") else dict(report)
    base = floors if floors is not None else baseline()
    declared: Dict[str, dict] = base.get("verifiers", {})

    out = GateReport()
    out.known_good = int(data.get("known_good", 0))
    out.fleet_false_positive_rate = float(
        data.get("fleet", {}).get("false_positive_rate", 0.0))

    for score in data.get("verifiers", []):
        name = score["name"]
        precision = score.get("precision")
        fp_codes = dict(score.get("fp_codes") or {})
        facing = is_model_facing(name)
        if facing:
            out.model_facing.append(name)
        out.measured[name] = precision

        decl = declared.get(name)
        if decl is None:
            out.violations.append(Violation(
                name, "unregistered",
                "verifier is in the fleet and NOT in the committed precision "
                "baseline. Measure it, write its floor down, and say why. An "
                "unaudited rule in the feedback channel has unbounded negative "
                "expected value (assets/pressure/report.md)."))
            continue

        floor = decl.get("precision_floor")
        out.floors[name] = floor

        if facing and floor is not None and floor < MODEL_FACING_FLOOR:
            out.violations.append(Violation(
                name, "below-floor",
                "verifier is MODEL-FACING (codes %r) and its committed floor is "
                "%.3f. A rule that may instruct the model may never be wrong; "
                "the floor for that channel is %.2f."
                % (model_facing_codes(name), floor, MODEL_FACING_FLOOR)))

        if precision is not None and floor is not None and precision + 1e-9 < floor:
            out.violations.append(Violation(
                name, "below-floor",
                "precision %.3f is below the committed floor %.3f (false "
                "positives on: %s)"
                % (precision, floor, ", ".join(score.get("false_positives") or []) or "-")))

        if _fp_is_model_facing(name, fp_codes) and score.get("fp", 0):
            parts = list(score.get("false_positives") or [])
            out.model_facing_false_positives[name] = parts
            out.violations.append(Violation(
                name, "model-facing-fp",
                "%d false positive(s) on the MODEL-FACING channel (codes %r, "
                "parts %s). A false diagnostic is an instruction, and a capable "
                "model obeys it: this is the exact mechanism that cost 8 briefs."
                % (score.get("fp", 0), sorted(fp_codes), ", ".join(parts) or "-")))

    return out


def measure(backend: str = "frep"):
    """Run the live fleet audit. Imported lazily: it drives a geometry engine."""
    from harnesscad.eval.selftest import fleet_audit

    return fleet_audit.run(backend=backend)


# ---------------------------------------------------------------------------
# Baseline authoring
# ---------------------------------------------------------------------------

def write_baseline(report: Any, path: str = BASELINE_PATH,
                   preserve_reasons: bool = True) -> dict:
    """Render a fleet audit into a committed baseline document.

    Model-facing verifiers ALWAYS get floor 1.0 regardless of what they measured
    — if one of them measures below 1.0 the correct response is to fix the rule
    or demote it to HEURISTIC, never to lower the floor. Heuristic verifiers get
    their measured precision (or null when they never fired).
    """
    data = report.to_dict() if hasattr(report, "to_dict") else dict(report)
    old: Dict[str, dict] = {}
    if preserve_reasons and os.path.exists(path):
        old = baseline(path).get("verifiers", {})

    verifiers: Dict[str, dict] = {}
    for score in sorted(data.get("verifiers", []), key=lambda s: s["name"]):
        name = score["name"]
        facing = is_model_facing(name)
        if facing:
            floor: Optional[float] = MODEL_FACING_FLOOR
        else:
            floor = score.get("precision")
        entry = {
            "model_facing": facing,
            "model_facing_codes": model_facing_codes(name),
            "precision_floor": floor,
            "measured_precision": score.get("precision"),
            "measured_recall": score.get("recall"),
            "fired": score.get("tp", 0) + score.get("fp", 0),
            "reason": old.get(name, {}).get("reason", ""),
        }
        verifiers[name] = entry

    doc = {
        "_": "COMMITTED PRECISION FLOORS. Enforced by "
             "harnesscad.eval.gates.precision_floor, which CI runs on every push. "
             "A model-facing verifier's floor is 1.0 and may not be lowered: fix "
             "the rule or demote it to HEURISTIC in verifiers/soundness.py. A "
             "heuristic verifier's floor is its measured precision, so it cannot "
             "silently regress. A verifier absent from this file fails the build.",
        "corpus": {
            "backend": data.get("backend"),
            "known_good": data.get("known_good"),
            "known_bad": data.get("known_bad"),
        },
        "fleet": data.get("fleet", {}),
        "verifiers": verifiers,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return doc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def format_text(report: GateReport) -> str:
    lines: List[str] = []
    lines.append("PRECISION FLOOR GATE")
    lines.append("=" * 72)
    lines.append("")
    lines.append("%-20s %-13s %10s %10s" % ("verifier", "channel", "precision", "floor"))
    lines.append("-" * 72)
    for name in sorted(report.measured):
        p = report.measured[name]
        f = report.floors.get(name)
        lines.append("%-20s %-13s %10s %10s" % (
            name,
            "MODEL-FACING" if name in report.model_facing else "human-only",
            "-" if p is None else "%.3f" % p,
            "-" if f is None else "%.3f" % f,
        ))
    lines.append("")
    lines.append("fleet false-positive rate        : %.1f%%"
                 % (100.0 * report.fleet_false_positive_rate))
    lines.append("MODEL-FACING false-positive rate : %.1f%%   <- the one that costs parts"
                 % (100.0 * report.model_facing_false_positive_rate))
    lines.append("")
    if report.ok:
        lines.append("PASS: no verifier below its committed floor; the model-facing "
                     "channel told the truth on every known-good part.")
    else:
        lines.append("FAIL: %d violation(s)." % len(report.violations))
        for v in report.violations:
            lines.append("  [%s] %s: %s" % (v.kind, v.verifier, v.message))
    return "\n".join(lines)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", default="frep",
                        help="engine the fleet audit builds its corpora on")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--update", action="store_true",
                        help="REWRITE the committed baseline from the live audit. "
                             "Deliberate act: the diff is the review.")


def run(args: argparse.Namespace) -> int:
    rep = measure(backend=getattr(args, "backend", "frep"))
    if getattr(args, "update", False):
        doc = write_baseline(rep)
        print("wrote %s (%d verifiers)" % (BASELINE_PATH, len(doc["verifiers"])))
        return 0
    gate = check(rep)
    if getattr(args, "as_json", False):
        print(json.dumps(gate.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_text(gate))
    return 0 if gate.ok else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="precision_floor",
        description="Fail the build if any verifier is below its committed precision floor.")
    add_arguments(parser)
    return run(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
