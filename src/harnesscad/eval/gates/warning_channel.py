"""THE WARNING-CHANNEL GATE. A verifier that warns on the majority of provably-
correct parts is not a verifier.

WHAT THE PRECISION FLOOR DOES NOT CATCH
---------------------------------------
``precision_floor`` scores the ERROR channel: a verifier that raises an ERROR on a
known-good part is a false positive, and a model-facing one must never do it. But a
diagnostic does not have to be an ERROR to do harm. A WARNING is still written into
the report, still read by a human triaging the build, and -- for any verifier whose
warning code is promoted to a model-facing tier -- still handed to the model as an
instruction. ``assets/pressure/report.md`` is the record of what an unproven
advisory channel costs: "advisory means all of the harm, none of the containment".

A rule that WARNS on almost everything it sees has the same defect as one that
ERRORS on everything (``fleet_audit.fires_on_everything``): its output is
uncorrelated with whether the part is actually wrong, so a reader learns nothing
from it and eventually learns to ignore it -- at which point the one time it is
right is lost in the noise it made every other time. The book's over-refusal metric
(H1 s1.17.5) does not distinguish severities; a benign part that gets flagged is a
false alarm whether the flag was red or yellow.

THE POLICY, CODIFIED
--------------------
Over a corpus of PROVABLY-CORRECT parts -- the ``fleet_audit`` KNOWN_GOOD set (each
builds watertight on the exact B-rep kernels; every warning on one is a false
alarm), optionally extended with red-team CERTIFIED attacks (``eval/redteam`` proves
each is a real, sound, buildable solid) -- this gate measures, per verifier, the
fraction of good parts it WARNS on. It FAILS the build when that fraction exceeds
:data:`MAJORITY` (one half): a verifier warning on more than half of parts that are
all fine is, by the definition this module exists to enforce, not a verifier.

WHAT THIS GATE DOES NOT DO
--------------------------
It does not edit, tier or tune any verifier -- another discipline owns
``eval/verifiers``. It is the instrument that would CATCH a bad one, built entirely
from the red-team / corpus infrastructure, so that the judgement "this warner is
noise" is a measured verdict and not an opinion. A verifier that warns on a
minority of good parts passes here and is reviewed by a human; only a majority
warner is failed outright.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from harnesscad.eval.selftest.fleet_audit import KNOWN_GOOD, Case
from harnesscad.eval.selftest.probe import plan_opdag, resolve
from harnesscad.eval.verifiers import registry as fleet_registry
from harnesscad.eval.verifiers.verify import Severity

__all__ = [
    "MAJORITY",
    "WarnScore",
    "WarningReport",
    "Violation",
    "GateReport",
    "provably_correct_corpus",
    "measure",
    "check",
    "format_text",
    "main",
]

#: The fraction of provably-correct parts above which a warner is declared noise.
#: One half: "the MAJORITY of provably-correct parts". A warner at or below this is
#: passed and left to human review; above it, the build fails.
MAJORITY = 0.5


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------

@dataclass
class WarnScore:
    """How one verifier behaved on the provably-correct corpus."""

    name: str
    tier: str = "lint"
    warned_on: List[str] = field(default_factory=list)   # good parts it WARNED on
    codes: Dict[str, int] = field(default_factory=dict)  # warning code -> count
    errored_on: List[str] = field(default_factory=list)  # good parts it ERRORED on
    crashed: int = 0

    def warn_rate(self, total: int) -> float:
        return len(self.warned_on) / float(total) if total else 0.0

    def to_dict(self, total: int) -> dict:
        return {"name": self.name, "tier": self.tier,
                "warned": len(self.warned_on), "warned_on": self.warned_on,
                "warn_rate": self.warn_rate(total),
                "codes": dict(sorted(self.codes.items())),
                "errored_on": self.errored_on, "crashed": self.crashed}


@dataclass
class WarningReport:
    backend: str = "frep"
    total_good: int = 0
    scores: List[WarnScore] = field(default_factory=list)
    skipped: str = ""
    corpus: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"oracle": "warning_channel", "backend": self.backend,
                "skipped": self.skipped, "total_good": self.total_good,
                "corpus": self.corpus, "majority_threshold": MAJORITY,
                "verifiers": [s.to_dict(self.total_good) for s in self.scores]}


def provably_correct_corpus(with_redteam: bool = False,
                            backend: str = "frep") -> Tuple[List[Case], List[str]]:
    """The parts every warning fired on is a false alarm.

    The KNOWN_GOOD fleet corpus is the base -- each part builds watertight on the
    exact B-rep kernels, so it is good by construction. With ``with_redteam`` the
    red-team CERTIFIED attacks are appended: ``eval/redteam`` only promotes an
    attack to CERTIFIED once arithmetic, an engine build at that volume, AND the
    output gate all agree it is a real, sound solid -- exactly the proof this gate
    needs to call a warning on it a false alarm.
    """
    notes: List[str] = ["fleet_audit.KNOWN_GOOD (%d parts)" % len(KNOWN_GOOD)]
    cases: List[Case] = list(KNOWN_GOOD)
    if with_redteam:
        from harnesscad.eval.redteam.attacks import generate as rt_generate
        from harnesscad.eval.redteam import oracle as rt_oracle
        certified = 0
        for atk in rt_generate():
            cert = rt_oracle.certify(atk, backend=backend)
            if cert.certified:
                cases.append(Case(name="redteam:" + atk.name,
                                  ops=tuple(atk.ops), good=True,
                                  why="red-team certified: " + cert.reason))
                certified += 1
        notes.append("redteam certified (%d attacks)" % certified)
    return cases, notes


def _diags_by_verifier(state: Any, fleet: Sequence[Any]
                       ) -> Dict[str, Tuple[List[str], List[str], bool]]:
    """{verifier: (warning codes, error codes, crashed)} on one part.

    The fleet is shown the WHOLE PLAN, all tiers, exactly as ``fleet_audit`` does,
    so a warning is counted on the same evidence the loop would give the rule.
    """
    out: Dict[str, Tuple[List[str], List[str], bool]] = {}
    for v in fleet:
        name = getattr(v, "name", type(v).__name__)
        diags = fleet_registry.run_all(state, tiers=fleet_registry.TIERS,
                                       only=[name], verifiers=[v])
        warns = [d.code for d in diags if d.severity is Severity.WARNING
                 and d.code != "verifier-error"]
        errs = [d.code for d in diags if d.severity is Severity.ERROR]
        crashed = any(d.code == "verifier-error" for d in diags)
        out[name] = (warns, errs, crashed)
    return out


def measure(backend: str = "frep", with_redteam: bool = False) -> WarningReport:
    """Run the whole fleet over the provably-correct corpus, counting WARNINGS."""
    report = WarningReport(backend=backend)
    be, skip = resolve(backend)
    if be is None:
        report.skipped = skip
        return report

    cases, notes = provably_correct_corpus(with_redteam=with_redteam, backend=backend)
    report.corpus = notes
    report.total_good = len(cases)

    fleet = fleet_registry.discover()
    scores: Dict[str, WarnScore] = {
        getattr(v, "name", type(v).__name__): WarnScore(
            getattr(v, "name", type(v).__name__), getattr(v, "tier", "lint"))
        for v in fleet
    }

    from harnesscad.core.loop import HarnessSession
    for case in cases:
        engine, _ = resolve(backend)
        session = HarnessSession(engine, verify_level="core")
        try:
            session.apply_ops(list(case.ops))
        except Exception:                                     # noqa: BLE001
            pass
        state = fleet_registry.model_state(engine, plan_opdag(case.ops))
        fired = _diags_by_verifier(state, fleet)
        for name, (warns, errs, crashed) in fired.items():
            sc = scores[name]
            if crashed:
                sc.crashed += 1
            if warns:
                sc.warned_on.append(case.name)
                for c in warns:
                    sc.codes[c] = sc.codes.get(c, 0) + 1
            if errs:
                sc.errored_on.append(case.name)

    report.scores = sorted(scores.values(), key=lambda s: (s.tier, s.name))
    return report


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Violation:
    verifier: str
    warn_rate: float
    warned_on: Tuple[str, ...]
    message: str

    def to_dict(self) -> dict:
        return {"verifier": self.verifier, "warn_rate": self.warn_rate,
                "warned_on": list(self.warned_on), "message": self.message}


@dataclass
class GateReport:
    violations: List[Violation] = field(default_factory=list)
    warn_rates: Dict[str, float] = field(default_factory=dict)
    total_good: int = 0
    skipped: str = ""

    @property
    def ok(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict:
        return {"ok": self.ok, "skipped": self.skipped,
                "total_good": self.total_good, "majority_threshold": MAJORITY,
                "warn_rates": self.warn_rates,
                "violations": [v.to_dict() for v in self.violations]}


def check(report: Any, majority: float = MAJORITY) -> GateReport:
    """Score a :class:`WarningReport` (or its ``to_dict()``) against the policy.

    Pure: a measurement in, a verdict out. This is the function the unit test drives
    with a synthetic report, so the policy can be tested without an engine.
    """
    data = report.to_dict() if hasattr(report, "to_dict") else dict(report)
    out = GateReport()
    out.skipped = data.get("skipped", "")
    total = int(data.get("total_good", 0))
    out.total_good = total
    for s in data.get("verifiers", []):
        name = s["name"]
        rate = float(s.get("warn_rate", 0.0))
        out.warn_rates[name] = rate
        if total and rate > majority + 1e-9:
            out.violations.append(Violation(
                name, rate, tuple(s.get("warned_on") or []),
                "verifier WARNS on %d of %d provably-correct parts (%.0f%%, over "
                "the %.0f%% majority line). A rule that flags most of the parts "
                "that are all fine is uncorrelated with correctness and trains its "
                "reader to ignore it; it is not a verifier."
                % (int(round(rate * total)), total, 100.0 * rate,
                   100.0 * majority)))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def format_text(gate: GateReport, report: Optional[WarningReport] = None) -> str:
    lines: List[str] = []
    lines.append("WARNING-CHANNEL GATE")
    lines.append("=" * 72)
    if gate.skipped:
        lines.append("skipped: " + gate.skipped)
        return "\n".join(lines)
    lines.append("%d provably-correct parts; every warning on one is a false alarm. "
                 "majority line = %.0f%%" % (gate.total_good, 100.0 * MAJORITY))
    if report is not None:
        lines.append("corpus: " + "; ".join(report.corpus))
    lines.append("")
    lines.append("%-24s %8s  %s" % ("verifier", "warn%", "warned on"))
    lines.append("-" * 72)
    warners = sorted((n for n, r in gate.warn_rates.items() if r > 0),
                     key=lambda n: -gate.warn_rates[n])
    if not warners:
        lines.append("  (no verifier warned on any provably-correct part)")
    for name in warners:
        parts = []
        if report is not None:
            for s in report.scores:
                if s.name == name:
                    parts = s.warned_on[:3]
                    break
        lines.append("%-24s %7.0f%%  %s"
                     % (name, 100.0 * gate.warn_rates[name], ", ".join(parts)))
    lines.append("")
    if gate.ok:
        lines.append("PASS: no verifier warns on the majority of provably-correct "
                     "parts.")
    else:
        lines.append("FAIL: %d verifier(s) over the majority line." % len(gate.violations))
        for v in gate.violations:
            lines.append("  [%s] %s" % (v.verifier, v.message))
    return "\n".join(lines)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", default="frep",
                        help="engine the corpus is built on")
    parser.add_argument("--with-redteam", action="store_true",
                        help="extend the provably-correct corpus with red-team "
                             "CERTIFIED attacks (slower: each is built and gated)")
    parser.add_argument("--majority", type=float, default=MAJORITY,
                        help="warn-rate above which a verifier fails (default 0.5)")
    parser.add_argument("--json", action="store_true", dest="as_json")


def run(args: argparse.Namespace) -> int:
    report = measure(backend=getattr(args, "backend", "frep"),
                     with_redteam=getattr(args, "with_redteam", False))
    gate = check(report, majority=getattr(args, "majority", MAJORITY))
    if getattr(args, "as_json", False):
        print(json.dumps({"measurement": report.to_dict(), "gate": gate.to_dict()},
                         indent=2, sort_keys=True))
    else:
        print(format_text(gate, report))
    if gate.skipped:
        # No engine here: the gate could not run. Do not pass silently; say so and
        # let the caller decide. CI runs this on frep, which is always available.
        return 0
    return 0 if gate.ok else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="warning_channel",
        description="Fail the build if any verifier warns on the majority of "
                    "provably-correct parts.")
    add_arguments(parser)
    return run(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
