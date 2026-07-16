"""THE LIVENESS FLOOR — no NEW op field may be silently ignored by a backend.

`eval/selftest/field_liveness.py` asks one question per (op, field, backend):
build two op streams differing ONLY in that field, and see whether the model
state changes. A DEAD cell means the backend read its own schema and threw the
value away. An op field the kernel ignores is a parameter the model cannot
control, the brief cannot request, and the grader cannot see — the same shape of
bug as the shell that grew the part while every verifier stayed silent.

`frep` had 15 dead fields when this gate was written. Failing the build on all 15
would only have meant the gate got switched off, so this gate is a RATCHET, not a
cliff:

  * the census below is COMMITTED (`liveness_baseline.json`);
  * a dead field that is NOT in the census fails the build;
  * a censused field that comes back to life is reported, and the census must be
    tightened in the same diff (the gate tells you the exact edit);
  * the census may only ever shrink.

The number in the census is a debt, and it is now a visible one. THE CENSUS IS
NOW EMPTY: all 15 were fixed, the ratchet caught the revivals and forced the
census down to zero, and the floor is now a real floor -- any dead field on frep
fails the build outright, with nothing to hide behind.

That makes the OTHER two failure modes the ones to understand, because an empty
census is not the same as a clean bill of health:

  * UNMAPPED is the failure that actually mattered. The gate can only see fields
    the oracle probes, and `field_liveness.unmapped()` fails the build when the op
    schema grows and the oracle does not. It earned its keep: three commits added
    twelve ops to CISP and taught the oracle none of them, and the 48 unprobed
    fields -- the newest, so the likeliest to be unwired -- were reported rather
    than passing green. Two of them (`thicken.both`, `mate`'s ports) were in fact
    dead. A gate that measures nothing must fail, not pass.
  * a REJ cell (see `_CAPABILITY_CODES` and the `INERT_FIELDS` allow-list in the
    oracle) is a backend turning away an op the schema calls legal. It is reported
    rather than absorbed, because "the backend refuses it" and "the field is dead"
    are different findings with different fixes.

    python -m harnesscad.eval.gates.liveness_floor            # the gate (CI)
    python -m harnesscad.eval.gates.liveness_floor --update   # deliberate re-census
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "BASELINE_PATH",
    "LivenessGateReport",
    "check",
    "measure",
    "write_baseline",
    "format_text",
    "main",
]

BASELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "liveness_baseline.json")


def _key(cell: Any) -> str:
    d = cell.to_dict() if hasattr(cell, "to_dict") else dict(cell)
    return "%s:%s.%s" % (d["backend"], d["op"], d["field"])


@dataclass
class LivenessGateReport:
    new_dead: List[str] = field(default_factory=list)
    revived: List[str] = field(default_factory=list)
    known_dead: List[str] = field(default_factory=list)
    unmapped: List[str] = field(default_factory=list)
    backends: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        # A revived field is GOOD news, and it still fails the gate: the census
        # is a committed number and it must be tightened in the same diff, or the
        # ratchet does not ratchet.
        return not self.new_dead and not self.revived and not self.unmapped

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "backends": self.backends,
            "new_dead": self.new_dead,
            "revived": self.revived,
            "known_dead": self.known_dead,
            "unmapped_fields": self.unmapped,
        }


def baseline(path: str = BASELINE_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def check(report: Any, census: Optional[dict] = None) -> LivenessGateReport:
    """Score a `field_liveness.FieldLivenessReport` against the committed census."""
    base = census if census is not None else baseline()
    known = set(base.get("dead", []))

    dead = {_key(c) for c in report.dead}
    unmapped = ["%s.%s" % (t, f) for t, f in getattr(report, "unmapped", [])]

    out = LivenessGateReport(backends=list(report.backends))
    out.known_dead = sorted(dead & known)
    out.new_dead = sorted(dead - known)
    # Only claim a revival for backends we actually measured this run.
    measured = set(report.backends)
    out.revived = sorted(k for k in known - dead if k.split(":", 1)[0] in measured)
    out.unmapped = unmapped
    return out


def measure(backends: Sequence[str] = ("frep",)):
    from harnesscad.eval.selftest import field_liveness

    return field_liveness.run(backends=list(backends))


def write_baseline(report: Any, path: str = BASELINE_PATH) -> dict:
    doc = {
        "_": "COMMITTED DEAD-FIELD CENSUS. Enforced by "
             "harnesscad.eval.gates.liveness_floor. Each entry is a "
             "backend:op.field the kernel currently IGNORES -- a parameter the "
             "model can set and the geometry will not honour. This list is a debt. "
             "It may shrink; a new entry fails the build.",
        "backends": list(report.backends),
        "dead": sorted(_key(c) for c in report.dead),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return doc


def format_text(report: LivenessGateReport) -> str:
    lines = ["LIVENESS FLOOR (dead-field ratchet)", "=" * 72, ""]
    lines.append("backends measured : %s" % ", ".join(report.backends))
    lines.append("known dead (debt) : %d" % len(report.known_dead))
    for k in report.known_dead:
        lines.append("    %s" % k)
    if report.new_dead:
        lines.append("")
        lines.append("NEW DEAD FIELDS -- the backend ignores a field it documents:")
        for k in report.new_dead:
            lines.append("    %s" % k)
    if report.revived:
        lines.append("")
        lines.append("REVIVED (good news; tighten the census in this same diff):")
        for k in report.revived:
            lines.append("    %s" % k)
    if report.unmapped:
        lines.append("")
        lines.append("UNMAPPED (the op schema grew and the oracle did not):")
        for k in report.unmapped:
            lines.append("    %s" % k)
    lines.append("")
    lines.append("PASS" if report.ok else "FAIL")
    return "\n".join(lines)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", action="append", dest="backends",
                        help="engines to measure (repeatable; default: frep)")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--update", action="store_true",
                        help="re-census. Deliberate act: the diff is the review.")


def run(args: argparse.Namespace) -> int:
    backends = tuple(getattr(args, "backends", None) or ("frep",))
    rep = measure(backends)
    if getattr(args, "update", False):
        doc = write_baseline(rep)
        print("wrote %s (%d dead field(s))" % (BASELINE_PATH, len(doc["dead"])))
        return 0
    gate = check(rep)
    if getattr(args, "as_json", False):
        print(json.dumps(gate.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_text(gate))
    return 0 if gate.ok else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="liveness_floor",
        description="Fail the build when a backend starts ignoring an op field.")
    add_arguments(parser)
    return run(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
