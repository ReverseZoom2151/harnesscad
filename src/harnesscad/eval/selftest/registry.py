"""``harnesscad selftest`` — the CLI surface of the four self-evaluation oracles.

    harnesscad selftest --all
    harnesscad selftest --differential [--backend cadquery --backend frep]
    harnesscad selftest --golden --json
    harnesscad selftest --fleet
    harnesscad selftest --properties --count 200 --seed 20260714

With no flag it runs ``--all``. Exit code is 0 even when an oracle finds
something: FINDING A BUG IS THIS COMMAND WORKING. Pass ``--strict`` to make a
finding fail the process (that is the CI switch, and the day the harness is
correct it can be turned on).
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional

from harnesscad.eval.selftest import differential, fleet_audit, golden, properties
from harnesscad.eval.selftest.probe import GEOMETRIC_BACKENDS, available

__all__ = ["add_arguments", "run"]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--differential", action="store_true",
                        help="run one op stream on every available engine and "
                             "report where they disagree (no ground truth needed)")
    parser.add_argument("--golden", action="store_true",
                        help="check every engine against parts whose volume/bbox/"
                             "genus are known in closed form")
    parser.add_argument("--fleet", action="store_true",
                        help="precision/recall/F1 PER VERIFIER over a known-good "
                             "and a known-bad corpus")
    parser.add_argument("--properties", action="store_true",
                        help="metamorphic laws (a shell must not grow the part, "
                             "scaling by k scales volume by k^3, ...) over a "
                             "seeded random corpus")
    parser.add_argument("--all", action="store_true",
                        help="run all four oracles (the default)")
    parser.add_argument("--backend", action="append", dest="backends",
                        metavar="NAME",
                        help="restrict to these engines (repeatable); default is "
                             "every one installed on this machine")
    parser.add_argument("--fleet-backend", default="frep",
                        help="the engine the fleet audit builds its corpora on "
                             "(default: frep, which always works)")
    parser.add_argument("--count", type=int, default=200,
                        help="property streams to generate (default: 200)")
    parser.add_argument("--seed", type=int, default=20260714,
                        help="property corpus seed (default: 20260714)")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="emit the whole report as JSON")
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero when an oracle finds something "
                             "(default: 0 -- a finding is this command WORKING)")


def run(args: argparse.Namespace) -> int:
    wanted = {
        "differential": bool(getattr(args, "differential", False)),
        "golden": bool(getattr(args, "golden", False)),
        "fleet": bool(getattr(args, "fleet", False)),
        "properties": bool(getattr(args, "properties", False)),
    }
    if getattr(args, "all", False) or not any(wanted.values()):
        wanted = {k: True for k in wanted}

    backends = list(getattr(args, "backends", None) or GEOMETRIC_BACKENDS)
    as_json = bool(getattr(args, "as_json", False))
    out: Dict[str, Any] = {}
    findings = 0
    text: List[str] = []

    if wanted["differential"]:
        rep = differential.run(backends=backends)
        out["differential"] = rep.to_dict()
        findings += len(rep.disagreements)
        text.append(differential.format_text(rep))

    if wanted["golden"]:
        rep_g = golden.run(backends=backends)
        out["golden"] = rep_g.to_dict()
        findings += len(rep_g.deviations)
        text.append(golden.format_text(rep_g))

    if wanted["fleet"]:
        rep_f = fleet_audit.run(backend=getattr(args, "fleet_backend", "frep"))
        out["fleet"] = rep_f.to_dict()
        findings += rep_f.fleet_fp + rep_f.fleet_fn
        text.append(fleet_audit.format_text(rep_f))

    if wanted["properties"]:
        # ONE engine by default. The laws are engine-independent, so a second engine
        # is more evidence -- but 200 streams x 6 measurements x every installed
        # kernel is hours of meshing, and a report nobody will wait for is a report
        # nobody reads. Name engines explicitly with --backend to widen it.
        if getattr(args, "backends", None):
            live = available(backends)
        else:
            live = [b for b in ("frep",) if available([b])] or ["frep"]
        rep_p = properties.run(backends=live, count=int(getattr(args, "count", 200)),
                               seed=int(getattr(args, "seed", 20260714)))
        out["properties"] = rep_p.to_dict()
        findings += len(rep_p.violations)
        text.append(properties.format_text(rep_p))

    out["findings"] = findings
    if as_json:
        print(json.dumps(out, indent=2, sort_keys=True, default=str))
    else:
        print("\n\n".join(text))
        print("")
        print("=" * 76)
        print("%d finding(s). A finding is this command WORKING: the oracles point "
              "INWARD,\nat the harness itself. Nothing here scores a model."
              % findings)
    if findings and bool(getattr(args, "strict", False)):
        return 1
    return 0
