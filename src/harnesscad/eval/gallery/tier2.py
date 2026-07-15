"""Tier-2 evidence: what can be claimed about a part with no closed form.

The claim, stated once, precisely
---------------------------------
Tier 1 (:mod:`harnesscad.eval.selftest.golden`) proves a part CORRECT: an
analytic volume, an engine's answer, ``delta 0.00e+00``. It is the strongest
statement in the repo and it is only available on shapes you can integrate by
hand -- which is why every verified part here is a box, a plate or a washer.

Tier 2 is what is available on everything else, and it is WEAKER. It runs one op
stream on six independently-built engines and asks three ground-truth-free
questions:

1. **Differential agreement** (:mod:`~harnesscad.eval.selftest.differential`).
   Do the engines return the same geometry? Six kernels -- an exact OCCT B-rep,
   a second OCCT B-rep, a CGAL mesher, a mesh kernel, a sampled distance field --
   written by different people on different mathematics. If they disagree, at
   least one is WRONG, and we found that out without knowing the answer.
2. **Metamorphic laws** (:mod:`~harnesscad.eval.selftest.properties`).
   Chiefly ``scale_is_cubic``: multiply every length in the plan by k, and the
   volume must rise by k^3. This relates TWO RUNS OF THE SAME ENGINE, so it holds
   even for an engine whose absolute numbers are all wrong -- it is the only
   check here that survives the case where every engine shares a bug.
3. **The output gate** (:mod:`harnesscad.io.gate`). Re-measure the solid that was
   actually WRITTEN TO DISK -- not the one the backend claims it built.

WHAT THIS PROVES:  nothing disagreed.
WHAT THIS DOES NOT PROVE:  that the part is correct.

Those are not the same sentence and this module will not let them be confused.
Agreement is not truth:

* Engines can SHARE a bug. Five wrong answers that match are still five wrong
  answers, and a consensus of them looks exactly like a consensus of right ones.
* The compared signature -- volume, bbox, genus, watertightness -- is
  MANY-TO-ONE. A part with every hole bored in the wrong place matches every
  number in the table.
* An op whose meaning is UNDERDETERMINED can have six engines confidently
  computing six answers to six different questions; consensus among them would
  measure nothing but their shared reading of an ambiguous spec.

So a green Tier-2 run is not a certificate. It is the absence of a
counterexample, from an oracle that is genuinely capable of producing one. A RED
Tier-2 run, by contrast, is unambiguous and is the reason to run it at all: a
disagreement between two independent kernels is a bug in one of them, full stop.
Tier 2 is a bug DETECTOR, not a correctness PROOF -- and that is what makes it
usable on a gyroid, where Tier 1 has nothing to say at all.

Process isolation
-----------------
Each stream is measured in its OWN SUBPROCESS. This is not defensive
programming, it is a measured requirement: the OCCT-backed engines can abort at
interpreter teardown (the repo's test suite already runs per-module for this
reason), and a long multi-engine run can exhaust memory such that a later
``import OCP`` fails and the cadquery backend reports itself UNAVAILABLE --
turning a healthy engine into a phantom capability gap. In one process, one
engine's corpse corrupts the verdict on every engine after it. In a subprocess
per stream, a crash is a data point instead of a contagion, and it is reported
as :attr:`StreamEvidence.crashed`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from harnesscad.eval.gallery import complex_parts
from harnesscad.eval.selftest import differential, properties
from harnesscad.eval.selftest.probe import GEOMETRIC_BACKENDS, available

__all__ = [
    "PROVES",
    "DOES_NOT_PROVE",
    "StreamEvidence",
    "Tier2Report",
    "measure_stream",
    "run",
    "format_text",
    "main",
]


#: Stated in the artifact, not only in this docstring, so a reader of the JSON
#: cannot mistake a green run for a proof of correctness.
PROVES: Tuple[str, ...] = (
    "six independently-built engines were run on the same op stream and their "
    "measured geometry (volume, bbox, genus, watertightness) did not disagree "
    "beyond each engine's physically-derived tolerance",
    "the metamorphic laws held -- in particular scale_is_cubic, which relates "
    "two runs of the SAME engine and therefore holds even for an engine whose "
    "absolute numbers are all wrong",
    "the solid that was WRITTEN TO DISK was re-measured by the output gate and "
    "was a closed, 2-manifold, positive-volume solid",
    "an engine that REFUSED the plan said so, rather than building a different "
    "part quietly",
)

#: The engines that measure IN-PROCESS (no subprocess fork per measurement); see
#: :mod:`harnesscad.eval.selftest.probe`. Only these are cheap enough to run the
#: per-op property laws on a deep stream.
IN_PROCESS_BACKENDS: Tuple[str, ...] = ("frep", "stub")

#: Above this op count, the per-op property laws run on the in-process engine
#: only (and say so). The differential and the gate always run on every engine.
PROPERTY_DEPTH_LIMIT: int = 12

DOES_NOT_PROVE: Tuple[str, ...] = (
    "that any part here is CORRECT -- there is no ground truth in this tier, by "
    "construction; that is the whole reason it can be pointed at a gyroid",
    "that the engines are right -- they can SHARE a bug, and a consensus of five "
    "wrong answers is indistinguishable here from a consensus of five right ones",
    "that the part matches any brief or intent -- the compared signature is "
    "MANY-TO-ONE, and a part with every hole bored in the wrong place matches "
    "every number in this report",
    "that an op whose meaning is UNDERDETERMINED was implemented correctly -- "
    "six engines can agree because they share a reading of an ambiguous spec",
    "anything at all about a feature finer than the coarsest engine's "
    "tessellation",
)


@dataclass
class StreamEvidence:
    """The three ground-truth-free oracles' verdict on one op stream."""

    name: str
    depth: int = 0
    ops: List[str] = field(default_factory=list)
    # -- differential
    engines: List[str] = field(default_factory=list)
    consensus: List[str] = field(default_factory=list)
    clusters: List[List[str]] = field(default_factory=list)
    volume_spread: Optional[float] = None
    disagreements: List[dict] = field(default_factory=list)
    refused: Dict[str, str] = field(default_factory=dict)
    crashed: Dict[str, str] = field(default_factory=dict)
    measurements: List[dict] = field(default_factory=list)
    # -- properties
    property_checks: int = 0
    property_engines: List[str] = field(default_factory=list)
    property_note: str = ""
    violations: List[dict] = field(default_factory=list)
    # -- gate
    gate: Dict[str, dict] = field(default_factory=dict)
    error: str = ""

    @property
    def structural(self) -> List[dict]:
        return [d for d in self.disagreements if d.get("structural")]

    @property
    def agreed(self) -> bool:
        """No disagreement, no crash, no broken law. NOT 'the part is correct'."""
        return (not self.disagreements and not self.crashed
                and not self.violations and not self.error)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "depth": self.depth, "ops": self.ops,
            "agreed": self.agreed,
            "engines": self.engines,
            "consensus": self.consensus, "clusters": self.clusters,
            "volume_spread": self.volume_spread,
            "disagreements": self.disagreements,
            "structural_disagreements": len(self.structural),
            "refused": self.refused, "crashed": self.crashed,
            "measurements": self.measurements,
            "property_checks": self.property_checks,
            "property_engines": self.property_engines,
            "property_note": self.property_note,
            "violations": self.violations,
            "gate": self.gate,
            "error": self.error,
        }


@dataclass
class Tier2Report:
    streams: List[StreamEvidence] = field(default_factory=list)
    engines: List[str] = field(default_factory=list)
    coverage: dict = field(default_factory=dict)

    @property
    def disagreements(self) -> List[Tuple[str, dict]]:
        return [(s.name, d) for s in self.streams for d in s.disagreements]

    @property
    def crashes(self) -> List[Tuple[str, str, str]]:
        return [(s.name, b, e) for s in self.streams
                for b, e in sorted(s.crashed.items())]

    @property
    def violations(self) -> List[Tuple[str, dict]]:
        return [(s.name, v) for s in self.streams for v in s.violations]

    @property
    def gate_failures(self) -> List[Tuple[str, str, dict]]:
        return [(s.name, b, g) for s in self.streams
                for b, g in sorted(s.gate.items()) if not g.get("ok", True)]

    @property
    def findings(self) -> int:
        return (len(self.disagreements) + len(self.crashes)
                + len(self.violations) + len(self.gate_failures))

    def to_dict(self) -> dict:
        return {
            "oracle": "tier2-differential",
            "tier": 2,
            "proves": list(PROVES),
            "does_not_prove": list(DOES_NOT_PROVE),
            "engines": self.engines,
            "coverage": self.coverage,
            "findings": self.findings,
            "counts": {
                "streams": len(self.streams),
                "agreed": sum(1 for s in self.streams if s.agreed),
                "disagreements": len(self.disagreements),
                "structural": sum(len(s.structural) for s in self.streams),
                "crashes": len(self.crashes),
                "property_violations": len(self.violations),
                "gate_failures": len(self.gate_failures),
            },
            "streams": [s.to_dict() for s in self.streams],
        }


# ---------------------------------------------------------------------------
# measurement (this is what runs INSIDE the worker subprocess)
# ---------------------------------------------------------------------------
def measure_stream(name: str, backends: Optional[Sequence[str]] = None,
                   gate_parts: bool = True) -> StreamEvidence:
    """Run all three ground-truth-free oracles on one Tier-2 stream.

    Called in-process. :func:`run` invokes it through a subprocess per stream;
    the unit tests call it directly on the in-process engines only.
    """
    part = complex_parts.get(name)
    ops = part.ops
    ev = StreamEvidence(name=name, depth=part.depth, ops=sorted(part.op_set))

    live = list(available(tuple(backends) if backends else GEOMETRIC_BACKENDS))
    ev.engines = live
    if len(live) < 2:
        ev.error = "fewer than two engines available; a differential needs two"
        return ev

    # 1. DIFFERENTIAL -- six engines, one plan, no ground truth.
    case = differential.compare(name, ops, backends=live)
    ev.consensus = list(case.consensus)
    ev.clusters = [list(c) for c in case.clusters]
    ev.volume_spread = case.volume_spread()
    ev.disagreements = [d.to_dict() for d in case.disagreements]
    ev.refused = dict(case.refused)
    ev.crashed = dict(case.crashed)
    ev.measurements = [
        {"backend": o.backend, "volume": o.volume, "bbox": list(o.bbox) if o.bbox else None,
         "genus": o.genus, "watertight": o.watertight, "ok": o.ok,
         "codes": o.codes, "error": o.error}
        for o in case.observations if o.available
    ]

    # 2. METAMORPHIC LAWS -- these need no second engine and no ground truth.
    #
    # COST, and an honest restriction. The property oracle measures the part
    # AFTER EVERY OP (``properties.observe_steps``) to check laws like
    # extrude_gives_height. On an IN-PROCESS engine (frep) that is cheap; on the
    # external kernels (freecad/blender/openscad) each of those measurements
    # FORKS A PROCESS, so a 22-op stream is ~22 forks PER ENGINE and the deepest
    # part in the corpus timed out a 30-minute worker. So above a depth
    # threshold the per-op laws run on the in-process engine only, and the fact
    # is RECORDED (``property_engines`` / ``property_note``) rather than hidden.
    # This is keyed on op COUNT, applies to any deep stream, and changes only
    # which engines get the most expensive oracle -- never what is measured, and
    # never anything shape-specific. The metamorphic law that matters most,
    # scale_is_cubic, is a whole-part relation and still runs on every engine
    # below the threshold; the differential and the gate always run on all.
    prop_engines = list(live)
    if part.depth > PROPERTY_DEPTH_LIMIT:
        prop_engines = [b for b in live if b in IN_PROCESS_BACKENDS]
        skipped = [b for b in live if b not in prop_engines]
        if skipped:
            ev.property_note = (
                "per-op property laws restricted to in-process engine(s) %s for "
                "this %d-op stream; %s each fork a process per op and would time "
                "the worker out. Differential and gate still ran on all %d engines."
                % (",".join(prop_engines), part.depth, ",".join(skipped), len(live)))
    ev.property_engines = prop_engines
    for b in prop_engines:
        try:
            vios, checks = properties.check_stream(name, ops, b)
        except Exception as exc:  # noqa: BLE001 - a crash in an oracle is a finding
            ev.violations.append({"property": "(oracle crashed)", "backend": b,
                                  "stream": name, "detail": "%s: %s"
                                  % (type(exc).__name__, exc), "ops": []})
            continue
        ev.property_checks += checks
        ev.violations.extend(v.to_dict() for v in vios)

    # 3. THE GATE -- re-measure the solid that was actually written.
    if gate_parts:
        for b in live:
            if b in ev.crashed or b in ev.refused:
                continue
            ev.gate[b] = _gate_stream(part, b)
    return ev


def _gate_stream(part: complex_parts.ComplexPart, backend: str) -> dict:
    """Export the built solid and re-measure the FILE through :mod:`io.gate`.

    The gate is the only oracle here that does not take the backend's word for
    anything: it reads back the mesh that was serialised and measures that. A
    backend that reports a beautiful solid and exports a broken one is invisible
    to every other check in this module.
    """
    from harnesscad.io import gate
    from harnesscad.io.formats import registry as formats
    from harnesscad.io.surfaces.server import CISPServer

    try:
        server = CISPServer(backend=backend)
        if server.backend_name != backend:
            return {"ok": False, "skipped": True,
                    "why": "backend unavailable (%s)" % server.backend_note}
        result = server.applyOps([dict(o) for o in part.raw])
        if not result.get("ok"):
            return {"ok": False, "skipped": True, "why": "backend refused the plan"}
        verts, faces = formats.to_mesh(server.backend).indexed()
        mesh = ([tuple(float(c) for c in v) for v in verts],
                [tuple(int(i) for i in f) for f in faces])
        report = gate.check(mesh, source=server.session)
        m = report.measurement if isinstance(report.measurement, dict) else {}
        return {
            "ok": bool(report.ok),
            "failures": [str(f) for f in (report.failures or [])],
            "volume": m.get("volume"),
            "watertight": m.get("watertight"),
            "genus": m.get("genus"),
        }
    except Exception as exc:  # noqa: BLE001 - a gate crash IS a gate failure
        return {"ok": False, "why": "%s: %s" % (type(exc).__name__, exc)}


# ---------------------------------------------------------------------------
# the run (parent process: one subprocess per stream)
# ---------------------------------------------------------------------------
def run(only: Optional[str] = None, backends: Optional[Sequence[str]] = None,
        isolate: bool = True, gate_parts: bool = True, log=None) -> Tier2Report:
    """Measure the whole Tier-2 corpus. One subprocess per stream by default."""
    say = log or (lambda _m: None)
    selected = [complex_parts.get(only)] if only else list(complex_parts.CORPUS)
    report = Tier2Report(coverage=complex_parts.coverage_report())

    for part in selected:
        if isolate:
            ev = _measure_isolated(part.name, backends, gate_parts)
        else:
            ev = measure_stream(part.name, backends, gate_parts)
        report.streams.append(ev)
        if not report.engines:
            report.engines = ev.engines
        flag = "ok  " if ev.agreed else "FIND"
        say("%s %-22s %2d ops  engines=%-2d  consensus=%-2d  disagree=%d "
            "crash=%d viol=%d%s"
            % (flag, ev.name, ev.depth, len(ev.engines), len(ev.consensus),
               len(ev.disagreements), len(ev.crashed), len(ev.violations),
               ("  refused: " + ",".join(sorted(ev.refused))) if ev.refused else ""))
    return report


def _measure_isolated(name: str, backends: Optional[Sequence[str]],
                      gate_parts: bool) -> StreamEvidence:
    """Run one stream in a fresh interpreter and read its JSON back.

    A worker that dies (OCCT can abort at teardown) is recorded as a crash of the
    STREAM, with its exit status -- never silently dropped, and never allowed to
    take the rest of the corpus with it.
    """
    cmd = [sys.executable, "-m", "harnesscad.eval.gallery.tier2",
           "--worker", name]
    if backends:
        cmd += ["--backend", ",".join(backends)]
    if not gate_parts:
        cmd.append("--no-gate")
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=1800, env=env)
    except subprocess.TimeoutExpired:
        return StreamEvidence(name=name, error="worker timed out after 1800s")
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return _from_dict(json.loads(line))
            except ValueError:
                continue
    return StreamEvidence(
        name=name,
        error="worker produced no result (exit %s): %s"
              % (proc.returncode, (proc.stderr or "").strip()[-300:]))


def _from_dict(d: dict) -> StreamEvidence:
    ev = StreamEvidence(name=d.get("name", "?"))
    for k, v in d.items():
        if hasattr(ev, k) and k not in ("agreed", "structural_disagreements"):
            setattr(ev, k, v)
    return ev


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def format_text(report: Tier2Report) -> str:
    lines: List[str] = []
    lines.append("TIER 2 -- complex shapes, no closed form, no ground truth")
    lines.append("=" * 74)
    lines.append("")
    lines.append("PROVES:")
    for p in PROVES:
        lines.append("  + " + _wrap(p, 70, "    "))
    lines.append("DOES NOT PROVE:")
    for p in DOES_NOT_PROVE:
        lines.append("  - " + _wrap(p, 70, "    "))
    lines.append("")
    cov = report.coverage
    lines.append("corpus: %d parts, depth %d max / %d median (golden's max is 8)"
                 % (cov.get("parts", 0), cov.get("max_depth", 0),
                    cov.get("median_depth", 0)))
    lines.append("        %d op pairs NEVER previously co-emitted (baseline had %d)"
                 % (cov.get("novel_pair_count", 0), cov.get("baseline_pair_count", 0)))
    lines.append("        ops used for the first time: %s"
                 % ", ".join(cov.get("novel_ops", [])) or "none")
    lines.append("engines: %s" % ", ".join(report.engines))
    lines.append("")

    lines.append("%-22s %5s %-30s %9s %s"
                 % ("stream", "ops", "consensus", "spread", "findings"))
    lines.append("-" * 74)
    for s in report.streams:
        spread = ("%.2f%%" % (100.0 * s.volume_spread)) if s.volume_spread else "-"
        bits = []
        if s.disagreements:
            bits.append("%d disagree" % len(s.disagreements))
        if s.crashed:
            bits.append("%d crash" % len(s.crashed))
        if s.violations:
            bits.append("%d law" % len(s.violations))
        if s.refused:
            bits.append("%d refused" % len(s.refused))
        lines.append("%-22s %5d %-30s %9s %s"
                     % (s.name, s.depth, ",".join(s.consensus) or "-", spread,
                        ", ".join(bits) if bits else "agreed"))
    lines.append("")

    if report.disagreements:
        lines.append("DISAGREEMENTS -- at least one engine is WRONG here")
        lines.append("-" * 74)
        for name, d in report.disagreements:
            lines.append("  %-22s %-8s %-9s consensus %s -> got %s (%s)%s"
                         % (name, d.get("metric", "?").upper(), d.get("backend"),
                            d.get("consensus_value"), d.get("value"),
                            d.get("delta", ""),
                            "  STRUCTURAL" if d.get("structural") else ""))
        lines.append("")
    if report.crashes:
        lines.append("CRASHES")
        lines.append("-" * 74)
        for name, b, err in report.crashes:
            lines.append("  %-22s %-9s %s" % (name, b, err[:60]))
        lines.append("")
    if report.violations:
        lines.append("BROKEN LAWS (metamorphic / property oracle)")
        lines.append("-" * 74)
        for name, v in report.violations:
            lines.append("  %-22s %-26s %-9s %s"
                         % (name, v.get("property"), v.get("backend"),
                            str(v.get("detail"))[:40]))
        lines.append("")
    if report.gate_failures:
        lines.append("GATE FAILURES -- the solid WRITTEN TO DISK did not pass")
        lines.append("-" * 74)
        for name, b, g in report.gate_failures:
            why = g.get("why") or "; ".join(g.get("failures", []))
            lines.append("  %-22s %-9s %s" % (name, b, str(why)[:50]))
        lines.append("")

    refusals = [(s.name, b, w) for s in report.streams
                for b, w in sorted(s.refused.items())]
    if refusals:
        lines.append("REFUSALS -- capability gaps, NOT bugs (the engine declined "
                     "rather than\n            building the part wrong)")
        lines.append("-" * 74)
        for name, b, w in refusals:
            lines.append("  %-22s %-9s %s" % (name, b, w[:44]))
        lines.append("")

    n = len(report.streams)
    agreed = sum(1 for s in report.streams if s.agreed)
    lines.append("%d/%d streams: no engine disagreed." % (agreed, n))
    lines.append("READ THAT PRECISELY. It means no counterexample was found by an "
                 "oracle\nthat is capable of producing one. It does NOT mean these "
                 "parts are correct:\nthere is no ground truth in this tier, the "
                 "engines can share a bug, and the\nsignature compared is "
                 "many-to-one. Tier 2 is a bug DETECTOR, not a proof.")
    return "\n".join(lines)


def _wrap(text: str, width: int, indent: str) -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w) if cur else w
    lines.append(cur)
    return ("\n" + indent).join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="tier2", description=__doc__)
    parser.add_argument("--worker", metavar="STREAM",
                        help="internal: measure ONE stream and print its JSON")
    parser.add_argument("--only", metavar="STREAM", help="measure just this stream")
    parser.add_argument("--backend", help="comma-separated engine subset")
    parser.add_argument("--no-gate", action="store_true", dest="no_gate")
    parser.add_argument("--no-isolate", action="store_true", dest="no_isolate",
                        help="run in-process (faster; one crash can poison the run)")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", metavar="PATH", help="write the JSON report here")
    args = parser.parse_args(argv)

    backends = args.backend.split(",") if args.backend else None

    if args.worker:
        ev = measure_stream(args.worker, backends, not args.no_gate)
        sys.stdout.write(json.dumps(ev.to_dict(), sort_keys=True) + "\n")
        sys.stdout.flush()
        # OCCT can abort at interpreter teardown; the JSON is already out.
        os._exit(0)

    report = run(only=args.only, backends=backends,
                 isolate=not args.no_isolate, gate_parts=not args.no_gate,
                 log=lambda m: print(m, flush=True))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(report.to_dict(), fh, indent=2, sort_keys=True)
            fh.write("\n")
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print()
        print(format_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
