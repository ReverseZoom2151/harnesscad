"""THE ORPHAN-PROVENANCE GATE. Every build step must move geometry; every measured
feature must have a step that made it.

WHAT IT ENFORCES
----------------
traceSDD (Panda, arXiv 2606.30689) turns hallucination detection into a set
difference over per-line REQ citations, and finds that every injected hallucination
still passed all functional tests -- traceability catches what testing cannot.
:mod:`harnesscad.core.cisp.provenance` carries that idea into CAD: replay a CISP op
stream, measure after each op, and attribute the measured geometry delta to the op
that caused it. Two defects fall out as set differences:

  * an ORPHAN OP changed nothing measurable -- a cited-but-nonexistent REQ, a
    hallucinated build step the measured-gate never notices (the part is still
    watertight and still the right volume; one op simply did no work);
  * an ORPHAN FEATURE is measured geometry no op claims -- the reverse difference,
    a face/edge/quantity in the artifact that the program cannot account for.

This gate is the standing wrapper. It FAILS the build when a CISP program contains
orphan ops or produces orphan features beyond a documented threshold.

THE POLICY, CODIFIED
--------------------
The default threshold for orphan ops is ZERO: any orphan op fails. The field-
liveness census (``eval/gates/liveness_floor``) already proves that individual op
FIELDS reach the kernel; by the time a program is assembled, a whole op that moves
nothing is not a tolerable debt, it is a dead step, so the floor is not "a minority"
(that is the warning-channel gate's shape) but "none". The orphan-FEATURE threshold
is likewise zero by default: geometry the op stream cannot cite is unexplained
geometry. Both are overridable for callers that knowingly accept a bounded number.

WHY A GATE AS WELL AS THE STRUCTURE
-----------------------------------
:func:`harnesscad.core.cisp.provenance.orphan_ops` is the measurement;
:func:`check` is the verdict. Keeping the verdict here -- pure, a provenance in and
a report out -- lets the policy be exercised with an injected synthetic program and
a fake ``measure_state`` (``--selfcheck``), with no model and no kernel, exactly as
``warning_channel`` is driven by a synthetic report. The gate degrades to a clean
PASS when no real session is available, so it never blocks a machine that has no
engine installed.

WHAT THIS GATE DOES NOT DO
--------------------------
It does not prove an op did the RIGHT work -- a non-orphan op provably moved some
measured quantity, not the intended one. Attribution is a floor, the same honest
caveat field-liveness carries. It catches the dead step and the unclaimed feature,
which measurement alone does not.

    python -m harnesscad.eval.gates.orphan_provenance --selfcheck   # no engine
    python -m harnesscad.eval.gates.orphan_provenance               # real session
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from harnesscad.core.cisp.ops import Op
from harnesscad.core.cisp.provenance import (Provenance, build_provenance,
                                             orphan_ops, unattributed_features)

__all__ = [
    "DEFAULT_MAX_ORPHAN_OPS",
    "DEFAULT_MAX_ORPHAN_FEATURES",
    "Violation",
    "GateReport",
    "check",
    "measure",
    "session_measure_state",
    "measurement_of",
    "format_text",
    "selfcheck",
    "main",
]

#: A live-field census should already have rejected dead fields, so a whole op that
#: moves nothing is a defect, not a debt: zero orphan ops are tolerated by default.
DEFAULT_MAX_ORPHAN_OPS: int = 0

#: Geometry no op cites is unexplained; none is tolerated by default.
DEFAULT_MAX_ORPHAN_FEATURES: int = 0

#: Measurement keys pulled from a backend, in the namespace the deltas diff over.
#: Kept flat and comparable (numbers, bools, a bbox tuple) so two states that
#: differ geometrically differ in at least one key.
_METRIC_KEYS: Tuple[str, ...] = (
    "volume", "surface_area", "area", "n_faces", "n_edges", "n_vertices",
    "n_triangles", "n_solids")
_VALIDITY_KEYS: Tuple[str, ...] = (
    "genus", "watertight", "manifold", "solid_present")


# ---------------------------------------------------------------------------
# Turning a live backend into a measurement (the real path).
# ---------------------------------------------------------------------------

def measurement_of(backend: Any) -> Dict[str, Any]:
    """A flat, comparable measurement of a backend's built state.

    Pulls a fixed set of scalar metrics + validity flags and a bbox tuple. Missing
    keys are simply absent (their appearance/disappearance is itself a delta). Any
    query that raises is skipped -- a hostile backend must not crash the gate.
    """
    out: Dict[str, Any] = {}
    for what, keys in (("metrics", _METRIC_KEYS), ("validity", _VALIDITY_KEYS)):
        try:
            res = backend.query(what)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(res, dict):
            continue
        for k in keys:
            if k in res and isinstance(res[k], (int, float, bool)):
                out[k] = res[k]
        bbox = res.get("bbox")
        if isinstance(bbox, (list, tuple)) and all(
                isinstance(v, (int, float)) for v in bbox):
            out["bbox"] = tuple(float(v) for v in bbox)
    return out


def session_measure_state(backend_name: str):
    """A ``measure_state`` closure over a real engine, or ``None`` when unavailable.

    Returns ``(measure_state, skip_reason)``. The closure rebuilds a fresh session
    for each prefix and measures it, which is the safe backend-agnostic contract
    (a session need not be incrementally re-measurable), mirroring how
    ``field_liveness`` rebuilds full streams.
    """
    from harnesscad.eval.selftest.probe import resolve

    engine, skip = resolve(backend_name)
    if engine is None:
        return None, skip

    from harnesscad.core.loop import HarnessSession

    def measure_state(prefix: Sequence[Op]) -> Dict[str, Any]:
        session = HarnessSession(engine, verify_level="core")
        try:
            session.apply_ops(list(prefix))
        except Exception:  # noqa: BLE001 - a refusal/crash still leaves a state
            pass
        return measurement_of(engine)

    return measure_state, ""


# ---------------------------------------------------------------------------
# The gate (pure): a provenance in, a verdict out.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Violation:
    kind: str          # "orphan-op" | "orphan-feature"
    ref: str           # op_id (for an op) or feature key (for a feature)
    detail: str

    def to_dict(self) -> dict:
        return {"kind": self.kind, "ref": self.ref, "detail": self.detail}


@dataclass
class GateReport:
    violations: List[Violation] = field(default_factory=list)
    n_ops: int = 0
    orphan_ops: List[str] = field(default_factory=list)       # op_ids
    orphan_features: List[str] = field(default_factory=list)   # feature keys
    max_orphan_ops: int = DEFAULT_MAX_ORPHAN_OPS
    max_orphan_features: int = DEFAULT_MAX_ORPHAN_FEATURES
    skipped: str = ""

    @property
    def ok(self) -> bool:
        # No session to measure => nothing to fail on. Degrade to PASS, say so.
        if self.skipped:
            return True
        return not self.violations

    def to_dict(self) -> dict:
        return {
            "gate": "orphan_provenance",
            "ok": self.ok,
            "skipped": self.skipped,
            "n_ops": self.n_ops,
            "max_orphan_ops": self.max_orphan_ops,
            "max_orphan_features": self.max_orphan_features,
            "orphan_ops": self.orphan_ops,
            "orphan_features": self.orphan_features,
            "violations": [v.to_dict() for v in self.violations],
        }


def check(prov: Provenance,
          measured_features: Sequence[str] = (),
          max_orphan_ops: int = DEFAULT_MAX_ORPHAN_OPS,
          max_orphan_features: int = DEFAULT_MAX_ORPHAN_FEATURES) -> GateReport:
    """Score a :class:`Provenance` against the orphan policy.

    Pure: this is the function ``--selfcheck`` and any unit test drive with a
    synthetic provenance, so the policy needs neither a model nor a kernel.
    ``measured_features`` is the artifact's feature set for the reverse
    (orphan-feature) difference; leave it empty to check orphan ops only.
    """
    out = GateReport(max_orphan_ops=max_orphan_ops,
                     max_orphan_features=max_orphan_features)
    out.skipped = prov.skipped
    out.n_ops = len(prov.deltas)
    if prov.skipped:
        return out

    orphans = orphan_ops(prov)
    out.orphan_ops = [d.op_id for d in orphans]
    features = unattributed_features(prov, measured_features)
    out.orphan_features = list(features)

    if len(orphans) > max_orphan_ops:
        for d in orphans:
            out.violations.append(Violation(
                "orphan-op", d.op_id,
                "op '%s' (index %d) changed nothing measurable -- a build step "
                "the geometry does not reflect (cited-but-nonexistent REQ)."
                % (d.op_tag, d.index)))
    if len(features) > max_orphan_features:
        for key in features:
            out.violations.append(Violation(
                "orphan-feature", key,
                "measured feature '%s' is claimed by no op -- geometry the op "
                "stream cannot account for (orphan REQ, reversed)." % key))
    return out


# ---------------------------------------------------------------------------
# The real path: build a provenance from a live session (degrades to skipped).
# ---------------------------------------------------------------------------

def measure(ops: Optional[Sequence[Op]] = None,
            backend: str = "frep") -> Provenance:
    """Build a provenance for ``ops`` on a real engine, or a skipped one.

    With no ops (nothing to check) or no available engine the provenance comes
    back ``skipped`` and the gate degrades to PASS. This gate has no standing
    corpus of its own -- the CISP program under test is supplied by the caller,
    exactly as the model would emit it.
    """
    if not ops:
        prov = Provenance()
        prov.skipped = "no CISP program supplied (nothing to attribute)"
        return prov
    measure_state, skip = session_measure_state(backend)
    if measure_state is None:
        prov = Provenance()
        prov.skipped = skip or ("backend %r unavailable" % backend)
        return prov
    return build_provenance(list(ops), measure_state)


# ---------------------------------------------------------------------------
# --selfcheck: the whole pipeline on a synthetic fixture, NO model, NO kernel.
# ---------------------------------------------------------------------------

def _fixture_ops() -> List[Op]:
    """A synthetic 3-op program whose middle op is a planted orphan."""
    from harnesscad.core.cisp.ops import Extrude, Fillet, Primitive

    # op 0 builds a box; op 1 is a no-op (a fillet whose fake measurement does not
    # move); op 2 changes the geometry again.
    return [Primitive("box", 10.0, 10.0, 10.0),
            Fillet((), 0.0),
            Extrude("sk1", 5.0)]


def _fake_measure_state(prefix: Sequence[Op]) -> Dict[str, Any]:
    """A kernel-free measurement table keyed by prefix length.

    Prefix length 2 repeats the length-1 measurement verbatim, so the second op
    (index 1) attributes to nothing -- a planted orphan op. ``genus`` is constant
    across every measurement (including the baseline), so no op ever moves it: a
    planted orphan feature, present in the artifact yet cited by nothing.
    """
    table: Dict[int, Dict[str, Any]] = {
        0: {"genus": 0},
        1: {"genus": 0, "volume": 1000.0, "n_faces": 6, "bbox": (10.0, 10.0, 10.0)},
        2: {"genus": 0, "volume": 1000.0, "n_faces": 6, "bbox": (10.0, 10.0, 10.0)},
        3: {"genus": 0, "volume": 1240.0, "n_faces": 6, "bbox": (10.0, 10.0, 15.0)},
    }
    return table[len(prefix)]


def selfcheck() -> Tuple[bool, str]:
    """Drive the pure pipeline on the synthetic fixture; assert it flags the plants.

    Returns ``(passed, message)``. ``passed`` is True when the gate behaves as
    designed on BOTH a dirty program (one planted orphan op + one planted orphan
    feature -> FAIL) and a clean one (thresholds raised to absorb them -> PASS).
    This exercises the whole machinery with no engine.
    """
    ops = _fixture_ops()
    prov = build_provenance(ops, _fake_measure_state)

    orphans = orphan_ops(prov)
    dirty = check(prov, measured_features=["volume", "n_faces", "genus", "bbox"])
    # genus is the planted orphan feature; the length-1 baseline established
    # volume/n_faces/bbox as attributed.
    clean = check(prov, measured_features=["volume", "n_faces", "genus", "bbox"],
                  max_orphan_ops=len(orphans),
                  max_orphan_features=len(dirty.orphan_features))

    expectations = [
        (len(orphans) == 1, "expected exactly one orphan op, got %d" % len(orphans)),
        (orphans and orphans[0].index == 1,
         "expected the middle op (index 1) to be the orphan"),
        (dirty.orphan_features == ["genus"],
         "expected 'genus' as the sole orphan feature, got %r"
         % dirty.orphan_features),
        (not dirty.ok, "dirty program should FAIL the gate"),
        (clean.ok, "raising the thresholds should PASS the gate"),
    ]
    for passed, msg in expectations:
        if not passed:
            return False, msg
    return True, ("orphan op at index 1 and orphan feature 'genus' both detected; "
                  "thresholds absorb them cleanly")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def format_text(gate: GateReport) -> str:
    lines: List[str] = []
    lines.append("ORPHAN-PROVENANCE GATE")
    lines.append("=" * 72)
    if gate.skipped:
        lines.append("skipped: " + gate.skipped)
        lines.append("")
        lines.append("PASS (degraded): no session to attribute; nothing to fail on.")
        return "\n".join(lines)
    lines.append("ops attributed        : %d" % gate.n_ops)
    lines.append("orphan-op threshold   : %d" % gate.max_orphan_ops)
    lines.append("orphan-feature thresh : %d" % gate.max_orphan_features)
    lines.append("orphan ops            : %d" % len(gate.orphan_ops))
    lines.append("orphan features       : %d" % len(gate.orphan_features))
    lines.append("")
    if gate.ok:
        lines.append("PASS: every op moved geometry and every measured feature is "
                     "cited by an op.")
    else:
        lines.append("FAIL: %d violation(s)." % len(gate.violations))
        for v in gate.violations:
            lines.append("  [%s] %s -- %s" % (v.kind, v.ref, v.detail))
    return "\n".join(lines)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", default="frep",
                        help="engine the program is replayed on (default frep)")
    parser.add_argument("--max-orphan-ops", type=int,
                        default=DEFAULT_MAX_ORPHAN_OPS,
                        help="orphan ops tolerated before failing (default 0)")
    parser.add_argument("--max-orphan-features", type=int,
                        default=DEFAULT_MAX_ORPHAN_FEATURES,
                        help="orphan features tolerated before failing (default 0)")
    parser.add_argument("--selfcheck", action="store_true",
                        help="run the kernel-free synthetic fixture and exit")
    parser.add_argument("--json", action="store_true", dest="as_json")


def run(args: argparse.Namespace) -> int:
    if getattr(args, "selfcheck", False):
        passed, message = selfcheck()
        if getattr(args, "as_json", False):
            print(json.dumps({"selfcheck": passed, "message": message},
                             indent=2, sort_keys=True))
        else:
            print("ORPHAN-PROVENANCE GATE -- selfcheck")
            print("=" * 72)
            print(("PASS: " if passed else "FAIL: ") + message)
        return 0 if passed else 1

    # No standing corpus: with no program supplied this degrades to skipped/PASS.
    prov = measure(ops=None, backend=getattr(args, "backend", "frep"))
    gate = check(prov,
                 max_orphan_ops=getattr(args, "max_orphan_ops",
                                        DEFAULT_MAX_ORPHAN_OPS),
                 max_orphan_features=getattr(args, "max_orphan_features",
                                             DEFAULT_MAX_ORPHAN_FEATURES))
    if getattr(args, "as_json", False):
        print(json.dumps({"provenance": prov.to_dict(), "gate": gate.to_dict()},
                         indent=2, sort_keys=True))
    else:
        print(format_text(gate))
    return 0 if gate.ok else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="orphan_provenance",
        description="Fail the build when a CISP program contains orphan ops "
                    "(steps that moved nothing) or orphan features (measured "
                    "geometry no op claims).")
    add_arguments(parser)
    return run(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
