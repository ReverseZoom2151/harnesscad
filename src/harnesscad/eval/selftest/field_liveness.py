"""Field-liveness oracle — does an op's DECLARED field actually reach the kernel?

The bug this exists to catch
----------------------------
Three independent audits of three backends found the SAME defect three times:
**an op silently discards its own declared fields.** ``_fillet`` took a tuple of
edge selectors and rounded EVERY edge. ``_shell`` took a face list and hard-coded
``">Z"``. FreeCAD's ``Hole`` ignored ``kind``, ``cbore_*`` and ``csk_*``, so a
counterbore, a countersink and a plain hole all produced the SAME CYLINDER. Every
one of them accepted a typed op, threw half of it away, returned a perfectly valid
solid, and emitted NO diagnostic. A dropped field is invisible: the part is
well-formed, watertight, manifold, and wrong.

Why the differential oracle cannot find this
---------------------------------------------
:mod:`harnesscad.eval.selftest.differential` runs one plan on six engines and
reports where they disagree. It is powerful and it is blind here, because ALL SIX
ENGINES DROP THE SAME FIELDS. They agree perfectly while all being wrong.
Cross-checking engines can only ever find bugs the engines do not share. That is
the deepest lesson of this bug class, and it is why this module is not a
differential test.

The property
------------
    For every op type, and every field on that op, changing that field MUST
    change the resulting model state.

Two op streams that differ in exactly one field, and produce an identical
measurement vector, prove the backend never read that field. There is no
tolerance to hide behind and no ground truth to argue about: the backend is
ignoring its own schema. That is a bug, unconditionally.

Derived from the schema, not from a list
-----------------------------------------
The fields are enumerated with ``dataclasses.fields`` over
:data:`harnesscad.core.cisp.ops._REGISTRY`. A hand-written list would rot the
moment somebody adds a field -- and the field they add is exactly the one nobody
wired up. :func:`unmapped` reports any (op, field) that has neither a variant nor
an allow-list entry, and the test suite FAILS on a non-empty result. Adding a
field to the schema and not to this file is itself a test failure.

What a cell means
------------------
    LIVE   the two streams produced different model state -- the field is read.
    DEAD   identical model state -- THE BACKEND IGNORED THE FIELD. A bug.
    ERR    one stream was refused / raised where the other was not. The field was
           READ and rejected: a typed error is an acceptable "difference".
           Silence is not.
    N/A    both streams were refused (the backend cannot do this op at all), or
           the field is on the inert allow-list.
    SKIP   the backend is not installed here.

What this proves, and what it does not
---------------------------------------
It proves a field REACHES the kernel. It does not prove the field is used
CORRECTLY: a backend that filleted the edges named in ``edges`` and a backend that
filleted a different four edges both score LIVE. Liveness is a floor, not a
ceiling -- but it is a floor that six engines were all below.
"""

from __future__ import annotations

import dataclasses
import json
import math
import threading
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from harnesscad.core.cisp.ops import (AddCircle, AddInstance, AddLine, AddPoint,
                                      AddRectangle, Boolean, Chamfer, Constrain,
                                      CircularPattern, Draft, Extrude, Fillet,
                                      Hole, LinearPattern, Loft, Mate, Mirror,
                                      NewSketch, Op, Revolve, SetParam, Shell,
                                      Sweep, _REGISTRY)
from harnesscad.core.loop import HarnessSession
from harnesscad.eval.selftest.probe import BACKENDS, BackendFactory, resolve

__all__ = [
    "LIVE", "DEAD", "ERR", "NA", "REJ", "SKIP",
    "INERT_FIELDS", "CASES", "CHEAP_BACKENDS", "DEFAULT_TIMEOUT_S",
    "Case", "Cell", "FieldLivenessReport",
    "op_fields", "unmapped", "signature", "check_field", "run", "format_text",
]

LIVE = "LIVE"
DEAD = "DEAD"
ERR = "ERR"
NA = "N/A"
REJ = "REJ"
SKIP = "SKIP"

#: Codes with which a backend is DECLARING a capability gap rather than failing.
#: An engine that answers "I do not implement loft" is honest, and a field on an
#: op it does not implement cannot be alive: that is N/A. An engine that answers
#: "bad-value" to the op as the SCHEMA DOCUMENTS IT is not declaring a gap -- it
#: is refusing a legal op, and that is a REJ, which is reported, not absorbed.
#: (This is how frep's shell showed up: ops.py specifies ``faces`` as CadQuery
#: selector strings -- ``(">Z",)`` -- and frep's vocabulary is ``("top",)``, so it
#: bad-values every documented shell. A silent N/A would have buried that.)
_CAPABILITY_CODES = frozenset({"unsupported-op", "not-supported", "no-solid"})

#: The in-process engines. Everything else forks a process (freecadcmd, blender,
#: openscad) or is a heavy kernel, so the six-backend matrix is opt-in.
CHEAP_BACKENDS: Tuple[str, ...] = ("stub", "frep")

#: The stub builds NO geometry and its ``query()`` exposes only counts and sketch
#: DOF, so almost every field is DEAD on it. That is not the bug this oracle
#: hunts -- it is what "the stub is a bookkeeping backend" MEANS. Its column is
#: reported (it is the only engine that can be checked in 0.1 s, and it is where a
#: broken FIXTURE shows up first) but it is excluded from the bug census and never
#: asserted on. Every other engine claims to build the solid, and is held to it.
NON_GEOMETRIC: Tuple[str, ...] = ("stub",)

#: Hard wall-clock ceiling for ONE backend running ONE op stream. A backend that
#: shells out must never be able to hang the suite.
DEFAULT_TIMEOUT_S: float = 90.0


# ---------------------------------------------------------------------------
# The inert allow-list.
#
# BE SUSPICIOUS OF THIS LIST. It is the one place in this file where a real bug
# can hide: every entry is a field this oracle agrees to expect DEAD. Each needs a
# reason that is a fact about GEOMETRY, not about an implementation. "The backend
# does not model it" is a bug report, not a justification, and does not belong
# here -- those stay DEAD in the matrix where they can be counted.
#
# It is currently TWO entries long, and both say the same thing: a point has no
# measure.
# ---------------------------------------------------------------------------
INERT_FIELDS: Dict[Tuple[str, str], str] = {
    ("add_point", "x"):
        "a sketch point is a zero-measure datum: no op consumes a point (no "
        "backend's profile builder or path wire reads one), so its abscissa "
        "cannot enter any solid. Inert by geometry, not by neglect.",
    ("add_point", "y"):
        "same as add_point.x -- a point contributes no area to a profile and no "
        "length to a path, so its ordinate has nothing to change.",
}


# ---------------------------------------------------------------------------
# Fixtures: for each op, a stream that exercises it, and one alternate value per
# field. Stream A = prelude + [op] + postlude. Stream B is identical except that
# ONE field of ``op`` is changed. Nothing else moves.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Case:
    """How to exercise one op type, field by field."""

    prelude: Tuple[Op, ...] = ()
    op: Optional[Op] = None
    postlude: Tuple[Op, ...] = ()
    #: field -> (alternate value, base-op overrides). The overrides let a field be
    #: exercised from a base the DEFAULT op would not reach -- cbore_depth only
    #: means anything on a hole whose kind is already "counterbore".
    variants: Dict[str, Tuple[Any, Dict[str, Any]]] = dc_field(default_factory=dict)

    def stream(self, changes: Optional[Dict[str, Any]] = None) -> List[Op]:
        op = dataclasses.replace(self.op, **(changes or {}))
        return list(self.prelude) + [op] + list(self.postlude)


def _v(alt: Any, **base: Any) -> Tuple[Any, Dict[str, Any]]:
    return (alt, base)


#: A 60 x 40 x 20 box (feature f1) -- the substrate for every solid-modifying op.
_BOX: Tuple[Op, ...] = (
    NewSketch("XY"), AddRectangle("sk1", 0.0, 0.0, 60.0, 40.0), Extrude("sk1", 20.0))

#: A CONCAVE (L-shaped) prism: 60 x 40 and 20 x 70 unioned in one sketch, extruded
#: 20. The reflex corner at (20, 40) is the whole point -- it is the only place a
#: shell's JOIN TYPE can manifest. See the "shell" Case.
_L_PRISM: Tuple[Op, ...] = (
    NewSketch("XY"), AddRectangle("sk1", 0.0, 0.0, 60.0, 40.0),
    AddRectangle("sk1", 0.0, 0.0, 20.0, 70.0), Extrude("sk1", 20.0))

#: An off-axis 20 x 20 x 10 block (f1) plus a fillet (f2) -- two features, so a
#: pattern/mirror ``feature`` reference has two distinct things to point at, and
#: the body is away from the origin so a rotation about Z actually moves it.
_TWO_FEATURES: Tuple[Op, ...] = (
    NewSketch("XY"), AddRectangle("sk1", 100.0, 0.0, 20.0, 20.0),
    Extrude("sk1", 10.0), Fillet(("|Z",), 2.0))


def _cases() -> Dict[str, Case]:
    return {
        "new_sketch": Case(
            op=NewSketch("XY"),
            postlude=(AddRectangle("sk1", 0.0, 0.0, 60.0, 20.0), Extrude("sk1", 10.0)),
            variants={"plane": _v("XZ")},
        ),
        "add_point": Case(
            prelude=(NewSketch("XY"), NewSketch("XY")),
            op=AddPoint("sk1", 5.0, 5.0),
            postlude=(AddRectangle("sk1", 0.0, 0.0, 60.0, 40.0), Extrude("sk1", 20.0)),
            variants={"sketch": _v("sk2"), "x": _v(25.0), "y": _v(25.0)},
        ),
        # A line is only ever consumed as a SWEEP PATH (no backend's profile
        # builder closes a lone line), so that is the only stream in which its
        # coordinates can be alive at all.
        "add_line": Case(
            prelude=(NewSketch("XY"), AddCircle("sk1", 0.0, 0.0, 5.0), NewSketch("XZ")),
            op=AddLine("sk2", 0.0, 0.0, 0.0, 40.0),
            postlude=(Sweep("sk1", "sk2"),),
            variants={"sketch": _v("sk1"), "x1": _v(15.0), "y1": _v(15.0),
                      "x2": _v(25.0), "y2": _v(80.0)},
        ),
        "add_circle": Case(
            prelude=(NewSketch("XY"), NewSketch("XY")),
            op=AddCircle("sk1", 0.0, 0.0, 10.0),
            postlude=(Extrude("sk1", 10.0),),
            variants={"sketch": _v("sk2"), "cx": _v(20.0), "cy": _v(20.0),
                      "r": _v(20.0)},
        ),
        "add_rectangle": Case(
            prelude=(NewSketch("XY"), NewSketch("XY")),
            op=AddRectangle("sk1", 0.0, 0.0, 60.0, 40.0),
            postlude=(Extrude("sk1", 20.0),),
            variants={"sketch": _v("sk2"), "x": _v(30.0), "y": _v(30.0),
                      "w": _v(30.0), "h": _v(20.0)},
        ),
        "constrain": Case(
            prelude=(NewSketch("XY"), AddCircle("sk1", 0.0, 0.0, 10.0),
                     NewSketch("XY"), AddCircle("sk2", 0.0, 0.0, 10.0)),
            op=Constrain("radius", "e1", None, 10.0),
            postlude=(Extrude("sk1", 10.0),),
            variants={"kind": _v("coincident"), "a": _v("e2"), "b": _v("e2"),
                      "value": _v(20.0)},
        ),
        "extrude": Case(
            prelude=(NewSketch("XY"), AddRectangle("sk1", 0.0, 0.0, 60.0, 40.0),
                     NewSketch("XY"), AddRectangle("sk2", 0.0, 0.0, 20.0, 20.0)),
            op=Extrude("sk1", 20.0),
            variants={"sketch": _v("sk2"), "distance": _v(40.0)},
        ),
        "fillet": Case(
            prelude=_BOX,
            op=Fillet(("|Z",), 3.0),
            variants={"edges": _v((">Z",)), "radius": _v(6.0)},
        ),
        # Three disjoint solids (f1 f2 f3). ``cut`` is non-commutative, so
        # swapping target/tool is a real change -- under ``union`` it would not be.
        "boolean": Case(
            prelude=(NewSketch("XY"), AddRectangle("sk1", 0.0, 0.0, 60.0, 40.0),
                     Extrude("sk1", 20.0),
                     NewSketch("XY"), AddRectangle("sk2", 10.0, 10.0, 20.0, 20.0),
                     Extrude("sk2", 40.0),
                     NewSketch("XY"), AddRectangle("sk3", 35.0, 10.0, 15.0, 15.0),
                     Extrude("sk3", 40.0)),
            op=Boolean("cut", "f1", "f2"),
            variants={"kind": _v("union"), "target": _v("f2"), "tool": _v("f3")},
        ),
        "revolve": Case(
            prelude=(NewSketch("XY"), AddRectangle("sk1", 20.0, 0.0, 10.0, 10.0),
                     NewSketch("XY"), AddRectangle("sk2", 30.0, 0.0, 20.0, 20.0)),
            op=Revolve("sk1", (0.0, 0.0, 0.0, 0.0, 1.0, 0.0), 360.0),
            variants={"sketch": _v("sk2"),
                      "axis": _v((0.0, 0.0, 0.0, 1.0, 0.0, 0.0)),
                      "angle": _v(180.0)},
        ),
        "chamfer": Case(
            prelude=_BOX,
            op=Chamfer(("|Z",), 3.0, None),
            variants={"edges": _v((">Z",)), "distance": _v(6.0),
                      "distance2": _v(6.0)},
        ),
        # The hole is the worst offender in the census (freecad collapsed simple /
        # counterbore / countersink onto one cylinder), so every one of its eleven
        # fields is exercised from a base on which it is MEANINGFUL: cbore_* from a
        # counterbore, csk_* from a countersink, depth from a blind hole.
        "hole": Case(
            prelude=_BOX,
            op=Hole("", 0.0, 0.0, 8.0, 10.0, False, "simple"),
            variants={
                "face_or_sketch": _v("<Z"),
                "x": _v(15.0),
                "y": _v(10.0),
                "diameter": _v(16.0),
                "depth": _v(5.0),
                "through": _v(True),
                "kind": _v("counterbore", kind="simple", through=True,
                           depth=None, cbore_diameter=16.0, cbore_depth=5.0),
                "cbore_diameter": _v(24.0, kind="counterbore", through=True,
                                     depth=None, cbore_diameter=16.0,
                                     cbore_depth=5.0),
                "cbore_depth": _v(10.0, kind="counterbore", through=True,
                                  depth=None, cbore_diameter=16.0,
                                  cbore_depth=5.0),
                "csk_diameter": _v(24.0, kind="countersink", through=True,
                                   depth=None, csk_diameter=16.0, csk_angle=82.0),
                "csk_angle": _v(120.0, kind="countersink", through=True,
                                depth=None, csk_diameter=16.0, csk_angle=82.0),
            },
        ),
        # THE FIXTURE WAS THE BUG, for shell.kind. ``kind`` is OCCT's JOIN TYPE, and
        # a join can only show itself where the offset faces do NOT meet cleanly. On
        # the convex _BOX this Case used to use, they meet cleanly: "arc" and
        # "intersection" produce THE SAME SOLID, and the DEAD cell that reported was
        # a weakness of the fixture, not a dropped field. So the substrate is now an
        # L-shaped prism (:data:`_L_PRISM`), whose reflex corner is exactly the
        # corner a join type has to decide about -- an arc rolls a radius round it,
        # an intersection runs the two offset faces on until they meet. A backend
        # that drops ``kind`` now produces identical state and is caught; one that
        # honours it is not. The wall is 5mm on a 70mm part so that frep's sampling
        # grid can resolve it (see frep.MIN_WALL_CELLS) -- a fixture below the cell
        # size would test the mesher, not the field.
        "shell": Case(
            prelude=_L_PRISM,
            op=Shell((">Z",), 5.0, "arc"),
            variants={"faces": _v(("<Z",)), "thickness": _v(8.0),
                      "kind": _v("intersection")},
        ),
        "draft": Case(
            prelude=_BOX,
            op=Draft((">X",), 5.0, "<Z"),
            variants={"faces": _v(("<X",)), "angle": _v(10.0),
                      "neutral_plane": _v(">Z")},
        ),
        # ``ruled`` is identity on a two-profile loft (a straight run between two
        # sections is the same surface either way), so it is exercised from a
        # THREE-profile base, which is the only place it can mean anything.
        "loft": Case(
            prelude=(NewSketch("XY"), AddRectangle("sk1", 0.0, 0.0, 40.0, 40.0),
                     NewSketch("XY"), AddRectangle("sk2", 0.0, 0.0, 20.0, 20.0),
                     NewSketch("XY"), AddRectangle("sk3", 0.0, 0.0, 10.0, 10.0)),
            op=Loft(("sk1", "sk2"), False, (0.0, 30.0)),
            variants={
                "sketches": _v(("sk1", "sk3")),
                "offsets": _v((0.0, 60.0)),
                "ruled": _v(True, sketches=("sk1", "sk2", "sk3"),
                            offsets=(0.0, 15.0, 30.0), ruled=False),
            },
        ),
        "sweep": Case(
            prelude=(NewSketch("XY"), AddCircle("sk1", 0.0, 0.0, 5.0),
                     NewSketch("XZ"), AddLine("sk2", 0.0, 0.0, 0.0, 40.0),
                     NewSketch("XY"), AddCircle("sk3", 0.0, 0.0, 12.0),
                     NewSketch("XZ"), AddLine("sk4", 0.0, 0.0, 0.0, 20.0)),
            op=Sweep("sk1", "sk2"),
            variants={"sketch": _v("sk3"), "path": _v("sk4")},
        ),
        "linear_pattern": Case(
            prelude=_TWO_FEATURES,
            op=LinearPattern("f1", (1.0, 0.0, 0.0), 3, 40.0),
            variants={"feature": _v("f2"), "direction": _v((0.0, 1.0, 0.0)),
                      "count": _v(5), "spacing": _v(60.0)},
        ),
        "circular_pattern": Case(
            prelude=_TWO_FEATURES,
            op=CircularPattern("f1", (0.0, 0.0, 0.0, 0.0, 0.0, 1.0), 4, 360.0),
            variants={"feature": _v("f2"),
                      "axis": _v((0.0, 0.0, 0.0, 1.0, 0.0, 0.0)),
                      "count": _v(6), "angle": _v(180.0)},
        ),
        "mirror": Case(
            prelude=_TWO_FEATURES,
            op=Mirror("f1", "XZ"),
            variants={"feature_or_body": _v("f2"), "plane": _v("YZ")},
        ),
        "add_instance": Case(
            prelude=_BOX,
            op=AddInstance("f1", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            variants={"part": _v("solid"), "x": _v(50.0), "y": _v(50.0),
                      "z": _v(50.0), "rx": _v(90.0), "ry": _v(90.0),
                      "rz": _v(90.0)},
        ),
        "mate": Case(
            prelude=_BOX + (AddInstance("f1", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                            AddInstance("f1", 50.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                            AddInstance("f1", 0.0, 50.0, 0.0, 0.0, 0.0, 0.0)),
            op=Mate("rigid", "i1", "i2", None),
            variants={"kind": _v("revolute"), "a": _v("i3"), "b": _v("i3"),
                      "value": _v(5.0)},
        ),
        # SetParam edits op #1 (the rectangle) and replays. Retargeting it at op #2
        # (the extrude, which has no 'w') is a typed bad-param error -- which is a
        # perfectly good way for the field to prove it was read.
        "set_param": Case(
            prelude=_BOX,
            op=SetParam(1, "w", 30.0),
            variants={"target": _v(2), "param": _v("h"), "value": _v(10.0)},
        ),
    }


CASES: Dict[str, Case] = _cases()


# ---------------------------------------------------------------------------
# Schema enumeration -- derived, never hand-listed.
# ---------------------------------------------------------------------------

def op_fields() -> List[Tuple[str, str]]:
    """Every (op tag, field name) in the CISP schema, in registry order.

    ``dataclasses.fields`` walks the class, so a field added to ops.py appears
    here the moment it is added -- and immediately shows up in :func:`unmapped`.
    """
    out: List[Tuple[str, str]] = []
    for tag, cls in _REGISTRY.items():
        for f in dataclasses.fields(cls):
            out.append((tag, f.name))
    return out


def unmapped() -> List[Tuple[str, str]]:
    """(op, field) pairs with neither a variant nor an inert justification.

    A non-empty result is a TEST FAILURE, not a warning. It is the anti-rot
    latch: the schema grew and this oracle did not, so the new field -- which is
    exactly the field most likely to be unwired -- would otherwise go unchecked.
    """
    missing: List[Tuple[str, str]] = []
    for tag, name in op_fields():
        if (tag, name) in INERT_FIELDS:
            continue
        case = CASES.get(tag)
        if case is None or name not in case.variants:
            missing.append((tag, name))
    return missing


# ---------------------------------------------------------------------------
# Measurement.
# ---------------------------------------------------------------------------

def _round(value: Any, ndigits: int = 6) -> Any:
    """Canonicalise for comparison. Floats are rounded so that a re-run of the
    SAME stream is bit-identical; -0.0 is folded onto 0.0 so a sign flip on a
    zero is not mistaken for a change."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        r = round(value, ndigits)
        return 0.0 if r == 0.0 else r
    if isinstance(value, (list, tuple)):
        return [_round(v, ndigits) for v in value]
    if isinstance(value, dict):
        return {str(k): _round(v, ndigits) for k, v in sorted(value.items())
                # an OCCT shape handle is an opaque object: it is not comparable
                # across runs and its geometry is already covered by bbox/volume.
                if k != "shape"}
    if isinstance(value, (int, str, type(None))):
        return value
    return repr(value)


def signature(backend: Any) -> str:
    """The measurement vector: everything the backend will tell us about its state.

    Deliberately NOT ``state_digest()``. Every backend's digest hashes the op log,
    so ANY field change moves it -- every field would score LIVE and the oracle
    would be a tautology. What is compared here is what the backend BUILT:

      * ``metrics``    volume, surface_area, bbox, centre of mass, tri/vert counts.
        Centre of mass matters: a hole moved 15 mm sideways changes neither volume
        nor bbox, and a loft with its profiles swapped changes neither either.
      * ``validity``   genus, watertight, manifold, solid_present -- topology, which
        a size-only vector would miss.
      * ``summary``    sketch / entity / feature counts.
      * ``sketch_dof`` the constraint bookkeeping (the only observable a sketch-only
        op has).
      * ``assembly``   placed instances, their transforms, and the mates.
    """
    state: Dict[str, Any] = {}
    for what in ("metrics", "measure", "validity", "summary", "sketch_dof",
                 "assembly"):
        try:
            res = backend.query(what)
        except Exception as exc:  # noqa: BLE001 - a hostile backend must not crash us
            res = {"query_raised": f"{type(exc).__name__}: {exc}"}
        state[what] = _round(res) if isinstance(res, dict) else _round(res)
    return json.dumps(state, sort_keys=True, separators=(",", ":"), default=repr)


@dataclass
class Outcome:
    """One op stream on one backend."""

    ok: bool = False
    rejected: Optional[str] = None
    codes: Tuple[str, ...] = ()
    error: str = ""
    sig: str = ""

    @property
    def refused(self) -> bool:
        return not self.ok and not self.error


def _run_stream(backend: Any, ops: Sequence[Op]) -> Outcome:
    out = Outcome()
    try:
        session = HarnessSession(backend, verify_level="core")
        result = session.apply_ops(list(ops))
    except Exception as exc:  # noqa: BLE001 - a crash IS an observation; record it
        out.error = f"{type(exc).__name__}: {exc}"
        return out
    out.ok = bool(getattr(result, "ok", False))
    rej = getattr(result, "rejected", None)
    out.rejected = str(rej.get("op")) if isinstance(rej, dict) else None
    out.codes = tuple(sorted({d.code for d in getattr(result, "diagnostics", [])}))
    out.sig = signature(backend)
    return out


def _run_stream_guarded(backend: Any, ops: Sequence[Op],
                        timeout_s: float) -> Outcome:
    """``_run_stream`` under a hard wall-clock ceiling.

    freecad and blender fork a process. A forked process can wedge. A test suite
    that can wedge is a test suite nobody runs, so the worker is a daemon thread
    and a timeout is an ERROR cell, not a hang. (The thread is abandoned, not
    killed -- Python cannot kill a thread -- but it is a daemon, so it cannot keep
    the interpreter alive.)
    """
    box: List[Outcome] = []

    def work() -> None:
        box.append(_run_stream(backend, ops))

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive() or not box:
        return Outcome(error=f"timeout after {timeout_s:g}s")
    return box[0]


# ---------------------------------------------------------------------------
# The check.
# ---------------------------------------------------------------------------

@dataclass
class Cell:
    """One (backend, op, field) verdict."""

    backend: str
    op: str
    field: str
    verdict: str = SKIP
    detail: str = ""

    @property
    def dead(self) -> bool:
        return self.verdict == DEAD

    def to_dict(self) -> dict:
        return {"backend": self.backend, "op": self.op, "field": self.field,
                "verdict": self.verdict, "detail": self.detail}


def check_field(backend_name: str, op_tag: str, field_name: str,
                factory: Optional[BackendFactory] = None,
                timeout_s: float = DEFAULT_TIMEOUT_S) -> Cell:
    """Run the base stream and the one-field-changed stream; compare."""
    cell = Cell(backend_name, op_tag, field_name)
    if (op_tag, field_name) in INERT_FIELDS:
        cell.verdict = NA
        cell.detail = "inert by allow-list"
        return cell
    case = CASES.get(op_tag)
    if case is None or field_name not in case.variants:
        cell.verdict = ERR
        cell.detail = "no fixture for this field (see unmapped())"
        return cell

    alt, base = case.variants[field_name]
    backend, skip = resolve(backend_name, factory)
    if backend is None:
        cell.verdict = SKIP
        cell.detail = skip
        return cell

    a = _run_stream_guarded(backend, case.stream(base), timeout_s)
    changes = dict(base)
    changes[field_name] = alt
    b = _run_stream_guarded(backend, case.stream(changes), timeout_s)

    if a.error or b.error:
        cell.verdict = ERR
        cell.detail = (a.error or b.error)[:120]
        return cell
    if a.refused and b.refused:
        # Both streams were refused, so the field was never exercised. WHY they
        # were refused decides whether that is fine.
        declared = set(a.codes) & _CAPABILITY_CODES
        cell.verdict = NA if declared else REJ
        cell.detail = "both streams refused (%s)" % (",".join(a.codes) or "no code")
        return cell
    if a.ok != b.ok or a.codes != b.codes:
        # One variant was refused and the other was not, or they were diagnosed
        # differently. The field was READ. A typed error is a difference.
        cell.verdict = ERR
        cell.detail = "refusal differs: base(ok=%s %s) alt(ok=%s %s)" % (
            a.ok, ",".join(a.codes) or "-", b.ok, ",".join(b.codes) or "-")
        return cell
    if a.sig != b.sig:
        cell.verdict = LIVE
        return cell
    cell.verdict = DEAD
    cell.detail = "identical model state -- the backend never read this field"
    return cell


@dataclass
class FieldLivenessReport:
    backends: List[str] = dc_field(default_factory=list)
    skipped_backends: Dict[str, str] = dc_field(default_factory=dict)
    cells: List[Cell] = dc_field(default_factory=list)
    unmapped: List[Tuple[str, str]] = dc_field(default_factory=list)

    @property
    def dead(self) -> List[Cell]:
        """The BUG CENSUS: dead fields on engines that claim to build geometry."""
        return [c for c in self.cells
                if c.dead and c.backend not in NON_GEOMETRIC]

    @property
    def dead_nongeometric(self) -> List[Cell]:
        """Dead cells on the stub. Informational: the stub models no geometry."""
        return [c for c in self.cells
                if c.dead and c.backend in NON_GEOMETRIC]

    @property
    def ok(self) -> bool:
        return not self.dead and not self.unmapped

    @property
    def rejected(self) -> List[Cell]:
        """The backend refused the op AS THE SCHEMA DOCUMENTS IT. Also a finding."""
        return [c for c in self.cells if c.verdict == REJ]

    def counts(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for c in self.cells:
            row = out.setdefault(c.backend, {LIVE: 0, DEAD: 0, ERR: 0, NA: 0,
                                             REJ: 0, SKIP: 0})
            row[c.verdict] = row.get(c.verdict, 0) + 1
        return out

    def to_dict(self) -> dict:
        return {
            "oracle": "field_liveness",
            "ok": self.ok,
            "backends": self.backends,
            "skipped_backends": self.skipped_backends,
            "unmapped_fields": [list(p) for p in self.unmapped],
            "inert_allow_list": {"%s.%s" % k: v for k, v in INERT_FIELDS.items()},
            "counts": self.counts(),
            "dead": [c.to_dict() for c in self.dead],
            "dead_nongeometric": [c.to_dict() for c in self.dead_nongeometric],
            "rejected": [c.to_dict() for c in self.rejected],
            "cells": [c.to_dict() for c in self.cells],
            "note": ("LIVE proves a field REACHES the kernel. It does not prove "
                     "the field is used CORRECTLY. Liveness is a floor."),
        }


def run(backends: Optional[Sequence[str]] = None,
        ops: Optional[Sequence[str]] = None,
        factory: Optional[BackendFactory] = None,
        timeout_s: float = DEFAULT_TIMEOUT_S) -> FieldLivenessReport:
    """The full matrix: every (op, field) x every requested backend."""
    wanted = tuple(backends) if backends is not None else BACKENDS
    report = FieldLivenessReport(unmapped=unmapped())
    live: List[str] = []
    for name in wanted:
        backend, skip = resolve(name, factory)
        if backend is None:
            report.skipped_backends[name] = skip
        else:
            live.append(name)
    report.backends = live

    pairs = [(t, f) for (t, f) in op_fields()
             if ops is None or t in set(ops)]
    for name in wanted:
        for tag, fname in pairs:
            if name not in live:
                report.cells.append(Cell(name, tag, fname, SKIP,
                                         report.skipped_backends.get(name, "")))
                continue
            report.cells.append(check_field(name, tag, fname, factory=factory,
                                            timeout_s=timeout_s))
    return report


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------

def format_text(report: FieldLivenessReport) -> str:
    order = [b for b in BACKENDS
             if b in report.backends or b in report.skipped_backends]
    by_key: Dict[Tuple[str, str], Dict[str, Cell]] = {}
    for c in report.cells:
        by_key.setdefault((c.op, c.field), {})[c.backend] = c

    lines: List[str] = []
    lines.append("FIELD LIVENESS -- does the backend READ the field it was given?")
    lines.append("=" * 78)
    lines.append("a DEAD cell means: two op streams differing ONLY in this field")
    lines.append("produced identical model state. The backend ignored its own schema.")
    lines.append("")
    if report.unmapped:
        lines.append("UNMAPPED FIELDS (the schema grew and this oracle did not):")
        for tag, name in report.unmapped:
            lines.append("  %s.%s" % (tag, name))
        lines.append("")
    for name, why in sorted(report.skipped_backends.items()):
        lines.append("  skipped %-9s %s" % (name, why))
    if report.skipped_backends:
        lines.append("")

    head = "%-18s %-16s" % ("op", "field")
    lines.append(head + " ".join("%-5s" % b[:5] for b in order))
    lines.append("-" * (len(head) + 6 * len(order)))
    for (tag, fname) in op_fields():
        row = by_key.get((tag, fname))
        if row is None:
            continue
        cells = [row.get(b) for b in order]
        lines.append("%-18s %-16s" % (tag, fname)
                     + " ".join("%-5s" % (c.verdict if c else "-") for c in cells))
    lines.append("")
    counts = report.counts()
    lines.append("%-10s %5s %5s %5s %5s %5s %5s"
                 % ("engine", "LIVE", "DEAD", "ERR", "N/A", "REJ", "SKIP"))
    lines.append("-" * 46)
    for b in order:
        row = counts.get(b, {})
        lines.append("%-10s %5d %5d %5d %5d %5d %5d"
                     % (b, row.get(LIVE, 0), row.get(DEAD, 0), row.get(ERR, 0),
                        row.get(NA, 0), row.get(REJ, 0), row.get(SKIP, 0)))
    lines.append("")
    if report.dead:
        lines.append("BUG CENSUS -- %d DEAD field(s) on engines that build geometry:"
                     % len(report.dead))
        for c in report.dead:
            lines.append("  %-9s %s.%s" % (c.backend, c.op, c.field))
    else:
        lines.append("no dead fields on the geometric engines checked.")
    if report.dead_nongeometric:
        lines.append("")
        lines.append("(%d dead cell(s) on the stub -- it models no geometry and its "
                     "query()\n exposes only counts and DOF, so this is by design "
                     "and is not counted.)" % len(report.dead_nongeometric))
    if report.rejected:
        lines.append("")
        lines.append("%d REJECTED cell(s) -- the engine refused the op as ops.py "
                     "DOCUMENTS it\n (not a declared capability gap; a legal op "
                     "turned away):" % len(report.rejected))
        for c in report.rejected:
            lines.append("  %-9s %s.%s  %s" % (c.backend, c.op, c.field, c.detail))
    lines.append("")
    lines.append("inert allow-list (%d entries -- fields a change CANNOT move, by "
                 "geometry):" % len(INERT_FIELDS))
    for (tag, fname), why in sorted(INERT_FIELDS.items()):
        lines.append("  %s.%s" % (tag, fname))
        lines.append("      %s" % why)
    return "\n".join(lines)
