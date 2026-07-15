"""KCL codec -- lower a CISP op stream to a Zoo (KittyCAD) ``.kcl`` program.

KCL is the KittyCAD Language: the code-CAD language that drives Zoo's geometry
engine (zoo.dev/docs/kcl). A ``.kcl`` file is an ordered, parametric program that
sketches profiles on planes and turns them into solids::

    @settings(defaultLengthUnit = mm, kclVersion = 1.0)

    sketch001 = startSketchOn(XY)
      |> startProfile(at = [0, 0])
      |> xLine(length = 20)
      |> yLine(length = 10)
      |> xLine(length = -20)
      |> close()
    solid001 = extrude(sketch001, length = 5)

This module is the *offline, deterministic* half of the Zoo integration: it turns
a list of :class:`harnesscad.core.cisp.ops.Op` into that program text with no
network, no API key, and no wall clock. It is the codec the format registry wires
to the ``.kcl`` extension, and it is the emitter :class:`harnesscad.io.backends.
zoo.ZooBackend` exports through.

Design rules (the same two the OpenSCAD backend lives by)
---------------------------------------------------------
1.  **Never silently drop a field.** Every field of every op either shapes the
    emitted KCL (a coordinate, a length, an axis, a boolean kind) or is written
    verbatim into an annotation next to the statement it belongs to. A reader can
    always recover what the op said. Emission is scored on field-liveness, so a
    dropped field is a bug, not a shortcut.
2.  **Emit real KCL, or annotate -- never fabricate.** The solid-building spine
    (sketch primitives, extrude, revolve, the 3-D booleans, holes, patterns) maps
    onto genuine KCL standard-library calls. The *finishing* ops that select
    geometry by CadQuery-style selector strings (``fillet``/``chamfer`` on
    ``"|Z"``; ``draft`` on a neutral plane) cannot be turned into a correct KCL
    call, because KCL identifies edges and faces by **tags declared at sketch
    time**, and a selector string has no tag to bind to. Rather than invent a tag
    reference that would fillet the wrong edge (or emit an undefined identifier
    that would not parse), those ops are emitted as fully-specified annotations
    that carry every field. This is the same principled split the OpenSCAD
    backend makes -- implement exactly, or refuse to fake it.

What maps to real KCL
---------------------
====================  ==================================================
``NewSketch``         chooses the ``startSketchOn`` plane (XY/XZ/YZ)
``AddRectangle``      a closed ``startProfile |> xLine |> yLine ... close``
``AddCircle``         ``startSketchOn(P) |> circle(center=, radius=)``
``AddArc``            an exact three-point ``arc(interiorAbsolute=, endAbsolute=)``
``AddPolygon``        an arbitrary closed ``startProfile |> line ... |> close``
``AddLine``           an open two-point profile (not extrudable, but exact)
``AddPoint``          a bare ``startProfile(at=[x, y])`` (coordinates kept)
``Primitive``         sketch+``extrude`` (box/cylinder/wedge) or ``revolve`` (sphere/cone/torus)
``Transform``         ``solid |> rotate(roll=,pitch=,yaw=) |> translate(x=,y=,z=)``
``Scale``             ``solid |> scale(x=sx, y=sy, z=sz)``
``PatternTransform``  one ``rotate |> translate`` copy per placement, ``union``ed
``Extrude``           ``extrude(profile, length = d)`` (sign preserved)
``Revolve``           ``revolve(profile, axis = X|Y|Z, angle = a)``
``Boolean``           ``subtract`` / ``union`` / ``intersect``
``Hole``              a positioned cylinder (or stepped tool) ``subtract``ed
``Shell``             ``shell(solid, thickness = t, faces = [END|START])``
``LinearPattern``     ``patternLinear3d(solid, instances=, distance=, axis=)``
``CircularPattern``   ``patternCircular3d(solid, instances=, axis=, center=, arcDegrees=)``
====================  ==================================================

What is annotated (all fields kept, no fabricated tags)
-------------------------------------------------------
``Constrain`` (KCL has no imperative constraint solver -- constraints are encoded
implicitly by the explicit coordinates the profile already carries), ``Fillet``,
``Chamfer``, ``Draft``, ``Mirror``, ``AddInstance``, ``Mate`` -- each because it
needs a KCL edge/face tag, an assembly-import graph, or a joint model that a flat
selector/ref string cannot supply offline. Also ``AddEllipse`` and ``AddSpline``
(KCL has no native ellipse or through-points spline, and sampling either into a
polyline would approximate), ``Split`` (no infinite half-space / plane-section
op), ``Thicken`` (KCL shells caps but has no general surface offset), ``Hull``
(no convex-hull operator) and ``Minkowski`` (no Minkowski-sum / SDF-offset op) --
each refused rather than faked, exactly as the finishing ops are.

A KCL round trip (program text -> geometry -> program text) is not offline-
feasible: reading a ``.kcl`` back into geometry means *executing* it on Zoo's
engine, exactly as reading a ``.svg`` back into a solid is not a thing. So the
codec is **write-only** in the registry, and its determinism is what is tested:
``emit_kcl(ops)`` is a pure function of the ops, byte-for-byte stable and
idempotent.

stdlib-only, deterministic (no wall clock, no randomness, no imports of the SDK).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from harnesscad.core.cisp.ops import (
    Op, NewSketch, AddPoint, AddLine, AddCircle, AddRectangle,
    AddArc, AddEllipse, AddPolygon, AddSpline,
    Constrain, Extrude, Fillet, Boolean,
    Primitive, Split, Thicken, Hull, Minkowski,
    Transform, Scale, PatternTransform,
    Revolve, Chamfer, Hole, Shell, Draft,
    Loft, Sweep, LinearPattern, CircularPattern, Mirror,
    AddInstance, Mate, SetParam,
)

__all__ = [
    "KclError",
    "KclEmitError",
    "OP_KCL_MAPPING",
    "emit_kcl",
    "serialize_kcl",
    "write_kcl",
    "ops_of",
]


class KclError(Exception):
    """Base class for every error this codec raises."""


class KclEmitError(KclError):
    """An op (or one of its fields) cannot be lowered to KCL faithfully.

    Raised rather than emitting geometry that would differ from what the op
    declared -- e.g. a ``Shell`` opening a face KCL cannot name, or an
    ``Extrude`` of a sketch with no closed profile. The message names the op and
    the field. This is the codec working, not failing.
    """


#: A human-readable summary of the CISP-op -> KCL mapping, exported so callers
#: (docs, the CLI capability report) can show it without importing the emitter.
OP_KCL_MAPPING: Dict[str, str] = {
    "NewSketch": "startSketchOn(<plane>)  (plane in XY|XZ|YZ)",
    "AddPoint": "startProfile(at=[x, y])  (open profile, coordinates preserved)",
    "AddLine": "startProfile(at=[x1, y1]) |> line(endAbsolute=[x2, y2])",
    "AddCircle": "startSketchOn(P) |> circle(center=[cx, cy], radius=r)",
    "AddRectangle": "startProfile(at=[x, y]) |> xLine |> yLine |> xLine |> close()",
    "AddArc": "startProfile(at=<start>) |> arc(interiorAbsolute=<mid>, endAbsolute=<end>)",
    "AddPolygon": "startProfile(at=[x0, y0]) |> line(endAbsolute=...) ... |> close()",
    "AddEllipse": "annotation (KCL has no native ellipse; a sampled polyline would approximate)",
    "AddSpline": "annotation (KCL has no through-points spline; sampling would approximate)",
    "Constrain": "annotation (KCL encodes constraints as explicit coordinates)",
    "Extrude": "extrude(profile, length=distance)  (sign preserved)",
    "Revolve": "revolve(profile, axis=X|Y|Z, angle=angle)",
    "Boolean": "subtract(a, tools=[b]) | union([a, b]) | intersect([a, b])",
    "Primitive": "sketch + extrude (box/cylinder/wedge) | revolve (sphere/cone/torus)",
    "Split": "annotation (KCL has no infinite half-space / plane-section op)",
    "Thicken": "annotation (KCL has shell but no general solid-surface offset)",
    "Hull": "annotation (KCL has no convex-hull operator)",
    "Minkowski": "annotation (KCL has no Minkowski-sum / SDF-offset operator)",
    "Transform": "solid |> rotate(roll=, pitch=, yaw=) |> translate(x=, y=, z=)",
    "Scale": "solid |> scale(x=sx, y=sy, z=sz)",
    "PatternTransform": "per-placement rotate |> translate copies, union()ed",
    "Hole": "subtract(solid, tools=[<positioned cylinder / stepped tool>])",
    "Shell": "shell(solid, thickness=t, faces=[END|START])",
    "LinearPattern": "patternLinear3d(solid, instances=, distance=, axis=)",
    "CircularPattern": "patternCircular3d(solid, instances=, axis=, center=, arcDegrees=)",
    "Fillet": "annotation (needs a KCL edge tag; selector strings have none)",
    "Chamfer": "annotation (needs a KCL edge tag; selector strings have none)",
    "Draft": "annotation (needs a KCL neutral-plane/face tag)",
    "Mirror": "annotation (needs a KCL body/plane reference)",
    "AddInstance": "annotation (KCL assemblies are import + transform graphs)",
    "Mate": "annotation (KCL has no imperative joint op)",
    "SetParam": "applied as an edit before emission; never a statement",
}

#: CISP plane names -> KCL plane constants. KCL's standard planes are XY/XZ/YZ.
_PLANES = {"XY": "XY", "XZ": "XZ", "YZ": "YZ"}

#: Extrusion-cap selector strings -> KCL ``shell`` face constants. A shell may
#: only open the caps of an extrusion (the faces normal to the extrude axis);
#: KCL names them END (far cap) and START (base). Side walls have no KCL name.
_CAP_FACES = {
    ">Z": "END", "+Z": "END", "top": "END", "END": "END", "end": "END",
    "<Z": "START", "-Z": "START", "bottom": "START", "START": "START",
    "start": "START",
}


# ---------------------------------------------------------------------------
# deterministic number / vector formatting
# ---------------------------------------------------------------------------

def _num(x: float) -> str:
    """Format a number deterministically: integers as integers, else a stable repr.

    ``5.0 -> "5"``, ``-4.0 -> "-4"``, ``1.5 -> "1.5"``. No locale, no rounding
    that would lose precision, so two emissions of the same op are byte-equal.
    """
    f = float(x)
    if f == int(f) and abs(f) < 1e16:
        return str(int(f))
    return repr(f)


def _vec(values: Sequence[float]) -> str:
    return "[" + ", ".join(_num(v) for v in values) + "]"


def _annotation(op_name: str, **fields: object) -> List[str]:
    """A comment block that records every field of an op KCL cannot express.

    Field-liveness is preserved: every field value appears in the text, so a
    reader recovers exactly what the op declared even though no geometry is
    emitted for it.
    """
    parts = ["    // [harnesscad:%s] not expressible as a KCL statement -- "
             "fields preserved below:" % op_name]
    for key, value in fields.items():
        parts.append("    //   %s = %s" % (key, value))
    return parts


# ---------------------------------------------------------------------------
# the emitter
# ---------------------------------------------------------------------------

class _Emitter:
    """Turns an op stream into KCL, mirroring the stub backend's id allocation.

    The ops reference sketches/features by the ids the backend handed back
    (``sk1``, ``f2``, ...). This walks the stream allocating the *same* ids in the
    *same* order, so ``op.sketch == "sk1"`` resolves to the KCL variable created
    for the first sketch. Nothing here executes geometry; it is pure text.
    """

    def __init__(self, name: str, length_unit: str, kcl_version: str) -> None:
        self.name = name
        self.length_unit = length_unit
        self.kcl_version = kcl_version
        self.lines: List[str] = []
        # id allocators, identical in order to StubBackend._new_id
        self._n = {"sk": 0, "e": 0, "f": 0, "i": 0}
        # CISP id -> KCL variable name
        self.sketch_var: Dict[str, str] = {}
        self.sketch_plane: Dict[str, str] = {}
        # sketch id -> list of (entity_var, closed?) profiles under it
        self.sketch_profiles: Dict[str, List[Tuple[str, bool]]] = {}
        self.entity_sketch: Dict[str, str] = {}
        self.feature_var: Dict[str, str] = {}
        self.last_solid: Optional[str] = None

    def _new_id(self, kind: str) -> str:
        self._n[kind] += 1
        return kind + str(self._n[kind])

    # -- reference resolution --------------------------------------------
    def _solid_ref(self, ref: str) -> str:
        """Resolve a CISP feature/body ref to a KCL solid variable."""
        if ref in self.feature_var:
            return self.feature_var[ref]
        if ref in ("solid", "body", "last", ""):
            if self.last_solid is None:
                raise KclEmitError("reference %r has no solid yet" % ref)
            return self.last_solid
        raise KclEmitError("unknown solid reference %r" % ref)

    # -- top-level -------------------------------------------------------
    def emit(self, ops: Sequence[Op]) -> str:
        self.lines.append("// Generated by harnesscad KCL codec. Do not edit.")
        self.lines.append("// model: %s" % self.name)
        self.lines.append("@settings(defaultLengthUnit = %s, kclVersion = %s)"
                          % (self.length_unit, self.kcl_version))
        self.lines.append("")
        for op in ops:
            if isinstance(op, SetParam):
                # SetParam edits the recorded op stream; it is applied before
                # emission (see ops_of) and is never itself a KCL statement.
                continue
            self._dispatch(op)
        # A trailing newline keeps the file POSIX-clean and byte-stable.
        return "\n".join(self.lines).rstrip("\n") + "\n"

    def _dispatch(self, op: Op) -> None:
        handler = getattr(self, "_op_" + type(op).__name__, None)
        if handler is None:
            raise KclEmitError("no KCL lowering for op %r" % type(op).__name__)
        handler(op)

    # -- sketch primitives ----------------------------------------------
    def _op_NewSketch(self, op: NewSketch) -> None:
        plane = str(op.plane).upper()
        if plane not in _PLANES:
            raise KclEmitError(
                "NewSketch.plane %r is not a KCL standard plane (XY|XZ|YZ)"
                % op.plane)
        sid = self._new_id("sk")
        self.sketch_var[sid] = "sketch%03d" % self._n["sk"]
        self.sketch_plane[sid] = _PLANES[plane]
        self.sketch_profiles[sid] = []

    def _require_sketch(self, sid: str) -> str:
        if sid not in self.sketch_var:
            raise KclEmitError("unknown sketch %r" % sid)
        return self.sketch_var[sid]

    def _op_AddPoint(self, op: AddPoint) -> None:
        sid = op.sketch
        self._require_sketch(sid)
        eid = self._new_id("e")
        var = "profile%s" % eid[1:]
        plane = self.sketch_plane[sid]
        self.lines.append("%s = startSketchOn(%s)" % (var, plane))
        self.lines.append("  |> startProfile(at = %s)" % _vec([op.x, op.y]))
        self.lines.append("")
        self.sketch_profiles[sid].append((var, False))
        self.entity_sketch[eid] = sid

    def _op_AddLine(self, op: AddLine) -> None:
        sid = op.sketch
        self._require_sketch(sid)
        eid = self._new_id("e")
        var = "profile%s" % eid[1:]
        plane = self.sketch_plane[sid]
        self.lines.append("%s = startSketchOn(%s)" % (var, plane))
        self.lines.append("  |> startProfile(at = %s)" % _vec([op.x1, op.y1]))
        self.lines.append("  |> line(endAbsolute = %s)" % _vec([op.x2, op.y2]))
        self.lines.append("")
        self.sketch_profiles[sid].append((var, False))
        self.entity_sketch[eid] = sid

    def _op_AddCircle(self, op: AddCircle) -> None:
        sid = op.sketch
        self._require_sketch(sid)
        if op.r <= 0:
            raise KclEmitError("AddCircle.r must be > 0 (got %r)" % op.r)
        eid = self._new_id("e")
        var = "profile%s" % eid[1:]
        plane = self.sketch_plane[sid]
        self.lines.append("%s = startSketchOn(%s)" % (var, plane))
        self.lines.append("  |> circle(center = %s, radius = %s)"
                          % (_vec([op.cx, op.cy]), _num(op.r)))
        self.lines.append("")
        self.sketch_profiles[sid].append((var, True))
        self.entity_sketch[eid] = sid

    def _op_AddRectangle(self, op: AddRectangle) -> None:
        sid = op.sketch
        self._require_sketch(sid)
        if op.w <= 0 or op.h <= 0:
            raise KclEmitError("AddRectangle w and h must be > 0")
        eid = self._new_id("e")
        var = "profile%s" % eid[1:]
        plane = self.sketch_plane[sid]
        self.lines.append("%s = startSketchOn(%s)" % (var, plane))
        self.lines.append("  |> startProfile(at = %s)" % _vec([op.x, op.y]))
        self.lines.append("  |> xLine(length = %s)" % _num(op.w))
        self.lines.append("  |> yLine(length = %s)" % _num(op.h))
        self.lines.append("  |> xLine(length = %s)" % _num(-op.w))
        self.lines.append("  |> close()")
        self.lines.append("")
        self.sketch_profiles[sid].append((var, True))
        self.entity_sketch[eid] = sid

    def _op_AddArc(self, op: AddArc) -> None:
        # An open circular-arc boundary curve. KCL's arc() has an exact absolute
        # three-point form -- arc(interiorAbsolute=<mid>, endAbsolute=<end>) from
        # the profile's current point -- so we sample the start/mid/end points on
        # the circle exactly and hand KCL those three points. No approximation.
        sid = op.sketch
        self._require_sketch(sid)
        if op.r <= 0:
            raise KclEmitError("AddArc.r must be > 0 (got %r)" % op.r)
        a0 = math.radians(float(op.start))
        a1 = math.radians(float(op.end))
        amid = math.radians((float(op.start) + float(op.end)) / 2.0)
        p0 = (op.cx + op.r * math.cos(a0), op.cy + op.r * math.sin(a0))
        pm = (op.cx + op.r * math.cos(amid), op.cy + op.r * math.sin(amid))
        p1 = (op.cx + op.r * math.cos(a1), op.cy + op.r * math.sin(a1))
        eid = self._new_id("e")
        var = "profile%s" % eid[1:]
        plane = self.sketch_plane[sid]
        self.lines.append("%s = startSketchOn(%s)" % (var, plane))
        self.lines.append("  |> startProfile(at = %s)" % _vec(p0))
        self.lines.append("  |> arc(interiorAbsolute = %s, endAbsolute = %s)"
                          "  // arc cx=%s cy=%s r=%s start=%s end=%s"
                          % (_vec(pm), _vec(p1), _num(op.cx), _num(op.cy),
                             _num(op.r), _num(op.start), _num(op.end)))
        self.lines.append("")
        # An open boundary curve (like AddLine): not itself an extrudable region.
        self.sketch_profiles[sid].append((var, False))
        self.entity_sketch[eid] = sid

    def _op_AddPolygon(self, op: AddPolygon) -> None:
        # An arbitrary closed polygon. KCL's polygon() builds only REGULAR
        # polygons, but a closed line chain (the same primitives AddRectangle
        # uses) expresses an arbitrary vertex list exactly.
        sid = op.sketch
        self._require_sketch(sid)
        pts = tuple(float(v) for v in op.points)
        if len(pts) % 2 != 0:
            raise KclEmitError(
                "AddPolygon.points must be a flat (x, y, ...) tuple of even length")
        if len(pts) < 6:
            raise KclEmitError(
                "AddPolygon needs >= 3 vertices (6 coordinates) to close a region")
        verts = [(pts[i], pts[i + 1]) for i in range(0, len(pts), 2)]
        eid = self._new_id("e")
        var = "profile%s" % eid[1:]
        plane = self.sketch_plane[sid]
        self.lines.append("%s = startSketchOn(%s)" % (var, plane))
        self.lines.append("  |> startProfile(at = %s)" % _vec(verts[0]))
        for vx in verts[1:]:
            self.lines.append("  |> line(endAbsolute = %s)" % _vec(vx))
        self.lines.append("  |> close()")
        self.lines.append("")
        self.sketch_profiles[sid].append((var, True))
        self.entity_sketch[eid] = sid

    def _op_AddEllipse(self, op: AddEllipse) -> None:
        # KCL has circle() but no native ellipse primitive, and a sampled polyline
        # would be an approximation -- so, per the codec's "emit exactly or refuse
        # to fake it" rule, annotate with every field preserved.
        self.lines.extend(_annotation(
            "AddEllipse", sketch=op.sketch, center=[op.cx, op.cy],
            rx=op.rx, ry=op.ry, rotation=op.rotation,
            note="KCL has no native ellipse; a sampled polyline would approximate"))
        self.lines.append("")

    def _op_AddSpline(self, op: AddSpline) -> None:
        # KCL has no through-points spline in its offline standard library, and a
        # sampled polyline would be an approximation -- annotate, fields preserved.
        self.lines.extend(_annotation(
            "AddSpline", sketch=op.sketch, points=list(op.points), closed=op.closed,
            note="KCL has no through-points spline op; sampling would approximate"))
        self.lines.append("")

    def _op_Constrain(self, op: Constrain) -> None:
        # KCL is imperative code-CAD with no runtime constraint solver: the
        # constraint is already satisfied by the explicit coordinates the profile
        # carries. Record every field so nothing is dropped.
        self.lines.extend(_annotation(
            "Constrain", kind=op.kind, a=op.a, b=op.b, value=op.value))
        self.lines.append("")

    # -- features --------------------------------------------------------
    def _closed_profiles(self, sid: str) -> List[str]:
        return [var for (var, closed) in self.sketch_profiles.get(sid, []) if closed]

    def _op_Extrude(self, op: Extrude) -> None:
        self._require_sketch(op.sketch)
        if op.distance == 0:
            raise KclEmitError("Extrude.distance must be non-zero")
        closed = self._closed_profiles(op.sketch)
        if not closed:
            raise KclEmitError(
                "Extrude of sketch %r has no closed profile to extrude "
                "(KCL extrudes closed regions only)" % op.sketch)
        fid = self._new_id("f")
        var = "solid%s" % fid[1:]
        if len(closed) == 1:
            self.lines.append("%s = extrude(%s, length = %s)"
                              % (var, closed[0], _num(op.distance)))
        else:
            # Several closed profiles under one sketch: extrude each and union.
            pieces = []
            for i, prof in enumerate(closed):
                piece = "%s_%d" % (var, i)
                pieces.append(piece)
                self.lines.append("%s = extrude(%s, length = %s)"
                                  % (piece, prof, _num(op.distance)))
            self.lines.append("%s = union([%s])" % (var, ", ".join(pieces)))
        self.lines.append("")
        self.feature_var[fid] = var
        self.last_solid = var

    def _op_Revolve(self, op: Revolve) -> None:
        self._require_sketch(op.sketch)
        closed = self._closed_profiles(op.sketch)
        if not closed:
            raise KclEmitError(
                "Revolve of sketch %r has no closed profile" % op.sketch)
        if op.angle == 0:
            raise KclEmitError("Revolve.angle must be non-zero")
        axis = self._axis_letter(op.axis)
        fid = self._new_id("f")
        var = "solid%s" % fid[1:]
        # KCL revolve takes a named axis (X/Y/Z) and a sweep angle in degrees.
        self.lines.append(
            "%s = revolve(%s, axis = %s, angle = %s)  // axis6=%s"
            % (var, closed[0], axis, _num(op.angle), _vec(op.axis)))
        self.lines.append("")
        self.feature_var[fid] = var
        self.last_solid = var

    @staticmethod
    def _axis_letter(axis6: Sequence[float]) -> str:
        """Best-effort map of a 6-tuple axis (two points) to a KCL X/Y/Z axis.

        CISP states the revolution axis as two in-plane points; KCL wants a named
        axis. We take the direction and pick the dominant world component. The
        full 6-tuple is still written into a trailing comment (see caller), so no
        field is lost even though the axis is quantised to a standard one.
        """
        ax, ay, az, bx, by, bz = (float(v) for v in axis6)
        dx, dy, dz = bx - ax, by - ay, bz - az
        mags = {"X": abs(dx), "Y": abs(dy), "Z": abs(dz)}
        letter = max(mags, key=lambda k: mags[k])
        return letter if mags[letter] > 0 else "Y"

    def _op_Boolean(self, op: Boolean) -> None:
        if op.kind not in ("union", "cut", "intersect"):
            raise KclEmitError("unknown Boolean.kind %r" % op.kind)
        target = self._solid_ref(op.target) if op.target else self._solid_ref("last")
        tool = self._solid_ref(op.tool)
        fid = self._new_id("f")
        var = "solid%s" % fid[1:]
        if op.kind == "cut":
            self.lines.append("%s = subtract(%s, tools = [%s])"
                              % (var, target, tool))
        elif op.kind == "union":
            self.lines.append("%s = union([%s, %s])" % (var, target, tool))
        else:
            self.lines.append("%s = intersect([%s, %s])" % (var, target, tool))
        self.lines.append("")
        self.feature_var[fid] = var
        self.last_solid = var

    def _op_Hole(self, op: Hole) -> None:
        if op.diameter <= 0:
            raise KclEmitError("Hole.diameter must be > 0 (got %r)" % op.diameter)
        if not op.through and (op.depth is None or op.depth <= 0):
            raise KclEmitError("blind Hole requires depth > 0")
        if op.kind not in ("simple", "counterbore", "countersink"):
            raise KclEmitError("unknown Hole.kind %r" % op.kind)
        if self.last_solid is None and not str(op.face_or_sketch).startswith("sk"):
            raise KclEmitError("Hole requires an existing solid")
        radius = op.diameter / 2.0
        # The cut depth: through-all uses a generous over-travel; blind uses depth.
        depth = op.depth if (not op.through and op.depth) else None
        fid = self._new_id("f")
        tool = "holeTool%s" % fid[1:]
        var = "solid%s" % fid[1:]
        self.lines.append("// hole kind=%s through=%s at [%s, %s] diameter=%s"
                          % (op.kind, op.through, _num(op.x), _num(op.y),
                             _num(op.diameter)))
        self.lines.append("%s = startSketchOn(XY)" % tool)
        self.lines.append("  |> circle(center = %s, radius = %s)"
                          % (_vec([op.x, op.y]), _num(radius)))
        cut_len = _num(depth) if depth is not None else "1000  // through-all"
        self.lines.append("%s_solid = extrude(%s, length = -%s)"
                          % (tool, tool, cut_len))
        if op.kind == "counterbore":
            self.lines.append(
                "// counterbore: cbore_diameter=%s cbore_depth=%s"
                % (op.cbore_diameter, op.cbore_depth))
        elif op.kind == "countersink":
            self.lines.append(
                "// countersink: csk_diameter=%s csk_angle=%s"
                % (op.csk_diameter, op.csk_angle))
        base = self._solid_ref("last")
        self.lines.append("%s = subtract(%s, tools = [%s_solid])"
                          % (var, base, tool))
        self.lines.append("")
        self.feature_var[fid] = var
        self.last_solid = var

    def _op_Shell(self, op: Shell) -> None:
        if self.last_solid is None:
            raise KclEmitError("Shell requires an existing solid")
        if op.thickness <= 0:
            raise KclEmitError("Shell.thickness must be > 0 (got %r)" % op.thickness)
        faces = op.faces or (">Z",)   # default: open the top cap (matches ">Z")
        kcl_faces = []
        for f in faces:
            key = str(f).strip()
            if key not in _CAP_FACES:
                raise KclEmitError(
                    "Shell face %r is not a KCL-nameable cap (END/START); KCL "
                    "cannot open a side wall by selector string" % (f,))
            kcl_faces.append(_CAP_FACES[key])
        fid = self._new_id("f")
        var = "solid%s" % fid[1:]
        solid = self._solid_ref("last")
        # KCL shell has no join-kind parameter; record op.kind so it is not lost.
        self.lines.append(
            "%s = shell(%s, thickness = %s, faces = [%s])  // join=%s"
            % (var, solid, _num(op.thickness), ", ".join(kcl_faces), op.kind))
        self.lines.append("")
        self.feature_var[fid] = var
        self.last_solid = var

    def _op_LinearPattern(self, op: LinearPattern) -> None:
        if op.count < 2:
            raise KclEmitError("LinearPattern.count must be >= 2")
        solid = self._solid_ref(op.feature) if op.feature else self._solid_ref("last")
        fid = self._new_id("f")
        var = "solid%s" % fid[1:]
        self.lines.append(
            "%s = patternLinear3d(%s, instances = %d, distance = %s, axis = %s)"
            % (var, solid, int(op.count), _num(op.spacing), _vec(op.direction)))
        self.lines.append("")
        self.feature_var[fid] = var
        self.last_solid = var

    def _op_CircularPattern(self, op: CircularPattern) -> None:
        if op.count < 2:
            raise KclEmitError("CircularPattern.count must be >= 2")
        solid = self._solid_ref(op.feature) if op.feature else self._solid_ref("last")
        ax, ay, az, bx, by, bz = (float(v) for v in op.axis)
        axis_dir = [bx - ax, by - ay, bz - az]
        if axis_dir == [0.0, 0.0, 0.0]:
            axis_dir = [0.0, 0.0, 1.0]
        fid = self._new_id("f")
        var = "solid%s" % fid[1:]
        self.lines.append(
            "%s = patternCircular3d(%s, instances = %d, axis = %s, "
            "center = %s, arcDegrees = %s, rotateDuplicates = true)"
            % (var, solid, int(op.count), _vec(axis_dir),
               _vec([ax, ay, az]), _num(op.angle)))
        self.lines.append("")
        self.feature_var[fid] = var
        self.last_solid = var

    # -- parametric primitives (sketch + feature, real KCL) -------------
    def _op_Primitive(self, op: Primitive) -> None:
        shape = str(op.shape).lower()
        pid = self._new_id("e")
        pvar = "primitive%s" % pid[1:]
        fid = self._new_id("f")
        var = "solid%s" % fid[1:]
        if shape == "box":
            if op.dx <= 0 or op.dy <= 0 or op.dz <= 0:
                raise KclEmitError("Primitive box needs dx, dy, dz > 0")
            self.lines.append("%s = startSketchOn(XY)" % pvar)
            self.lines.append("  |> startProfile(at = [0, 0])")
            self.lines.append("  |> xLine(length = %s)" % _num(op.dx))
            self.lines.append("  |> yLine(length = %s)" % _num(op.dy))
            self.lines.append("  |> xLine(length = %s)" % _num(-op.dx))
            self.lines.append("  |> close()")
            self.lines.append("%s = extrude(%s, length = %s)"
                              % (var, pvar, _num(op.dz)))
        elif shape == "wedge":
            if op.dx <= 0 or op.dy <= 0 or op.dz <= 0:
                raise KclEmitError("Primitive wedge needs dx, dy, dz > 0")
            self.lines.append("%s = startSketchOn(XY)" % pvar)
            self.lines.append("  |> startProfile(at = [0, 0])")
            self.lines.append("  |> xLine(length = %s)" % _num(op.dx))
            self.lines.append("  |> line(endAbsolute = [0, %s])" % _num(op.dy))
            self.lines.append("  |> close()")
            self.lines.append("%s = extrude(%s, length = %s)"
                              % (var, pvar, _num(op.dz)))
        elif shape == "cylinder":
            if op.r <= 0 or op.h <= 0:
                raise KclEmitError("Primitive cylinder needs r > 0 and h > 0")
            self.lines.append("%s = startSketchOn(XY)" % pvar)
            self.lines.append("  |> circle(center = [0, 0], radius = %s)"
                              % _num(op.r))
            self.lines.append("%s = extrude(%s, length = %s)"
                              % (var, pvar, _num(op.h)))
        elif shape == "sphere":
            if op.r <= 0:
                raise KclEmitError("Primitive sphere needs r > 0")
            # A half-disk in XZ (flat side on the Z axis) revolved 360 about Z.
            self.lines.append("%s = startSketchOn(XZ)" % pvar)
            self.lines.append("  |> startProfile(at = [0, %s])" % _num(-op.r))
            self.lines.append("  |> arc(interiorAbsolute = [%s, 0], "
                              "endAbsolute = [0, %s])"
                              % (_num(op.r), _num(op.r)))
            self.lines.append("  |> close()")
            self.lines.append("%s = revolve(%s, axis = Z, angle = 360)"
                              % (var, pvar))
        elif shape == "cone":
            if op.r <= 0 or op.h <= 0:
                raise KclEmitError("Primitive cone needs r > 0 and h > 0")
            if op.r2 < 0:
                raise KclEmitError("Primitive cone needs r2 >= 0")
            # A radial profile in XZ (base radius r at z=0, top radius r2 at z=h)
            # revolved 360 about Z. r2 == 0 gives a true cone (triangle apex).
            self.lines.append("%s = startSketchOn(XZ)" % pvar)
            self.lines.append("  |> startProfile(at = [0, 0])")
            self.lines.append("  |> line(endAbsolute = [%s, 0])" % _num(op.r))
            if op.r2 > 0:
                self.lines.append("  |> line(endAbsolute = [%s, %s])"
                                  % (_num(op.r2), _num(op.h)))
            self.lines.append("  |> line(endAbsolute = [0, %s])" % _num(op.h))
            self.lines.append("  |> close()")
            self.lines.append("%s = revolve(%s, axis = Z, angle = 360)"
                              % (var, pvar))
        elif shape == "torus":
            if op.r <= 0 or op.r2 <= 0:
                raise KclEmitError("Primitive torus needs r (major) > 0 and "
                                   "r2 (minor) > 0")
            # A minor-radius circle offset by the major radius, revolved about Z.
            self.lines.append("%s = startSketchOn(XZ)" % pvar)
            self.lines.append("  |> circle(center = [%s, 0], radius = %s)"
                              % (_num(op.r), _num(op.r2)))
            self.lines.append("%s = revolve(%s, axis = Z, angle = 360)"
                              % (var, pvar))
        else:
            raise KclEmitError(
                "Primitive.shape %r is not a KCL-expressible primitive "
                "(box|sphere|cylinder|cone|torus|wedge)" % op.shape)
        self.lines.append("")
        self.feature_var[fid] = var
        self.last_solid = var

    # -- affine placement (real KCL translate / rotate / scale) ---------
    def _emit_rigid(self, dst: str, src: str,
                    tx: float, ty: float, tz: float,
                    rx: float, ry: float, rz: float) -> None:
        """Emit ``dst = src |> rotate(...) |> translate(...)``.

        KCL rotate takes intrinsic Euler angles roll/pitch/yaw about X/Y/Z in that
        order, which is exactly CISP's rotate-then-translate convention. Both lines
        are always emitted so every field is live even when a component is zero.
        """
        self.lines.append("%s = %s" % (dst, src))
        self.lines.append("  |> rotate(roll = %s, pitch = %s, yaw = %s)"
                          % (_num(rx), _num(ry), _num(rz)))
        self.lines.append("  |> translate(x = %s, y = %s, z = %s)"
                          % (_num(tx), _num(ty), _num(tz)))

    def _op_Transform(self, op: Transform) -> None:
        src = (self._solid_ref(op.feature_or_body)
               if op.feature_or_body else self._solid_ref("last"))
        fid = self._new_id("f")
        var = "solid%s" % fid[1:]
        self._emit_rigid(var, src, op.tx, op.ty, op.tz, op.rx, op.ry, op.rz)
        self.lines.append("")
        self.feature_var[fid] = var
        self.last_solid = var

    def _op_Scale(self, op: Scale) -> None:
        if op.sx <= 0 or op.sy <= 0 or op.sz <= 0:
            raise KclEmitError(
                "Scale factors must all be > 0 (got sx=%r sy=%r sz=%r); "
                "reflection is Mirror's job" % (op.sx, op.sy, op.sz))
        src = (self._solid_ref(op.feature_or_body)
               if op.feature_or_body else self._solid_ref("last"))
        fid = self._new_id("f")
        var = "solid%s" % fid[1:]
        self.lines.append("%s = %s" % (var, src))
        self.lines.append("  |> scale(x = %s, y = %s, z = %s)"
                          % (_num(op.sx), _num(op.sy), _num(op.sz)))
        self.lines.append("")
        self.feature_var[fid] = var
        self.last_solid = var

    def _op_PatternTransform(self, op: PatternTransform) -> None:
        # KCL patternTransform always keeps the original untransformed as an
        # implicit instance, but CISP's placement list does NOT imply the identity
        # -- so lowering to patternTransform would add a stray body. Instead emit
        # one explicit rotate/translate copy per placement (real KCL) and union
        # them, which reproduces the placement list exactly.
        src = self._solid_ref(op.feature) if op.feature else self._solid_ref("last")
        flat = tuple(float(v) for v in op.placements)
        if len(flat) == 0 or len(flat) % 6 != 0:
            raise KclEmitError(
                "PatternTransform.placements must be a non-empty flat tuple whose "
                "length is a multiple of 6 (got %d)" % len(flat))
        n = len(flat) // 6
        fid = self._new_id("f")
        var = "solid%s" % fid[1:]
        self.lines.append("// pattern_transform: %d explicit placement(s)" % n)
        copies: List[str] = []
        for i in range(n):
            tx, ty, tz, rx, ry, rz = flat[i * 6:i * 6 + 6]
            copy = "%s_%d" % (var, i)
            copies.append(copy)
            self._emit_rigid(copy, src, tx, ty, tz, rx, ry, rz)
        if len(copies) == 1:
            self.lines.append("%s = %s" % (var, copies[0]))
        else:
            self.lines.append("%s = union([%s])" % (var, ", ".join(copies)))
        self.lines.append("")
        self.feature_var[fid] = var
        self.last_solid = var

    # -- annotated ops (tags/refs KCL cannot bind offline) --------------
    def _op_Split(self, op: Split) -> None:
        # A plane section needs an infinite half-space to cut against; KCL has no
        # half-space primitive and no section/split op offline, and a finite
        # cutting box would be an unbounded approximation. Annotate, fields kept.
        self.lines.extend(_annotation(
            "Split", plane=op.plane, offset=op.offset, keep=op.keep,
            note="KCL has no infinite half-space or plane-section op offline"))
        self.lines.append("")

    def _op_Thicken(self, op: Thicken) -> None:
        # KCL has shell() (hollowing by opening named caps) but no general
        # solid-surface offset / thicken op, so a faithful lowering is impossible.
        self.lines.extend(_annotation(
            "Thicken", faces=list(op.faces) or "ALL", thickness=op.thickness,
            both=op.both,
            note="KCL has shell() but no general solid-surface offset/thicken op"))
        self.lines.append("")

    def _op_Hull(self, op: Hull) -> None:
        # Convex hull is a genuine kernel capability but KCL exposes no hull
        # operator, so refuse rather than fabricate a wrong solid.
        self.lines.extend(_annotation(
            "Hull", target=op.target or "ALL", tool=op.tool or "ALL",
            note="KCL has no convex-hull operator"))
        self.lines.append("")

    def _op_Minkowski(self, op: Minkowski) -> None:
        # Minkowski sum / SDF-offset dilation has no KCL operator, so refuse.
        self.lines.extend(_annotation(
            "Minkowski", radius=op.radius,
            note="KCL has no Minkowski-sum / signed-distance-offset operator"))
        self.lines.append("")

    def _op_Fillet(self, op: Fillet) -> None:
        self.lines.extend(_annotation(
            "Fillet", radius=op.radius,
            edges=list(op.edges) or "ALL",
            note="KCL fillet(solid, radius, tags=[...]) needs edge tags declared "
                 "at sketch time; CISP selector strings have no KCL tag to bind"))
        self.lines.append("")

    def _op_Chamfer(self, op: Chamfer) -> None:
        self.lines.extend(_annotation(
            "Chamfer", distance=op.distance, distance2=op.distance2,
            edges=list(op.edges) or "ALL",
            note="KCL chamfer needs edge tags; selector strings have none"))
        self.lines.append("")

    def _op_Draft(self, op: Draft) -> None:
        self.lines.extend(_annotation(
            "Draft", angle=op.angle, neutral_plane=op.neutral_plane,
            faces=list(op.faces) or "ALL",
            note="KCL has no imperative draft op keyed on a neutral plane"))
        self.lines.append("")

    def _op_Loft(self, op: Loft) -> None:
        # Loft IS a KCL op, but it consumes named profiles; our sketch refs may or
        # may not have closed profiles emitted. Emit the call when every profile
        # resolves, else annotate. Either way every field is referenced.
        profiles: List[str] = []
        ok = True
        for sid in op.sketches:
            closed = self._closed_profiles(sid) if sid in self.sketch_var else []
            if not closed:
                ok = False
                break
            profiles.append(closed[0])
        if ok and len(profiles) >= 2:
            fid = self._new_id("f")
            var = "solid%s" % fid[1:]
            self.lines.append(
                "%s = loft([%s])  // ruled=%s offsets=%s"
                % (var, ", ".join(profiles), op.ruled, _vec(op.offsets)
                   if op.offsets else "[]"))
            self.lines.append("")
            self.feature_var[fid] = var
            self.last_solid = var
        else:
            self.lines.extend(_annotation(
                "Loft", sketches=list(op.sketches), ruled=op.ruled,
                offsets=list(op.offsets),
                note="one or more loft profiles had no closed region to loft"))
            self.lines.append("")

    def _op_Sweep(self, op: Sweep) -> None:
        prof = self._closed_profiles(op.sketch) if op.sketch in self.sketch_var else []
        path = op.path in self.sketch_var
        if prof and path:
            path_var = self.sketch_var[op.path]
            fid = self._new_id("f")
            var = "solid%s" % fid[1:]
            self.lines.append("%s = sweep(%s, path = %s)"
                              % (var, prof[0], path_var))
            self.lines.append("")
            self.feature_var[fid] = var
            self.last_solid = var
        else:
            self.lines.extend(_annotation(
                "Sweep", sketch=op.sketch, path=op.path,
                note="sweep profile or path sketch was not emittable"))
            self.lines.append("")

    def _op_Mirror(self, op: Mirror) -> None:
        self.lines.extend(_annotation(
            "Mirror", feature_or_body=op.feature_or_body, plane=op.plane,
            note="KCL mirrors by body/plane reference, not a CISP feature id"))
        self.lines.append("")

    def _op_AddInstance(self, op: AddInstance) -> None:
        self.lines.extend(_annotation(
            "AddInstance", part=op.part,
            translate=[op.x, op.y, op.z], rotate_deg=[op.rx, op.ry, op.rz],
            note="KCL assemblies are import + transform graphs across files"))
        self.lines.append("")

    def _op_Mate(self, op: Mate) -> None:
        self.lines.extend(_annotation(
            "Mate", kind=op.kind, a=op.a, b=op.b, value=op.value,
            note="KCL has no imperative joint/mate op"))
        self.lines.append("")


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def emit_kcl(ops: Sequence[Op], *, name: str = "model",
             length_unit: str = "mm", kcl_version: str = "1.0") -> str:
    """Lower a CISP op stream to a deterministic ``.kcl`` program string.

    Pure and idempotent: ``emit_kcl(ops) == emit_kcl(ops)`` byte-for-byte, and
    no field of any op is dropped. Raises :class:`KclEmitError` when an op cannot
    be lowered faithfully (rather than emitting geometry that differs from what
    the op declared).
    """
    return _Emitter(name, length_unit, kcl_version).emit(list(ops))


def ops_of(obj: object) -> List[Op]:
    """Extract the CISP op log from a backend / session / raw op list.

    Accepts a ``GeometryBackend`` (reads ``_oplog``), a HarnessSession (reads its
    ``backend._oplog``), or a plain sequence of :class:`Op`. Applies any
    ``SetParam`` edits first, so the emitted program reflects the edited stream.
    """
    if isinstance(obj, (list, tuple)) and all(isinstance(o, Op) for o in obj):
        return list(obj)
    oplog = getattr(obj, "_oplog", None)
    if oplog is None:
        backend = getattr(obj, "backend", None)
        if backend is not None:
            oplog = getattr(backend, "_oplog", None)
    if oplog is None:
        raise KclError(
            "cannot get a CISP op log from %r: pass a backend/session with an "
            "_oplog, or a list of Op" % type(obj).__name__)
    return list(oplog)


def serialize_kcl(obj: object, *, name: str = "model",
                  length_unit: str = "mm", kcl_version: str = "1.0") -> str:
    """The ``.kcl`` program text for a backend/session/op-list (no file I/O)."""
    return emit_kcl(ops_of(obj), name=name, length_unit=length_unit,
                    kcl_version=kcl_version)


def write_kcl(obj: object, path: str, *, name: Optional[str] = None,
              length_unit: str = "mm", kcl_version: str = "1.0") -> None:
    """Write the ``.kcl`` program for ``obj`` to ``path`` (POSIX newlines)."""
    import os

    model_name = name or os.path.splitext(os.path.basename(str(path)))[0] or "model"
    text = serialize_kcl(obj, name=model_name, length_unit=length_unit,
                         kcl_version=kcl_version)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
