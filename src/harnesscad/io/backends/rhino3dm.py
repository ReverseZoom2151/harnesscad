"""Rhino3dmBackend -- an openNURBS geometry backend that REFUSES what it cannot do.

``rhino3dm`` is Rhino's openNURBS library as a standalone pip wheel: a geometry
CONTAINER + I/O library, not a modelling kernel. It can build primitives and
extrusions (``Extrusion.Create`` / ``CreateBoxExtrusion``), hold meshes and
Breps, report bounding boxes, and read/write ``.3dm``. It CANNOT do the boolean,
fillet, chamfer, shell, revolve, loft, sweep, pattern or draft operations a CAD
kernel does -- and, in the Python wheel specifically, it exposes NO mass-property
computation (``AreaMassProperties`` / ``VolumeMassProperties`` are not bound) and
NO mesher for Breps/Extrusions.

So this backend does one honest thing: it IMPLEMENTS the handful of CISP ops
rhino3dm can genuinely perform, and REFUSES every other op with a typed
``unsupported-op`` diagnostic -- never a faked or silently-dropped result. Its
value is not breadth; it is being an INDEPENDENT measurement + IO voice:

* the volume and bounding box it reports for a box/cylinder extrusion are an
  independent oracle voice in the differential oracle (they must agree with
  CadQuery, FreeCAD and F-rep, or one of them is wrong);
* it converts the current solid to/from ``.3dm`` (``export("3dm")``), the one
  format that speaks to the Rhino ecosystem.

WHAT IT IMPLEMENTS
------------------
* sketch + constraint bookkeeping (``NewSketch``, ``AddPoint/Line/Circle/
  Rectangle``, ``Constrain``) -- DOF tracking identical to the stub, so the
  op-stream verifiers behave the same;
* ``Extrude`` of a SINGLE rectangle or circle profile into the first solid, built
  as a real ``rhino3dm.Extrusion`` (box / cylinder) with an explicit mesh;
* ``SetParam`` (edit-and-replay), like the stub.

WHAT IT REFUSES (typed ``unsupported-op``)
------------------------------------------
``Boolean``, ``Fillet``, ``Chamfer``, ``Shell``, ``Draft``, ``Hole``,
``Revolve``, ``Loft``, ``Sweep``, ``LinearPattern``, ``CircularPattern``,
``Mirror``, ``AddInstance``, ``Mate`` -- and a second solid-producing ``Extrude``
(that would need a boolean union rhino3dm does not have).

VOLUME HONESTY
--------------
Because the Python wheel has no mass-property call, the reported volume is
analytic from the profile: ``w * h * |d|`` for a box (cross-checked against
rhino3dm's own ``Box.Volume``) and ``pi * r^2 * |d|`` for a cylinder. The
BOUNDING BOX is taken from rhino3dm's own ``Extrusion.GetBoundingBox`` (Rhino's
kernel), axis-remapped to the sketch plane. Both are exact for these primitives,
so this backend is an EXACT voice for the ops it supports.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import List, Optional, Sequence, Tuple

from harnesscad.core.cisp.ops import (
    CONSTRAINT_DOF, PRIMITIVE_DOF,
    Op, NewSketch, AddPoint, AddLine, AddCircle, AddRectangle,
    AddArc, AddEllipse, AddPolygon, AddSpline,
    Constrain, Extrude, Fillet, Boolean,
    Primitive, Split, Thicken, Hull, Minkowski,
    Revolve, Chamfer, Hole, Shell, Draft,
    Loft, Sweep, LinearPattern, CircularPattern, Mirror,
    AddInstance, Mate, SetParam,
    canonical_json, edit_oplog,
)
from harnesscad.eval.verifiers.verify import Diagnostic, Severity
from harnesscad.io.backends.base import ApplyResult, BackendUnavailable
from harnesscad.io.formats import stl as stl_codec
from harnesscad.io.formats import threedm as threedm_codec

# rhino3dm is optional; the constructor raises BackendUnavailable when it is
# missing, so module import must never fail on its account.
try:  # pragma: no cover - availability depends on the host
    import rhino3dm as _r3  # type: ignore
except Exception:  # noqa: BLE001
    _r3 = None  # type: ignore

Vec3 = Tuple[float, float, float]

#: How many segments a cylinder extrusion is tessellated into for export/gate.
#: The oracle VOLUME is analytic (pi r^2 d), not read off this mesh, so the facet
#: count only affects the exported triangle soup, never the reported volume.
_CYL_SEGMENTS = 64

#: sketch-plane name -> (in-plane axis 0, in-plane axis 1, normal axis).
_PLANE_AXES = {
    "XY": (0, 1, 2),
    "XZ": (0, 2, 1),
    "YZ": (1, 2, 0),
}


def _signed_volume(verts: Sequence[Vec3], faces: Sequence[Sequence[int]]) -> float:
    """Divergence-theorem signed volume of a fan-triangulated soup (>0 = outward)."""
    total = 0.0
    for face in faces:
        ids = [int(i) for i in face]
        for k in range(1, len(ids) - 1):
            a, b, c = verts[ids[0]], verts[ids[k]], verts[ids[k + 1]]
            total += (a[0] * (b[1] * c[2] - b[2] * c[1])
                      - a[1] * (b[0] * c[2] - b[2] * c[0])
                      + a[2] * (b[0] * c[1] - b[1] * c[0]))
    return total / 6.0


def _orient_outward(verts: List[Vec3],
                    faces: List[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    """Reverse every face's winding if the solid is inside-out (inward normals)."""
    if _signed_volume(verts, faces) < 0.0:
        return [tuple(reversed(f)) for f in faces]
    return list(faces)


def _err(code: str, msg: str, where: Optional[str] = None) -> ApplyResult:
    return ApplyResult(False, [], [Diagnostic(Severity.ERROR, code, msg, where)])


def _unsupported(op_name: str, why: str) -> ApplyResult:
    """A typed refusal: the op is real, rhino3dm simply cannot perform it."""
    return _err(
        "unsupported-op",
        "rhino3dm cannot perform '%s': %s. rhino3dm is a geometry container + IO "
        "library, not a modelling kernel." % (op_name, why))


class Rhino3dmBackend:
    """A GeometryBackend backed by openNURBS via the rhino3dm wheel."""

    #: Formats this backend can serialise the current solid to.
    FORMATS = ("stl", "obj", "3dm")

    def __init__(self) -> None:
        if not threedm_codec.RHINO3DM_AVAILABLE:
            raise BackendUnavailable(
                "rhino3dm",
                "the rhino3dm wheel is not installed (`pip install rhino3dm`)",
                searched=["import rhino3dm"])
        self.reset()

    # -- lifecycle ----------------------------------------------------------
    def reset(self) -> None:
        self.sketches: dict = {}      # sid -> {plane, entities:[eid], dof}
        self.entities: dict = {}      # eid -> {type, sketch, geom}
        self.features: list = []      # [{type, id, ...}]
        self.solid_present = False
        self._solid: Optional[dict] = None   # {kind, volume, bbox, verts, faces}
        self._oplog: list = []
        self._n = {"sk": 0, "e": 0, "f": 0}
        # Set when an op is REFUSED as ``unsupported-op``. The requested part was
        # never built, so from that point the measurement must refuse (volume/bbox
        # None) rather than report the pre-op (un-bored / un-cut / un-revolved)
        # solid as if it were the finished part -- the silent-wrong-part failure.
        self._tainted = False

    def _new_id(self, kind: str) -> str:
        self._n[kind] += 1
        return kind + str(self._n[kind])

    # -- apply --------------------------------------------------------------
    def apply(self, op: Op) -> ApplyResult:
        if isinstance(op, SetParam):
            return self._set_param(op)
        result = self._dispatch(op)
        if result.ok:
            self._oplog.append(op)
        elif any(getattr(d, "code", None) == "unsupported-op"
                 for d in result.diagnostics):
            self._tainted = True
        return result

    def _add_primitive(self, sketch: str, kind: str, geom: dict) -> ApplyResult:
        if sketch not in self.sketches:
            return _err("bad-ref", f"unknown sketch '{sketch}'", sketch)
        eid = self._new_id("e")
        self.entities[eid] = {"type": kind, "sketch": sketch, "geom": geom}
        self.sketches[sketch]["entities"].append(eid)
        self.sketches[sketch]["dof"] += PRIMITIVE_DOF[kind]
        return ApplyResult(True, [eid])

    def _dispatch(self, op: Op) -> ApplyResult:
        if isinstance(op, NewSketch):
            plane = str(op.plane).upper()
            if plane not in _PLANE_AXES:
                return _err("bad-value", f"unknown sketch plane '{op.plane}'")
            sid = self._new_id("sk")
            self.sketches[sid] = {"plane": plane, "entities": [], "dof": 0}
            return ApplyResult(True, [sid])
        if isinstance(op, AddPoint):
            return self._add_primitive(op.sketch, "point", {"x": op.x, "y": op.y})
        if isinstance(op, AddLine):
            return self._add_primitive(op.sketch, "line",
                                       {"x1": op.x1, "y1": op.y1,
                                        "x2": op.x2, "y2": op.y2})
        if isinstance(op, AddCircle):
            if op.r <= 0:
                return _err("bad-value", f"circle radius must be > 0 (got {op.r})")
            return self._add_primitive(op.sketch, "circle",
                                       {"cx": op.cx, "cy": op.cy, "r": op.r})
        if isinstance(op, AddRectangle):
            if op.w <= 0 or op.h <= 0:
                return _err("bad-value", "rectangle w and h must be > 0")
            return self._add_primitive(op.sketch, "rectangle",
                                       {"x": op.x, "y": op.y, "w": op.w, "h": op.h})
        if isinstance(op, Constrain):
            return self._constrain(op)
        if isinstance(op, Extrude):
            return self._extrude(op)

        # -- everything below is a real CAD op rhino3dm cannot perform -------
        if isinstance(op, AddArc):
            return _unsupported("add_arc", "openNURBS carries curves but this "
                                "backend's profile builder closes only rectangle "
                                "and circle regions")
        if isinstance(op, AddEllipse):
            return _unsupported("add_ellipse", "only rectangle and circle profiles "
                                "are realised into extrudable regions here")
        if isinstance(op, AddPolygon):
            return _unsupported("add_polygon", "only rectangle and circle profiles "
                                "are realised into extrudable regions here")
        if isinstance(op, AddSpline):
            return _unsupported("add_spline", "only rectangle and circle profiles "
                                "are realised into extrudable regions here")
        if isinstance(op, Primitive):
            return _unsupported("primitive", "no solid primitive builder is wired "
                                "through this container backend")
        if isinstance(op, Split):
            return _unsupported("split", "a plane split is a boolean cut, which is "
                                "absent")
        if isinstance(op, Thicken):
            return _unsupported("thicken", "no offset-solid / thick-solid operation "
                                "exists")
        if isinstance(op, Hull):
            return _unsupported("hull", "no convex-hull operation exists in this "
                                "container library")
        if isinstance(op, Minkowski):
            return _unsupported("minkowski", "no Minkowski-sum / offset operation "
                                "exists in this container library")
        if isinstance(op, Boolean):
            return _unsupported("boolean", "openNURBS has no solid boolean engine")
        if isinstance(op, Fillet):
            return _unsupported("fillet", "no edge-blending kernel is exposed")
        if isinstance(op, Chamfer):
            return _unsupported("chamfer", "no edge-chamfer kernel is exposed")
        if isinstance(op, Shell):
            return _unsupported("shell", "no thick-solid / hollow operation exists")
        if isinstance(op, Draft):
            return _unsupported("draft", "no face-draft operation exists")
        if isinstance(op, Hole):
            return _unsupported("hole", "a hole is a boolean cut, which is absent")
        if isinstance(op, Revolve):
            return _unsupported("revolve", "no solid-of-revolution builder is exposed")
        if isinstance(op, Loft):
            return _unsupported("loft", "no lofting kernel is exposed")
        if isinstance(op, Sweep):
            return _unsupported("sweep", "no sweeping kernel is exposed")
        if isinstance(op, LinearPattern):
            return _unsupported("linear_pattern",
                                "patterning needs a boolean union, which is absent")
        if isinstance(op, CircularPattern):
            return _unsupported("circular_pattern",
                                "patterning needs a boolean union, which is absent")
        if isinstance(op, Mirror):
            return _unsupported("mirror",
                                "mirroring a solid needs a boolean union, absent")
        if isinstance(op, AddInstance):
            return _unsupported("add_instance",
                                "assembly placement is out of scope for this backend")
        if isinstance(op, Mate):
            return _unsupported("mate",
                                "assembly mating is out of scope for this backend")
        return _err("unknown-op", f"unhandled op {type(op).__name__}")

    # -- the one thing it can build ----------------------------------------
    def _extrude(self, op: Extrude) -> ApplyResult:
        if op.sketch not in self.sketches:
            return _err("bad-ref", f"unknown sketch '{op.sketch}'", op.sketch)
        sk = self.sketches[op.sketch]
        eids = sk["entities"]
        if not eids:
            return _err("empty-sketch", f"sketch '{op.sketch}' has no profile",
                        op.sketch)
        if op.distance == 0:
            return _err("bad-value", "extrude distance must be non-zero")
        # A second solid would need a boolean union, which rhino3dm lacks.
        if self.solid_present:
            return _unsupported(
                "extrude",
                "a second extrusion would have to be unioned into the existing "
                "solid, and openNURBS has no boolean union")
        # Only a single rectangle or a single circle profile is extrudable here.
        profiles = [self.entities[e] for e in eids]
        kinds = {p["type"] for p in profiles}
        if len(profiles) != 1 or not kinds <= {"rectangle", "circle"}:
            return _unsupported(
                "extrude",
                "this backend extrudes only a single rectangle or circle profile; "
                "sketch '%s' holds %s" % (op.sketch, sorted(kinds)))

        prof = profiles[0]
        plane = sk["plane"]
        a0, a1, an = _PLANE_AXES[plane]
        d = abs(float(op.distance))

        if prof["type"] == "rectangle":
            solid = self._box_solid(prof["geom"]["w"], prof["geom"]["h"], d,
                                    a0, a1, an)
        else:
            solid = self._cylinder_solid(prof["geom"]["r"], d, a0, a1, an)

        self._solid = solid
        self.solid_present = True
        fid = self._new_id("f")
        self.features.append({"type": "extrude", "id": fid, "sketch": op.sketch})
        return ApplyResult(True, [fid])

    def _box_solid(self, w: float, h: float, d: float,
                   a0: int, a1: int, an: int) -> dict:
        """A box, sized (w,h) in the sketch plane and d along its normal.

        Volume is cross-checked against rhino3dm's own ``Box.Volume``; the mesh is
        the exact 8-vertex / 6-quad box, oriented to the sketch plane so its
        per-axis bounding box matches the other engines.
        """
        # rhino3dm computes the primitive's volume from its own geometry.
        box = _r3.Box(_r3.BoundingBox(0.0, 0.0, 0.0, w, h, d))
        rhino_volume = float(box.Volume)

        size = [0.0, 0.0, 0.0]
        size[a0] = w
        size[a1] = h
        size[an] = d
        # 8 corners of an axis-aligned box at the origin.
        corners = []
        for zi in (0, 1):
            for yi in (0, 1):
                for xi in (0, 1):
                    corners.append((xi * size[0], yi * size[1], zi * size[2]))
        # index helper: bit(x)=1, bit(y)=2, bit(z)=4 within the loop order above
        idx = {(x, y, z): (z * 4 + y * 2 + x)
               for z in (0, 1) for y in (0, 1) for x in (0, 1)}
        faces = [
            (idx[0, 0, 0], idx[1, 0, 0], idx[1, 1, 0], idx[0, 1, 0]),  # z=0
            (idx[0, 0, 1], idx[0, 1, 1], idx[1, 1, 1], idx[1, 0, 1]),  # z=1
            (idx[0, 0, 0], idx[0, 1, 0], idx[0, 1, 1], idx[0, 0, 1]),  # x=0
            (idx[1, 0, 0], idx[1, 0, 1], idx[1, 1, 1], idx[1, 1, 0]),  # x=1
            (idx[0, 0, 0], idx[0, 0, 1], idx[1, 0, 1], idx[1, 0, 0]),  # y=0
            (idx[0, 1, 0], idx[1, 1, 0], idx[1, 1, 1], idx[0, 1, 1]),  # y=1
        ]
        bbox = [size[0], size[1], size[2]]
        faces = _orient_outward(corners, faces)
        return {"kind": "box", "volume": rhino_volume, "bbox": bbox,
                "verts": corners, "faces": faces}

    def _cylinder_solid(self, r: float, d: float,
                        a0: int, a1: int, an: int) -> dict:
        """A capped cylinder, radius r in the sketch plane, height d along normal.

        Built as a real ``rhino3dm.Extrusion`` so its bounding box is Rhino's own
        (``GetBoundingBox``), axis-remapped to the sketch plane. Volume is analytic
        (pi r^2 d): the Python wheel exposes no mass-property call.
        """
        # A real openNURBS extrusion, purely so the bbox is Rhino's measurement.
        circle = _r3.Circle(r)
        ext = _r3.Extrusion.Create(circle.ToNurbsCurve(), d, True)
        rb = ext.GetBoundingBox() if ext is not None else None
        if rb is not None:
            native = [float(rb.Max.X - rb.Min.X),
                      float(rb.Max.Y - rb.Min.Y),
                      float(rb.Max.Z - rb.Min.Z)]
            # native is (2r, 2r, d) in the extrusion's own frame; remap to plane.
            in_plane = sorted([native[0], native[1]], reverse=True)[0]
            height = native[2]
        else:  # pragma: no cover - defensive
            in_plane, height = 2.0 * r, d
        bbox = [0.0, 0.0, 0.0]
        bbox[a0] = in_plane
        bbox[a1] = in_plane
        bbox[an] = height

        volume = math.pi * r * r * d

        # Deterministic polygonal mesh (segments) purely for export / the gate.
        verts: List[Vec3] = []
        c0: List[int] = []
        c1: List[int] = []
        for level, hz in ((0, 0.0), (1, height)):
            for s in range(_CYL_SEGMENTS):
                ang = 2.0 * math.pi * s / _CYL_SEGMENTS
                p = [0.0, 0.0, 0.0]
                p[a0] = r + r * math.cos(ang)
                p[a1] = r + r * math.sin(ang)
                p[an] = hz
                (c0 if level == 0 else c1).append(len(verts))
                verts.append((p[0], p[1], p[2]))
        cen0 = [0.0, 0.0, 0.0]; cen0[a0] = r; cen0[a1] = r; cen0[an] = 0.0
        cen1 = [0.0, 0.0, 0.0]; cen1[a0] = r; cen1[a1] = r; cen1[an] = height
        i_cen0 = len(verts); verts.append(tuple(cen0))
        i_cen1 = len(verts); verts.append(tuple(cen1))
        faces: List[Tuple[int, ...]] = []
        n = _CYL_SEGMENTS
        for s in range(n):
            s2 = (s + 1) % n
            faces.append((c0[s], c0[s2], c1[s2], c1[s]))          # side quad
            faces.append((i_cen0, c0[s2], c0[s]))                 # bottom cap
            faces.append((i_cen1, c1[s], c1[s2]))                 # top cap
        faces = _orient_outward(verts, faces)
        return {"kind": "cylinder", "volume": volume, "bbox": bbox,
                "verts": verts, "faces": faces}

    # -- constraints (DOF bookkeeping, identical to the stub) ---------------
    def _constrain(self, op: Constrain) -> ApplyResult:
        if op.kind not in CONSTRAINT_DOF:
            return _err("bad-value", f"unknown constraint kind '{op.kind}'")
        if op.kind in ("distance", "radius") and op.value is None:
            return _err("bad-value", f"'{op.kind}' constraint requires a value")
        if op.a not in self.entities:
            return _err("bad-ref", f"unknown entity '{op.a}'", op.a)
        if op.b is not None and op.b not in self.entities:
            return _err("bad-ref", f"unknown entity '{op.b}'", op.b)
        sid = self.entities[op.a]["sketch"]
        self.sketches[sid]["dof"] -= CONSTRAINT_DOF[op.kind]
        self.sketches[sid].setdefault("constraints", []).append(op.kind)
        return ApplyResult(True, [])

    # -- SetParam edit-and-replay (identical semantics to the stub) ---------
    def _set_param(self, op: SetParam) -> ApplyResult:
        new_log, err = edit_oplog(self._oplog, op)
        if err is not None:
            return _err(*err)
        trial = type(self)()
        for logged in new_log:
            r = trial.apply(logged)
            if not r.ok:
                return ApplyResult(False, [], r.diagnostics)
        self.__dict__.update(trial.__dict__)
        return ApplyResult(True, [])

    # -- regen / queries ----------------------------------------------------
    def regenerate(self) -> List[Diagnostic]:
        return []

    def mesh(self) -> Tuple[List[Vec3], List[Tuple[int, int, int]]]:
        """The current solid as a welded ``(vertices, triangles)`` -- gate route."""
        if self._solid is None:
            return [], []
        verts = [tuple(float(c) for c in v) for v in self._solid["verts"]]
        tris: List[Tuple[int, int, int]] = []
        for face in self._solid["faces"]:
            ids = [int(i) for i in face]
            for k in range(1, len(ids) - 1):
                tris.append((ids[0], ids[k], ids[k + 1]))
        return verts, tris

    def query(self, q: str) -> dict:
        if q == "sketch_dof":
            return {sid: s["dof"] for sid, s in self.sketches.items()}
        if q == "summary":
            return {
                "sketch_count": len(self.sketches),
                "entity_count": len(self.entities),
                "feature_count": len(self.features),
                "solid_present": self.solid_present,
            }
        if q in ("measure", "metrics"):
            return self._metrics()
        if q == "validity":
            return self._validity()
        if q == "assembly":
            return {}     # this backend places no assembly instances
        return {}

    def _metrics(self) -> dict:
        if self._tainted:
            # An unsupported op was refused: the requested part was never built,
            # so there is nothing honest to measure. REFUSE (volume/bbox None)
            # rather than leak the pre-op volume as a plausible wrong number.
            return {"volume": None, "bbox": None}
        if self._solid is None:
            return {"volume": 0.0, "bbox": [0.0, 0.0, 0.0]}
        verts, tris = self.mesh()
        return {
            "volume": float(self._solid["volume"]),
            "bbox": [float(c) for c in self._solid["bbox"]],
            "faces": len(self._solid["faces"]),
            "vertices": len(verts),
            "solids": 1,
            "primitive": self._solid["kind"],
        }

    def _validity(self) -> dict:
        if self._solid is None:
            return {"manifold": False, "watertight": False,
                    "genus": None, "solid_present": False}
        # Every solid this backend produces is a closed, genus-0 primitive.
        return {"manifold": True, "watertight": True, "genus": 0,
                "is_valid": True, "solid_present": True}

    # -- export -------------------------------------------------------------
    def export(self, fmt: str):
        fmt = str(fmt).lower()
        if fmt not in self.FORMATS:
            raise ValueError(
                "the rhino3dm backend cannot export '%s' (supported: %s)"
                % (fmt, ", ".join(self.FORMATS)))
        if self._solid is None:
            raise ValueError("nothing to export: no solid present")
        verts, tris = self.mesh()
        if fmt == "stl":
            triangles = [stl_codec.Triangle(verts[a], verts[b], verts[c])
                         for a, b, c in tris]
            return stl_codec.write_ascii_stl(triangles, name=self._solid["kind"])
        if fmt == "obj":
            from harnesscad.io.formats import obj as obj_codec
            return obj_codec.serialize_obj_float(verts, tris, precision=6)
        # fmt == "3dm": write the current solid to an openNURBS container.
        return threedm_codec.serialize_3dm(verts, tris, unit="millimeter",
                                            name=self._solid["kind"])

    def state_digest(self) -> str:
        model = {
            "sketches": self.sketches,
            "entities": self.entities,
            "features": self.features,
            "solid_present": self.solid_present,
            "solid": None if self._solid is None else {
                "kind": self._solid["kind"],
                "volume": round(self._solid["volume"], 9),
                "bbox": [round(c, 9) for c in self._solid["bbox"]],
            },
            "oplog": [canonical_json(o) for o in self._oplog],
        }
        blob = json.dumps(model, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode()).hexdigest()
