"""The output gate -- the one door every artifact leaves the harness through.

The harness cannot promise it will always PRODUCE a part: a model may simply
fail, a field may be empty, a boolean may annihilate its target. What it CAN
promise is that it never SHIPS a wrong one. The invariant this module enforces
is *soundness*, not completeness::

    Every artifact leaving the harness is either
        (a) verified valid, or
        (b) refused with a reason.
    There is no third outcome. Silence is not success.

Why this exists: the F-rep ``shell`` used to DILATE the part instead of hollowing
it (Curv's two-sided ``|f| - t/2``). A 60x40x20 box shelled at t=3 came out
63x43x23 -- 3 mm oversize in every dimension, watertight, beautifully rendered,
and *no verifier complained*, because nothing ever compared the built geometry
against the declared intent. The op fix (``shell_inward``) closes that one bug.
This gate closes the *class* of bug: the harness now measures what it actually
built, compares it against what it was asked to build, and refuses to write the
file when the two disagree.

Two families of check:

*   **MEASURED** -- read off the built geometry alone, no intent needed:
    non-empty, no degenerate faces, watertight, 2-manifold, outward normals,
    non-degenerate volume, finite/sane bounding box, and (when the mesh is
    small enough for it to be cheap) no self-intersection.

*   **DECLARED** -- the op stream states an intent; the geometry must honour it.
    The op log is replayed on a fresh backend of the same class and the geometry
    is measured either side of every intent-bearing op:

    ===============  =========================================================
    ``shell(t)``     must NOT grow the bounding box, and must not add volume.
    ``cut``          must NOT increase volume.
    ``extrude(d)``   the first solid's extent along the sketch normal is ~ ``d``.
    ===============  =========================================================

On failure the gate raises :class:`InvalidArtifact` naming exactly the
measurement that failed, and **the file is not written**. A refusal is a success
of the gate, not a failure of it.

``force=True`` overrides for debugging. When forced, the artifact is written AND
a sidecar ``<name>.INVALID.json`` is written beside it naming exactly what
failed. Never silently: a forced artifact is always accompanied by its own
indictment.

stdlib-only, deterministic (no wall clock, no randomness).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from harnesscad.domain.geometry.mesh.halfedge import HalfedgeMesh
from harnesscad.io.formats import stl as stl_codec

__all__ = [
    "GateError",
    "InvalidArtifact",
    "Failure",
    "GateReport",
    "SIDECAR_SUFFIX",
    "measure",
    "measured_failures",
    "declared_failures",
    "check",
    "guard",
    "gated_write",
    "sidecar_path",
    "write_sidecar",
]

Vec3 = Tuple[float, float, float]

#: A forced-through artifact is always accompanied by one of these.
SIDECAR_SUFFIX = ".INVALID.json"

# --- tolerances -------------------------------------------------------------
#: Below this, a triangle has no area and is a degenerate face.
AREA_EPS = 1e-10
#: Below this, the solid has no volume worth shipping (mm^3).
VOLUME_EPS = 1e-6
#: A bounding-box extent below this is a zero-thickness part.
EXTENT_EPS = 1e-6
#: A bounding box wider than this is a runaway field, not a part (mm).
EXTENT_MAX = 1.0e7
#: Triangle-triangle self-intersection is quadratic-ish even with a broadphase;
#: above this we say so honestly (``self_intersection_checked = False``) rather
#: than pretend or hang.
SELF_INTERSECT_MAX_TRIS = 3000
#: Relative slack on the DECLARED comparisons: both sides are measured off the
#: same marching-cubes grid, so they carry the same discretisation error.
INTENT_REL_TOL = 0.01
INTENT_ABS_TOL = 1e-6

#: THE REFERENCE TESSELLATION. Not a detail -- it is the difference between
#: measuring our GEOMETRY and measuring our TESSELLATION.
#:
#: CadQuery's ``exporters.export(..., tolerance=0.1)`` default is 100x coarser
#: than ``Shape.exportStl``'s 1e-3, and the harness passed no tolerance at all:
#: every mesh-based measurement silently inherited a 0.1 mm linear deflection.
#: On a 3 mm wall that is a 3% error before anything has gone wrong -- the same
#: size as the errors the gate exists to catch.
#:
#: The gate does NOT re-tessellate at this value (that would certify a mesh that
#: never reaches the file). It is the yardstick the *backend's* pinned setting is
#: reported against: coarser than this and the report carries a warning saying so.
MEASURE_LINEAR_DEFLECTION = 0.01     # mm
MEASURE_ANGULAR_DEFLECTION = 0.1     # radians

#: How far a measured wall may drift from its declared thickness before the gate
#: calls it a wrong part. A shell can preserve the bounding box EXACTLY and still
#: leave the wall 42% too thin, so the envelope check alone proves nothing about
#: the wall; this is the tolerance on the wall check that does.
WALL_REL_TOL = 0.12
#: The wall probe's tolerance also carries a term in the mesh's own edge length,
#: because no probe resolves a defect finer than the mesh it reads. Calibrated
#: against a known-good 3 mm wall from grid resolution 18 to 48 (observed error
#: <0.25 mm, ~0.07 of an edge); 0.3 keeps ~4x margin on that.
WALL_EDGE_FACTOR = 0.3
#: How many post-shell vertices the wall probe samples (deterministic stride).
WALL_SAMPLE_POINTS = 240


# ---------------------------------------------------------------------------
# Errors and reports
# ---------------------------------------------------------------------------

class GateError(Exception):
    """Base class for every error the output gate raises."""


@dataclass(frozen=True)
class Failure:
    """One check that failed, and the measurement that failed it."""

    check: str          # stable machine code, e.g. "shell-grew-bbox"
    family: str         # "measured" | "declared"
    detail: str         # a sentence a human can act on
    measured: Any = None
    expected: Any = None

    def to_dict(self) -> dict:
        return {
            "check": self.check,
            "family": self.family,
            "detail": self.detail,
            "measured": self.measured,
            "expected": self.expected,
        }

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.check}: {self.detail}"


@dataclass(frozen=True)
class GateReport:
    """What the gate found. ``ok`` is the total-correctness claim for one file."""

    path: Optional[str]
    ok: bool
    failures: Tuple[Failure, ...] = ()
    measurement: Dict[str, Any] = dc_field(default_factory=dict)
    declared: Tuple[Dict[str, Any], ...] = ()
    forced: bool = False

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "ok": self.ok,
            "forced": self.forced,
            "failures": [f.to_dict() for f in self.failures],
            "measurement": self.measurement,
            "declared": list(self.declared),
        }


class InvalidArtifact(GateError):
    """The gate refused to write an artifact. The file does NOT exist.

    ``.failures`` names exactly what failed; ``.report`` carries the full
    measurement. This exception IS the gate working.
    """

    def __init__(self, report: GateReport) -> None:
        self.report = report
        self.path = report.path
        self.failures: Tuple[Failure, ...] = tuple(report.failures)
        codes = ", ".join(f.check for f in self.failures) or "unknown"
        where = f" for {report.path!r}" if report.path else ""
        detail = "; ".join(f.detail for f in self.failures)
        super().__init__(
            f"refused to write an invalid artifact{where} [{codes}]: {detail}")

    def to_dict(self) -> dict:
        return self.report.to_dict()


# ---------------------------------------------------------------------------
# Coercion -- duck-typed, so this module never imports the format registry
# (the registry imports *us*; a cycle would make the gate skippable).
# ---------------------------------------------------------------------------

def _backend_of(obj: Any) -> Any:
    """The GeometryBackend behind a HarnessSession / backend / server, or None."""
    if obj is None:
        return None
    for attr in ("backend", "session"):
        inner = getattr(obj, attr, None)
        if inner is not None:
            found = _backend_of(inner)
            if found is not None:
                return found
    if hasattr(obj, "export") and hasattr(obj, "state_digest"):
        return obj
    return None


def _indexed(verts: Sequence[Sequence[float]],
             faces: Sequence[Sequence[int]]) -> Tuple[List[Vec3], List[Tuple[int, int, int]]]:
    return ([tuple(float(c) for c in v) for v in verts],
            [tuple(int(i) for i in f) for f in faces])


def _weld(triangles: Sequence[Any]) -> Tuple[List[Vec3], List[Tuple[int, int, int]]]:
    lookup: Dict[Vec3, int] = {}
    verts: List[Vec3] = []
    faces: List[Tuple[int, int, int]] = []
    for t in triangles:
        idx: List[int] = []
        for v in (t.v0, t.v1, t.v2):
            key = (float(v[0]), float(v[1]), float(v[2]))
            if key not in lookup:
                lookup[key] = len(verts)
                verts.append(key)
            idx.append(lookup[key])
        faces.append((idx[0], idx[1], idx[2]))
    return verts, faces


#: Set by :func:`_backend_geometry`, read by :func:`check` -- the provenance of
#: the numbers in the last measurement (which tessellation they came off).
_LAST_TESSELLATION: Dict[str, Any] = {}


def _tessellation_of(backend: Any) -> Dict[str, Any]:
    """Name the tessellation this backend meshes at. An unnameable one is a finding.

    The gate must not *impose* a tolerance -- if it re-meshed at its own setting
    it would be certifying a tessellation nobody exports, while the file on disk
    carries a different one. What it must do is REFUSE TO MEASURE BLIND: it reads
    the backend's pinned, reviewable tessellation control and records it on every
    report, so every number the gate prints can be attributed to a known mesh.

    Two shapes of control exist in this repo:

    * an F-rep field, meshed on a grid -- the control is ``resolution``, and the
      field itself is exact, so the only error is the grid;
    * an OCCT BRep, tessellated by deflection -- the control is
      ``LINEAR_DEFLECTION`` / ``ANGULAR_DEFLECTION``. CadQuery's *export default*
      is 0.1 mm, 100x coarser than ``Shape.exportStl``'s 1e-3, and the harness was
      passing no tolerance at all -- so every mesh measurement silently inherited
      an error the same size as the ones the gate is hunting. The backend now pins
      it; the gate reads the pin and says so.

    ``controlled`` is False when neither is declared, which the report carries as
    a caveat on every number in it. That is the honest answer, not a pass.
    """
    record: Dict[str, Any] = {"controlled": False, "kind": None}
    lin = getattr(backend, "LINEAR_DEFLECTION", None)
    if lin is not None:
        record.update(kind="deflection", controlled=True,
                      linear_deflection=float(lin),
                      angular_deflection=float(
                          getattr(backend, "ANGULAR_DEFLECTION", 0.0)) or None)
        if float(lin) > MEASURE_LINEAR_DEFLECTION:
            record["warning"] = (
                "the backend tessellates at %g mm, coarser than the gate's "
                "reference %g mm: defects finer than that are invisible here"
                % (float(lin), MEASURE_LINEAR_DEFLECTION))
        return record
    res = getattr(backend, "resolution", None)
    if res is not None:
        record.update(kind="grid", controlled=True, resolution=int(res),
                      mesher=str(getattr(backend, "mesher", "")) or None)
        return record
    record["warning"] = (
        "this backend declares no tessellation control, so the gate cannot say "
        "what mesh its numbers came off; treat every measurement below as having "
        "an unknown discretisation error")
    return record


def _backend_geometry(backend: Any) -> Optional[Tuple[List[Vec3],
                                                      List[Tuple[int, int, int]]]]:
    """The backend's mesh -- the SAME one it exports -- plus its provenance.

    Deliberately does NOT pass a tolerance of its own. The gate measures the mesh
    the harness actually ships; re-tessellating at a private setting would mean
    certifying geometry that never reaches the file. It records which tessellation
    that was instead (see :func:`_tessellation_of`).
    """
    global _LAST_TESSELLATION
    _LAST_TESSELLATION = _tessellation_of(backend)

    mesher = getattr(backend, "mesh", None)
    if callable(mesher):
        try:
            verts, faces = mesher()
            _LAST_TESSELLATION["route"] = "backend.mesh()"
            return _indexed(verts, faces)
        except Exception:  # noqa: BLE001 -- fall through to the STL route
            pass

    try:
        payload = backend.export("stl")
    except Exception:  # noqa: BLE001
        return None
    _LAST_TESSELLATION["route"] = "backend.export('stl')"
    data = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
    try:
        return _weld(stl_codec.parse_stl(data))
    except Exception:  # noqa: BLE001
        return None


def _geometry(model: Any, source: Any = None) -> Optional[Tuple[List[Vec3],
                                                                List[Tuple[int, int, int]]]]:
    """The (vertices, faces) the gate will measure, or None if not derivable.

    A backend/session is *always* preferred over the already-coerced payload:
    that is what the harness actually built, and for a BRep target (STEP) the
    payload is text the gate could not measure at all.
    """
    backend = _backend_of(source) or _backend_of(model)
    if backend is not None:
        return _backend_geometry(backend)

    # Raw geometry payloads.
    tris = getattr(model, "triangles", None)          # formats.Mesh
    if tris is not None and not callable(tris):
        return _weld(tuple(tris))
    if hasattr(model, "vertices") and hasattr(model, "faces"):   # Polyhedron
        v, f = model.vertices, model.faces
        if not callable(v) and not callable(f):
            fan: List[Tuple[int, int, int]] = []
            for face in f:
                ids = [int(i) for i in face]
                for k in range(1, len(ids) - 1):
                    fan.append((ids[0], ids[k], ids[k + 1]))
            return _indexed(v, fan)
    if isinstance(model, (bytes, bytearray)):
        try:
            return _weld(stl_codec.parse_stl(bytes(model)))
        except Exception:  # noqa: BLE001
            return None
    if isinstance(model, (list, tuple)):
        if not model:
            return ([], [])
        if all(isinstance(t, stl_codec.Triangle) for t in model):
            return _weld(model)
        if len(model) == 2:
            verts, faces = model
            try:
                return _indexed(verts, faces)
            except Exception:  # noqa: BLE001
                return None
    return None


# ---------------------------------------------------------------------------
# MEASURED -- read off the geometry alone
# ---------------------------------------------------------------------------

def _finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


def _tri_area(a: Vec3, b: Vec3, c: Vec3) -> float:
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    cx, cy, cz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    return 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)


def _signed_volume(verts: Sequence[Vec3], faces: Sequence[Sequence[int]]) -> float:
    """Divergence theorem: positive when the winding puts the normals outward."""
    total = 0.0
    for f in faces:
        a, b, c = verts[f[0]], verts[f[1]], verts[f[2]]
        total += (a[0] * (b[1] * c[2] - b[2] * c[1])
                  - a[1] * (b[0] * c[2] - b[2] * c[0])
                  + a[2] * (b[0] * c[1] - b[1] * c[0]))
    return total / 6.0


def _edge_census(faces: Sequence[Sequence[int]]) -> Tuple[int, int, int]:
    """(boundary edges, non-manifold edges, inconsistently-wound edges)."""
    undirected: Dict[Tuple[int, int], int] = {}
    directed: Dict[Tuple[int, int], int] = {}
    for f in faces:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            key = (a, b) if a < b else (b, a)
            undirected[key] = undirected.get(key, 0) + 1
            directed[(a, b)] = directed.get((a, b), 0) + 1
    boundary = sum(1 for n in undirected.values() if n == 1)
    non_manifold = sum(1 for n in undirected.values() if n > 2)
    # A consistently wound closed surface uses every directed edge exactly once.
    inconsistent = sum(1 for n in directed.values() if n > 1)
    return boundary, non_manifold, inconsistent


def _tri_tri_intersect(p: Sequence[Vec3], q: Sequence[Vec3], eps: float) -> bool:
    """Moller's separating-axis triangle-triangle overlap test."""
    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0])

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    n1 = cross(sub(p[1], p[0]), sub(p[2], p[0]))
    d1 = -dot(n1, p[0])
    dq = [dot(n1, v) + d1 for v in q]
    if all(v > eps for v in dq) or all(v < -eps for v in dq):
        return False

    n2 = cross(sub(q[1], q[0]), sub(q[2], q[0]))
    d2 = -dot(n2, q[0])
    dp = [dot(n2, v) + d2 for v in p]
    if all(v > eps for v in dp) or all(v < -eps for v in dp):
        return False

    axis = cross(n1, n2)
    if dot(axis, axis) < eps * eps:
        return False        # coplanar: the closed-surface checks own this case

    # Project both triangles onto the intersection line and compare intervals.
    def interval(tri, d):
        idx = max(range(3), key=lambda i: abs(axis[i]))
        proj = [tri[i][idx] for i in range(3)]
        # the lone vertex is the one whose signed distance differs in sign
        lo = hi = None
        for i in range(3):
            j, k = (i + 1) % 3, (i + 2) % 3
            if (d[i] > eps and d[j] <= eps and d[k] <= eps) or \
               (d[i] < -eps and d[j] >= -eps and d[k] >= -eps):
                t1 = proj[i] + (proj[j] - proj[i]) * (d[i] / (d[i] - d[j]))
                t2 = proj[i] + (proj[k] - proj[i]) * (d[i] / (d[i] - d[k]))
                lo, hi = min(t1, t2), max(t1, t2)
                break
        return lo, hi

    a0, a1 = interval(p, dp)
    b0, b1 = interval(q, dq)
    if a0 is None or b0 is None:
        return False
    return not (a1 < b0 - eps or b1 < a0 - eps)


def _self_intersects(verts: Sequence[Vec3],
                     faces: Sequence[Sequence[int]]) -> Optional[int]:
    """Count self-intersecting triangle pairs, or None when too big to be cheap.

    A uniform-grid broadphase over triangle AABBs; triangles that share a vertex
    are skipped (they touch by construction, not by defect).
    """
    n = len(faces)
    if n == 0:
        return 0
    if n > SELF_INTERSECT_MAX_TRIS:
        return None

    boxes = []
    for f in faces:
        tri = (verts[f[0]], verts[f[1]], verts[f[2]])
        lo = tuple(min(v[i] for v in tri) for i in range(3))
        hi = tuple(max(v[i] for v in tri) for i in range(3))
        boxes.append((lo, hi, tri))

    span = [max(b[1][i] for b in boxes) - min(b[0][i] for b in boxes) for i in range(3)]
    scale = max(span) or 1.0
    cell = max(scale / 32.0, 1e-9)
    origin = tuple(min(b[0][i] for b in boxes) for i in range(3))
    eps = 1e-9 * scale

    grid: Dict[Tuple[int, int, int], List[int]] = {}
    for fi, (lo, hi, _tri) in enumerate(boxes):
        c0 = tuple(int((lo[i] - origin[i]) // cell) for i in range(3))
        c1 = tuple(int((hi[i] - origin[i]) // cell) for i in range(3))
        for x in range(c0[0], c1[0] + 1):
            for y in range(c0[1], c1[1] + 1):
                for z in range(c0[2], c1[2] + 1):
                    grid.setdefault((x, y, z), []).append(fi)

    hits = 0
    seen: set = set()
    for members in grid.values():
        for ai in range(len(members)):
            for bi in range(ai + 1, len(members)):
                i, j = members[ai], members[bi]
                key = (i, j)
                if key in seen:
                    continue
                seen.add(key)
                if set(faces[i]) & set(faces[j]):
                    continue        # adjacent by construction
                lo_i, hi_i, tri_i = boxes[i]
                lo_j, hi_j, tri_j = boxes[j]
                if any(hi_i[k] < lo_j[k] - eps or hi_j[k] < lo_i[k] - eps
                       for k in range(3)):
                    continue
                if _tri_tri_intersect(tri_i, tri_j, eps):
                    hits += 1
    return hits


def measure(verts: Sequence[Sequence[float]],
            faces: Sequence[Sequence[int]]) -> Dict[str, Any]:
    """Every number the gate judges on. Pure; no opinion, just measurement."""
    v, f = _indexed(verts, faces)
    out: Dict[str, Any] = {
        "vertex_count": len(v),
        "triangle_count": len(f),
    }
    if not f:
        out.update({
            "watertight": False, "manifold": False, "consistently_wound": False,
            "boundary_edges": 0, "non_manifold_edges": 0, "inconsistent_edges": 0,
            "degenerate_faces": 0, "volume": 0.0, "signed_volume": 0.0,
            "bbox": [0.0, 0.0, 0.0], "bbox_min": [0.0, 0.0, 0.0],
            "bbox_max": [0.0, 0.0, 0.0], "finite": True,
            "self_intersections": 0, "self_intersection_checked": True,
        })
        return out

    finite = all(_finite(c) for vert in v for c in vert)
    out["finite"] = finite

    out["degenerate_faces"] = sum(
        1 for t in f if _tri_area(v[t[0]], v[t[1]], v[t[2]]) <= AREA_EPS) if finite else 0

    boundary, non_manifold, inconsistent = _edge_census(f)
    out["boundary_edges"] = boundary
    out["non_manifold_edges"] = non_manifold
    out["inconsistent_edges"] = inconsistent
    out["watertight"] = boundary == 0
    out["consistently_wound"] = inconsistent == 0

    # The half-edge structure is the authority on 2-manifoldness; it can only be
    # built when the edge census is sane, so guard it.
    if non_manifold == 0 and inconsistent == 0:
        try:
            he = HalfedgeMesh(v, f)
            ok, issues = he.is_2manifold()
            out["manifold"] = bool(ok)
            out["manifold_issues"] = len(issues)
            out["euler_characteristic"] = he.euler_characteristic()
        except Exception as exc:  # noqa: BLE001
            out["manifold"] = False
            out["manifold_issues"] = -1
            out["manifold_error"] = str(exc)
    else:
        out["manifold"] = False
        out["manifold_issues"] = non_manifold + inconsistent

    if finite:
        sv = _signed_volume(v, f)
        out["signed_volume"] = float(sv)
        out["volume"] = float(abs(sv))
        lo = [min(vert[i] for vert in v) for i in range(3)]
        hi = [max(vert[i] for vert in v) for i in range(3)]
        out["bbox_min"] = [float(c) for c in lo]
        out["bbox_max"] = [float(c) for c in hi]
        out["bbox"] = [float(hi[i] - lo[i]) for i in range(3)]
    else:
        out["signed_volume"] = float("nan")
        out["volume"] = float("nan")
        out["bbox_min"] = out["bbox_max"] = out["bbox"] = [float("nan")] * 3

    si = _self_intersects(v, f) if finite else None
    out["self_intersections"] = si
    out["self_intersection_checked"] = si is not None
    return out


def measured_failures(m: Dict[str, Any]) -> List[Failure]:
    """The MEASURED verdict: what is wrong with this geometry, on its own terms."""
    out: List[Failure] = []

    if m["triangle_count"] == 0:
        out.append(Failure("empty-geometry", "measured",
                           "the artifact has no triangles: there is no part here",
                           measured=0, expected="> 0"))
        return out                      # everything downstream is vacuous

    if not m.get("finite", True):
        out.append(Failure("non-finite-coordinates", "measured",
                           "the geometry contains NaN or infinite coordinates",
                           measured="nan/inf", expected="finite"))
        return out

    if m["degenerate_faces"]:
        out.append(Failure("degenerate-faces", "measured",
                           f"{m['degenerate_faces']} triangle(s) have zero area",
                           measured=m["degenerate_faces"], expected=0))

    if not m["watertight"]:
        out.append(Failure("not-watertight", "measured",
                           f"the surface is open: {m['boundary_edges']} boundary "
                           f"edge(s) belong to a single face",
                           measured=m["boundary_edges"], expected=0))

    if m["non_manifold_edges"]:
        out.append(Failure("not-2-manifold", "measured",
                           f"{m['non_manifold_edges']} edge(s) are shared by more "
                           f"than two faces",
                           measured=m["non_manifold_edges"], expected=0))
    elif not m["manifold"]:
        out.append(Failure("not-2-manifold", "measured",
                           f"the surface is not a 2-manifold "
                           f"({m.get('manifold_issues')} issue(s))",
                           measured=m.get("manifold_issues"), expected=0))

    if not m["consistently_wound"]:
        out.append(Failure("inconsistent-winding", "measured",
                           f"{m['inconsistent_edges']} directed edge(s) are used "
                           f"twice: the face winding is not consistent",
                           measured=m["inconsistent_edges"], expected=0))

    if m["volume"] <= VOLUME_EPS:
        out.append(Failure("degenerate-volume", "measured",
                           f"the solid encloses no volume ({m['volume']:.6g} mm3)",
                           measured=m["volume"], expected=f"> {VOLUME_EPS}"))
    elif m["signed_volume"] < 0.0:
        out.append(Failure("inverted-normals", "measured",
                           f"the face normals point inward (signed volume "
                           f"{m['signed_volume']:.6g} mm3 < 0)",
                           measured=m["signed_volume"], expected="> 0"))

    bbox = m["bbox"]
    if any(e <= EXTENT_EPS for e in bbox):
        out.append(Failure("degenerate-bbox", "measured",
                           f"the part is flat: bounding box {bbox} has a "
                           f"zero-thickness extent",
                           measured=bbox, expected=f"every extent > {EXTENT_EPS}"))
    elif any(e > EXTENT_MAX for e in bbox):
        out.append(Failure("runaway-bbox", "measured",
                           f"the part is implausibly large: bounding box {bbox}",
                           measured=bbox, expected=f"every extent <= {EXTENT_MAX}"))

    if m.get("self_intersections"):
        out.append(Failure("self-intersecting", "measured",
                           f"{m['self_intersections']} triangle pair(s) intersect "
                           f"each other",
                           measured=m["self_intersections"], expected=0))

    return out


# ---------------------------------------------------------------------------
# DECLARED -- the op stream said what it wanted; did it get it?
# ---------------------------------------------------------------------------

_PLANE_NORMAL_AXIS = {"XY": 2, "YZ": 0, "XZ": 1}


def _op_name(op: Any) -> str:
    return type(op).__name__


def _fresh_backend(backend: Any) -> Optional[Any]:
    """A same-class, same-settings backend to replay the op log on."""
    cls = type(backend)
    kwargs = {}
    for key in ("resolution", "mesher", "normals", "prune"):
        if hasattr(backend, key):
            kwargs[key] = getattr(backend, key)
    for attempt in (kwargs, {}):
        try:
            fresh = cls(**attempt)
            fresh.reset()
            return fresh
        except Exception:  # noqa: BLE001
            continue
    return None


def _snapshot(backend: Any) -> Optional[Dict[str, Any]]:
    """(volume, bbox) of the backend's current solid, or None when it has none."""
    try:
        m = backend.query("metrics")
    except Exception:  # noqa: BLE001
        return None
    if not m or "bbox" not in m:
        return None
    return {"volume": float(m.get("volume", 0.0)),
            "bbox": [float(c) for c in m["bbox"]]}


def _tol(reference: float) -> float:
    return abs(reference) * INTENT_REL_TOL + INTENT_ABS_TOL


def _point_triangle_distance(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> float:
    """Exact unsigned distance from a point to a triangle (Ericson, RTCD 5.1.5)."""
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    ap = (p[0] - a[0], p[1] - a[1], p[2] - a[2])
    d1 = ab[0] * ap[0] + ab[1] * ap[1] + ab[2] * ap[2]
    d2 = ac[0] * ap[0] + ac[1] * ap[1] + ac[2] * ap[2]
    if d1 <= 0.0 and d2 <= 0.0:
        q = a
    else:
        bp = (p[0] - b[0], p[1] - b[1], p[2] - b[2])
        d3 = ab[0] * bp[0] + ab[1] * bp[1] + ab[2] * bp[2]
        d4 = ac[0] * bp[0] + ac[1] * bp[1] + ac[2] * bp[2]
        if d3 >= 0.0 and d4 <= d3:
            q = b
        else:
            vc = d1 * d4 - d3 * d2
            if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                v = d1 / (d1 - d3) if (d1 - d3) else 0.0
                q = (a[0] + ab[0] * v, a[1] + ab[1] * v, a[2] + ab[2] * v)
            else:
                cp = (p[0] - c[0], p[1] - c[1], p[2] - c[2])
                d5 = ab[0] * cp[0] + ab[1] * cp[1] + ab[2] * cp[2]
                d6 = ac[0] * cp[0] + ac[1] * cp[1] + ac[2] * cp[2]
                if d6 >= 0.0 and d5 <= d6:
                    q = c
                else:
                    vb = d5 * d2 - d1 * d6
                    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                        w = d2 / (d2 - d6) if (d2 - d6) else 0.0
                        q = (a[0] + ac[0] * w, a[1] + ac[1] * w, a[2] + ac[2] * w)
                    else:
                        va = d3 * d6 - d5 * d4
                        denom = (d4 - d3) + (d5 - d6)
                        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0 and denom:
                            w = (d4 - d3) / denom
                            q = (b[0] + (c[0] - b[0]) * w,
                                 b[1] + (c[1] - b[1]) * w,
                                 b[2] + (c[2] - b[2]) * w)
                        else:
                            den = va + vb + vc
                            if den <= 0.0:
                                q = a
                            else:
                                v, w = vb / den, vc / den
                                q = (a[0] + ab[0] * v + ac[0] * w,
                                     a[1] + ab[1] * v + ac[1] * w,
                                     a[2] + ab[2] * v + ac[2] * w)
    dx, dy, dz = p[0] - q[0], p[1] - q[1], p[2] - q[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _distance_to_surface(p: Vec3, verts: Sequence[Vec3],
                         faces: Sequence[Sequence[int]]) -> float:
    best = float("inf")
    for f in faces:
        d = _point_triangle_distance(p, verts[f[0]], verts[f[1]], verts[f[2]])
        if d < best:
            best = d
            if best <= 0.0:
                break
    return best


def _characteristic_length(verts: Sequence[Vec3],
                           faces: Sequence[Sequence[int]]) -> float:
    """Mean triangle edge length: how finely this mesh can resolve anything.

    The wall probe cannot be sharper than the tessellation it reads, so the wall
    tolerance is widened to this. It is the honest floor on what we can claim.
    """
    if not faces:
        return 0.0
    total = 0.0
    n = 0
    step = max(1, len(faces) // 200)
    for f in faces[::step]:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            va, vb = verts[a], verts[b]
            total += math.dist(va, vb)
            n += 1
    return total / n if n else 0.0


def _wall_probe(pre: Tuple[List[Vec3], List[Tuple[int, int, int]]],
                post: Tuple[List[Vec3], List[Tuple[int, int, int]]],
                thickness: float) -> Optional[Dict[str, Any]]:
    """Measure the WALL a shell actually left, not just the envelope it kept.

    A bounding box proves nothing about a shell. An inward shell can preserve the
    envelope EXACTLY and still leave the wall 42% too thin -- the outer surface is
    untouched by construction, so the one number the bbox check reads is the one
    number a broken shell is guaranteed to get right.

    So we probe the wall directly. After a correct ``shell(t)``, every point of
    the new solid lies within ``t`` of the ORIGINAL surface: the outer surface
    sits at distance 0, and the cavity surface it opened sits at distance exactly
    ``t``. So we take the post-shell vertices, measure each one's distance to the
    pre-shell surface, and look at the far end of that distribution: it must peak
    at ``t``. A wall left at 0.58t puts the cavity at 0.58t and is caught; a
    dilating shell puts material at negative distance (outside the original) and
    is caught by the envelope check above.

    HONESTY: this is a SAMPLE of the vertices, not a proof over the continuum,
    and its resolving power is bounded by the tessellation (see
    ``tolerance`` in the returned record). It can miss a wall defect smaller than
    one triangle edge. It is a strong check, not a theorem.
    """
    pre_v, pre_f = pre
    post_v, post_f = post
    if not pre_f or not post_f or thickness <= 0.0:
        return None

    char = _characteristic_length(post_v, post_f)
    step = max(1, len(post_v) // WALL_SAMPLE_POINTS)
    sample = post_v[::step]
    if not sample:
        return None

    distances = [_distance_to_surface(p, pre_v, pre_f) for p in sample]
    deepest = max(distances)
    # The probe can never be finer than the mesh it reads, so the tolerance
    # carries a term in the mesh's own edge length. CALIBRATED, not guessed: the
    # deepest-material figure is a MAX over many cavity vertices, so it is far
    # better resolved than a single facet is. Measured against a known-correct
    # 3 mm wall, it errs by <0.25 mm from grid resolution 18 up to 48 -- roughly
    # 0.07 of an edge. WALL_EDGE_FACTOR keeps ~4x margin on that while still
    # resolving the 42%-too-thin wall (1.75 mm against a declared 3 mm) that a
    # bounding-box check waves straight through.
    tol = max(WALL_REL_TOL * thickness, WALL_EDGE_FACTOR * char)
    return {
        "declared_thickness": float(thickness),
        "deepest_material": round(deepest, 6),
        "tolerance": round(tol, 6),
        "characteristic_edge": round(char, 6),
        "samples": len(sample),
        "too_thin": deepest < thickness - tol,
        "too_thick": deepest > thickness + tol,
    }


def declared_failures(source: Any) -> Tuple[List[Failure], List[Dict[str, Any]]]:
    """Replay the op log and check the geometry against what each op declared.

    Returns ``(failures, checks)``. ``checks`` is the audit trail -- every intent
    comparison the gate actually made, including the ones that passed -- so a
    clean report can prove it *looked*, rather than merely not complaining.

    When there is no op log (a raw mesh, a foreign STEP file) there is no
    declared intent, and this returns ``([], [])``: the MEASURED checks stand
    alone. That is stated in the report, never implied.
    """
    backend = _backend_of(source)
    if backend is None:
        return [], []
    oplog = getattr(backend, "_oplog", None)
    if not oplog:
        return [], []

    fresh = _fresh_backend(backend)
    if fresh is None:
        return [], [{"check": "replay", "status": "unavailable",
                     "detail": f"{type(backend).__name__} could not be reconstructed "
                               f"for an intent replay"}]

    failures: List[Failure] = []
    checks: List[Dict[str, Any]] = []
    # sketch id -> plane, so an Extrude knows which axis it grows along
    planes: Dict[str, str] = {}
    sketch_seq = 0

    for op in oplog:
        name = _op_name(op)

        if name == "NewSketch":
            sketch_seq += 1
            planes[f"sk{sketch_seq}"] = str(getattr(op, "plane", "XY")).upper()

        interesting = (
            name == "Shell"
            or (name == "Boolean" and str(getattr(op, "kind", "")).lower() == "cut")
            or name == "Extrude"
        )
        before = _snapshot(fresh) if interesting else None
        had_solid = bool(getattr(fresh, "_bodies", None)) if interesting else False
        pre_mesh = _backend_geometry(fresh) if name == "Shell" else None

        try:
            result = fresh.apply(op)
        except Exception as exc:  # noqa: BLE001
            checks.append({"check": "replay", "status": "diverged", "op": name,
                           "detail": f"replay raised {type(exc).__name__}: {exc}"})
            break
        if not getattr(result, "ok", True):
            checks.append({"check": "replay", "status": "diverged", "op": name,
                           "detail": "the op log does not replay cleanly"})
            break
        if not interesting:
            continue

        after = _snapshot(fresh)
        if after is None:
            continue

        if name == "Shell":
            t = float(getattr(op, "thickness", 0.0))
            if before is None:
                continue
            grown = [i for i in range(3)
                     if after["bbox"][i] > before["bbox"][i] + _tol(before["bbox"][i])]
            checks.append({"check": "shell-preserves-bbox", "op": "Shell",
                           "thickness": t,
                           "bbox_before": [round(c, 6) for c in before["bbox"]],
                           "bbox_after": [round(c, 6) for c in after["bbox"]],
                           "status": "fail" if grown else "pass"})
            if grown:
                axes = ", ".join("XYZ"[i] for i in grown)
                failures.append(Failure(
                    "shell-grew-bbox", "declared",
                    f"shell(t={t}) GREW the part along {axes}: bounding box "
                    f"{[round(c, 3) for c in before['bbox']]} -> "
                    f"{[round(c, 3) for c in after['bbox']]}. A CAD shell hollows "
                    f"inward and must never change the outer surface.",
                    measured=[round(c, 6) for c in after["bbox"]],
                    expected=f"<= {[round(c, 6) for c in before['bbox']]}"))

            if after["volume"] > before["volume"] + _tol(before["volume"]):
                checks.append({"check": "shell-removes-material", "op": "Shell",
                               "volume_before": before["volume"],
                               "volume_after": after["volume"], "status": "fail"})
                failures.append(Failure(
                    "shell-added-volume", "declared",
                    f"shell(t={t}) ADDED material: volume "
                    f"{before['volume']:.3f} -> {after['volume']:.3f} mm3. "
                    f"A shell only ever removes material.",
                    measured=after["volume"], expected=f"<= {before['volume']}"))

            # THE WALL. The envelope check above is satisfied by construction on
            # any inward shell -- it cannot tell a 3 mm wall from a 1.7 mm one.
            post_mesh = _backend_geometry(fresh)
            cavity_opened = after["volume"] < before["volume"] - _tol(before["volume"])
            probe = (_wall_probe(pre_mesh, post_mesh, t)
                     if (pre_mesh and post_mesh and cavity_opened) else None)
            if probe is None:
                checks.append({"check": "shell-wall-thickness", "op": "Shell",
                               "thickness": t, "status": "not-checked",
                               "detail": ("no cavity opened (the solid is thinner "
                                          "than 2t)" if not cavity_opened
                                          else "no mesh available to probe")})
            else:
                bad = probe["too_thin"] or probe["too_thick"]
                record = {"check": "shell-wall-thickness", "op": "Shell",
                          "status": "fail" if bad else "pass"}
                record.update(probe)
                checks.append(record)
                if bad:
                    how = "THIN" if probe["too_thin"] else "THICK"
                    failures.append(Failure(
                        "shell-wrong-wall", "declared",
                        f"shell(t={t}) left a wall that is too {how}: the deepest "
                        f"material sits {probe['deepest_material']} mm from the "
                        f"original surface, but a {t} mm wall must put it at {t} "
                        f"mm (tolerance {probe['tolerance']}). The bounding box is "
                        f"unchanged, so the envelope check alone would have passed "
                        f"this part.",
                        measured=probe["deepest_material"], expected=t))

        elif name == "Boolean":
            if before is None:
                continue
            checks.append({"check": "cut-removes-material", "op": "Boolean(cut)",
                           "volume_before": before["volume"],
                           "volume_after": after["volume"],
                           "status": "fail"
                                     if after["volume"] > before["volume"]
                                        + _tol(before["volume"]) else "pass"})
            if after["volume"] > before["volume"] + _tol(before["volume"]):
                failures.append(Failure(
                    "cut-increased-volume", "declared",
                    f"a boolean CUT increased the volume: "
                    f"{before['volume']:.3f} -> {after['volume']:.3f} mm3. "
                    f"A cut only ever removes material.",
                    measured=after["volume"], expected=f"<= {before['volume']}"))

        elif name == "Extrude":
            # Only the solid-creating extrude has an unambiguous declared height:
            # once a body exists, a further extrude unions into it and the overall
            # extent is no longer the extrude's own distance.
            if had_solid:
                continue
            d = abs(float(getattr(op, "distance", 0.0)))
            plane = planes.get(str(getattr(op, "sketch", "")), "XY")
            axis = _PLANE_NORMAL_AXIS.get(plane, 2)
            got = after["bbox"][axis]
            ok = abs(got - d) <= max(_tol(d), 2e-2 * max(d, 1.0))
            checks.append({"check": "extrude-height", "op": "Extrude",
                           "distance": d, "axis": "XYZ"[axis], "measured": round(got, 6),
                           "status": "pass" if ok else "fail"})
            if not ok:
                failures.append(Failure(
                    "extrude-wrong-height", "declared",
                    f"extrude(distance={d}) produced a part {got:.3f} mm tall "
                    f"along {'XYZ'[axis]} (the {plane} sketch normal)",
                    measured=round(got, 6), expected=d))

    return failures, checks


# ---------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------

#: What a PASS from this gate actually means. Kept next to the code that earns
#: it, and copied into every report, because the failure mode of a verifier is
#: not that it is wrong -- it is that it is BELIEVED past what it checked.
PROVES = (
    "the artifact is a closed, 2-manifold, consistently-wound solid",
    "it encloses a positive, non-degenerate volume with outward normals",
    "its bounding box is finite and physically plausible",
    "no two of its triangles intersect (only when the mesh is small enough to "
    "check cheaply; see 'self_intersection_checked')",
    "the declared op invariants that were checked HELD: a shell did not grow the "
    "envelope and left a wall of the declared thickness; a cut did not add "
    "volume; the first extrude produced a part of the declared height",
    "the measurement was taken on the SAME mesh the harness exports, at a "
    "tessellation the report names explicitly (see 'tessellation')",
)

#: What a PASS from this gate does NOT mean. THE ORACLE IS MANY-TO-ONE: volume,
#: bounding box and genus do not pin down a part. Two very different solids can
#: agree on all three. A hole in the wrong place, a fillet on the wrong edge, a
#: feature at the wrong coordinate -- all of these can score perfectly here.
DOES_NOT_PROVE = (
    "that the part matches the BRIEF -- the gate never reads the brief, and no "
    "combination of volume, bbox and genus identifies a part uniquely",
    "that features are in the right PLACE: a hole bored at the wrong coordinate "
    "changes no measured quantity the gate checks",
    "that ops with no declared invariant (fillet, chamfer, draft, pattern, "
    "revolve, loft, sweep) did what they were asked -- they are measured as "
    "geometry, not compared against an intent",
    "anything about a defect finer than the tessellation it measured on",
    "that a foreign payload (a STEP file read from disk, a raw CSG tree) is a "
    "valid part -- with no backend behind it there is no geometry to measure, "
    "and the report says so ('geometry_source': 'none')",
)


def claims() -> Dict[str, Any]:
    """Exactly what this gate does and does not establish. No more, no less."""
    return {
        "proves": list(PROVES),
        "does_not_prove": list(DOES_NOT_PROVE),
        "linear_deflection": MEASURE_LINEAR_DEFLECTION,
        "angular_deflection": MEASURE_ANGULAR_DEFLECTION,
    }


def _document_state(m: Dict[str, Any], body: str):
    """A validation_rules_md DocumentState built from the gate's measurement.

    validation_rules_md's BodyState is documented as "a neutral measured-state
    model replacing the live FreeCAD document, so the checks compose with the
    harness's own measurement layer instead of a running CAD host". This gate IS
    that measurement layer, so this is the adapter the module was written for.

    Only genuinely measured quantities are filled in. `None` in a BodyState
    means "not measured", and the checks FAIL on it with an explicit message
    rather than passing silently -- so a field the gate cannot honestly supply
    is left None on purpose, not defaulted into a fake pass. The gate measures
    ONE mesh, so the document holds one body under `body`, and `total_bodies`
    is 1 by construction.
    """
    from harnesscad.eval.verifiers.validation_rules_md import (BodyState,
                                                               DocumentState)

    bbox = m.get("bbox")
    bbox_t = (tuple(float(x) for x in bbox)
              if isinstance(bbox, (list, tuple)) and len(bbox) == 3 else None)
    lo, hi = m.get("bbox_min"), m.get("bbox_max")
    z_range = ((float(lo[2]), float(hi[2]))
               if isinstance(lo, (list, tuple)) and isinstance(hi, (list, tuple))
               and len(lo) == 3 and len(hi) == 3 else None)
    volume = m.get("volume")
    # "Valid solid" in the gate's own terms: closed, 2-manifold, and not
    # self-intersecting. When self-intersection could not be checked (the mesh
    # was too big to do it honestly) validity is left UNMEASURED rather than
    # asserted from the other two -- the gate does not certify what it skipped.
    valid: Optional[bool] = None
    if m.get("self_intersection_checked"):
        valid = bool(m.get("watertight") and m.get("manifold")
                     and not m.get("self_intersections"))
    doc = DocumentState()
    doc.add(BodyState(
        label=body,
        bbox=bbox_t,
        z_range=z_range,
        volume=float(volume) if isinstance(volume, (int, float)) else None,
        solid_count=1 if m.get("triangle_count") else 0,
        is_valid_solid=valid,
    ))
    return doc


def _rule_failures(m: Dict[str, Any], validation_rules: str,
                   validation_params: Optional[Dict[str, Any]],
                   body: str) -> Tuple[List[Failure], List[Dict[str, Any]]]:
    """Run a caller's VALIDATION.md against the measurement. (failures, results).

    A VALIDATION.md is a DECLARED intent -- the caller states what the part must
    be -- so a failing rule is a `declared` Failure and the gate refuses, which
    is the same treatment the op stream's declared intent already gets.

    A rules document that cannot be parsed or run is itself a failure, not a
    silent pass: this gate exists so that "no complaint" can never mean "nobody
    looked".
    """
    from harnesscad.eval.verifiers.validation_rules_md import validate

    try:
        results = validate(_document_state(m, body), validation_params or {},
                           validation_rules)
    except Exception as exc:  # noqa: BLE001 - a broken contract is a refusal
        return [Failure("validation-rules-error", "declared",
                        f"the supplied validation rules could not be run: "
                        f"{type(exc).__name__}: {exc}",
                        measured=None, expected="a runnable VALIDATION.md")], []
    failures = [
        Failure(f"validation-rule:{r.check}", "declared", r.message,
                measured=r.actual, expected=r.expected)
        for r in results if not r.passed and not r.skipped
    ]
    return failures, [r.to_dict() for r in results]


def check(model: Any, path: Optional[str] = None, *, source: Any = None,
          forced: bool = False, validation_rules: Optional[str] = None,
          validation_params: Optional[Dict[str, Any]] = None,
          validation_body: str = "Body") -> GateReport:
    """Measure the artifact and judge it. Never raises, never writes; pure.

    ``validation_rules`` optionally supplies a freecad-ai VALIDATION.md
    (eval/verifiers/validation_rules_md.py): a declarative per-body acceptance
    contract -- bbox, volume formula, solid count, validity -- run against this
    gate's own measurement, with ``validation_params`` binding its parameter
    block and ``validation_body`` naming the single measured body its
    ``### <BodyLabel>`` rules target. A failing rule refuses the artifact like
    any other declared intent. ``None`` -- the default -- runs no rules and
    imports nothing, so every existing caller is unaffected.
    """
    geom = _geometry(model, source)
    if geom is None:
        # Nothing mesh-shaped: a StepFile / part-21 text / CSG tree handed
        # straight to a codec with no backend behind it. There is no built
        # geometry to measure, so there is no part to get wrong -- this is
        # transport, not emission. Say so explicitly; do not imply a pass.
        fails, meas = _structural_measurement(model)
        if validation_rules is not None:
            # The caller DECLARED an acceptance contract, and there is no
            # geometry to evaluate it against. Passing here would hand back
            # ok=True for a contract nobody ran -- the exact "silence is not
            # success" this gate exists to prevent. Refuse instead.
            fails = list(fails) + [Failure(
                "validation-rules-unevaluated", "declared",
                "validation rules were supplied but there is no measurable "
                "geometry to run them against; the contract is unverified",
                measured=None, expected="a measurable solid")]
            meas["declared_intent"] = "unevaluated"
        return GateReport(path=path, ok=not fails, failures=tuple(fails),
                          measurement=meas, declared=(), forced=forced)

    verts, faces = geom
    from_backend = bool(_backend_of(source) or _backend_of(model))
    tessellation = dict(_LAST_TESSELLATION) if from_backend else {
        "route": "caller-supplied mesh", "tolerance_controlled": False,
        "linear_deflection": None}
    m = measure(verts, faces)
    m["geometry_source"] = "backend" if from_backend else "payload"
    # Which tessellation these numbers came off. A measurement whose tolerance is
    # unknown is not a measurement, so the report always carries it -- including
    # when the gate could NOT pin it (tolerance_controlled: false), which is a
    # caveat on every number above, not a footnote.
    m["tessellation"] = tessellation
    failures = measured_failures(m)
    declared, checks = declared_failures(source if source is not None else model)
    failures.extend(declared)
    if validation_rules is not None:
        rule_failures, rule_results = _rule_failures(
            m, validation_rules, validation_params, validation_body)
        failures.extend(rule_failures)
        m["validation_rules"] = {"body": validation_body,
                                 "results": rule_results}
        checks = list(checks) + rule_results
    m["declared_intent"] = "checked" if checks else "none-declared"
    m["proves"] = list(PROVES)
    m["does_not_prove"] = list(DOES_NOT_PROVE)
    return GateReport(path=path, ok=not failures, failures=tuple(failures),
                      measurement=m, declared=tuple(checks), forced=forced)


def _structural_measurement(model: Any) -> Tuple[List[Failure], Dict[str, Any]]:
    """The most the gate can honestly say about a non-mesh payload."""
    kind = type(model).__name__
    meas: Dict[str, Any] = {"geometry_source": "none",
                            "payload_type": kind,
                            "declared_intent": "none-declared",
                            "measured": "structural"}
    fails: List[Failure] = []
    if model is None:
        fails.append(Failure("empty-artifact", "measured",
                             "there is nothing to write", measured=None,
                             expected="a model"))
        return fails, meas
    if isinstance(model, str):
        meas["length"] = len(model)
        if not model.strip():
            fails.append(Failure("empty-artifact", "measured",
                                 "the payload is empty text", measured=0,
                                 expected="> 0 bytes"))
    entities = getattr(model, "entities", None)
    if entities is not None and not callable(entities):
        meas["entity_count"] = len(entities)
        if not entities:
            fails.append(Failure("empty-artifact", "measured",
                                 f"the {kind} carries no entities", measured=0,
                                 expected="> 0"))
    return fails, meas


def sidecar_path(path: str) -> str:
    """``part.stl`` -> ``part.INVALID.json``."""
    root, _ext = os.path.splitext(str(path))
    return root + SIDECAR_SUFFIX


def write_sidecar(path: str, report: GateReport) -> str:
    """Name, on disk and beside the artifact, exactly what is wrong with it."""
    target = sidecar_path(path)
    payload = report.to_dict()
    payload["artifact"] = os.path.basename(str(path))
    payload["warning"] = (
        "This artifact FAILED the harness output gate and was written only "
        "because --force was given. It is not a valid part. The failures below "
        "name exactly what is wrong with it.")
    with open(target, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(payload, indent=2, sort_keys=True, default=str))
        fh.write("\n")
    return target


def guard(model: Any, path: Optional[str] = None, *, source: Any = None,
          force: bool = False, validation_rules: Optional[str] = None,
          validation_params: Optional[Dict[str, Any]] = None,
          validation_body: str = "Body") -> GateReport:
    """Judge an artifact on its way out. Raise :class:`InvalidArtifact` or pass.

    This is the single call every write path makes *before* it opens the file.
    With ``force=True`` a failing artifact is allowed through, but the returned
    report says ``ok=False`` and the caller MUST write the sidecar (which
    :func:`gated_write` does for you).

    ``validation_rules`` / ``validation_params`` / ``validation_body`` are
    passed through to :func:`check` -- this is the door every write goes
    through, so a declarative acceptance contract has to be reachable from it.
    """
    report = check(model, path, source=source, forced=force,
                   validation_rules=validation_rules,
                   validation_params=validation_params,
                   validation_body=validation_body)
    if not report.ok and not force:
        raise InvalidArtifact(report)
    return report


def gated_write(writer, model: Any, path: str, *, source: Any = None,
                force: bool = False, **options: Any) -> str:
    """Gate, then write, then (if forced through a failure) indict.

    ``writer(model, path, **options)`` is only ever called once the gate has
    passed -- or once ``force`` has explicitly overridden it, in which case the
    ``<name>.INVALID.json`` sidecar lands beside the file. There is no path
    through this function that writes an unverified artifact silently.
    """
    report = guard(model, path, source=source, force=force)
    writer(model, path, **options)
    if not report.ok:
        write_sidecar(path, report)
    return str(path)
