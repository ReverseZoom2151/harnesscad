"""Polyhedron construction, validation and mass properties (AngelCAD polyhedron3d).

AngelCAD's ``polyhedron`` primitive is the escape hatch of the language: a solid
given as an explicit vertex list plus a list of ``pface`` index tuples.  Because
the boolean engine downstream cannot repair a broken input, the class exposes
``verify()`` (``polyhedron3d::verify_polyhedron``), ``face_area()``, ``volume()``,
``flip_face()`` / ``flip_faces()`` and a bounding box -- i.e. a *validator* and
mass-property routines that run before any kernel is involved.

This module reimplements that layer in stdlib Python:

* :class:`Polyhedron` -- vertices + faces (any planar polygon, not just
  triangles), with Newell face normals so a quad face works;
* :func:`verify` -- a full closed-orientable-manifold check producing structured
  :class:`Issue` records: index out of range, degenerate face (< 3 vertices,
  repeated index, zero area), non-planar face, boundary edge (used once, so the
  surface is open), non-manifold edge (used > 2 times), inconsistent orientation
  (an edge traversed the same way by both its faces), unused vertex, and inward
  orientation (negative signed volume);
* mass properties by the divergence theorem over a fan triangulation of each
  face: :meth:`Polyhedron.volume` (signed), :meth:`Polyhedron.surface_area`,
  :meth:`Polyhedron.centroid` and :meth:`Polyhedron.inertia_tensor` (about the
  centroid, unit density), plus :meth:`Polyhedron.bounds`;
* repair helpers :meth:`Polyhedron.flip_face`, :meth:`Polyhedron.flip_faces` and
  :meth:`Polyhedron.oriented_outward`.

Nothing in the harness validated an explicit polyhedron: the mesh modules
(``formats.t2cdean_stl_codec``, ``geometry.mesh_sampling``) assume triangle soup
and do not check manifoldness or orientation, and no module computes a solid's
inertia tensor.  A single ``verify`` call is exactly the offline gate that
catches an LLM emitting a ``polyhedron(...)`` with a face wound the wrong way --
the classic OpenSCAD/AngelCAD failure that only shows up at render time.

Pure stdlib, deterministic.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "Issue",
    "PolyhedronError",
    "Polyhedron",
    "verify",
    "tetrahedron",
    "unit_cube",
]

Point = Tuple[float, float, float]
Face = Tuple[int, ...]


class Issue:
    """One validation finding."""

    __slots__ = ("code", "message", "face", "edge")

    def __init__(
        self,
        code: str,
        message: str,
        face: Optional[int] = None,
        edge: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.code = code
        self.message = message
        self.face = face
        self.edge = edge

    def key(self):
        return (self.code, self.message, self.face, self.edge)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Issue) and other.key() == self.key()

    def __hash__(self) -> int:
        return hash(self.key())

    def __repr__(self) -> str:
        return "Issue(%s: %s)" % (self.code, self.message)

    __str__ = __repr__


class PolyhedronError(Exception):
    def __init__(self, issues: Sequence[Issue]) -> None:
        self.issues = list(issues)
        super().__init__(
            "%d issue(s):\n%s" % (len(self.issues), "\n".join(str(i) for i in self.issues))
        )


def _sub(a: Sequence[float], b: Sequence[float]) -> Point:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Sequence[float], b: Sequence[float]) -> Point:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _length(v: Sequence[float]) -> float:
    return math.sqrt(_dot(v, v))


class Polyhedron:
    """A polyhedron: vertices plus index faces (counter-clockwise seen from outside)."""

    __slots__ = ("vertices", "faces")

    def __init__(
        self,
        vertices: Sequence[Sequence[float]],
        faces: Sequence[Sequence[int]],
    ) -> None:
        self.vertices: List[Point] = [
            (float(p[0]), float(p[1]), float(p[2])) for p in vertices
        ]
        self.faces: List[Face] = [tuple(int(i) for i in f) for f in faces]

    # -- basics -----------------------------------------------------------
    def nvert(self) -> int:
        return len(self.vertices)

    def nface(self) -> int:
        return len(self.faces)

    def face_points(self, iface: int) -> List[Point]:
        return [self.vertices[i] for i in self.faces[iface]]

    def face_normal(self, iface: int) -> Point:
        """Newell area-vector; its length is twice the face area."""
        pts = self.face_points(iface)
        n = len(pts)
        nx = ny = nz = 0.0
        for i in range(n):
            a = pts[i]
            b = pts[(i + 1) % n]
            nx += (a[1] - b[1]) * (a[2] + b[2])
            ny += (a[2] - b[2]) * (a[0] + b[0])
            nz += (a[0] - b[0]) * (a[1] + b[1])
        return (nx, ny, nz)

    def face_area(self, iface: int) -> float:
        return _length(self.face_normal(iface)) * 0.5

    def face_unit_normal(self, iface: int) -> Point:
        n = self.face_normal(iface)
        ln = _length(n)
        if ln == 0.0:
            raise PolyhedronError([Issue("face-degenerate", "face %d has zero area" % iface, iface)])
        return (n[0] / ln, n[1] / ln, n[2] / ln)

    def face_centroid(self, iface: int) -> Point:
        pts = self.face_points(iface)
        n = float(len(pts))
        return (
            sum(p[0] for p in pts) / n,
            sum(p[1] for p in pts) / n,
            sum(p[2] for p in pts) / n,
        )

    def face_planarity(self, iface: int) -> float:
        """Max distance of a face vertex from the face's mean plane."""
        pts = self.face_points(iface)
        if len(pts) <= 3:
            return 0.0
        try:
            n = self.face_unit_normal(iface)
        except PolyhedronError:
            return float("inf")
        c = self.face_centroid(iface)
        return max(abs(_dot(_sub(p, c), n)) for p in pts)

    def triangles(self) -> List[Tuple[int, int, int]]:
        """Fan-triangulate every face (deterministic)."""
        tris: List[Tuple[int, int, int]] = []
        for face in self.faces:
            for k in range(1, len(face) - 1):
                tris.append((face[0], face[k], face[k + 1]))
        return tris

    def edges(self) -> Dict[Tuple[int, int], List[int]]:
        """Undirected edge -> list of face indices using it."""
        table: Dict[Tuple[int, int], List[int]] = {}
        for fi, face in enumerate(self.faces):
            n = len(face)
            for i in range(n):
                a, b = face[i], face[(i + 1) % n]
                key = (a, b) if a <= b else (b, a)
                table.setdefault(key, []).append(fi)
        return table

    # -- mass properties --------------------------------------------------
    def bounds(self) -> Tuple[Point, Point]:
        if not self.vertices:
            raise ValueError("empty polyhedron has no bounds")
        xs = [p[0] for p in self.vertices]
        ys = [p[1] for p in self.vertices]
        zs = [p[2] for p in self.vertices]
        return ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))

    def surface_area(self) -> float:
        return sum(self.face_area(i) for i in range(self.nface()))

    def volume(self) -> float:
        """Signed volume (positive when faces are wound outward)."""
        total = 0.0
        for a, b, c in self.triangles():
            pa, pb, pc = self.vertices[a], self.vertices[b], self.vertices[c]
            total += _dot(pa, _cross(pb, pc))
        return total / 6.0

    def centroid(self) -> Point:
        """Volume centroid (unit density)."""
        vol = 0.0
        cx = cy = cz = 0.0
        for a, b, c in self.triangles():
            pa, pb, pc = self.vertices[a], self.vertices[b], self.vertices[c]
            v = _dot(pa, _cross(pb, pc)) / 6.0
            vol += v
            cx += v * (pa[0] + pb[0] + pc[0]) / 4.0
            cy += v * (pa[1] + pb[1] + pc[1]) / 4.0
            cz += v * (pa[2] + pb[2] + pc[2]) / 4.0
        if vol == 0.0:
            raise ValueError("degenerate polyhedron: zero volume")
        return (cx / vol, cy / vol, cz / vol)

    def inertia_tensor(self, about_centroid: bool = True) -> Tuple[Tuple[float, ...], ...]:
        """Inertia tensor for unit density, by tetrahedron decomposition about the origin.

        Each fan triangle plus the origin forms a signed tetrahedron; the tensor of
        a tetrahedron with one vertex at the origin has a closed form.  Optionally
        shifted to the centroid with the parallel-axis theorem.
        """
        xx = yy = zz = xy = xz = yz = 0.0
        vol = 0.0
        for a, b, c in self.triangles():
            pa, pb, pc = self.vertices[a], self.vertices[b], self.vertices[c]
            det = _dot(pa, _cross(pb, pc))
            vol += det / 6.0
            pts = (pa, pb, pc)

            # closed form for the tetra (origin, pa, pb, pc):
            #   integral of x_i x_j dV = det/120 * sum_k sum_l w_kl * p_k[i] * p_l[j]
            # with w_kl = 2 on the diagonal and 1 off it.
            def prod(i: int, j: int) -> float:
                s = 0.0
                for k in range(3):
                    for l in range(3):
                        s += pts[k][i] * pts[l][j] * (2.0 if k == l else 1.0)
                return det * s / 120.0

            ixx = prod(0, 0)
            iyy = prod(1, 1)
            izz = prod(2, 2)
            xx += iyy + izz
            yy += ixx + izz
            zz += ixx + iyy
            xy -= prod(0, 1)
            xz -= prod(0, 2)
            yz -= prod(1, 2)
        tensor = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
        if about_centroid:
            if vol == 0.0:
                raise ValueError("degenerate polyhedron: zero volume")
            cx, cy, cz = self.centroid()
            m = vol
            d = (cx, cy, cz)
            d2 = _dot(d, d)
            for i in range(3):
                for j in range(3):
                    tensor[i][j] -= m * ((d2 if i == j else 0.0) - d[i] * d[j])
        return tuple(tuple(row) for row in tensor)

    # -- repair -----------------------------------------------------------
    def flip_face(self, iface: int) -> None:
        self.faces[iface] = tuple(reversed(self.faces[iface]))

    def flip_faces(self) -> None:
        for i in range(self.nface()):
            self.flip_face(i)

    def oriented_outward(self) -> "Polyhedron":
        """Return a copy whose signed volume is positive (flip all faces if needed)."""
        out = Polyhedron(self.vertices, self.faces)
        if out.volume() < 0.0:
            out.flip_faces()
        return out

    # -- validation -------------------------------------------------------
    def verify(self, planarity_tol: float = 1e-9) -> List[Issue]:
        return verify(self, planarity_tol)

    def check(self, planarity_tol: float = 1e-9) -> None:
        issues = self.verify(planarity_tol)
        if issues:
            raise PolyhedronError(issues)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Polyhedron)
            and other.vertices == self.vertices
            and other.faces == self.faces
        )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "Polyhedron(%d vertices, %d faces)" % (self.nvert(), self.nface())


def verify(poly: Polyhedron, planarity_tol: float = 1e-9) -> List[Issue]:
    """Full validation of an explicit polyhedron.  Returns issues in stable order."""
    issues: List[Issue] = []
    nv = poly.nvert()

    if nv == 0:
        issues.append(Issue("empty", "polyhedron has no vertices"))
    if poly.nface() == 0:
        issues.append(Issue("empty", "polyhedron has no faces"))
    if issues:
        return issues

    index_ok = True
    for fi, face in enumerate(poly.faces):
        if len(face) < 3:
            issues.append(
                Issue("face-degenerate", "face %d has only %d vertices" % (fi, len(face)), fi)
            )
            index_ok = False
            continue
        if len(set(face)) != len(face):
            issues.append(Issue("face-degenerate", "face %d repeats a vertex index" % fi, fi))
            index_ok = False
        for iv in face:
            if iv < 0 or iv >= nv:
                issues.append(
                    Issue(
                        "index-range",
                        "face %d references vertex %d, valid range 0..%d" % (fi, iv, nv - 1),
                        fi,
                    )
                )
                index_ok = False

    if not index_ok:
        return issues

    for fi in range(poly.nface()):
        if poly.face_area(fi) <= 0.0:
            issues.append(Issue("face-degenerate", "face %d has zero area" % fi, fi))
        elif poly.face_planarity(fi) > planarity_tol:
            issues.append(
                Issue(
                    "face-nonplanar",
                    "face %d is non-planar (deviation %.3g > %.3g)"
                    % (fi, poly.face_planarity(fi), planarity_tol),
                    fi,
                )
            )

    used = set()
    directed: Dict[Tuple[int, int], List[int]] = {}
    for fi, face in enumerate(poly.faces):
        n = len(face)
        for i in range(n):
            a, b = face[i], face[(i + 1) % n]
            used.add(a)
            directed.setdefault((a, b), []).append(fi)

    for key in sorted(directed):
        a, b = key
        if a > b:
            continue
        fwd = directed.get((a, b), [])
        rev = directed.get((b, a), [])
        total = len(fwd) + len(rev)
        if total == 1:
            issues.append(
                Issue("edge-boundary", "edge (%d,%d) belongs to one face: surface is open" % (a, b), fwd[0] if fwd else rev[0], (a, b))
            )
        elif total > 2:
            issues.append(
                Issue("edge-nonmanifold", "edge (%d,%d) is used by %d faces" % (a, b, total), None, (a, b))
            )
        elif len(fwd) == 2 or len(rev) == 2:
            issues.append(
                Issue(
                    "orientation",
                    "edge (%d,%d) is traversed in the same direction by both faces"
                    % (a, b),
                    None,
                    (a, b),
                )
            )

    for iv in range(nv):
        if iv not in used:
            issues.append(Issue("vertex-unused", "vertex %d is not used by any face" % iv))

    if not any(i.code in ("edge-boundary", "edge-nonmanifold", "orientation") for i in issues):
        vol = poly.volume()
        if vol < 0.0:
            issues.append(
                Issue(
                    "orientation-inward",
                    "signed volume is negative (%.6g): faces are wound inward" % vol,
                )
            )
        elif vol == 0.0:
            issues.append(Issue("degenerate", "signed volume is zero"))

    return issues


# --------------------------------------------------------------------------
# canonical solids (handy for tests and for seeding a polyhedron primitive)
# --------------------------------------------------------------------------


def tetrahedron(size: float = 1.0) -> Polyhedron:
    s = float(size)
    return Polyhedron(
        [(0, 0, 0), (s, 0, 0), (0, s, 0), (0, 0, s)],
        [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)],
    )


def unit_cube(size: float = 1.0) -> Polyhedron:
    s = float(size)
    v = [
        (0, 0, 0),
        (s, 0, 0),
        (s, s, 0),
        (0, s, 0),
        (0, 0, s),
        (s, 0, s),
        (s, s, s),
        (0, s, s),
    ]
    f = [
        (0, 3, 2, 1),  # bottom, -z
        (4, 5, 6, 7),  # top, +z
        (0, 1, 5, 4),  # -y
        (1, 2, 6, 5),  # +x
        (2, 3, 7, 6),  # +y
        (3, 0, 4, 7),  # -x
    ]
    return Polyhedron(v, f)
