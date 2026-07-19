"""Geometric and topological mesh discrepancy metrics.

The premise is that point-cloud metrics such as Chamfer distance obscure fine
structural detail, so this introduces four richer structural/topological
indicators to compare a *predicted* reconstruction against its *ground-truth*
mesh:

  * Watertightness      -- a closed 2-manifold surface (every edge shared by
                           exactly two faces, no boundary/non-manifold edges),
                           the geometric-validity gate for the other metrics.
  * Sphericity          -- Wadell (1932) compactness ``psi = pi^(1/3)(6V)^(2/3)/s``
                           in ``(0, 1]``; the Sphericity Discrepancy (SD) metric is
                           ``|psi_pred - psi_gt|``.
  * Discrete mean curvature -- per-vertex sum of signed dihedral angles of the
                           mesh edges within a ball of radius ``r`` (Cohen-Steiner &
                           Morvan, 2003); the Discrete Mean Curvature Difference
                           (DMCD) is ``|mean(kappa)_pred - mean(kappa)_gt|``.
  * Euler characteristic -- ``chi = V - E + F`` (Richeson, 2012); the Exact Euler
                           Characteristic Match (EECM) is ``1`` iff the two chi
                           agree, else ``0``.

Per the paper, SD, EECM and DMCD are computed *only if both* the predicted and the
ground-truth mesh are watertight. Everything here is pure-stdlib and deterministic:
a mesh is ``(vertices, faces)`` with ``vertices`` a sequence of 3-tuples and
``faces`` a sequence of vertex-index tuples (triangles or convex polygons).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import atan2, pi, sqrt
from typing import Optional, Sequence

Point = Sequence[float]
Face = Sequence[int]


# --------------------------------------------------------------------------- #
# vector helpers
# --------------------------------------------------------------------------- #
def _sub(a: Point, b: Point) -> tuple:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Point, b: Point) -> tuple:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a: Point) -> float:
    return sqrt(_dot(a, a))


def _midpoint(a: Point, b: Point) -> tuple:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0)


# --------------------------------------------------------------------------- #
# mesh
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Mesh:
    """A polygon-soup mesh: vertices and faces (index tuples, >= 3 indices)."""

    vertices: tuple
    faces: tuple

    @classmethod
    def of(cls, vertices: Sequence[Point], faces: Sequence[Face]) -> "Mesh":
        verts = tuple(tuple(float(c) for c in v) for v in vertices)
        if any(len(v) != 3 for v in verts):
            raise ValueError("every vertex must have exactly 3 coordinates")
        fs = tuple(tuple(int(i) for i in f) for f in faces)
        n = len(verts)
        for face in fs:
            if len(face) < 3:
                raise ValueError("every face needs at least 3 vertices")
            if any(i < 0 or i >= n for i in face):
                raise ValueError("face references an out-of-range vertex")
        return cls(verts, fs)

    def _triangles(self):
        """Fan-triangulate each polygon face into index triples."""
        for face in self.faces:
            for k in range(1, len(face) - 1):
                yield (face[0], face[k], face[k + 1])


def undirected_edges(mesh: Mesh) -> Counter:
    """Count how many faces each undirected edge belongs to."""
    counts: Counter = Counter()
    for face in mesh.faces:
        m = len(face)
        for k in range(m):
            a, b = face[k], face[(k + 1) % m]
            counts[(a, b) if a < b else (b, a)] += 1
    return counts


# --------------------------------------------------------------------------- #
# surface area, volume, sphericity
# --------------------------------------------------------------------------- #
def surface_area(mesh: Mesh) -> float:
    v = mesh.vertices
    total = 0.0
    for i, j, k in mesh._triangles():
        total += 0.5 * _norm(_cross(_sub(v[j], v[i]), _sub(v[k], v[i])))
    return total


def signed_volume(mesh: Mesh) -> float:
    """Signed volume via the divergence theorem over fan-triangulated faces."""
    v = mesh.vertices
    total = 0.0
    for i, j, k in mesh._triangles():
        total += _dot(v[i], _cross(v[j], v[k])) / 6.0
    return total


def volume(mesh: Mesh) -> float:
    return abs(signed_volume(mesh))


def sphericity(mesh: Mesh) -> float:
    """Wadell sphericity ``psi = pi^(1/3) (6V)^(2/3) / s`` in ``(0, 1]``."""
    s = surface_area(mesh)
    vol = volume(mesh)
    if s <= 0.0 or vol <= 0.0:
        raise ValueError("sphericity needs positive surface area and volume")
    return (pi ** (1.0 / 3.0)) * ((6.0 * vol) ** (2.0 / 3.0)) / s


def sphericity_discrepancy(pred: Mesh, gt: Mesh) -> float:
    """SD metric: ``|psi_pred - psi_gt|`` (lower is better)."""
    return abs(sphericity(pred) - sphericity(gt))


# --------------------------------------------------------------------------- #
# Euler characteristic
# --------------------------------------------------------------------------- #
def euler_characteristic(mesh: Mesh) -> int:
    """``chi = V - E + F`` with V/E/F counted on referenced vertices only."""
    used = {i for face in mesh.faces for i in face}
    edges = undirected_edges(mesh)
    return len(used) - len(edges) + len(mesh.faces)


def euler_characteristic_match(pred: Mesh, gt: Mesh) -> int:
    """EECM metric: ``1`` iff the Euler characteristics agree, else ``0``."""
    return 1 if euler_characteristic(pred) == euler_characteristic(gt) else 0


# --------------------------------------------------------------------------- #
# watertightness
# --------------------------------------------------------------------------- #
def is_watertight(mesh: Mesh) -> bool:
    """Closed 2-manifold: non-empty and every edge shared by exactly two faces."""
    if not mesh.faces:
        return False
    return all(count == 2 for count in undirected_edges(mesh).values())


# --------------------------------------------------------------------------- #
# discrete mean curvature (Cohen-Steiner & Morvan normal cycle)
# --------------------------------------------------------------------------- #
def _face_normals(mesh: Mesh) -> dict:
    """Unit normal per face (fan-based); degenerate faces are skipped."""
    v = mesh.vertices
    normals = {}
    for f_index, face in enumerate(mesh.faces):
        n = (0.0, 0.0, 0.0)
        for k in range(1, len(face) - 1):
            c = _cross(_sub(v[face[k]], v[face[0]]), _sub(v[face[k + 1]], v[face[0]]))
            n = (n[0] + c[0], n[1] + c[1], n[2] + c[2])
        length = _norm(n)
        if length > 0.0:
            normals[f_index] = (n[0] / length, n[1] / length, n[2] / length)
    return normals


def _signed_dihedral_angles(mesh: Mesh) -> dict:
    """Signed dihedral angle for each edge shared by exactly two faces.

    The magnitude is the turn between the two outward face normals; the sign is
    a geometric convexity test (positive/convex when each face's off-edge vertex
    lies on the interior side of the other face), so it is independent of vertex
    and face iteration order. Edges not shared by exactly two (non-degenerate)
    faces are omitted.
    """
    v = mesh.vertices
    normals = _face_normals(mesh)
    incident = defaultdict(list)
    for f_index, face in enumerate(mesh.faces):
        if f_index not in normals:
            continue
        m = len(face)
        for k in range(m):
            a, b = face[k], face[(k + 1) % m]
            key = (a, b) if a < b else (b, a)
            incident[key].append(f_index)
    angles = {}
    for (a, b), faces in incident.items():
        if len(faces) != 2:
            continue
        f0, f1 = faces
        n0, n1 = normals[f0], normals[f1]
        cos_term = max(-1.0, min(1.0, _dot(n0, n1)))
        magnitude = atan2(sqrt(max(0.0, 1.0 - cos_term * cos_term)), cos_term)
        if magnitude == 0.0:
            angles[(a, b)] = 0.0
            continue
        off = next((i for i in mesh.faces[f1] if i != a and i != b), None)
        if off is None:
            continue
        # Convex edge: the neighbouring face's off-edge vertex sits below this
        # face's outward normal plane.
        sign = -1.0 if _dot(n0, _sub(v[off], v[a])) > 0.0 else 1.0
        angles[(a, b)] = sign * magnitude
    return angles


def discrete_mean_curvatures(mesh: Mesh, radius: float) -> dict:
    """Per-vertex discrete mean curvature ``kappa``.

    For each vertex, sum the signed dihedral angles of all mesh edges whose
    midpoint lies within ``radius`` of that vertex. The value is dimensionless.
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    v = mesh.vertices
    angles = _signed_dihedral_angles(mesh)
    edge_mid = {e: _midpoint(v[e[0]], v[e[1]]) for e in angles}
    used = sorted({i for face in mesh.faces for i in face})
    kappa = {}
    r2 = radius * radius
    for i in used:
        vi = v[i]
        total = 0.0
        for e, angle in angles.items():
            m = edge_mid[e]
            d = _sub(m, vi)
            if _dot(d, d) <= r2:
                total += angle
        kappa[i] = total
    return kappa


def mean_curvature(mesh: Mesh, radius: float) -> float:
    """Average per-vertex discrete mean curvature ``mean(kappa)``."""
    kappa = discrete_mean_curvatures(mesh, radius)
    if not kappa:
        return 0.0
    return sum(kappa.values()) / len(kappa)


def discrete_mean_curvature_difference(pred: Mesh, gt: Mesh, radius: float) -> float:
    """DMCD metric: ``|mean(kappa)_pred - mean(kappa)_gt|`` (lower is better)."""
    return abs(mean_curvature(pred, radius) - mean_curvature(gt, radius))


# --------------------------------------------------------------------------- #
# combined report
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MeshComparison:
    """Bundled structural discrepancy metrics.

    ``sphericity_discrepancy``, ``euler_match`` and ``dmcd`` are ``None`` unless
    *both* meshes are watertight, exactly as this protocol gates them.
    """

    pred_watertight: bool
    gt_watertight: bool
    both_watertight: bool
    pred_euler: int
    gt_euler: int
    euler_match: Optional[int]
    sphericity_discrepancy: Optional[float]
    dmcd: Optional[float]


def compare(pred: Mesh, gt: Mesh, *, radius: float = 0.1) -> MeshComparison:
    pw, gw = is_watertight(pred), is_watertight(gt)
    both = pw and gw
    return MeshComparison(
        pred_watertight=pw,
        gt_watertight=gw,
        both_watertight=both,
        pred_euler=euler_characteristic(pred),
        gt_euler=euler_characteristic(gt),
        euler_match=euler_characteristic_match(pred, gt) if both else None,
        sphericity_discrepancy=sphericity_discrepancy(pred, gt) if both else None,
        dmcd=discrete_mean_curvature_difference(pred, gt, radius) if both else None,
    )
