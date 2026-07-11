"""Template-sphere deformation for Magic3DSketch's mesh backbone (Zang et al. 2024).

Magic3DSketch's decoder does not regress vertex positions directly: it predicts
per-vertex *offsets* on a fixed **sphere template mesh**, which is then deformed
to obtain the output (paper Sec. 3.2).  Because the output is a deformation of a
genus-0 sphere, the method cannot represent non-zero-genus shapes -- a stated
limitation (Sec. 4.9).  Realism is encouraged by the flatten loss and Laplacian
smooth loss L_r (paper Sec. 3.5, Eq. 5).

This module implements the deterministic pieces of that backbone:

* an **icosphere** template generator (recursive subdivision of a regular
  icosahedron, vertices projected to the unit sphere) -- the fixed base mesh;
* **offset deformation**, both free (per-vertex 3-vectors) and along the vertex
  *normal* direction (the paper's "displacement along the surface normal"
  parametrisation reused in its stylization branch, Sec. 3.6);
* the **flatten loss**: mean of ``(1 - cos(theta))`` over mesh edges, where
  ``theta`` is the dihedral angle between the two faces sharing an edge; it is 0
  for a perfectly flat (coplanar) neighbourhood and grows as faces fold, so
  minimising it smooths creases.  This is distinct from Laplacian smoothing
  (which pulls vertices toward neighbourhood centroids) and complements the
  existing ``geometry.craftsman_relative_laplacian`` operators.

Stdlib-only, deterministic.  The learned decoder weights, differentiable
renderer and CLIP guidance are external.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Face = Tuple[int, int, int]


def _normalize(v: Vec3) -> Vec3:
    x, y, z = v
    n = math.sqrt(x * x + y * y + z * z)
    if n == 0.0:
        return (0.0, 0.0, 0.0)
    return (x / n, y / n, z / n)


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def icosphere(subdivisions: int = 0) -> Tuple[List[Vec3], List[Face]]:
    """Return ``(vertices, faces)`` of a unit icosphere.

    ``subdivisions=0`` gives the 12-vertex, 20-face icosahedron; each additional
    level splits every triangle into four and reprojects new vertices onto the
    unit sphere.  Vertex order is deterministic.
    """
    if subdivisions < 0:
        raise ValueError("subdivisions must be >= 0")
    t = (1.0 + math.sqrt(5.0)) / 2.0
    base = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ]
    verts: List[Vec3] = [_normalize((float(x), float(y), float(z))) for x, y, z in base]
    faces: List[Face] = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    for _ in range(subdivisions):
        midcache: Dict[Tuple[int, int], int] = {}
        new_faces: List[Face] = []

        def midpoint(i: int, j: int) -> int:
            key = (i, j) if i < j else (j, i)
            if key in midcache:
                return midcache[key]
            vi, vj = verts[i], verts[j]
            m = _normalize(
                ((vi[0] + vj[0]) / 2.0, (vi[1] + vj[1]) / 2.0, (vi[2] + vj[2]) / 2.0)
            )
            verts.append(m)
            idx = len(verts) - 1
            midcache[key] = idx
            return idx

        for a, b, c in faces:
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_faces.extend([(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)])
        faces = new_faces
    return verts, faces


def apply_offsets(
    vertices: Sequence[Vec3], offsets: Sequence[Vec3]
) -> List[Vec3]:
    """Add a per-vertex 3-vector offset field to the template vertices."""
    if len(vertices) != len(offsets):
        raise ValueError("offsets must match vertex count")
    out: List[Vec3] = []
    for v, o in zip(vertices, offsets):
        out.append((v[0] + o[0], v[1] + o[1], v[2] + o[2]))
    return out


def vertex_normals(
    vertices: Sequence[Vec3], faces: Sequence[Face]
) -> List[Vec3]:
    """Area-weighted vertex normals (unit length) from face geometry."""
    acc: List[List[float]] = [[0.0, 0.0, 0.0] for _ in vertices]
    for a, b, c in faces:
        va, vb, vc = vertices[a], vertices[b], vertices[c]
        n = _cross(_sub(vb, va), _sub(vc, va))  # magnitude = 2 * area
        for idx in (a, b, c):
            acc[idx][0] += n[0]
            acc[idx][1] += n[1]
            acc[idx][2] += n[2]
    return [_normalize((v[0], v[1], v[2])) for v in acc]


def apply_normal_displacement(
    vertices: Sequence[Vec3],
    faces: Sequence[Face],
    displacements: Sequence[float],
) -> List[Vec3]:
    """Displace each vertex by a scalar amount along its surface normal.

    This mirrors the stylization branch's ``d_p`` displacement along the vertex
    normal (paper Sec. 3.6).
    """
    if len(vertices) != len(displacements):
        raise ValueError("displacements must match vertex count")
    normals = vertex_normals(vertices, faces)
    out: List[Vec3] = []
    for v, n, d in zip(vertices, normals, displacements):
        out.append((v[0] + n[0] * d, v[1] + n[1] * d, v[2] + n[2] * d))
    return out


def _face_normal(vertices: Sequence[Vec3], f: Face) -> Vec3:
    va, vb, vc = vertices[f[0]], vertices[f[1]], vertices[f[2]]
    return _normalize(_cross(_sub(vb, va), _sub(vc, va)))


def _edge_faces(faces: Sequence[Face]) -> Dict[Tuple[int, int], List[int]]:
    edges: Dict[Tuple[int, int], List[int]] = {}
    for fi, (a, b, c) in enumerate(faces):
        for i, j in ((a, b), (b, c), (c, a)):
            key = (i, j) if i < j else (j, i)
            edges.setdefault(key, []).append(fi)
    return edges


def flatten_loss(vertices: Sequence[Vec3], faces: Sequence[Face]) -> float:
    """Mean ``1 - cos(dihedral)`` over interior edges (0 = flat, up to 2 = folded).

    For every edge shared by exactly two faces we take the angle between the two
    face normals; coplanar faces contribute 0.  Boundary edges (one face) and
    non-manifold edges (>2 faces) are skipped.  A mesh with no interior edges
    returns 0.
    """
    edges = _edge_faces(faces)
    normals = [_face_normal(vertices, f) for f in faces]
    total = 0.0
    count = 0
    for fis in edges.values():
        if len(fis) != 2:
            continue
        n0 = normals[fis[0]]
        n1 = normals[fis[1]]
        cos = max(-1.0, min(1.0, _dot(n0, n1)))
        total += 1.0 - cos
        count += 1
    if count == 0:
        return 0.0
    return total / count
