"""Mesh normal and base-flatness metrics used by physical refinement."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence

Point3 = tuple[float, float, float]
Face = tuple[int, int, int]


@dataclass(frozen=True)
class MeshStabilityMetrics:
    normal_inconsistency: float | None
    adjacent_pairs: int
    bottom_laplacian: float | None
    bottom_roughness: float | None
    bottom_vertices: int
    diagnostics: tuple[str, ...]


def _normal(a: Point3, b: Point3, c: Point3) -> Point3 | None:
    u = tuple(b[i] - a[i] for i in range(3))
    v = tuple(c[i] - a[i] for i in range(3))
    n = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
    length = sqrt(sum(x*x for x in n))
    return tuple(x / length for x in n) if length else None


def mesh_stability_metrics(
    vertices: Sequence[Point3],
    faces: Sequence[Face],
    *,
    bottom_height: float = 0.0,
) -> MeshStabilityMetrics:
    if bottom_height < 0:
        raise ValueError("bottom_height must be non-negative")
    diagnostics: list[str] = []
    normals = [_normal(*(vertices[i] for i in face)) for face in faces]
    edges: dict[tuple[int, int], list[int]] = {}
    neighbours = [set() for _ in vertices]
    for fi, face in enumerate(faces):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = tuple(sorted((a, b)))
            edges.setdefault(edge, []).append(fi)
            neighbours[a].add(b); neighbours[b].add(a)
    penalties = []
    for owners in edges.values():
        if len(owners) == 2:
            n1, n2 = normals[owners[0]], normals[owners[1]]
            if n1 is not None and n2 is not None:
                penalties.append(1.0 - sum(a*b for a, b in zip(n1, n2)))
    if any(n is None for n in normals):
        diagnostics.append("degenerate_faces")
    if not penalties:
        diagnostics.append("no_adjacent_face_pairs")

    if vertices:
        z0 = min(v[2] for v in vertices)
        bottom = [i for i, v in enumerate(vertices) if v[2] <= z0 + bottom_height]
    else:
        bottom = []
    lap = []
    for i in bottom:
        ns = neighbours[i]
        if ns:
            avg = tuple(sum(vertices[j][k] for j in ns) / len(ns) for k in range(3))
            lap.append(sqrt(sum((vertices[i][k] - avg[k])**2 for k in range(3))))
    if not bottom:
        diagnostics.append("no_bottom_vertices")
    zs = [vertices[i][2] for i in bottom]
    roughness = (sqrt(sum((z - sum(zs)/len(zs))**2 for z in zs) / len(zs))
                 if zs else None)
    return MeshStabilityMetrics(
        sum(penalties)/len(penalties) if penalties else None,
        len(penalties),
        sum(lap)/len(lap) if lap else None,
        roughness,
        len(bottom),
        tuple(diagnostics),
    )
