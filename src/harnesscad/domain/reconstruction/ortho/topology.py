"""Wireframe graph, deterministic planar cycles, clustering and manifold gate."""

from __future__ import annotations

from collections import defaultdict
import math

from .model import Diagnostic, Edge3D, FaceCluster, FaceLoop


def wireframe_graph(edges: tuple[Edge3D, ...], tolerance: float):
    def q(point):
        return tuple(round(value / tolerance) for value in point)
    representative = {}
    adjacency = defaultdict(list)
    for index, edge in enumerate(edges):
        a, b = q(edge.start), q(edge.end)
        representative.setdefault(a, edge.start)
        representative.setdefault(b, edge.end)
        adjacency[a].append((b, index))
        adjacency[b].append((a, index))
    return representative, {key: tuple(sorted(value)) for key, value in sorted(adjacency.items())}


def _plane(points, tolerance: float):
    a = points[0]
    for i in range(1, len(points) - 1):
        u = tuple(points[i][j] - a[j] for j in range(3))
        v = tuple(points[i + 1][j] - a[j] for j in range(3))
        n = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
        norm = math.sqrt(sum(x*x for x in n))
        if norm > tolerance:
            n = tuple(x / norm for x in n)
            if next(x for x in n if abs(x) > tolerance) < 0:
                n = tuple(-x for x in n)
            d = -sum(n[j] * a[j] for j in range(3))
            return (*n, d)
    return None


def _coplanar(points, tolerance: float):
    plane = _plane(points, tolerance)
    return plane is not None and all(
        abs(sum(plane[j]*p[j] for j in range(3)) + plane[3]) <= tolerance
        for p in points
    )


def find_face_loops(edges: tuple[Edge3D, ...], tolerance: float) -> tuple[FaceLoop, ...]:
    reps, graph = wireframe_graph(edges, tolerance)
    cycles = {}
    for start in graph:
        def walk(node, path, edge_path):
            for nxt, edge_index in graph[node]:
                if edge_index in edge_path:
                    continue
                if nxt == start and len(path) >= 3:
                    keys = path[:]
                    rotations = [tuple(keys[i:] + keys[:i]) for i in range(len(keys))]
                    rev = list(reversed(keys))
                    rotations += [tuple(rev[i:] + rev[:i]) for i in range(len(keys))]
                    canonical = min(rotations)
                    points = tuple(reps[key] for key in canonical)
                    if _coplanar(points, tolerance):
                        plane = _plane(points, tolerance)
                        cycles.setdefault(canonical, FaceLoop(
                            points, tuple(sorted(edge_path + [edge_index])), plane))
                    continue
                if nxt in path or len(path) >= len(graph):
                    continue
                candidate = [reps[key] for key in path + [nxt]]
                if len(candidate) < 4 or _coplanar(candidate, tolerance):
                    walk(nxt, path + [nxt], edge_path + [edge_index])
        walk(start, [start], [])
    # Chordless cycles are face candidates; supersets containing a smaller cycle
    # with the same plane are not minimal boundaries.
    ordered = sorted(cycles.values(), key=lambda loop: (len(loop.vertices), loop.vertices))
    minimal = []
    for loop in ordered:
        edge_set = set(loop.edge_indices)
        if any(set(other.edge_indices) < edge_set and _same_plane(other.plane, loop.plane, tolerance)
               for other in minimal):
            continue
        minimal.append(loop)
    return tuple(minimal)


def _same_plane(a, b, tolerance):
    return all(abs(x-y) <= tolerance for x, y in zip(a, b))


def _project(loop: FaceLoop):
    normal = loop.plane[:3]
    drop = max(range(3), key=lambda i: abs(normal[i]))
    axes = [i for i in range(3) if i != drop]
    return tuple((point[axes[0]], point[axes[1]]) for point in loop.vertices)


def _area(poly):
    return abs(sum(a[0]*b[1] - b[0]*a[1] for a, b in zip(poly, poly[1:]+poly[:1]))) / 2


def _contains(outer, inner, tolerance):
    # Bounding containment is sufficient for the axis-aligned nesting
    # rule and deliberately treats touching boundaries as not nested.
    ob = (min(x for x, _ in outer), min(y for _, y in outer),
          max(x for x, _ in outer), max(y for _, y in outer))
    ib = (min(x for x, _ in inner), min(y for _, y in inner),
          max(x for x, _ in inner), max(y for _, y in inner))
    return (ob[0] + tolerance < ib[0] and ob[1] + tolerance < ib[1]
            and ob[2] - tolerance > ib[2] and ob[3] - tolerance > ib[3])


def cluster_planar_loops(loops: tuple[FaceLoop, ...], tolerance: float):
    remaining = list(loops)
    faces = []
    while remaining:
        outer = max(remaining, key=lambda loop: (_area(_project(loop)), loop.vertices))
        remaining.remove(outer)
        inner = [loop for loop in remaining
                 if _same_plane(outer.plane, loop.plane, tolerance)
                 and _contains(_project(outer), _project(loop), tolerance)]
        for loop in inner:
            remaining.remove(loop)
        faces.append(FaceCluster(outer, tuple(sorted(inner, key=lambda x: x.vertices))))
    return tuple(sorted(faces, key=lambda face: face.outer.vertices))


def manifold_gate(faces: tuple[FaceCluster, ...], edge_count: int):
    incidence = [0] * edge_count
    for face in faces:
        for loop in (face.outer, *face.inner):
            for index in loop.edge_indices:
                incidence[index] += 1
    bad = tuple((index, count) for index, count in enumerate(incidence) if count != 2)
    diagnostics = (() if not bad else (Diagnostic(
        "non-manifold-edge-incidence", "every wireframe edge must bound exactly two faces",
        context={"incidence": bad}),))
    return not bad, diagnostics
