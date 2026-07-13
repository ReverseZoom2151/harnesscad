"""Mesh-based Betti numbers and the CADGenBench topology-match score.

Ported (deterministically, stdlib-only) from the CADGenBench "Topology
Match" axis. Two pieces:

1. Betti numbers ``(b0, b1, b2)`` of a *solid* recovered from its
   tessellated boundary mesh:

   - ``b0``: connected solid components (pieces of material),
   - ``b1``: independent through-handles (through-holes),
   - ``b2``: enclosed internal voids (cavities).

   Pipeline: union-find components of the triangle mesh; each component is
   classified as an outer shell of a solid or the inner shell of a void by
   the parity of how many *other* components contain an interior probe point
   (even/odd ray casting); ``b1`` is then recovered from the surface Euler
   characteristic via ``chi(boundary) = 2 * (b0 - b1 + b2)``, i.e.

       b1 = b0 + b2 - chi / 2

   This is representation-invariant: the same physical part meshed two ways
   yields the same triple. Blind features (pockets, fillets, chamfers) are
   topologically trivial and leave the triple unchanged, by design.

2. Score. Per-axis *fuzzy log-ratio*, sharpened, then the **product**:

       s_i  = exp(-alpha * |log((b_cand_i + 1) / (b_gt_i + 1))|)
            = ((min + 1) / (max + 1)) ** alpha
       topo = s_0 * s_1 * s_2

   with ``alpha = BETTI_SHARPNESS = 2.0``. The ``+1`` shift keeps the ratio
   finite at zero counts; the product (not the mean) means one wrong count
   collapses the aggregate, because topology is discrete and "two of three
   invariants right" is not a partial match.

How this differs from what the harness already has: ``bench.topodiff_topology_consistency``
scores Betti agreement as *exact-match indicators* over **voxel** grids, and
``bench.evocad_topology_metrics`` only compares the Euler characteristic of a
surface. Neither derives ``(b0, b1, b2)`` of a solid from a boundary mesh, and
neither uses the graded log-ratio product score defined here.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

Vertex = Tuple[float, float, float]
Triangle = Tuple[int, int, int]

# Sharpness exponent on the per-axis fuzzy log-ratio. At 1.0 the score is
# (min + 1) / (max + 1); 2.0 is deliberately strict, so doubling a count
# (2 -> 4 holes) scores 0.36 rather than 0.60.
BETTI_SHARPNESS = 2.0

_EPS = 1e-12
_GRAZE_EPS = 1e-9

# Deterministic ray directions, tried in order. The axis ray comes first;
# the rest are fixed off-axis directions used when the axis ray grazes a
# triangle edge (which would poison the even/odd parity count).
_RAY_DIRECTIONS: Tuple[Vertex, ...] = (
    (1.0, 0.0, 0.0),
    (0.7371, 0.4293, 0.5213),
    (-0.3179, 0.8137, 0.4871),
    (0.5531, -0.2687, 0.7885),
    (-0.6217, -0.5493, 0.5581),
    (0.2143, 0.9091, -0.3573),
    (-0.8419, 0.3271, -0.4293),
    (0.4127, -0.7919, -0.4499),
)


class MeshGateError(ValueError):
    """The mesh is not a closed, orientable, manifold triangle surface."""


@dataclass(frozen=True)
class MeshSurface:
    """A welded triangle surface: shared corners must share a vertex index."""

    vertices: Sequence[Vertex]
    triangles: Sequence[Triangle]

    @property
    def n_triangles(self) -> int:
        return len(self.triangles)

    @property
    def n_vertices(self) -> int:
        return len(set(i for tri in self.triangles for i in tri))


@dataclass(frozen=True)
class BettiResult:
    """Betti numbers of one solid plus the diagnostics that produced them."""

    b0: int
    b1: int
    b2: int
    chi_surface: int
    n_components: int
    n_triangles: int
    n_vertices: int

    def as_vector(self) -> Tuple[int, int, int]:
        return (self.b0, self.b1, self.b2)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TopoMatchResult:
    candidate: BettiResult
    gt: BettiResult
    per_axis_scores: Dict[str, float]
    score: float

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate.to_dict(),
            "gt": self.gt.to_dict(),
            "per_axis_scores": dict(self.per_axis_scores),
            "score": self.score,
        }


# ---------------------------------------------------------------------------
# Mesh gate
# ---------------------------------------------------------------------------


def mesh_gate_errors(mesh: MeshSurface) -> List[str]:
    """Return the reasons *mesh* is not a closed orientable manifold ([] = ok).

    Three conditions, matching the CADGenBench validity gate's mesh stage:

    - manifold: every undirected edge is used by at most two triangles,
    - closed: every undirected edge is used by exactly two (equivalently 3F = 2E),
    - orientation-consistent: the two triangles on a shared edge traverse it in
      opposite directions.
    """
    errors: List[str] = []
    if not mesh.triangles:
        return ["mesh has no triangles"]

    directed: Dict[Tuple[int, int], int] = {}
    undirected: Dict[Tuple[int, int], int] = {}
    for tri in mesh.triangles:
        a, b, c = tri
        if a == b or b == c or a == c:
            errors.append(f"degenerate triangle {tri}")
            continue
        for u, v in ((a, b), (b, c), (c, a)):
            directed[(u, v)] = directed.get((u, v), 0) + 1
            key = (u, v) if u < v else (v, u)
            undirected[key] = undirected.get(key, 0) + 1

    non_manifold = sum(1 for n in undirected.values() if n > 2)
    if non_manifold:
        errors.append(f"non-manifold: {non_manifold} edge(s) in more than 2 triangles")
    naked = sum(1 for n in undirected.values() if n < 2)
    if naked:
        errors.append(f"not closed: {naked} boundary edge(s) in fewer than 2 triangles")

    flipped = sum(1 for n in directed.values() if n > 1)
    if flipped:
        errors.append(
            f"orientation inconsistent: {flipped} half-edge(s) traversed twice "
            "in the same direction"
        )
    return errors


def assert_mesh_gate(mesh: MeshSurface) -> None:
    """Raise :class:`MeshGateError` when the mesh fails the gate."""
    errors = mesh_gate_errors(mesh)
    if errors:
        raise MeshGateError("; ".join(errors))


# ---------------------------------------------------------------------------
# Betti numbers
# ---------------------------------------------------------------------------


def euler_characteristic(mesh: MeshSurface) -> int:
    """chi = V - E + F of the welded surface (V counts referenced vertices)."""
    used = set()
    edges = set()
    for a, b, c in mesh.triangles:
        used.update((a, b, c))
        for u, v in ((a, b), (b, c), (c, a)):
            edges.add((u, v) if u < v else (v, u))
    return len(used) - len(edges) + len(mesh.triangles)


def triangle_components(mesh: MeshSurface) -> List[List[Triangle]]:
    """Group triangles into connected components (union-find on vertices)."""
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for a, b, c in mesh.triangles:
        union(a, b)
        union(b, c)

    groups: Dict[int, List[Triangle]] = {}
    for tri in mesh.triangles:
        groups.setdefault(find(tri[0]), []).append(tuple(tri))
    # Sort by first-appearance order for determinism.
    order = []
    seen = set()
    for tri in mesh.triangles:
        root = find(tri[0])
        if root not in seen:
            seen.add(root)
            order.append(root)
    return [groups[root] for root in order]


def compute_betti(mesh: MeshSurface) -> BettiResult:
    """Betti numbers of the solid bounded by *mesh* (gate is a precondition)."""
    assert_mesh_gate(mesh)
    chi = euler_characteristic(mesh)
    components = triangle_components(mesh)
    b0, b2 = classify_components(mesh.vertices, components)
    b1 = b0 + b2 - chi // 2
    return BettiResult(
        b0=b0,
        b1=b1,
        b2=b2,
        chi_surface=chi,
        n_components=len(components),
        n_triangles=mesh.n_triangles,
        n_vertices=mesh.n_vertices,
    )


def classify_components(
    vertices: Sequence[Vertex], components: Sequence[Sequence[Triangle]]
) -> Tuple[int, int]:
    """Split components into ``(b0, b2)`` by containment parity.

    A component whose interior probe is contained by an *even* number of other
    components is the outer shell of a distinct solid (counts toward ``b0``);
    an odd count means it is the inner shell of a void (counts toward ``b2``).
    """
    probes = [interior_point(vertices, comp) for comp in components]
    boxes = [_component_aabb(vertices, comp) for comp in components]
    b0 = 0
    b2 = 0
    for i, probe in enumerate(probes):
        depth = 0
        for j, comp in enumerate(components):
            if i == j:
                continue
            lo, hi = boxes[j]
            if any(probe[k] < lo[k] - 1e-9 or probe[k] > hi[k] + 1e-9 for k in range(3)):
                continue
            if point_in_component(probe, vertices, comp):
                depth += 1
        if depth % 2 == 0:
            b0 += 1
        else:
            b2 += 1
    return b0, b2


def interior_point(
    vertices: Sequence[Vertex], component: Sequence[Triangle]
) -> Vertex:
    """A probe point demonstrably inside a closed component.

    Takes the centroid of a seed triangle and nudges it a small step along the
    triangle normal in both directions, returning whichever side the ray test
    calls "inside". Falls back to the centroid for degenerate components.
    """
    lo, hi = _component_aabb(vertices, component)
    diag = math.sqrt(sum((hi[k] - lo[k]) ** 2 for k in range(3)))
    step = max(1e-6, 1e-4 * diag)

    seed = component[len(component) // 2]
    a, b, c = (vertices[seed[0]], vertices[seed[1]], vertices[seed[2]])
    centroid = tuple((a[k] + b[k] + c[k]) / 3.0 for k in range(3))
    n = _cross(_sub(b, a), _sub(c, a))
    length = math.sqrt(sum(x * x for x in n))
    if length <= _EPS:
        return centroid  # type: ignore[return-value]
    n = tuple(x / length for x in n)
    for sign in (-1.0, 1.0):
        probe = tuple(centroid[k] + sign * step * n[k] for k in range(3))
        if point_in_component(probe, vertices, component):  # type: ignore[arg-type]
            return probe  # type: ignore[return-value]
    return centroid  # type: ignore[return-value]


def point_in_component(
    point: Vertex, vertices: Sequence[Vertex], component: Sequence[Triangle]
) -> bool:
    """Even/odd ray casting (Moller-Trumbore) against a closed component.

    Deterministic: directions are tried from a fixed table, moving on whenever
    a triangle is *grazed* (a barycentric coordinate within ``1e-9`` of an
    edge), so a degenerate intersection never corrupts the parity count.
    """
    for direction in _RAY_DIRECTIONS:
        hits = 0
        grazed = False
        for tri in component:
            code, hit = _ray_triangle(point, direction, vertices, tri)
            if code == "graze":
                grazed = True
                break
            if hit:
                hits += 1
        if grazed:
            continue
        return hits % 2 == 1
    return False


def _ray_triangle(
    origin: Vertex,
    direction: Vertex,
    vertices: Sequence[Vertex],
    tri: Triangle,
) -> Tuple[str, bool]:
    """Return ``("ok", hit)`` or ``("graze", False)`` for one ray/triangle pair."""
    v0 = _sub(vertices[tri[0]], origin)
    v1 = _sub(vertices[tri[1]], origin)
    v2 = _sub(vertices[tri[2]], origin)
    edge1 = _sub(v1, v0)
    edge2 = _sub(v2, v0)
    h = _cross(direction, edge2)
    a = _dot(edge1, h)
    if abs(a) <= _EPS:
        return ("ok", False)  # ray parallel to the triangle plane
    f = 1.0 / a
    s = tuple(-x for x in v0)
    u = f * _dot(s, h)  # type: ignore[arg-type]
    q = _cross(s, edge1)  # type: ignore[arg-type]
    v = f * _dot(direction, q)
    t = f * _dot(edge2, q)
    if (
        abs(u) < _GRAZE_EPS
        or abs(u - 1.0) < _GRAZE_EPS
        or abs(v) < _GRAZE_EPS
        or abs(u + v - 1.0) < _GRAZE_EPS
    ):
        return ("graze", False)
    hit = 0.0 <= u <= 1.0 and v >= 0.0 and u + v <= 1.0 and t > _GRAZE_EPS
    return ("ok", hit)


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------


def betti_axis_score(
    b_cand: int, b_gt: int, *, sharpness: float = BETTI_SHARPNESS
) -> float:
    """Fuzzy log-ratio score for one Betti axis, in ``[0, 1]``.

    A negative count is not a real Betti number (it means the candidate mesh is
    degenerate), and scores ``0`` rather than feeding ``log`` a bad argument.
    """
    if b_cand < 0 or b_gt < 0:
        return 0.0
    return math.exp(-sharpness * abs(math.log((b_cand + 1) / (b_gt + 1))))


def topo_match_score(
    candidate: BettiResult, gt: BettiResult, *, sharpness: float = BETTI_SHARPNESS
) -> Tuple[float, Dict[str, float]]:
    """Return ``(score, per_axis_scores)``; score is the product over the axes."""
    per_axis = {
        "b0": betti_axis_score(candidate.b0, gt.b0, sharpness=sharpness),
        "b1": betti_axis_score(candidate.b1, gt.b1, sharpness=sharpness),
        "b2": betti_axis_score(candidate.b2, gt.b2, sharpness=sharpness),
    }
    score = per_axis["b0"] * per_axis["b1"] * per_axis["b2"]
    return score, per_axis


def topo_match(
    candidate: MeshSurface, gt: MeshSurface, *, sharpness: float = BETTI_SHARPNESS
) -> TopoMatchResult:
    """End-to-end: Betti both meshes, score the triple."""
    cand_betti = compute_betti(candidate)
    gt_betti = compute_betti(gt)
    score, per_axis = topo_match_score(cand_betti, gt_betti, sharpness=sharpness)
    return TopoMatchResult(
        candidate=cand_betti, gt=gt_betti, per_axis_scores=per_axis, score=score
    )


# ---------------------------------------------------------------------------
# Small vector helpers
# ---------------------------------------------------------------------------


def _sub(a: Sequence[float], b: Sequence[float]) -> Vertex:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Sequence[float], b: Sequence[float]) -> Vertex:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _component_aabb(
    vertices: Sequence[Vertex], component: Sequence[Triangle]
) -> Tuple[Vertex, Vertex]:
    used = set(i for tri in component for i in tri)
    pts = [vertices[i] for i in used]
    lo = tuple(min(p[k] for p in pts) for k in range(3))
    hi = tuple(max(p[k] for p in pts) for k in range(3))
    return lo, hi  # type: ignore[return-value]
