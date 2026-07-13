"""Deterministic mesh Laplacian smoothing operators.

Extracted from the *only* non-learned, locally-reproducible primitive in
CraftsMan (Li et al.), namely the **relative Laplacian smoothing** term used to
stabilise the surface-normal-driven vertex optimisation (paper Eq. 6):

    x  <-  x_init + lambda * v * W(x - x_init)

where ``x_init`` is the position of a vertex on the *coarse* input mesh, ``W`` is
the combinatorial (umbrella) Laplacian operator, ``lambda`` is a smoothing
weight and ``v`` a per-vertex relative speed.  The crucial idea is that the
Laplacian is applied to the *relative displacement* ``x - x_init`` rather than to
the absolute positions.  Standard Laplacian smoothing (``x <- x + lambda*W(x)``)
drags every vertex toward the centroid of its neighbourhood, which causes thin
structures to diminish / collapse.  By smoothing only the displacement away from
the coarse mesh, the relative variant constrains vertices to the proximity of
the coarse mesh and avoids that collapse (paper Sec. 4.4, Fig. 8d).

Everything here is stdlib-only, deterministic (no randomness, no wall clock) and
operates on plain Python sequences of ``(x, y, z)`` vertices plus integer face
tuples.  The learned components of CraftsMan (3D latent-set diffusion,
multi-view diffusion, ControlNet-tile normal refiner, differentiable rendering)
are research-heavy/external and intentionally not reproduced here.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

Vector = Tuple[float, float, float]
Face = Sequence[int]


def _as_points(vertices: Sequence[Sequence[float]]) -> List[Vector]:
    points: List[Vector] = []
    for vertex in vertices:
        coords = tuple(float(component) for component in vertex)
        if len(coords) != 3:
            raise ValueError("each vertex must have exactly 3 coordinates")
        points.append((coords[0], coords[1], coords[2]))
    return points


def vertex_adjacency(num_vertices: int, faces: Sequence[Face]) -> List[List[int]]:
    """Return, for each vertex, the sorted list of distinct neighbour indices.

    Edges are derived from the consecutive-vertex pairs of every face, so the
    routine works for triangles, quads or arbitrary polygonal faces.  The result
    is fully determined by the connectivity (sorted, de-duplicated) and thus
    reproducible.
    """
    if num_vertices < 0:
        raise ValueError("num_vertices must be non-negative")
    neighbours: List[set] = [set() for _ in range(num_vertices)]
    for face in faces:
        indices = list(face)
        degree = len(indices)
        if degree < 2:
            continue
        for position in range(degree):
            a = indices[position]
            b = indices[(position + 1) % degree]
            if not (0 <= a < num_vertices and 0 <= b < num_vertices):
                raise IndexError("face references a vertex outside range")
            if a == b:
                continue
            neighbours[a].add(b)
            neighbours[b].add(a)
    return [sorted(group) for group in neighbours]


def uniform_laplacian(
    points: Sequence[Sequence[float]],
    adjacency: Sequence[Sequence[int]],
) -> List[Vector]:
    """Umbrella (combinatorial, degree-normalised) Laplacian of a field.

    For each vertex ``i`` returns ``mean(points[j] for j in neighbours(i)) -
    points[i]``.  Isolated vertices (no neighbours) map to the zero vector.  This
    is the operator ``W`` used by both the standard and relative smoothers below.
    """
    pts = _as_points(points)
    if len(adjacency) != len(pts):
        raise ValueError("adjacency length must match number of points")
    deltas: List[Vector] = []
    for index, point in enumerate(pts):
        neighbours = adjacency[index]
        if not neighbours:
            deltas.append((0.0, 0.0, 0.0))
            continue
        accum = [0.0, 0.0, 0.0]
        for neighbour in neighbours:
            other = pts[neighbour]
            accum[0] += other[0]
            accum[1] += other[1]
            accum[2] += other[2]
        inv = 1.0 / len(neighbours)
        deltas.append((
            accum[0] * inv - point[0],
            accum[1] * inv - point[1],
            accum[2] * inv - point[2],
        ))
    return deltas


def laplacian_smooth(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Face],
    *,
    iterations: int = 1,
    lam: float = 0.5,
    adjacency: Sequence[Sequence[int]] | None = None,
) -> List[Vector]:
    """Standard iterative Laplacian smoothing ``x <- x + lam * W(x)``.

    This is the baseline the CraftsMan paper contrasts against; it shrinks and
    can collapse thin features.  Provided for comparison and as a building block.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must lie in [0, 1]")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    points = _as_points(vertices)
    if adjacency is None:
        adjacency = vertex_adjacency(len(points), faces)
    for _ in range(iterations):
        deltas = uniform_laplacian(points, adjacency)
        points = [(
            point[0] + lam * delta[0],
            point[1] + lam * delta[1],
            point[2] + lam * delta[2],
        ) for point, delta in zip(points, deltas)]
    return points


def relative_laplacian_smooth(
    vertices: Sequence[Sequence[float]],
    initial: Sequence[Sequence[float]],
    faces: Sequence[Face],
    *,
    iterations: int = 1,
    lam: float = 0.5,
    speed: float | Sequence[float] = 1.0,
    adjacency: Sequence[Sequence[int]] | None = None,
) -> List[Vector]:
    """Relative Laplacian smoothing (CraftsMan Eq. 6).

    Applies ``x <- x_init + lam * v * W(x - x_init)`` where ``x_init`` is the
    corresponding coarse-mesh vertex, ``W`` the umbrella Laplacian and ``v`` a
    per-vertex (or scalar) relative speed.  Because the Laplacian acts on the
    displacement field ``x - x_init``, vertices are pulled toward a smoothed
    version of their *offset* from the coarse mesh, keeping them near the coarse
    surface and preventing the thin-feature collapse of absolute smoothing.

    ``vertices`` and ``initial`` must have the same length; ``initial`` is held
    fixed across all iterations.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must lie in [0, 1]")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    points = _as_points(vertices)
    anchors = _as_points(initial)
    if len(points) != len(anchors):
        raise ValueError("vertices and initial must have equal length")
    if isinstance(speed, (int, float)):
        speeds = [float(speed)] * len(points)
    else:
        speeds = [float(value) for value in speed]
        if len(speeds) != len(points):
            raise ValueError("per-vertex speed length must match vertices")
    if adjacency is None:
        adjacency = vertex_adjacency(len(points), faces)
    for _ in range(iterations):
        displacement = [(
            point[0] - anchor[0],
            point[1] - anchor[1],
            point[2] - anchor[2],
        ) for point, anchor in zip(points, anchors)]
        smoothed = uniform_laplacian(displacement, adjacency)
        points = [(
            anchor[0] + lam * speeds[i] * smoothed[i][0],
            anchor[1] + lam * speeds[i] * smoothed[i][1],
            anchor[2] + lam * speeds[i] * smoothed[i][2],
        ) for i, anchor in enumerate(anchors)]
    return points


def taubin_smooth(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Face],
    *,
    iterations: int = 1,
    lam: float = 0.5,
    mu: float = -0.53,
    adjacency: Sequence[Sequence[int]] | None = None,
) -> List[Vector]:
    """Taubin lambda|mu shrink-free smoothing.

    A deterministic, volume-preserving alternative to plain Laplacian smoothing:
    each iteration performs a shrinking pass with weight ``lam`` followed by an
    inflating pass with negative weight ``mu`` (``mu < -lam < 0``).  Included as a
    complementary in-scope mesh operator for the harness.
    """
    if not 0.0 < lam < 1.0:
        raise ValueError("lam must lie in (0, 1)")
    if not mu < -lam:
        raise ValueError("mu must satisfy mu < -lam < 0")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    points = _as_points(vertices)
    if adjacency is None:
        adjacency = vertex_adjacency(len(points), faces)

    def _apply(current: List[Vector], weight: float) -> List[Vector]:
        deltas = uniform_laplacian(current, adjacency)
        return [(
            point[0] + weight * delta[0],
            point[1] + weight * delta[1],
            point[2] + weight * delta[2],
        ) for point, delta in zip(current, deltas)]

    for _ in range(iterations):
        points = _apply(points, lam)
        points = _apply(points, mu)
    return points


def mean_displacement(
    points_a: Sequence[Sequence[float]],
    points_b: Sequence[Sequence[float]],
) -> float:
    """Mean Euclidean distance between two equally sized vertex sets.

    Convenience metric for quantifying how far a smoothing pass moved vertices
    from a reference (e.g. the coarse mesh), used to demonstrate the
    collapse-resistance of the relative operator.
    """
    a = _as_points(points_a)
    b = _as_points(points_b)
    if len(a) != len(b):
        raise ValueError("point sets must have equal length")
    if not a:
        return 0.0
    total = 0.0
    for p, q in zip(a, b):
        dx = p[0] - q[0]
        dy = p[1] - q[1]
        dz = p[2] - q[2]
        total += (dx * dx + dy * dy + dz * dz) ** 0.5
    return total / len(a)
