"""The B-Rep chain complex of ComplexGen (SIGGRAPH 2022) as a checkable structure.

ComplexGen models a B-Rep as a *chain complex*: three ordered cells --- corners
(0-cells), curves (1-cells) and patches (2-cells) --- plus the two incidence
matrices ``EV`` (curve x corner) and ``FE`` (patch x curve).  The paper's
complex-extraction stage solves a global optimisation whose *constraints* encode
exactly what makes such a complex structurally valid; those constraints are
deterministic and are what this module implements (the ILP/solver itself, and the
neural generator, are external).

Constraints lifted from ``PostProcess/complex_extraction.py``:

  * an **open** curve is incident to exactly 2 corners; a **closed** curve to 0;
  * every curve is incident to exactly 2 patches (watertight, orientable-ready);
  * the patch-corner incidence is the composition ``FC = FE . EV / 2`` -- i.e.
    each corner of a patch is shared by exactly 2 of that patch's curves, so
    ``FE . EV`` must be even entry-wise.  This is the chain-complex condition
    ``d1 . d2 = 0`` (mod 2) and it is what forces closed boundary loops;
  * a corner must have degree >= ``min_corner_degree`` (3 in a CAD B-Rep);
  * incidence entries may only exist between cells that are both present.

On top of the constraint checks the module derives structure:

  * :func:`patch_corner_incidence` -- the composed ``FC``;
  * :func:`patch_loops`            -- walk each patch's curve set into closed
    loops (a closed curve is its own loop);
  * :func:`euler_characteristic`   -- ``V - E + F``;
  * :func:`is_watertight`, :func:`is_connected`, :func:`check` / :func:`is_valid`.

Geometry-topology consistency (``check_geom_topo_cons`` in the reference) is
also available via :func:`check_geometry`: an incident curve endpoint must lie
within ``tol`` of its corner, an incident curve must lie within ``tol`` of its
patch sample grid, and an incident corner within ``tol`` of the patch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

Point = tuple[float, float, float]
Matrix = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Curve:
    """A 1-cell: an ordered polyline of sample points, open or closed."""
    points: tuple[Point, ...]
    closed: bool = False

    def endpoints(self) -> tuple[Point, Point]:
        return self.points[0], self.points[-1]


@dataclass(frozen=True)
class Patch:
    """A 2-cell: an unordered cloud of surface sample points (the paper's 20x20 grid)."""
    points: tuple[Point, ...]


@dataclass(frozen=True)
class ChainComplex:
    """Corners / curves / patches plus the ``EV`` and ``FE`` incidence matrices.

    ``curve_corner[i][j] == 1`` iff curve ``i`` is bounded by corner ``j``.
    ``patch_curve[k][i] == 1`` iff patch ``k`` is bounded by curve ``i``.
    """
    corners: tuple[Point, ...]
    curves: tuple[Curve, ...]
    patches: tuple[Patch, ...]
    curve_corner: Matrix
    patch_curve: Matrix

    def __post_init__(self):
        if len(self.curve_corner) != len(self.curves):
            raise ValueError("curve_corner must have one row per curve")
        for row in self.curve_corner:
            if len(row) != len(self.corners):
                raise ValueError("curve_corner rows must have one column per corner")
        if len(self.patch_curve) != len(self.patches):
            raise ValueError("patch_curve must have one row per patch")
        for row in self.patch_curve:
            if len(row) != len(self.curves):
                raise ValueError("patch_curve rows must have one column per curve")

    @property
    def n_corners(self) -> int:
        return len(self.corners)

    @property
    def n_curves(self) -> int:
        return len(self.curves)

    @property
    def n_patches(self) -> int:
        return len(self.patches)


@dataclass
class Diagnostic:
    """Result of :func:`check` -- ``valid`` plus the list of violations."""
    valid: bool
    violations: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.valid


def make_complex(corners, curves, patches, curve_corner, patch_curve) -> ChainComplex:
    """Build a :class:`ChainComplex` from loose sequences (normalising to tuples)."""
    return ChainComplex(
        corners=tuple(tuple(float(x) for x in c) for c in corners),
        curves=tuple(
            c if isinstance(c, Curve)
            else Curve(tuple(tuple(float(x) for x in p) for p in c[0]), bool(c[1]))
            for c in curves),
        patches=tuple(
            p if isinstance(p, Patch)
            else Patch(tuple(tuple(float(x) for x in q) for q in p))
            for p in patches),
        curve_corner=tuple(tuple(int(bool(v)) for v in row) for row in curve_corner),
        patch_curve=tuple(tuple(int(bool(v)) for v in row) for row in patch_curve),
    )


# --------------------------------------------------------------------------- #
# derived structure
# --------------------------------------------------------------------------- #
def corners_of_curve(cx: ChainComplex, curve: int) -> tuple[int, ...]:
    return tuple(j for j, v in enumerate(cx.curve_corner[curve]) if v)


def curves_of_patch(cx: ChainComplex, patch: int) -> tuple[int, ...]:
    return tuple(i for i, v in enumerate(cx.patch_curve[patch]) if v)


def patches_of_curve(cx: ChainComplex, curve: int) -> tuple[int, ...]:
    return tuple(k for k in range(cx.n_patches) if cx.patch_curve[k][curve])


def corner_degree(cx: ChainComplex, corner: int) -> int:
    return sum(cx.curve_corner[i][corner] for i in range(cx.n_curves))


def patch_corner_product(cx: ChainComplex) -> tuple[tuple[int, ...], ...]:
    """The raw composition ``FE . EV`` (patch x corner), before halving."""
    out = []
    for k in range(cx.n_patches):
        row = [0] * cx.n_corners
        for i in range(cx.n_curves):
            if not cx.patch_curve[k][i]:
                continue
            for j in range(cx.n_corners):
                row[j] += cx.curve_corner[i][j]
        out.append(tuple(row))
    return tuple(out)


def patch_corner_incidence(cx: ChainComplex) -> Matrix:
    """``FC = FE . EV / 2`` -- the induced patch-corner incidence.

    Raises :class:`ValueError` if the product has an odd entry (i.e. the complex
    fails ``d1 . d2 = 0`` and a patch boundary is not a union of closed loops).
    """
    prod = patch_corner_product(cx)
    out = []
    for k, row in enumerate(prod):
        new = []
        for j, v in enumerate(row):
            if v % 2 != 0:
                raise ValueError(
                    f"patch {k} touches corner {j} an odd number of times ({v}); "
                    "boundary is not a union of closed loops")
            new.append(v // 2)
        out.append(tuple(new))
    return tuple(out)


def patch_loops(cx: ChainComplex, patch: int) -> tuple[tuple[int, ...], ...]:
    """Decompose a patch boundary into closed loops of curve indices.

    A closed curve forms a loop on its own.  Open curves are chained corner to
    corner; if a chain cannot be closed, :class:`ValueError` is raised.  Loops are
    returned in a deterministic order (ascending smallest curve index) and each
    loop starts at its own smallest curve index.
    """
    curves = list(curves_of_patch(cx, patch))
    loops: list[tuple[int, ...]] = []
    open_curves = []
    for i in curves:
        if cx.curves[i].closed:
            loops.append((i,))
        else:
            open_curves.append(i)

    remaining = set(open_curves)
    while remaining:
        start = min(remaining)
        ends = corners_of_curve(cx, start)
        if len(ends) != 2:
            raise ValueError(f"open curve {start} does not have exactly 2 corners")
        remaining.discard(start)
        loop = [start]
        first_corner, cursor = ends
        while cursor != first_corner:
            nxt = None
            for i in sorted(remaining):
                if cx.curve_corner[i][cursor]:
                    nxt = i
                    break
            if nxt is None:
                raise ValueError(f"patch {patch}: boundary chain from curve {start} is open")
            remaining.discard(nxt)
            loop.append(nxt)
            ends = corners_of_curve(cx, nxt)
            if len(ends) != 2:
                raise ValueError(f"open curve {nxt} does not have exactly 2 corners")
            cursor = ends[0] if ends[1] == cursor else ends[1]
        loops.append(tuple(loop))
    return tuple(sorted(loops, key=lambda l: min(l)))


def euler_characteristic(cx: ChainComplex) -> int:
    return cx.n_corners - cx.n_curves + cx.n_patches


def is_watertight(cx: ChainComplex) -> bool:
    """Every curve bounds exactly 2 patches (no free/naked edge)."""
    return all(len(patches_of_curve(cx, i)) == 2 for i in range(cx.n_curves))


def is_connected(cx: ChainComplex) -> bool:
    """Connectivity of the corner-curve-patch incidence graph."""
    total = cx.n_corners + cx.n_curves + cx.n_patches
    if total == 0:
        return True
    adj: dict[int, set[int]] = {n: set() for n in range(total)}

    def curve_node(i):
        return cx.n_corners + i

    def patch_node(k):
        return cx.n_corners + cx.n_curves + k

    for i in range(cx.n_curves):
        for j in corners_of_curve(cx, i):
            adj[curve_node(i)].add(j)
            adj[j].add(curve_node(i))
    for k in range(cx.n_patches):
        for i in curves_of_patch(cx, k):
            adj[patch_node(k)].add(curve_node(i))
            adj[curve_node(i)].add(patch_node(k))

    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for nb in adj[node]:
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == total


# --------------------------------------------------------------------------- #
# validity
# --------------------------------------------------------------------------- #
def check(cx: ChainComplex, min_corner_degree: int = 3,
          require_watertight: bool = True) -> Diagnostic:
    """Run every structural constraint; return a :class:`Diagnostic`."""
    problems: list[str] = []

    for i, curve in enumerate(cx.curves):
        deg = len(corners_of_curve(cx, i))
        want = 0 if curve.closed else 2
        if deg != want:
            kind = "closed" if curve.closed else "open"
            problems.append(f"curve {i} ({kind}) has {deg} corners, expected {want}")

    if require_watertight:
        for i in range(cx.n_curves):
            npatch = len(patches_of_curve(cx, i))
            if npatch != 2:
                problems.append(f"curve {i} bounds {npatch} patches, expected 2")

    for j in range(cx.n_corners):
        deg = corner_degree(cx, j)
        if deg < min_corner_degree:
            problems.append(f"corner {j} has degree {deg} < {min_corner_degree}")

    prod = patch_corner_product(cx)
    for k, row in enumerate(prod):
        for j, v in enumerate(row):
            if v % 2 != 0:
                problems.append(
                    f"incidence inconsistency: patch {k} meets corner {j} {v} times (odd)")

    for k in range(cx.n_patches):
        if not curves_of_patch(cx, k):
            problems.append(f"patch {k} has no boundary curves")
            continue
        try:
            patch_loops(cx, k)
        except ValueError as exc:
            problems.append(str(exc))

    return Diagnostic(valid=not problems, violations=problems)


def is_valid(cx: ChainComplex, min_corner_degree: int = 3,
             require_watertight: bool = True) -> bool:
    return check(cx, min_corner_degree, require_watertight).valid


# --------------------------------------------------------------------------- #
# geometry-topology consistency (check_geom_topo_cons)
# --------------------------------------------------------------------------- #
def _dist(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _point_to_cloud(p, cloud) -> float:
    return min(_dist(p, q) for q in cloud)


def _mean_cloud_to_cloud(src, dst) -> float:
    return sum(_point_to_cloud(p, dst) for p in src) / len(src)


def check_geometry(cx: ChainComplex, tol: float = 0.1) -> Diagnostic:
    """Geometric consistency of the declared incidences.

    * an open curve incident to a corner must have an endpoint within ``tol``;
    * a curve incident to a patch must lie within mean distance ``tol`` of it;
    * a corner incident to a patch must be within ``tol`` of the patch samples.
    """
    problems: list[str] = []

    for i, curve in enumerate(cx.curves):
        if curve.closed:
            continue
        e1, e2 = curve.endpoints()
        for j in corners_of_curve(cx, i):
            corner = cx.corners[j]
            if min(_dist(e1, corner), _dist(e2, corner)) > tol:
                problems.append(f"curve {i} endpoint is farther than {tol} from corner {j}")

    for k, patch in enumerate(cx.patches):
        if not patch.points:
            continue
        for i in curves_of_patch(cx, k):
            d = _mean_cloud_to_cloud(cx.curves[i].points, patch.points)
            if d > tol:
                problems.append(f"curve {i} is {d:.4f} from patch {k} (> {tol})")

    try:
        fc = patch_corner_incidence(cx)
    except ValueError as exc:
        problems.append(str(exc))
        fc = None
    if fc is not None:
        for k, patch in enumerate(cx.patches):
            if not patch.points:
                continue
            for j, v in enumerate(fc[k]):
                if not v:
                    continue
                d = _point_to_cloud(cx.corners[j], patch.points)
                if d > tol:
                    problems.append(f"corner {j} is {d:.4f} from patch {k} (> {tol})")

    return Diagnostic(valid=not problems, violations=problems)


# --------------------------------------------------------------------------- #
# extraction helpers: thresholding a probabilistic complex into a definite one
# --------------------------------------------------------------------------- #
def threshold_incidence(similarity, threshold: float = 0.5) -> Matrix:
    """Round a probability / similarity matrix into a 0-1 incidence matrix."""
    return tuple(tuple(1 if v > threshold else 0 for v in row) for row in similarity)


def geometric_similarity(distance: float, sigma: float) -> float:
    """The paper's geometric similarity ``exp(-d^2 / sigma^2)`` (Sec. 4)."""
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    return math.exp(-(distance * distance) / (sigma * sigma))


def curve_corner_similarity(cx: ChainComplex, sigma: float = 0.2) -> tuple[tuple[float, ...], ...]:
    """``exp(-d^2/sigma^2)`` on the min distance from a curve's endpoints to each corner."""
    out = []
    for curve in cx.curves:
        e1, e2 = curve.endpoints()
        out.append(tuple(geometric_similarity(min(_dist(e1, c), _dist(e2, c)), sigma)
                         for c in cx.corners))
    return tuple(out)


def patch_curve_similarity(cx: ChainComplex, sigma: float = 0.1) -> tuple[tuple[float, ...], ...]:
    """``exp(-d^2/sigma^2)`` on the mean curve-to-patch sample distance."""
    out = []
    for patch in cx.patches:
        row = []
        for curve in cx.curves:
            d = _mean_cloud_to_cloud(curve.points, patch.points) if patch.points else float("inf")
            row.append(geometric_similarity(d, sigma) if patch.points else 0.0)
        out.append(tuple(row))
    return tuple(out)
