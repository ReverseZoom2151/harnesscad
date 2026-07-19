"""Edge-convexity classification and Attributed Adjacency Graph (deterministic, stdlib-only).

Ported from the Hierarchical-CADNet edge routines used by QueryCAD
(``src/cad_service/utils/hierarchical_cadnet_utils.py``). QueryCAD grounds a
natural-language part query onto B-rep faces, and its first geometric step is to
build a *face adjacency graph* in which every shared edge between two faces is
labelled **convex**, **concave**, or **smooth**. That single per-edge label is
what lets a downstream feature detector tell a pocket (bounded by concave edges)
from a boss (bounded by convex edges) without any learned model.

The original computes the label with an OpenCascade kernel: it samples the
outward unit normals ``n0``, ``n1`` of the two faces at the edge midpoint and the
edge tangent ``t`` there, then takes the sign of ``dot(cross(n0, n1), t)`` (with
the cross-product order flipped when the edge is reversed). That sign is +1 for a
convex edge, -1 for a concave edge and 0 for a smooth/tangent edge. The kernel is
only a *supplier* of the three vectors; the classification itself is pure
arithmetic, reimplemented here over plain 3-vectors so it works with any
geometry source.

On top of the per-edge rule this module builds an **Attributed Adjacency Graph**
(AAG): faces are nodes, each shared edge is an arc carrying its convexity label,
so callers can query "which faces border ``f`` across a concave edge" -- the
canonical primitive for machining-feature recognition.

Everything is deterministic and stdlib-only; no NumPy, no kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from harnesscad.domain.geometry.topology import brep_entity_ids

Vec3 = Tuple[float, float, float]

CONVEX = "convex"
CONCAVE = "concave"
SMOOTH = "smooth"

_EPS = 1e-9


def _as_vec3(v: Sequence[float]) -> Vec3:
    if len(v) != 3:
        raise ValueError("expected a 3-component vector, got %d" % len(v))
    return (float(v[0]), float(v[1]), float(v[2]))


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _length(v: Vec3) -> float:
    return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) ** 0.5


def classify_edge_convexity(
    normal_a: Sequence[float],
    normal_b: Sequence[float],
    tangent: Sequence[float],
    forward: bool = True,
    tolerance: float = 1e-6,
) -> str:
    """Classify a shared edge as ``convex``, ``concave`` or ``smooth``.

    ``normal_a`` / ``normal_b`` are the outward face normals sampled at a common
    point on the edge (order = face ``a`` then face ``b``); ``tangent`` is the
    edge tangent at that point. ``forward`` mirrors OpenCascade's edge
    orientation flag -- a reversed edge swaps the cross-product operands, exactly
    as in the source.

    The convexity is the sign of ``dot(cross(n_a, n_b), t)``. The vectors need
    not be unit length; the magnitude is scaled out by ``tolerance``, which is
    compared against the value normalised by the operand lengths so it behaves
    like an angular dead-band around the tangent (smooth) case.
    """
    na = _as_vec3(normal_a)
    nb = _as_vec3(normal_b)
    t = _as_vec3(tangent)

    if forward:
        cp = _cross(na, nb)
    else:
        cp = _cross(nb, na)

    r = _dot(cp, t)

    # Normalise so ``tolerance`` is a scale-free dead-band: divide by the product
    # of operand magnitudes (|n_a| |n_b| |t|) which bounds |dot(cross,.)|.
    scale = _length(na) * _length(nb) * _length(t)
    if scale <= _EPS:
        return SMOOTH
    rn = r / scale

    if rn > tolerance:
        return CONVEX
    if rn < -tolerance:
        return CONCAVE
    return SMOOTH


# The continuous three-way sign is a strict *subset* of JoinABLe's discrete
# six-state ``Convexity`` enum. These helpers lift a label (or a full
# classification) into that discrete id via the enum module's authoritative
# ``EDGE_CONVEXITY_TO_ID`` bridge -- never a second mapping -- so a caller that
# wants JoinABLe-compatible integer ids can get them. The three states the
# continuous sign cannot express (``None`` for faces, ``Non-manifold``,
# ``Degenerate``) are not producible from a sign and remain reachable through
# ``brep_entity_ids.classify`` by their wire names.


def discrete_convexity(label: str) -> "brep_entity_ids.Convexity":
    """Lift a continuous convexity label to its discrete :class:`Convexity` state.

    ``label`` is one of :data:`CONVEX`, :data:`CONCAVE`, :data:`SMOOTH` (exactly
    the strings :func:`classify_edge_convexity` returns). The mapping is the enum
    module's :data:`~...brep_entity_ids.EDGE_CONVEXITY_TO_ID` bridge. Raises
    ``KeyError`` for any label outside the continuous three-way set.
    """
    cid = brep_entity_ids.EDGE_CONVEXITY_TO_ID[label]
    return brep_entity_ids.Convexity(cid)


def classify_edge_convexity_id(
    normal_a: Sequence[float],
    normal_b: Sequence[float],
    tangent: Sequence[float],
    forward: bool = True,
    tolerance: float = 1e-6,
) -> "brep_entity_ids.Convexity":
    """Discrete-id variant of :func:`classify_edge_convexity`.

    Runs the identical sign classification and returns the JoinABLe discrete
    :class:`Convexity` state (a ``Convexity`` ``IntEnum``, so it is also its
    integer id) instead of the continuous string label. The string API and its
    return type are unchanged; this is an additive path for callers that want the
    discrete taxonomy, with ``None`` / ``Non-manifold`` / ``Degenerate`` still
    available directly from :mod:`...brep_entity_ids`.
    """
    label = classify_edge_convexity(
        normal_a, normal_b, tangent, forward=forward, tolerance=tolerance
    )
    return discrete_convexity(label)


def dihedral_angle(
    normal_a: Sequence[float],
    normal_b: Sequence[float],
) -> float:
    """Return the interior dihedral-complement angle (radians) between two faces.

    This is the unsigned angle between the two outward normals, i.e.
    ``acos(dot(na, nb) / (|na||nb|))`` clamped to ``[0, pi]``. A flat/smooth
    joint gives ~0; a sharp right-angle edge gives ~pi/2. Convexity sign is not
    encoded here -- use :func:`classify_edge_convexity` for that.
    """
    na = _as_vec3(normal_a)
    nb = _as_vec3(normal_b)
    la = _length(na)
    lb = _length(nb)
    if la <= _EPS or lb <= _EPS:
        return 0.0
    c = _dot(na, nb) / (la * lb)
    if c > 1.0:
        c = 1.0
    elif c < -1.0:
        c = -1.0
    # math.acos without importing math at module top-level for the hot path
    import math

    return math.acos(c)


@dataclass
class EdgeArc:
    """A labelled arc between two faces in the Attributed Adjacency Graph."""

    face_a: int
    face_b: int
    convexity: str


@dataclass
class AttributedAdjacencyGraph:
    """Faces as nodes; shared edges as convexity-labelled arcs.

    The graph is undirected: an arc ``(a, b, label)`` is stored once but reachable
    from either endpoint via :meth:`neighbors`. Face ids are arbitrary hashables
    (typically ``int`` face indices).
    """

    faces: List[int] = field(default_factory=list)
    arcs: List[EdgeArc] = field(default_factory=list)
    _adj: Dict[int, List[Tuple[int, str]]] = field(default_factory=dict, repr=False)

    def add_face(self, face: int) -> None:
        if face not in self._adj:
            self.faces.append(face)
            self._adj[face] = []

    def add_edge(self, face_a: int, face_b: int, convexity: str) -> None:
        if convexity not in (CONVEX, CONCAVE, SMOOTH):
            raise ValueError("unknown convexity label: %r" % (convexity,))
        self.add_face(face_a)
        self.add_face(face_b)
        self.arcs.append(EdgeArc(face_a, face_b, convexity))
        self._adj[face_a].append((face_b, convexity))
        self._adj[face_b].append((face_a, convexity))

    def neighbors(self, face: int, convexity: str | None = None) -> List[int]:
        """Faces adjacent to ``face``; optionally filtered by arc label.

        Result is de-duplicated and returned in first-seen order for
        determinism.
        """
        out: List[int] = []
        seen = set()
        for other, label in self._adj.get(face, ()):  # noqa: B007
            if convexity is not None and label != convexity:
                continue
            if other in seen:
                continue
            seen.add(other)
            out.append(other)
        return out

    def concave_neighbors(self, face: int) -> List[int]:
        return self.neighbors(face, CONCAVE)

    def convex_neighbors(self, face: int) -> List[int]:
        return self.neighbors(face, CONVEX)

    def convexity_histogram(self) -> Dict[str, int]:
        """Count arcs per label -- a cheap global convexity fingerprint."""
        hist = {CONVEX: 0, CONCAVE: 0, SMOOTH: 0}
        for arc in self.arcs:
            hist[arc.convexity] += 1
        return hist


def build_aag(edge_records: Sequence[dict]) -> AttributedAdjacencyGraph:
    """Build an :class:`AttributedAdjacencyGraph` from raw edge samples.

    Each record is a mapping with keys ``face_a``, ``face_b``, ``normal_a``,
    ``normal_b``, ``tangent`` and an optional ``forward`` flag (default True).
    Edges with fewer than two adjacent faces (seam/boundary) should be omitted by
    the caller; they carry no convexity. Processing order follows the input for
    determinism.
    """
    g = AttributedAdjacencyGraph()
    for rec in edge_records:
        label = classify_edge_convexity(
            rec["normal_a"],
            rec["normal_b"],
            rec["tangent"],
            forward=rec.get("forward", True),
        )
        g.add_edge(int(rec["face_a"]), int(rec["face_b"]), label)
    return g
