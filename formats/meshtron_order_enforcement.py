"""Meshtron robust-sampling / mesh-sequence ordering enforcement (Sec. 3.4).

To keep autoregressive generation inside the data distribution, Meshtron
constrains sampling so that a generated mesh sequence obeys the ordering it was
built with (Sec. 2, see ``meshtron_ordering``):

  * vertex coordinates within a face follow a lexicographic ascending order,
  * the coordinates of a new face are lexicographically >= the previous face,
  * end-of-sequence tokens may appear only at the start of a new face.

For each token position the model would otherwise place mass on categories that
violate this order; the paper reports the enforcement prevents ~32% invalid
predictions at 1024-level quantization. This module reproduces the *checker*:

  * validity of a fully decoded stream,
  * the count / mask of invalid categories the constraint forbids at a given
    coordinate position (the quantity benchmarked in Sec. 3.4),
  * an aggregate "fraction of the categorical distribution that is invalid"
    over a whole sequence, matching the paper's evaluation protocol.

Coordinates are integer quantization bins in ``[0, num_bins)`` and each vertex
is a triple emitted in y-z-x priority order (as in ``meshtron_ordering``), so
that plain tuple comparison is exactly the enforced lexicographic order.

Pure stdlib, deterministic (no wall clock, no RNG).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

Triple = Tuple[int, int, int]

COORDS_PER_VERTEX = 3
VERTICES_PER_FACE = 3


# --------------------------------------------------------------------------- #
# Vertex / face level validity
# --------------------------------------------------------------------------- #
def is_face_vertices_ordered(vertices: Sequence[Triple]) -> bool:
    """True if the vertices of one face are lexicographically non-decreasing."""
    return all(tuple(vertices[i]) <= tuple(vertices[i + 1])
               for i in range(len(vertices) - 1))


def _face_key(face: Sequence[Triple]) -> Tuple[Triple, ...]:
    return tuple(tuple(v) for v in face)


def is_stream_ordered(faces: Sequence[Sequence[Triple]]) -> bool:
    """True if every face is internally ordered and faces are non-decreasing."""
    for face in faces:
        if len(face) != VERTICES_PER_FACE:
            raise ValueError("each face must have exactly 3 vertices")
        if not is_face_vertices_ordered(face):
            return False
    keys = [_face_key(f) for f in faces]
    return all(keys[i] <= keys[i + 1] for i in range(len(keys) - 1))


# --------------------------------------------------------------------------- #
# Per-coordinate invalid-category counting (the Sec. 3.4 benchmark)
# --------------------------------------------------------------------------- #
def invalid_category_count(
    lower_bound: Triple, prefix: Sequence[int], num_bins: int
) -> int:
    """Count forbidden categories for the next coordinate of a vertex.

    A vertex triple must be lexicographically ``>= lower_bound`` (the previous
    vertex it must not precede). Given the coordinates chosen so far
    (``prefix``, length 0/1/2) the next coordinate's valid range is constrained
    only while the prefix exactly matches ``lower_bound``'s prefix:

    * prefix equals ``lower_bound[:len(prefix)]`` -> next value must be
      ``>= lower_bound[len(prefix)]``; the lower values are invalid,
    * prefix already lexicographically greater -> nothing is forbidden.

    Returns how many of the ``num_bins`` categories are invalid.
    """
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    k = len(prefix)
    if k >= COORDS_PER_VERTEX:
        raise ValueError("prefix must be shorter than a full vertex")
    lead = tuple(lower_bound[:k])
    if tuple(prefix) < lead:
        # Should not happen for a valid partial generation.
        raise ValueError("prefix is below the lower bound (already invalid)")
    if tuple(prefix) > lead:
        return 0  # unconstrained: the vertex is already strictly greater
    forbidden = lower_bound[k]
    if forbidden < 0:
        forbidden = 0
    if forbidden > num_bins:
        forbidden = num_bins
    return forbidden


def invalid_category_mask(
    lower_bound: Triple, prefix: Sequence[int], num_bins: int
) -> List[bool]:
    """Boolean mask (length ``num_bins``): True where a category is forbidden."""
    count = invalid_category_count(lower_bound, prefix, num_bins)
    return [b < count for b in range(num_bins)]


def eos_allowed(coord_position: int) -> bool:
    """True if an end-of-sequence token may be emitted at ``coord_position``.

    EOS is permitted only at the start of a new face, i.e. at coordinate
    positions that are a multiple of 9 (a whole number of triangles emitted).
    """
    if coord_position < 0:
        raise ValueError("coord_position must be non-negative")
    return coord_position % (COORDS_PER_VERTEX * VERTICES_PER_FACE) == 0


# --------------------------------------------------------------------------- #
# Aggregate invalid fraction over a whole (valid) sequence
# --------------------------------------------------------------------------- #
def _lower_bound_for_vertex(
    faces: Sequence[Sequence[Triple]], fi: int, vj: int
) -> Triple:
    """Lower bound a vertex must satisfy given previously emitted vertices."""
    if vj > 0:
        return tuple(faces[fi][vj - 1])          # previous vertex in same face
    if fi > 0:
        return tuple(faces[fi - 1][0])           # first vertex of previous face
    return (0, 0, 0)                              # very first vertex


def invalid_fraction(
    faces: Sequence[Sequence[Triple]], num_bins: int
) -> float:
    """Fraction of the categorical space forbidden across the whole sequence.

    Simulates generation of every coordinate of an already-ordered mesh and
    averages ``invalid_category_count / num_bins`` over all coordinate
    positions -- the metric reported in Sec. 3.4 (e.g. ~0.32 at 1024 bins).
    Raises if the input is not itself ordered.
    """
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    if not is_stream_ordered(faces):
        raise ValueError("faces must already be in enforced order")
    total_invalid = 0
    total_positions = 0
    for fi, face in enumerate(faces):
        for vj, vertex in enumerate(face):
            lb = _lower_bound_for_vertex(faces, fi, vj)
            for k in range(COORDS_PER_VERTEX):
                total_invalid += invalid_category_count(lb, vertex[:k], num_bins)
                total_positions += 1
    if total_positions == 0:
        return 0.0
    return total_invalid / (total_positions * num_bins)
