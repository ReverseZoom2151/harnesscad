"""Meshtron y-z-x mesh-sequence ordering convention (Hao et al., 2024).

Meshtron (Sec. 2, "From unordered to ordered mesh sequences") adopts the
PolyGen / Nash et al. (2020) convention to turn an *unordered* set of faces into
a single deterministic sequence an autoregressive model can consume:

  1. Vertices are arranged in **y-z-x order**, where ``y`` is the vertical axis
     (the paper renders meshes bottom-to-top, so ``y`` sorts first). This is the
     load-bearing difference from LLaMA-Mesh (paper 124,
     ``formats/llamamesh_tokenization.py``), which sorts vertices ``z``-``y``-``x``.
  2. Within each face, vertices are cyclically rotated so the lowest y-z-x
     vertex leads (winding / orientation preserved).
  3. Faces are sorted in ascending y-z-x order using the tuple of their
     (rotated) vertex keys.

Because the sort priority is ``(y, z, x)``, this module also emits the flattened
coordinate stream in that same priority order per vertex (``y`` then ``z`` then
``x``) so that lexicographic comparison of consecutive emitted vertices is
exactly the y-z-x key comparison the ordering enforces. This keeps the ordering
convention and the order-enforcement checker (``meshtron_order_enforcement``)
consistent.

Pure stdlib, deterministic (no wall clock, no RNG).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

Vertex = Tuple[float, float, float]
Face = Tuple[int, ...]
Key = Tuple[float, float, float]


# --------------------------------------------------------------------------- #
# y-z-x keys
# --------------------------------------------------------------------------- #
def yzx_key(vertex: Sequence[float]) -> Key:
    """Return the ``(y, z, x)`` sort key of a ``(x, y, z)`` vertex.

    ``y`` (the vertical axis) has the highest priority, then ``z``, then ``x``.
    This is deliberately distinct from LLaMA-Mesh's ``(z, y, x)`` key.
    """
    if len(vertex) != 3:
        raise ValueError("each vertex must have exactly 3 coordinates")
    return (vertex[1], vertex[2], vertex[0])


def sort_vertices_yzx(
    vertices: Sequence[Sequence[float]],
) -> Tuple[List[Vertex], Dict[int, int]]:
    """Sort ``vertices`` ascending by y-z-x key.

    Returns ``(sorted_vertices, remap)`` where ``remap`` maps each old index to
    its new (sorted) index. Ordering is stable on ties.
    """
    order = sorted(range(len(vertices)), key=lambda i: yzx_key(vertices[i]))
    remap = {old: new for new, old in enumerate(order)}
    sorted_vertices = [tuple(float(c) for c in vertices[old]) for old in order]
    return sorted_vertices, remap


def _rotate_min_index_first(face: Sequence[int]) -> Face:
    """Cyclically rotate ``face`` so its smallest index leads (winding kept).

    After :func:`sort_vertices_yzx` the vertex index order equals the y-z-x key
    order, so the smallest index is exactly the lowest y-z-x vertex.
    """
    if len(face) < 3:
        raise ValueError("a face needs at least 3 vertices")
    pivot = min(range(len(face)), key=lambda i: face[i])
    return tuple(face[pivot:]) + tuple(face[:pivot])


def face_sort_key(
    face: Sequence[int], sorted_vertices: Sequence[Sequence[float]]
) -> Tuple[Key, ...]:
    """Return the tuple of per-vertex y-z-x keys used to order faces."""
    return tuple(yzx_key(sorted_vertices[i]) for i in face)


def canonicalize_mesh_yzx(
    vertices: Sequence[Sequence[float]], faces: Sequence[Sequence[int]]
) -> Tuple[List[Vertex], List[Face]]:
    """Apply the full Meshtron ordering: sort vertices, rotate + sort faces.

    ``faces`` use 0-based indices. Returns ``(sorted_vertices, sorted_faces)``
    with vertices in ascending y-z-x order and faces rotated (lowest y-z-x
    vertex first) then sorted by their vertex-key tuples.
    """
    sorted_vertices, remap = sort_vertices_yzx(vertices)
    rotated: List[Face] = []
    for face in faces:
        if len(face) < 3:
            raise ValueError("a face needs at least 3 vertices")
        remapped = tuple(remap[idx] for idx in face)
        rotated.append(_rotate_min_index_first(remapped))
    rotated.sort(key=lambda f: face_sort_key(f, sorted_vertices))
    return sorted_vertices, rotated


# --------------------------------------------------------------------------- #
# Flattened coordinate / vertex / face streams (Eq. 1 levels of abstraction)
# --------------------------------------------------------------------------- #
def vertex_stream(
    vertices: Sequence[Vertex], faces: Sequence[Sequence[int]]
) -> List[Vertex]:
    """Vertex-level stream: face vertices concatenated in face order.

    A vertex shared by several faces appears once per face (Eq. 1, vertex
    level), matching how the autoregressive sequence is built.
    """
    out: List[Vertex] = []
    for face in faces:
        for idx in face:
            out.append(vertices[idx])
    return out


def coordinate_stream(
    vertices: Sequence[Vertex], faces: Sequence[Sequence[int]]
) -> List[float]:
    """Coordinate-level stream in y-z-x priority order per vertex.

    Each vertex contributes three values in ``(y, z, x)`` order so that
    lexicographic comparison of consecutive vertices equals their y-z-x key
    comparison. For a triangle mesh with ``N`` faces this yields ``9N`` values.
    """
    out: List[float] = []
    for vertex in vertex_stream(vertices, faces):
        y, z, x = vertex[1], vertex[2], vertex[0]
        out.extend((y, z, x))
    return out


def is_vertices_sorted_yzx(vertices: Sequence[Sequence[float]]) -> bool:
    """True if ``vertices`` are already in non-decreasing y-z-x order."""
    keys = [yzx_key(v) for v in vertices]
    return all(keys[i] <= keys[i + 1] for i in range(len(keys) - 1))
