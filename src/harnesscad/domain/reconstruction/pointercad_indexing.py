"""Stable B-rep entity addressing for Pointer-CAD pointers.

A Pointer-CAD pointer is an *index* into the current B-rep's face list ``S_f`` or
edge list ``S_e`` (paper Sec. 4.1.1 / Sec. 10). For that index to be meaningful the
two lists must have a **deterministic, well-defined order** -- otherwise the same
pointer integer would address different geometry on different runs. This module
provides that addressing scheme.

Design (following the paper):

  * ``S_f`` always begins with the three *base planes* -- Right, Front, Top -- which
    Pointer-CAD encodes as distinct learnable slots that exist before any geometry
    is drawn (Sec. 4.1.1: "including the three base planes"). Model faces follow.
  * Each B-rep *edge* is the shared boundary of two faces ``i, j`` (Sec. 4.1.1:
    "the edge shared by the i-th and j-th faces"). We key an edge by the *sorted*
    face-index pair so orientation never changes its identity.
  * Ordering is canonical: base planes first (fixed order), then model faces by a
    stable sort key, then edges by their ``(min_face, max_face)`` key. Indices are
    assigned ``0..n-1`` in that order and are the pointer values.

Pure stdlib; kernel-agnostic. Faces/edges are described by small immutable records,
not an OCCT handle, so the indexing is reproducible and testable offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The three base planes are always present and always first, in this fixed order.
BASE_PLANES: tuple[str, ...] = ("Right", "Front", "Top")


class PointerIndexError(ValueError):
    """Raised when a face/edge cannot be addressed or a build is inconsistent."""


@dataclass(frozen=True)
class FaceRecord:
    """A B-rep face candidate for a face pointer.

    ``key`` is a caller-supplied stable identity (e.g. a construction id). ``plane``
    is a canonical plane signature ``(nx, ny, nz, d)`` used later for coplanarity;
    ``is_base`` flags the three synthetic base planes.
    """
    key: str
    plane: tuple[float, float, float, float] | None = None
    is_base: bool = False


@dataclass(frozen=True)
class EdgeRecord:
    """A B-rep edge = shared boundary of two faces (referenced by their keys)."""
    key: str
    face_keys: tuple[str, str]  # unordered pair; stored sorted
    line: tuple[float, float, float, float, float, float] | None = None  # canonical


def _base_face_records() -> list[FaceRecord]:
    return [FaceRecord(key=f"base::{name}", is_base=True) for name in BASE_PLANES]


@dataclass
class EntityIndex:
    """A stable, addressable snapshot of a B-rep's faces and edges.

    Faces are ordered ``[Right, Front, Top, *sorted model faces]``; edges follow the
    faces, ordered by their sorted owning-face indices. ``face_index`` / ``edge_index``
    map a key to its pointer value; the reverse lists give O(1) resolution.
    """
    faces: list[FaceRecord] = field(default_factory=list)
    edges: list[EdgeRecord] = field(default_factory=list)
    _face_index: dict[str, int] = field(default_factory=dict, repr=False)
    _edge_index: dict[str, int] = field(default_factory=dict, repr=False)

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def face_pointer(self, key: str) -> int:
        if key not in self._face_index:
            raise PointerIndexError(f"no face with key {key!r}")
        return self._face_index[key]

    def edge_pointer(self, key: str) -> int:
        if key not in self._edge_index:
            raise PointerIndexError(f"no edge with key {key!r}")
        return self._edge_index[key]

    def resolve_face(self, pointer: int) -> FaceRecord:
        if not 0 <= pointer < len(self.faces):
            raise PointerIndexError(f"face pointer {pointer} out of range [0,{len(self.faces)})")
        return self.faces[pointer]

    def resolve_edge(self, pointer: int) -> EdgeRecord:
        if not 0 <= pointer < len(self.edges):
            raise PointerIndexError(f"edge pointer {pointer} out of range [0,{len(self.edges)})")
        return self.edges[pointer]

    def face_pointer_valid(self, pointer: int) -> bool:
        return 0 <= pointer < len(self.faces)

    def edge_pointer_valid(self, pointer: int) -> bool:
        return 0 <= pointer < len(self.edges)


def build_index(
    model_faces: list[FaceRecord] | None = None,
    edges: list[EdgeRecord] | None = None,
    include_base_planes: bool = True,
) -> EntityIndex:
    """Assemble an :class:`EntityIndex` with the canonical Pointer-CAD ordering.

    ``model_faces`` are the non-base faces; they are appended after the three base
    planes and sorted by ``key`` for determinism. ``edges`` must reference existing
    face keys; they are ordered by their sorted owning-face *indices*.
    """
    model_faces = list(model_faces or [])
    edges = list(edges or [])

    faces: list[FaceRecord] = _base_face_records() if include_base_planes else []
    # Reject accidental duplicate/base collisions among supplied model faces.
    for f in sorted(model_faces, key=lambda r: r.key):
        if f.is_base:
            raise PointerIndexError(f"model face {f.key!r} must not be flagged is_base")
        faces.append(f)

    face_index: dict[str, int] = {}
    for i, f in enumerate(faces):
        if f.key in face_index:
            raise PointerIndexError(f"duplicate face key {f.key!r}")
        face_index[f.key] = i

    # Order edges by (min owning-face index, max owning-face index, key).
    def edge_key(e: EdgeRecord) -> tuple[int, int, str]:
        a, b = e.face_keys
        if a not in face_index or b not in face_index:
            raise PointerIndexError(f"edge {e.key!r} references unknown face(s) {e.face_keys}")
        ia, ib = face_index[a], face_index[b]
        return (min(ia, ib), max(ia, ib), e.key)

    ordered_edges = sorted(edges, key=edge_key)
    edge_index: dict[str, int] = {}
    normed_edges: list[EdgeRecord] = []
    for i, e in enumerate(ordered_edges):
        if e.key in edge_index:
            raise PointerIndexError(f"duplicate edge key {e.key!r}")
        a, b = sorted(e.face_keys)
        normed_edges.append(EdgeRecord(key=e.key, face_keys=(a, b), line=e.line))
        edge_index[e.key] = i

    return EntityIndex(
        faces=faces,
        edges=normed_edges,
        _face_index=face_index,
        _edge_index=edge_index,
    )


def face_incidence(index: EntityIndex) -> dict[int, list[int]]:
    """Map each face pointer to the list of edge pointers bounded by that face.

    This is the raw material for non-manifold detection: a well-formed manifold
    edge is shared by exactly two faces, so every edge contributes to exactly two
    faces' incidence lists.
    """
    inc: dict[int, list[int]] = {i: [] for i in range(index.num_faces)}
    for ep, e in enumerate(index.edges):
        for fk in e.face_keys:
            inc[index.face_pointer(fk)].append(ep)
    return inc
