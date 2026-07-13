"""Pointer-based commands, resolution and dangling-pointer detection.

This is the heart of Pointer-CAD's contribution: a command that *references* an
existing B-rep entity by index rather than by coordinates -- "fillet THIS edge",
"sketch FROM THIS face" (paper Sec. 3, Table 13). Concretely:

  * ``[Sketch] = <ss> <pe> [CS] [Profile]+`` -- a face pointer chooses the sketch
    plane;
  * ``[Chamfer] = <sc> <nv> <pe>+`` / ``[Fillet] = <sf> <nv> <pe>+`` -- one or more
    edge pointers choose the target edges, then a single numeric parameter.

A pointer is *valid* only if it addresses an entity that currently exists in the
B-rep; a **dangling pointer** is one that indexes past the entity list or references
geometry that has been removed (Sec. 9.1 -- the paper measures dangling edges as a
topological-soundness signal). This module builds the command records, resolves
them against a :class:`~reconstruction.pointercad_indexing.EntityIndex`, and reports
dangling pointers.

It also encodes the paper's *geometric special cases* (Sec. 10.3): because coplanar
faces / collinear edges are interchangeable, the ground-truth pointer is a **set** of
valid candidates, and a predicted pointer counts as correct if it lands anywhere in
that set. Non-manifold edges (bounded by >2 faces, Sec. 10.4) are flagged as
ambiguous. Pure stdlib.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from harnesscad.domain.reconstruction.brep.entity_index import EntityIndex, face_incidence

# Command kinds that carry pointers.
SKETCH = "sketch"    # one face pointer (plane)
CHAMFER = "chamfer"  # one or more edge pointers
FILLET = "fillet"    # one or more edge pointers

# Pointer targets.
FACE = "face"
EDGE = "edge"


class PointerError(ValueError):
    pass


@dataclass(frozen=True)
class PointerCommand:
    """A command plus the pointer indices it references.

    ``kind`` is one of :data:`SKETCH`, :data:`CHAMFER`, :data:`FILLET`.
    ``face_pointers`` / ``edge_pointers`` are integer indices into the B-rep's face
    / edge lists. ``param`` is the single numeric parameter (chamfer distance /
    fillet radius; ignored for a sketch).
    """
    kind: str
    face_pointers: tuple[int, ...] = ()
    edge_pointers: tuple[int, ...] = ()
    param: float | None = None

    def __post_init__(self) -> None:
        if self.kind == SKETCH:
            if len(self.face_pointers) != 1 or self.edge_pointers:
                raise PointerError("a sketch command needs exactly one face pointer")
        elif self.kind in (CHAMFER, FILLET):
            if not self.edge_pointers or self.face_pointers:
                raise PointerError(f"a {self.kind} command needs >=1 edge pointer")
        else:
            raise PointerError(f"unknown pointer-command kind {self.kind!r}")

    @property
    def target(self) -> str:
        return FACE if self.kind == SKETCH else EDGE


@dataclass
class ResolutionResult:
    """Outcome of resolving one command's pointers against a B-rep."""
    dangling_faces: tuple[int, ...] = ()
    dangling_edges: tuple[int, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.dangling_faces and not self.dangling_edges


def resolve_command(cmd: PointerCommand, index: EntityIndex) -> ResolutionResult:
    """Resolve a command's pointers; collect any that dangle (out of range)."""
    df = tuple(p for p in cmd.face_pointers if not index.face_pointer_valid(p))
    de = tuple(p for p in cmd.edge_pointers if not index.edge_pointer_valid(p))
    return ResolutionResult(dangling_faces=df, dangling_edges=de)


def validate_command(cmd: PointerCommand, index: EntityIndex) -> None:
    """Raise :class:`PointerError` if any pointer in ``cmd`` dangles."""
    res = resolve_command(cmd, index)
    if not res.is_valid:
        raise PointerError(
            f"dangling pointers in {cmd.kind}: faces={res.dangling_faces} "
            f"edges={res.dangling_edges}"
        )


def dangling_pointers(cmds: list[PointerCommand], index: EntityIndex) -> list[tuple[int, ResolutionResult]]:
    """Return ``(command_position, result)`` for every command that has a dangling
    pointer, checked against a single (static) index snapshot."""
    bad: list[tuple[int, ResolutionResult]] = []
    for i, cmd in enumerate(cmds):
        res = resolve_command(cmd, index)
        if not res.is_valid:
            bad.append((i, res))
    return bad


# --- geometric special cases (Sec. 10.3) -------------------------------------
def _canon_plane(plane: tuple[float, float, float, float], tol: float) -> tuple[int, ...]:
    """Canonicalise a plane ``(nx,ny,nz,d)`` so coplanar planes hash equal.

    The normal is unit-normalised and sign-fixed (first non-zero component made
    positive, flipping ``d`` accordingly) so a plane and its reverse-oriented twin
    collapse to one key. Components are snapped to a ``tol`` grid.
    """
    nx, ny, nz, d = plane
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm == 0.0:
        raise PointerError("degenerate plane normal")
    nx, ny, nz, d = nx / norm, ny / norm, nz / norm, d / norm
    for c in (nx, ny, nz):
        if abs(c) > tol:
            if c < 0:
                nx, ny, nz, d = -nx, -ny, -nz, -d
            break
    q = lambda x: int(round(x / tol))
    return (q(nx), q(ny), q(nz), q(d))


def coplanar_face_groups(index: EntityIndex, tol: float = 1e-6) -> list[tuple[int, ...]]:
    """Group model faces (those with a ``plane`` signature) that are coplanar.

    Per Sec. 10.3, selecting any face in a coplanar group yields the same sketch
    plane, so all are valid candidates for a face pointer. Returns groups (each a
    sorted tuple of face pointers) that contain more than one member.
    """
    buckets: dict[tuple[int, ...], list[int]] = {}
    for fp, face in enumerate(index.faces):
        if face.plane is None:
            continue
        key = _canon_plane(face.plane, tol)
        buckets.setdefault(key, []).append(fp)
    return [tuple(sorted(v)) for v in buckets.values() if len(v) > 1]


def _canon_line(line: tuple[float, ...], tol: float) -> tuple[int, ...]:
    """Canonicalise a line ``(px,py,pz, dx,dy,dz)`` so collinear lines hash equal."""
    px, py, pz, dx, dy, dz = line
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm == 0.0:
        raise PointerError("degenerate line direction")
    dx, dy, dz = dx / norm, dy / norm, dz / norm
    # Sign-fix direction.
    for c in (dx, dy, dz):
        if abs(c) > tol:
            if c < 0:
                dx, dy, dz = -dx, -dy, -dz
            break
    # Represent the line by its direction plus the point on it closest to origin
    # (removes the free translation along the direction).
    t = px * dx + py * dy + pz * dz
    ox, oy, oz = px - t * dx, py - t * dy, pz - t * dz
    q = lambda x: int(round(x / tol))
    return (q(dx), q(dy), q(dz), q(ox), q(oy), q(oz))


def collinear_edge_groups(index: EntityIndex, tol: float = 1e-6) -> list[tuple[int, ...]]:
    """Group edges (those with a ``line`` signature) that are collinear.

    Per Sec. 10.3, snapping to any edge in a collinear group produces the same
    result, so all are valid candidates for an edge pointer.
    """
    buckets: dict[tuple[int, ...], list[int]] = {}
    for ep, edge in enumerate(index.edges):
        if edge.line is None:
            continue
        key = _canon_line(edge.line, tol)
        buckets.setdefault(key, []).append(ep)
    return [tuple(sorted(v)) for v in buckets.values() if len(v) > 1]


def non_manifold_edges(index: EntityIndex) -> list[int]:
    """Edges bounded by more than two faces (Sec. 10.4 -- ambiguous for pointers).

    We reconstruct face incidence from the index and flag edges whose owning-face
    multiplicity exceeds two. (In the base :class:`EntityIndex` an edge stores a
    2-tuple, so non-manifold input must be supplied as coincident edges sharing a
    line -- these are detected here via the incidence multiplicity of that line.)
    """
    # Count how many distinct faces touch each collinear/coincident edge cluster.
    inc = face_incidence(index)
    # An edge is non-manifold if the same geometric edge is bounded by >2 faces.
    # With one EdgeRecord per boundary, coincident records sharing a line signal it.
    clusters: dict[tuple[int, ...], set[str]] = {}
    for edge in index.edges:
        if edge.line is None:
            continue
        key = _canon_line(edge.line, 1e-6)
        clusters.setdefault(key, set()).update(edge.face_keys)
    bad_lines = {k for k, faces in clusters.items() if len(faces) > 2}
    result: list[int] = []
    for ep, edge in enumerate(index.edges):
        if edge.line is not None and _canon_line(edge.line, 1e-6) in bad_lines:
            result.append(ep)
    return sorted(result)
