"""Unified B-rep <-> command-sequence linkage for Pointer-CAD.

Pointer-CAD's core claim is that a command sequence and a B-rep are *one* object:
each step's pointers are resolved against the B-rep accumulated so far, and applying
the step mutates that B-rep so the next step's pointers address the new geometry
(paper Sec. 4.1: "the B-rep is incrementally updated after each operation"). This
module makes that loop explicit and deterministic:

  * :class:`BrepState` holds the current model faces/edges and can produce a fresh
    :class:`EntityIndex` (stable pointer addressing) on demand.
  * :func:`apply_sketch_extrude` resolves a *face* pointer (the sketch plane) against
    the current index, then grows the B-rep with the newly created faces/edges.
  * :func:`apply_chamfer` / :func:`apply_fillet` resolve *edge* pointers against the
    current index, remove the referenced edges, and add the chamfer/fillet faces --
    exactly the "select THIS edge, then refine it" operation that plain command
    sequences cannot express.

Every operation returns a :class:`StepResult` recording which existing entities were
referenced and which were added/removed, so a whole sequence can be replayed and
audited. Geometry is tracked at the topology-bookkeeping level (kernel-agnostic);
pointer semantics come from :mod:`reconstruction.pointercad_pointer`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harnesscad.domain.reconstruction.brep.pointercad_indexing import EdgeRecord, EntityIndex, FaceRecord, build_index
from harnesscad.domain.reconstruction.sequences.pointercad_pointer import (
    CHAMFER,
    FILLET,
    PointerCommand,
    PointerError,
    validate_command,
)


@dataclass
class BrepState:
    """The evolving B-rep: model faces + edges (base planes are added by the index)."""
    faces: list[FaceRecord] = field(default_factory=list)
    edges: list[EdgeRecord] = field(default_factory=list)

    def index(self) -> EntityIndex:
        """Build the canonical, addressable index for the current geometry."""
        return build_index(self.faces, self.edges)

    def copy(self) -> "BrepState":
        return BrepState(faces=list(self.faces), edges=list(self.edges))


@dataclass(frozen=True)
class StepResult:
    """What a single applied step did to the B-rep."""
    operation: str
    referenced_faces: tuple[str, ...] = ()   # existing face keys the step pointed at
    referenced_edges: tuple[str, ...] = ()   # existing edge keys the step pointed at
    added_faces: tuple[str, ...] = ()
    added_edges: tuple[str, ...] = ()
    removed_edges: tuple[str, ...] = ()


def apply_sketch_extrude(
    state: BrepState,
    plane_pointer: int,
    new_faces: list[FaceRecord],
    new_edges: list[EdgeRecord],
) -> tuple[BrepState, StepResult]:
    """Apply a sketch-extrude: the plane pointer selects an existing face, then the
    extruded prism's ``new_faces`` / ``new_edges`` are grafted onto the B-rep.

    The plane pointer is resolved against the *current* index (which includes the
    three base planes, so the first step can sketch on a base plane even when the
    model is empty). New edges may reference any existing or newly added face key.
    """
    index = state.index()
    if not index.face_pointer_valid(plane_pointer):
        raise PointerError(f"sketch plane pointer {plane_pointer} dangles")
    plane_face = index.resolve_face(plane_pointer)

    for f in new_faces:
        if f.is_base:
            raise PointerError("extruded faces must not be base planes")

    new = state.copy()
    existing_face_keys = {f.key for f in new.faces} | set(
        f"base::{p}" for p in ("Right", "Front", "Top")
    )
    added_face_keys = {f.key for f in new_faces}
    valid_face_keys = existing_face_keys | added_face_keys
    for e in new_edges:
        for fk in e.face_keys:
            if fk not in valid_face_keys:
                raise PointerError(f"new edge {e.key!r} references unknown face {fk!r}")

    new.faces.extend(new_faces)
    new.edges.extend(new_edges)
    result = StepResult(
        operation="sketch_extrude",
        referenced_faces=(plane_face.key,),
        added_faces=tuple(f.key for f in new_faces),
        added_edges=tuple(e.key for e in new_edges),
    )
    return new, result


def _apply_edge_refinement(
    state: BrepState,
    cmd: PointerCommand,
    op_name: str,
    face_prefix: str,
) -> tuple[BrepState, StepResult]:
    index = state.index()
    validate_command(cmd, index)  # raises on any dangling edge pointer

    referenced = tuple(index.resolve_edge(p).key for p in cmd.edge_pointers)
    new = state.copy()
    remaining = [e for e in new.edges if e.key not in set(referenced)]

    # Each refined edge becomes a new (chamfer/fillet) face replacing it.
    added_faces: list[FaceRecord] = []
    for ep in cmd.edge_pointers:
        edge = index.resolve_edge(ep)
        added_faces.append(FaceRecord(key=f"{face_prefix}::{edge.key}"))

    new.edges = remaining
    new.faces = list(new.faces) + added_faces
    result = StepResult(
        operation=op_name,
        referenced_edges=referenced,
        added_faces=tuple(f.key for f in added_faces),
        removed_edges=referenced,
    )
    return new, result


def apply_chamfer(state: BrepState, cmd: PointerCommand) -> tuple[BrepState, StepResult]:
    """Apply a chamfer: resolve the edge pointers, replace those edges with faces."""
    if cmd.kind != CHAMFER:
        raise PointerError(f"expected a chamfer command, got {cmd.kind!r}")
    return _apply_edge_refinement(state, cmd, "chamfer", "chamfer_face")


def apply_fillet(state: BrepState, cmd: PointerCommand) -> tuple[BrepState, StepResult]:
    """Apply a fillet: resolve the edge pointers, replace those edges with faces."""
    if cmd.kind != FILLET:
        raise PointerError(f"expected a fillet command, got {cmd.kind!r}")
    return _apply_edge_refinement(state, cmd, "fillet", "fillet_face")


def replay(
    state: BrepState,
    steps: list[tuple[str, object]],
) -> tuple[BrepState, list[StepResult]]:
    """Replay a heterogeneous step list, threading the B-rep through each step.

    Each step is ``(op, payload)`` where ``op`` is ``"sketch_extrude"`` (payload is
    ``(plane_pointer, new_faces, new_edges)``), ``"chamfer"`` or ``"fillet"``
    (payload is a :class:`PointerCommand`). Pointers in every step are resolved
    against the B-rep produced by the previous steps -- the unified linkage.
    """
    results: list[StepResult] = []
    cur = state
    for op, payload in steps:
        if op == "sketch_extrude":
            plane_pointer, new_faces, new_edges = payload  # type: ignore[misc]
            cur, res = apply_sketch_extrude(cur, plane_pointer, new_faces, new_edges)
        elif op == "chamfer":
            cur, res = apply_chamfer(cur, payload)  # type: ignore[arg-type]
        elif op == "fillet":
            cur, res = apply_fillet(cur, payload)  # type: ignore[arg-type]
        else:
            raise PointerError(f"unknown replay op {op!r}")
        results.append(res)
    return cur, results
