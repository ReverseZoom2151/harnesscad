"""Synthesis of reversible face-delete/add training pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from editing.brep import FaceDescriptor, canonicalize_faces


@dataclass(frozen=True)
class BrepEditPair:
    before: Any
    after: Any
    face: FaceDescriptor
    forward: dict
    inverse: dict


def synthesize_delete_add_pairs(
    shape: Any,
    faces: Iterable[FaceDescriptor],
    *,
    delete_face: Callable[[Any, FaceDescriptor], Any],
    add_face: Callable[[Any, FaceDescriptor], Any],
    is_valid: Callable[[Any], bool],
    equivalent: Callable[[Any, Any], bool],
) -> tuple[BrepEditPair, ...]:
    """Keep only valid deletes whose inverse reconstructs equivalent geometry."""
    out = []
    for index, face in enumerate(canonicalize_faces(faces)):
        deleted = delete_face(shape, face)
        if deleted is None or not is_valid(deleted):
            continue
        restored = add_face(deleted, face)
        if restored is None or not is_valid(restored) or not equivalent(shape, restored):
            continue
        ref = {"canonical_face": index, "signature": face.signature()}
        out.append(BrepEditPair(
            shape, deleted, face,
            {"operation": "delete_face", **ref},
            {"operation": "add_face", **ref},
        ))
    return tuple(out)
