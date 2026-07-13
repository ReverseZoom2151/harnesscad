"""Deterministic, history-free editing of imported boundary representations."""

from harnesscad.domain.editing.brep import (
    EditCandidate, EditProvider, FaceDescriptor, canonicalize_faces,
    generate_candidates, rank_candidates,
)

__all__ = [
    "EditCandidate", "EditProvider", "FaceDescriptor", "canonicalize_faces",
    "generate_candidates", "rank_candidates",
]
