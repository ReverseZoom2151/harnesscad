"""Chain-complex validation for pointwise scan-to-B-rep labels."""
from dataclasses import dataclass

@dataclass(frozen=True)
class ScanLabel:
    point_id: str
    face_id: str | None = None
    boundary_id: str | None = None
    junction_id: str | None = None
    parameters: tuple[float, ...] = ()

@dataclass(frozen=True)
class ChainComplex:
    faces: frozenset[str]
    boundaries: dict[str, tuple[str, ...]]
    junctions: frozenset[str]

def validate(labels, complex):
    issues = []
    for x in labels:
        if x.face_id and x.face_id not in complex.faces: issues.append(f"unknown_face:{x.point_id}")
        if x.boundary_id and x.boundary_id not in complex.boundaries: issues.append(f"unknown_boundary:{x.point_id}")
        if x.junction_id and not x.boundary_id: issues.append(f"junction_without_boundary:{x.point_id}")
        if x.junction_id and x.junction_id not in complex.junctions: issues.append(f"unknown_junction:{x.point_id}")
    for key, loop in complex.boundaries.items():
        if len(loop) < 3 or loop[0] != loop[-1]: issues.append(f"open_loop:{key}")
        if any(face not in complex.faces for face in loop): issues.append(f"unknown_loop_face:{key}")
    return tuple(sorted(set(issues)))
