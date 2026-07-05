"""Stable failure taxonomy for reconstructed B-reps."""
from enum import Enum
class Failure(str, Enum):
    MISSING_FACE="missing_face"; NON_WATERTIGHT="non_watertight"
    SELF_INTERSECTION="self_intersection"; SURFACE_EDGE_INCONSISTENT="surface_edge_inconsistent"
    OVERMERGE="overmerge"
def classify(*, watertight=True, missing_faces=0, self_intersections=0,
             consistency_error=0.0, overmerged=0):
    out=[]
    if missing_faces: out.append(Failure.MISSING_FACE)
    if not watertight: out.append(Failure.NON_WATERTIGHT)
    if self_intersections: out.append(Failure.SELF_INTERSECTION)
    if consistency_error > 0: out.append(Failure.SURFACE_EDGE_INCONSISTENT)
    if overmerged: out.append(Failure.OVERMERGE)
    return tuple(out)
