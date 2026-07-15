"""FDM overhang detection, build-orientation search, and stability descriptors.

Deterministic Design-for-Additive-Manufacturing (DFAM) rules distilled from
**AgentsCAD** (George, Keefe, Pak, Barati Farimani, 2026). The paper's LLM reasoning
is out of scope, but its geometry layer is a "deterministic floor" of rule-based
checks (their Sec. 4.2.1) that this module reimplements from face descriptors:

* **Overhang detection** (Sec. 1, 4.2.2). A downward-facing surface tilted beyond ~45
  degrees from vertical cannot print without support. AgentsCAD flags a face when
  ``theta = arccos(n_face . z_plane) <= 0`` after resolving sign; concretely a face is
  an overhang when the angle between its outward normal and the build direction
  ``+z`` exceeds ``90 + threshold`` degrees (the normal points downward-and-out).
  :func:`overhang_faces` returns the flagged faces and their severity.

* **Orientation search** (Sec. 4.2.2, "a scripted orientation sub-phase evaluates up
  to four candidate rotations and commits to the best one"). :func:`best_orientation`
  scores candidate build directions by total overhang area and picks the minimiser --
  the deterministic ``check_orientation_overhangs`` / ``lay_face_to_build_surface``
  grounding tool, without the LLM.

* **Stability descriptors** (Sec. 4.1). Per-face ``radius_of_gyration`` (footprint
  size) and ``elongation_index`` (how needle-like the footprint is) plus a part-level
  base-area / bounding-box aspect proxy used to reason about print stability.

Faces are lightweight dicts/objects carrying an outward unit normal, an area and a
centroid. Angles in degrees. Stdlib only, fully deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

__all__ = [
    "DEFAULT_OVERHANG_THRESHOLD_DEG",
    "FaceOverhang",
    "overhang_angle_deg",
    "is_overhang",
    "overhang_faces",
    "total_overhang_area",
    "best_orientation",
    "radius_of_gyration",
    "elongation_index",
]

DEFAULT_OVERHANG_THRESHOLD_DEG = 45.0

# The six axis-aligned build directions searched by default.
_AXIS_DIRECTIONS: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
    (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
)


def _normalize(v: Sequence[float]) -> tuple[float, float, float]:
    n = math.sqrt(sum(c * c for c in v))
    if n < 1e-12:
        raise ValueError("cannot normalize a zero-length normal")
    return (v[0] / n, v[1] / n, v[2] / n)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _get(face, key: str, default=None):
    if isinstance(face, Mapping):
        return face.get(key, default)
    return getattr(face, key, default)


@dataclass(frozen=True)
class FaceOverhang:
    """An overhang finding for one face."""

    face_id: object
    angle_from_up_deg: float   # angle between outward normal and build direction
    overhang_deg: float        # how far past horizontal-downward the face tilts
    area: float
    is_bed_face: bool          # normal points straight down: rests on the bed, no support


def overhang_angle_deg(normal: Sequence[float], build_dir: Sequence[float] = (0.0, 0.0, 1.0)) -> float:
    """Angle in degrees between an outward face normal and the build direction."""
    n = _normalize(normal)
    b = _normalize(build_dir)
    c = max(-1.0, min(1.0, _dot(n, b)))
    return math.degrees(math.acos(c))


def is_overhang(normal: Sequence[float], build_dir: Sequence[float] = (0.0, 0.0, 1.0),
                threshold_deg: float = DEFAULT_OVERHANG_THRESHOLD_DEG,
                *, bed_tol_deg: float = 1.0) -> bool:
    """True when a face is an unsupported overhang for the given build direction.

    A face's outward normal at angle ``a`` from ``+build_dir``: horizontal walls sit at
    90 degrees (fine); downward-facing surfaces exceed 90. An overhang occurs once the
    downward tilt passes the threshold, i.e. ``a > 90 + threshold``. A face pointing
    straight *down* (``a`` near 180) rests flat on the bed and needs no support, so it
    is excluded.
    """
    a = overhang_angle_deg(normal, build_dir)
    if a >= 180.0 - bed_tol_deg:
        return False  # bed face
    return a > 90.0 + threshold_deg


def overhang_faces(faces: Iterable, build_dir: Sequence[float] = (0.0, 0.0, 1.0),
                   threshold_deg: float = DEFAULT_OVERHANG_THRESHOLD_DEG,
                   *, bed_tol_deg: float = 1.0) -> tuple[FaceOverhang, ...]:
    """Flag every actionable overhang face (AgentsCAD deterministic floor)."""
    out: list[FaceOverhang] = []
    for idx, face in enumerate(faces):
        normal = _get(face, "normal")
        if normal is None:
            raise ValueError(f"face {idx} has no 'normal'")
        area = float(_get(face, "area", 0.0) or 0.0)
        fid = _get(face, "id", idx)
        a = overhang_angle_deg(normal, build_dir)
        is_bed = a >= 180.0 - bed_tol_deg
        if is_overhang(normal, build_dir, threshold_deg, bed_tol_deg=bed_tol_deg):
            out.append(FaceOverhang(
                face_id=fid, angle_from_up_deg=a,
                overhang_deg=a - 90.0, area=area, is_bed_face=False,
            ))
    return tuple(out)


def total_overhang_area(faces: Iterable, build_dir: Sequence[float] = (0.0, 0.0, 1.0),
                        threshold_deg: float = DEFAULT_OVERHANG_THRESHOLD_DEG) -> float:
    """Sum of the areas of all overhang faces for a build direction."""
    return sum(f.area for f in overhang_faces(faces, build_dir, threshold_deg))


def best_orientation(faces: Sequence,
                     candidates: Sequence[Sequence[float]] = _AXIS_DIRECTIONS,
                     threshold_deg: float = DEFAULT_OVERHANG_THRESHOLD_DEG) -> tuple[tuple[float, float, float], float]:
    """Pick the build direction minimising total overhang area (AgentsCAD orientation).

    Returns ``(best_direction, overhang_area)``. Ties are broken by candidate order so
    the result is deterministic.
    """
    faces = list(faces)
    best_dir = None
    best_area = math.inf
    for cand in candidates:
        area = total_overhang_area(faces, cand, threshold_deg)
        if area < best_area - 1e-12:
            best_area = area
            best_dir = _normalize(cand)
    if best_dir is None:
        raise ValueError("no candidate build directions provided")
    return best_dir, best_area


def radius_of_gyration(area: float, second_moment: float) -> float:
    """Footprint radius of gyration ``sqrt(I / A)`` (AgentsCAD stability proxy)."""
    if area <= 0.0:
        raise ValueError("area must be positive")
    if second_moment < 0.0:
        raise ValueError("second moment must be non-negative")
    return math.sqrt(second_moment / area)


def elongation_index(i_major: float, i_minor: float) -> float:
    """How needle-like a footprint is: ``sqrt(I_major / I_minor) >= 1`` (AgentsCAD).

    An index of 1 is a rotationally symmetric footprint; large values are lopsided,
    unstable prints.
    """
    if i_minor <= 0.0:
        raise ValueError("minor principal moment must be positive")
    if i_major < i_minor:
        i_major, i_minor = i_minor, i_major
    return math.sqrt(i_major / i_minor)
