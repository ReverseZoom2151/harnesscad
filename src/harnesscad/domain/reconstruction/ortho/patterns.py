"""Table-driven orthographic projection-pattern matching."""

from __future__ import annotations

from dataclasses import dataclass
import itertools

from .edges import projection_feature
from .model import Diagnostic, Edge2D, Edge3D

# Paper taxonomy, mapped to our per-view H/V/I/A vocabulary.  Pt projections
# have no explicit edge and are represented by ``None``.
PATTERNS = {
    "L1": ("H", "H", None), "L2": ("V", None, "V"), "L3": (None, "V", "H"),
    "L4": ("V", "V", "I"), "L5": ("H", "I", "H"), "L6": ("I", "H", "V"),
    "L7": ("I", "I", "I"),
    "C1": ("A", "H", "V"), "C2": ("H", "A", "H"), "C3": ("V", "V", "A"),
    "C4": ("I", "A", "A"), "C5": ("A", "I", "A"), "C6": ("A", "A", "I"),
    "C7": ("A", "A", "A"),
}


def _range(edge: Edge2D, axis: int) -> tuple[float, float]:
    values = [point[axis] for point in edge.points]
    return min(values), max(values)


def _close_range(a, b, tolerance: float) -> bool:
    return abs(a[0] - b[0]) <= tolerance and abs(a[1] - b[1]) <= tolerance


def _compatible(front: Edge2D, bottom: Edge2D, left: Edge2D, tolerance: float) -> bool:
    # shared X: front[0] == bottom[0], shared Y: front[1] == left[1],
    # shared Z: bottom[1] == left[0].
    return (_close_range(_range(front, 0), _range(bottom, 0), tolerance)
            and _close_range(_range(front, 1), _range(left, 1), tolerance)
            and _close_range(_range(bottom, 1), _range(left, 0), tolerance))


def _point3(front: tuple[float, float], bottom: tuple[float, float]) -> tuple[float, float, float]:
    return front[0], front[1], bottom[1]


def match_patterns(views, tolerance: float):
    """Match complete three-edge patterns; emit diagnostics for ambiguity.

    Pt-bearing L1–L3 remain named in :data:`PATTERNS`, but require vertex data
    not expressible as an edge-only SVG and are reported as unsupported here.
    """
    candidates: list[Edge3D] = []
    diagnostics: list[Diagnostic] = []
    front, bottom, left = (views[name] for name in ("front", "bottom", "left"))
    by_sig = {}
    for triple in itertools.product(front, bottom, left):
        sig = tuple(projection_feature(edge, tolerance) for edge in triple)
        names = [name for name, pattern in PATTERNS.items() if pattern == sig]
        if not names or not _compatible(*triple, tolerance):
            continue
        pattern = names[0]
        f, b, _ = triple
        # Match endpoint correspondence by shared X; reversed orientation is legal.
        pairing = ((f.start, b.start, f.end, b.end)
                   if abs(f.start[0] - b.start[0]) <= tolerance
                   else (f.start, b.end, f.end, b.start))
        edge = Edge3D(_point3(pairing[0], pairing[1]),
                      _point3(pairing[2], pairing[3]),
                      "curve" if pattern.startswith("C") else "line",
                      pattern, tuple(item.source_id for item in triple))
        key = edge.canonical(tolerance)
        by_sig.setdefault(key, []).append(edge)
    for key in sorted(by_sig):
        group = by_sig[key]
        candidates.append(sorted(group, key=lambda e: (e.pattern, e.sources))[0])
        distinct = {(item.pattern, item.sources) for item in group}
        if len(distinct) > 1:
            diagnostics.append(Diagnostic(
                "ambiguous-edge-match",
                f"{len(distinct)} projection matches reconstruct the same 3D edge",
                "warning", {"edge": key, "candidates": tuple(sorted(distinct))},
            ))
    return tuple(candidates), tuple(diagnostics)
