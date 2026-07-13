"""Portable value objects shared by the orthographic reconstruction stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

Point2 = tuple[float, float]
Point3 = tuple[float, float, float]
ViewName = Literal["front", "bottom", "left"]


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    severity: Literal["info", "warning", "error"] = "error"
    context: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Edge2D:
    view: ViewName
    kind: Literal["line", "arc", "circle"]
    points: tuple[Point2, ...]
    hidden: bool = False
    source_id: str = ""

    @property
    def start(self) -> Point2:
        return self.points[0]

    @property
    def end(self) -> Point2:
        return self.points[-1]

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        xs, ys = zip(*self.points)
        return min(xs), min(ys), max(xs), max(ys)


@dataclass(frozen=True)
class View2D:
    name: ViewName
    edges: tuple[Edge2D, ...]


@dataclass(frozen=True)
class OrthographicInput:
    views: Mapping[ViewName, View2D]
    scale: float = 1.0
    tolerance: float = 0.005


@dataclass(frozen=True)
class Edge3D:
    start: Point3
    end: Point3
    kind: Literal["line", "curve"] = "line"
    pattern: str = ""
    sources: tuple[str, ...] = ()

    def canonical(self, tolerance: float = 1e-9) -> tuple[Point3, Point3]:
        q = lambda p: tuple(round(v / tolerance) for v in p)
        return (self.start, self.end) if q(self.start) <= q(self.end) else (self.end, self.start)


@dataclass(frozen=True)
class FaceLoop:
    vertices: tuple[Point3, ...]
    edge_indices: tuple[int, ...]
    plane: tuple[float, float, float, float]


@dataclass(frozen=True)
class FaceCluster:
    outer: FaceLoop
    inner: tuple[FaceLoop, ...] = ()


@dataclass(frozen=True)
class StageReport:
    name: str
    input_count: int
    output_count: int
    diagnostics: tuple[Diagnostic, ...] = ()


@dataclass(frozen=True)
class StitchStatus:
    status: Literal["not_requested", "unavailable", "succeeded", "failed"]
    message: str = ""
    artifact: object = None


@dataclass(frozen=True)
class ReconstructionResult:
    input: OrthographicInput | None
    edges: tuple[Edge3D, ...]
    loops: tuple[FaceLoop, ...]
    faces: tuple[FaceCluster, ...]
    manifold: bool
    reports: tuple[StageReport, ...]
    diagnostics: tuple[Diagnostic, ...]
    stitch: StitchStatus

    @property
    def ok(self) -> bool:
        return self.input is not None and not any(
            item.severity == "error" for item in self.diagnostics
        )
