"""Kernel-independent two-level attributed adjacency graphs (TAAG).

Extraction deliberately stops at topology and local geometry attributes.
Recognition consumes that immutable graph and may return overlapping,
set-valued feature hypotheses; it never mutates or silently simplifies the
extracted evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Sequence


class EdgeSense(str, Enum):
    """Local dihedral classification at a topological edge."""

    CONVEX = "convex"
    CONCAVE = "concave"
    TRANSITORY = "transitory"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TopologyRecord:
    """Abstract topology input, normally produced by a CAD-kernel adapter."""

    vertices: tuple[Mapping[str, Any], ...] = ()
    edges: tuple[Mapping[str, Any], ...] = ()
    faces: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class TopologyNode:
    id: str
    kind: str
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Incidence:
    source: str
    target: str
    relation: str


@dataclass(frozen=True)
class FaceAdjacency:
    face_a: str
    face_b: str
    through_edge: str
    sense: EdgeSense


@dataclass(frozen=True)
class AttributedBRepGraph:
    """Two graph levels: raw incidence and derived face adjacency."""

    nodes: tuple[TopologyNode, ...]
    incidence: tuple[Incidence, ...]
    face_adjacency: tuple[FaceAdjacency, ...]

    def node(self, node_id: str) -> TopologyNode:
        for item in self.nodes:
            if item.id == node_id:
                return item
        raise KeyError(node_id)

    def adjacent_faces(self, face_id: str) -> tuple[FaceAdjacency, ...]:
        return tuple(
            item
            for item in self.face_adjacency
            if face_id in (item.face_a, item.face_b)
        )


def _ids(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _sense(record: Mapping[str, Any]) -> EdgeSense:
    raw = str(record.get("sense", record.get("convexity", "unknown"))).lower()
    try:
        return EdgeSense(raw)
    except ValueError:
        return EdgeSense.UNKNOWN


class TopologyExtractor:
    """Convert abstract records to topology evidence; performs no recognition."""

    def extract(self, source: TopologyRecord | Mapping[str, Any]) -> AttributedBRepGraph:
        if isinstance(source, Mapping):
            source = TopologyRecord(
                tuple(source.get("vertices", ())),
                tuple(source.get("edges", ())),
                tuple(source.get("faces", ())),
            )

        nodes: list[TopologyNode] = []
        incidence: list[Incidence] = []
        edge_faces: dict[str, tuple[str, ...]] = {}
        senses: dict[str, EdgeSense] = {}
        seen: set[str] = set()

        for kind, records in (
            ("vertex", source.vertices),
            ("edge", source.edges),
            ("face", source.faces),
        ):
            for record in records:
                node_id = str(record["id"])
                if node_id in seen:
                    raise ValueError(f"duplicate topology id: {node_id}")
                seen.add(node_id)
                excluded = {"id", "vertices", "edges", "faces"}
                attributes = {k: v for k, v in record.items() if k not in excluded}
                if kind == "edge":
                    attributes["sense"] = _sense(record).value
                    senses[node_id] = _sense(record)
                    for vertex in _ids(record.get("vertices")):
                        incidence.append(Incidence(node_id, vertex, "bounded-by"))
                    edge_faces[node_id] = _ids(record.get("faces"))
                elif kind == "face":
                    for edge in _ids(record.get("edges")):
                        incidence.append(Incidence(node_id, edge, "bounded-by"))
                nodes.append(TopologyNode(node_id, kind, attributes))

        unknown = {
            endpoint
            for link in incidence
            for endpoint in (link.source, link.target)
            if endpoint not in seen
        }
        if unknown:
            raise ValueError(f"unknown topology references: {sorted(unknown)}")

        adjacency: list[FaceAdjacency] = []
        for edge_id, faces in edge_faces.items():
            # Non-manifold edges are represented pairwise without discarding data.
            for i, face_a in enumerate(faces):
                for face_b in faces[i + 1 :]:
                    if face_a not in seen or face_b not in seen:
                        raise ValueError(f"edge {edge_id} references unknown face")
                    a, b = sorted((face_a, face_b))
                    adjacency.append(FaceAdjacency(a, b, edge_id, senses[edge_id]))

        return AttributedBRepGraph(
            tuple(sorted(nodes, key=lambda item: (item.kind, item.id))),
            tuple(sorted(incidence, key=lambda item: (item.source, item.target))),
            tuple(sorted(adjacency, key=lambda item: (item.face_a, item.face_b, item.through_edge))),
        )


@dataclass(frozen=True)
class FeatureHypothesis:
    """A recognition candidate; topology membership may overlap other candidates."""

    id: str
    feature_type: str
    topology_ids: frozenset[str]
    confidence: float
    recognizer: str
    evidence: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not self.topology_ids:
            raise ValueError("a hypothesis must cite topology evidence")


@dataclass(frozen=True)
class HypothesisSet:
    """Set-valued recognition output; no forced winner or partition."""

    hypotheses: tuple[FeatureHypothesis, ...]

    def overlapping(self, hypothesis_id: str) -> tuple[FeatureHypothesis, ...]:
        selected = next(item for item in self.hypotheses if item.id == hypothesis_id)
        return tuple(
            item
            for item in self.hypotheses
            if item.id != hypothesis_id and item.topology_ids & selected.topology_ids
        )

    def for_topology(self, topology_id: str) -> tuple[FeatureHypothesis, ...]:
        return tuple(item for item in self.hypotheses if topology_id in item.topology_ids)


RecognizerRule = Callable[[AttributedBRepGraph], Iterable[FeatureHypothesis]]


class FeatureRecognizer:
    """Run explicit recognition rules against already-extracted evidence."""

    def __init__(self, rules: Sequence[RecognizerRule]) -> None:
        self.rules = tuple(rules)

    def recognize(self, graph: AttributedBRepGraph) -> HypothesisSet:
        known = {item.id for item in graph.nodes}
        found: list[FeatureHypothesis] = []
        ids: set[str] = set()
        for rule in self.rules:
            for hypothesis in rule(graph):
                if hypothesis.id in ids:
                    raise ValueError(f"duplicate hypothesis id: {hypothesis.id}")
                missing = hypothesis.topology_ids - known
                if missing:
                    raise ValueError(
                        f"hypothesis {hypothesis.id} cites unknown topology: {sorted(missing)}"
                    )
                ids.add(hypothesis.id)
                found.append(hypothesis)
        return HypothesisSet(tuple(sorted(found, key=lambda item: item.id)))
