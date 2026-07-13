"""Semantic feature / adjacency graph for a HarnessCAD part.

This is a *distinct* view from the op history (state/opdag.py): where the op-DAG
is an append-only event log ("git for CAD"), the feature graph is a queryable
model of *what the part is* -- its sketches, bodies and modifying features, and
the semantic relations between them ("this hole goes through that wall", "this
fillet is on that body", "this pattern replicates that feature").

Two construction sources are supported and produce the same node/edge model:

  * an :class:`state.opdag.OpDAG` (anything exposing ``ops()``) -- the rich path,
    replaying op semantics deterministically (mirroring the backends' id scheme
    so node ids line up: ``sk1``, ``f1``, ...);
  * a geometry backend (``StubBackend`` / ``CadQueryBackend``) -- read straight
    from its ``sketches`` / ``features`` state.

When a CadQuery backend is available (either passed as the primary object or via
the ``backend=`` argument), the graph is *optionally* enriched with real B-rep
face/edge adjacency (faces, edges, and edge-bounds-face relations) pulled from
the OCCT solid. That path is fully guarded: any kernel hiccup, or a backend with
no solid, simply yields the feature-level graph alone.

Pure and deterministic: same ops (or same backend state) -> same graph. Stdlib
only; OCCT is touched lazily and behind a broad guard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from harnesscad.core.cisp.ops import (
    Op, NewSketch, AddPoint, AddLine, AddCircle, AddRectangle,
    Constrain, Extrude, Fillet, Boolean,
    Revolve, Chamfer, Hole, Shell, Draft,
    Loft, Sweep, LinearPattern, CircularPattern, Mirror,
)

# Feature families used by the graph builder.
_BODY_TYPES = ("extrude", "revolve", "loft", "sweep")
_MODIFIER_TYPES = ("fillet", "chamfer", "shell", "draft")
_PATTERN_TYPES = ("linear_pattern", "circular_pattern")
# Everything the graph treats as a "feature" node (i.e. not a sketch/face/edge).
_FEATURE_TYPES = _BODY_TYPES + _MODIFIER_TYPES + _PATTERN_TYPES + (
    "boolean", "hole", "mirror",
)


# --- graph model -----------------------------------------------------------
@dataclass(frozen=True)
class FeatureNode:
    """A node in the feature graph: a sketch, a feature, or a B-rep face/edge."""

    id: str
    type: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "type": self.type, "params": dict(self.params)}


@dataclass(frozen=True)
class FeatureEdge:
    """A directed, typed relation between two nodes (e.g. 'hole-through-wall')."""

    source: str
    target: str
    relation: str

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "target": self.target, "relation": self.relation}


class FeatureGraph:
    """A queryable graph of a part's features and their semantic relations."""

    def __init__(self, nodes: List[FeatureNode], edges: List[FeatureEdge]) -> None:
        self.nodes: List[FeatureNode] = list(nodes)
        self.edges: List[FeatureEdge] = list(edges)
        self._by_id: Dict[str, FeatureNode] = {n.id: n for n in self.nodes}

    # -- lookups ------------------------------------------------------------
    def get(self, node_id: str) -> Optional[FeatureNode]:
        return self._by_id.get(node_id)

    def find(self, type: str) -> List[FeatureNode]:
        """All nodes of a given type (e.g. find('hole'))."""
        return [n for n in self.nodes if n.type == type]

    def find_features(self) -> List[FeatureNode]:
        """All feature nodes (bodies + modifiers; excludes sketches/faces/edges)."""
        return [n for n in self.nodes if n.type in _FEATURE_TYPES]

    def neighbors(self, node_id: str) -> List[FeatureNode]:
        """Nodes connected to ``node_id`` by an edge in either direction."""
        out: List[FeatureNode] = []
        seen = set()
        for e in self.edges:
            other = None
            if e.source == node_id:
                other = e.target
            elif e.target == node_id:
                other = e.source
            if other is not None and other not in seen and other in self._by_id:
                seen.add(other)
                out.append(self._by_id[other])
        return out

    def relations(self, node_id: str) -> List[FeatureEdge]:
        """Edges incident to ``node_id`` (either direction)."""
        return [e for e in self.edges if e.source == node_id or e.target == node_id]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }


# --- extraction: op-DAG ----------------------------------------------------
def _extract_from_ops(ops: List[Op]) -> Tuple[Dict[str, dict], List[dict]]:
    """Replay op semantics into (sketches, features), mirroring backend ids.

    Assumes every op in the stream was accepted (the op-DAG only records applied
    ops), so id counters advance exactly as the backends' ``_new_id`` does.
    """
    sketches: Dict[str, dict] = {}
    entity_sketch: Dict[str, str] = {}
    features: List[dict] = []
    n = {"sk": 0, "e": 0, "f": 0}

    def _nid(kind: str) -> str:
        n[kind] += 1
        prefix = {"sk": "sk", "e": "e", "f": "f"}[kind]
        return prefix + str(n[kind])

    def _add_primitive(sketch: str, kind: str, params: dict) -> None:
        eid = _nid("e")
        entity_sketch[eid] = sketch
        if sketch in sketches:
            sketches[sketch]["primitives"].append({"type": kind, **params})

    for op in ops:
        if isinstance(op, NewSketch):
            sid = _nid("sk")
            sketches[sid] = {"plane": op.plane, "primitives": [], "constraints": []}
        elif isinstance(op, AddPoint):
            _add_primitive(op.sketch, "point", {"x": op.x, "y": op.y})
        elif isinstance(op, AddLine):
            _add_primitive(op.sketch, "line",
                           {"x1": op.x1, "y1": op.y1, "x2": op.x2, "y2": op.y2})
        elif isinstance(op, AddCircle):
            _add_primitive(op.sketch, "circle", {"cx": op.cx, "cy": op.cy, "r": op.r})
        elif isinstance(op, AddRectangle):
            _add_primitive(op.sketch, "rectangle",
                           {"x": op.x, "y": op.y, "w": op.w, "h": op.h})
        elif isinstance(op, Constrain):
            sid = entity_sketch.get(op.a)
            if sid in sketches:
                sketches[sid]["constraints"].append(op.kind)
        elif isinstance(op, Extrude):
            features.append({"type": "extrude", "id": _nid("f"),
                             "sketch": op.sketch, "distance": op.distance})
        elif isinstance(op, Fillet):
            features.append({"type": "fillet", "id": _nid("f"),
                             "edges": list(op.edges), "radius": op.radius})
        elif isinstance(op, Chamfer):
            features.append({"type": "chamfer", "id": _nid("f"),
                             "edges": list(op.edges), "distance": op.distance})
        elif isinstance(op, Boolean):
            features.append({"type": "boolean", "id": _nid("f"), "kind": op.kind})
        elif isinstance(op, Revolve):
            features.append({"type": "revolve", "id": _nid("f"),
                             "sketch": op.sketch, "angle": op.angle})
        elif isinstance(op, Hole):
            features.append({"type": "hole", "id": _nid("f"),
                             "ref": op.face_or_sketch, "diameter": op.diameter,
                             "depth": op.depth, "through": op.through,
                             "kind": op.kind, "x": op.x, "y": op.y})
        elif isinstance(op, Shell):
            features.append({"type": "shell", "id": _nid("f"),
                             "faces": list(op.faces), "thickness": op.thickness})
        elif isinstance(op, Draft):
            features.append({"type": "draft", "id": _nid("f"),
                             "faces": list(op.faces), "angle": op.angle,
                             "neutral_plane": op.neutral_plane})
        elif isinstance(op, Loft):
            features.append({"type": "loft", "id": _nid("f"),
                             "sketches": list(op.sketches)})
        elif isinstance(op, Sweep):
            features.append({"type": "sweep", "id": _nid("f"),
                             "sketch": op.sketch, "path": op.path})
        elif isinstance(op, LinearPattern):
            features.append({"type": "linear_pattern", "id": _nid("f"),
                             "feature": op.feature, "count": op.count,
                             "spacing": op.spacing})
        elif isinstance(op, CircularPattern):
            features.append({"type": "circular_pattern", "id": _nid("f"),
                             "feature": op.feature, "count": op.count,
                             "angle": op.angle})
        elif isinstance(op, Mirror):
            features.append({"type": "mirror", "id": _nid("f"),
                             "feature": op.feature_or_body, "plane": op.plane})
        # unknown ops are ignored (never appended by the loop)
    return sketches, features


# --- extraction: backend state ---------------------------------------------
def _extract_from_backend(backend: Any) -> Tuple[Dict[str, dict], List[dict]]:
    """Read (sketches, features) straight off a StubBackend / CadQueryBackend."""
    entities = getattr(backend, "entities", {}) or {}
    raw_sketches = getattr(backend, "sketches", {}) or {}
    sketches: Dict[str, dict] = {}
    for sid, s in raw_sketches.items():
        prims = []
        for eid in s.get("entities", []):
            ent = entities.get(eid, {})
            prims.append({"type": ent.get("type", "?"), **(ent.get("params", {}) or {})})
        sketches[sid] = {
            "plane": s.get("plane", "XY"),
            "primitives": prims,
            "constraints": list(s.get("constraints", [])),
        }
    features = [dict(f) for f in getattr(backend, "features", [])]
    # Normalise the hole 'ref' key (backends store it as 'ref').
    return sketches, features


# --- builder ---------------------------------------------------------------
def _build(sketches: Dict[str, dict],
           features: List[dict]) -> Tuple[List[FeatureNode], List[FeatureEdge]]:
    nodes: List[FeatureNode] = []
    edges: List[FeatureEdge] = []
    node_ids = set()

    def add_node(nid: str, ntype: str, params: dict) -> None:
        if nid in node_ids:
            return
        nodes.append(FeatureNode(nid, ntype, params))
        node_ids.add(nid)

    def add_edge(src: str, dst: str, rel: str) -> None:
        if src in node_ids and dst in node_ids:
            edges.append(FeatureEdge(src, dst, rel))

    # sketch nodes first (deterministic insertion order)
    for sid, s in sketches.items():
        prims = s.get("primitives", [])
        prim_types = sorted({p.get("type", "?") for p in prims})
        add_node(sid, "sketch", {
            "plane": s.get("plane", "XY"),
            "primitive_count": len(prims),
            "primitive_types": prim_types,
            "constraints": list(s.get("constraints", [])),
        })

    current_solid: Optional[str] = None
    for feat in features:
        fid = feat.get("id")
        ftype = feat.get("type", "?")
        if fid is None:
            continue
        # node params: everything except the bookkeeping keys
        params = {k: v for k, v in feat.items() if k not in ("id", "type")}
        add_node(fid, ftype, params)

        if ftype in _BODY_TYPES:
            if ftype in ("extrude", "revolve"):
                sref = feat.get("sketch")
                if sref:
                    add_edge(sref, fid, "profile-of")
            elif ftype == "loft":
                for sref in feat.get("sketches", []):
                    add_edge(sref, fid, "profile-of")
            elif ftype == "sweep":
                if feat.get("sketch"):
                    add_edge(feat["sketch"], fid, "profile-of")
                if feat.get("path"):
                    add_edge(feat["path"], fid, "path-of")
            current_solid = fid
        elif ftype in _MODIFIER_TYPES:
            if current_solid is not None:
                add_edge(fid, current_solid, ftype + "-on")
        elif ftype == "hole":
            ref = feat.get("ref", "") or ""
            if ref.startswith("sk") and ref in node_ids:
                add_edge(fid, ref, "hole-in-sketch")
            elif current_solid is not None:
                rel = "hole-through-wall" if feat.get("through", True) else "hole-in-face"
                add_edge(fid, current_solid, rel)
            if current_solid is None:
                current_solid = fid
        elif ftype in _PATTERN_TYPES:
            tgt = feat.get("feature") or current_solid
            if tgt in node_ids:
                add_edge(fid, tgt, "pattern-of")
            elif current_solid is not None:
                add_edge(fid, current_solid, "pattern-of")
        elif ftype == "mirror":
            tgt = feat.get("feature") or current_solid
            if tgt in node_ids:
                add_edge(fid, tgt, "mirror-of")
            elif current_solid is not None:
                add_edge(fid, current_solid, "mirror-of")
        elif ftype == "boolean":
            if current_solid is not None:
                add_edge(fid, current_solid, "combines")
            current_solid = fid

    return nodes, edges


# --- optional B-rep adjacency enrichment -----------------------------------
def _maybe_enrich_brep(graph: FeatureGraph, backend: Any) -> None:
    """Add real OCCT face/edge nodes + edge-bounds-face relations, if available.

    Fully guarded: only a CadQuery-style backend exposes ``_combined``; any kernel
    error or absent solid leaves ``graph`` untouched (feature-level graph only).
    """
    combined = getattr(backend, "_combined", None)
    if not callable(combined):
        return
    try:
        shape = combined()
        if shape is None:
            return
        faces = shape.Faces()
        b_edges = shape.Edges()

        def _ekey(e) -> tuple:
            c = e.Center()
            return (round(e.Length(), 3),
                    round(c.x, 3), round(c.y, 3), round(c.z, 3))

        # edge nodes + a geometric key -> node-id map for adjacency matching
        edge_id_by_key: Dict[tuple, str] = {}
        for j, e in enumerate(b_edges):
            eid = "E%d" % j
            graph.nodes.append(FeatureNode(eid, "edge", {"length": round(e.Length(), 4)}))
            graph._by_id[eid] = graph.nodes[-1]
            edge_id_by_key.setdefault(_ekey(e), eid)

        for i, f in enumerate(faces):
            fid = "F%d" % i
            graph.nodes.append(FeatureNode(fid, "face", {"area": round(f.Area(), 4)}))
            graph._by_id[fid] = graph.nodes[-1]
            for fe in f.Edges():
                eid = edge_id_by_key.get(_ekey(fe))
                if eid is not None:
                    graph.edges.append(FeatureEdge(eid, fid, "bounds"))
    except Exception:  # noqa: BLE001 - enrichment must never break the graph
        return


# --- public entry point ----------------------------------------------------
def build_feature_graph(backend_or_opdag: Any, backend: Any = None) -> FeatureGraph:
    """Build a :class:`FeatureGraph` from an op-DAG or a geometry backend.

    ``backend_or_opdag`` may be an :class:`state.opdag.OpDAG` (anything exposing
    ``ops()``) or a backend. When an op-DAG is passed you may also pass a live
    ``backend=`` to unlock optional B-rep adjacency enrichment; when a backend is
    passed directly it is used both for structure and enrichment.
    """
    obj = backend_or_opdag
    if hasattr(obj, "ops") and callable(getattr(obj, "ops")):
        sketches, features = _extract_from_ops(list(obj.ops()))
        enrich_backend = backend
    else:
        sketches, features = _extract_from_backend(obj)
        enrich_backend = obj if backend is None else backend

    nodes, edges = _build(sketches, features)
    graph = FeatureGraph(nodes, edges)
    if enrich_backend is not None:
        _maybe_enrich_brep(graph, enrich_backend)
    return graph
