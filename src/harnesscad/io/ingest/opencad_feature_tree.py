"""Ingest an OpenCAD feature-tree JSON document into CISP ops (family: opencad).

OpenCAD persists a parametric model as a feature-tree JSON: ``{"root_id": ...,
"nodes": {id: {operation, parameters, parent_id/tool_refs (or a derived
``depends_on``), suppressed, ...}}}``, produced by its fluent runtime, its agent
tool runtime, and its CLI (``opencad build model.json``). The vocabulary is
close to CISP but not CISP: operation names differ (``add_sketch`` vs
``new_sketch``, ``fillet`` vs ``fillet_edges``), booleans reference *feature
node ids* (``base_id``/``tool_id``) rather than backend body ids, and sketches
carry an *entity map* plus an optional ``profile_order`` list that fixes the
loop ordering independent of dict iteration (OpenCAD
``ToolRuntime._entities_to_sketch_segments`` and ``kernel_adapter.py``).

This adapter reproduces those translation rules deterministically:

* nodes are emitted in the tree's topological order (alphabetical tie-breaks);
* sketch entities are emitted in ``profile_order`` first, then the remainder in
  sorted-id order; a legacy *point-only* sketch is chained into a closed
  polyline of ``AddLine`` ops (the OpenCAD fallback);
* feature references are resolved to the ids the harness backends will
  allocate: sketches become ``sk1, sk2, ...`` and solid features ``f1, f2, ...``
  in emission order, matching the deterministic id allocators every harness
  backend uses -- so a boolean's ``base_id``/``tool_id`` land as concrete
  ``target``/``tool`` refs;
* suppressed nodes are skipped, and any node depending on a skipped node is
  skipped too, with a note (never silently dropped);
* what cannot be lowered faithfully is recorded in ``notes`` (OpenCAD edge/face
  id lists have no CISP selector equivalent and are widened to "all edges" /
  default faces; assembly mates use a different mate vocabulary and are
  skipped), so the caller sees exactly where fidelity was lost.

Family discipline: like the token quantisers, this adapter is *family-tagged*.
A document carrying an explicit ``"family"`` key other than ``"opencad"`` is
refused -- op vocabularies are never silently mixed
(:mod:`harnesscad.domain.reconstruction.ingest_pipeline` holds the same line for
token families).

Stdlib-only, deterministic. No kernel is touched: the output is a CISP op list
ready for ``HarnessSession.apply_ops`` on any backend.

Public API
----------
``ingest_opencad_tree``, ``ingest_opencad_oplog``, ``OpenCadIngestResult``
``OpenCadIngestError``, ``FAMILY``
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

from harnesscad.core.cisp.ops import (
    AddArc,
    AddCircle,
    AddLine,
    AddRectangle,
    Boolean,
    Chamfer,
    CircularPattern,
    Draft,
    Extrude,
    Fillet,
    LinearPattern,
    Loft,
    Mirror,
    NewSketch,
    Op,
    Primitive,
    Revolve,
    Shell,
    Sweep,
    Thicken,
    Transform,
)

__all__ = [
    "FAMILY",
    "OpenCadIngestError",
    "OpenCadIngestResult",
    "ingest_opencad_tree",
    "ingest_opencad_oplog",
]

FAMILY = "opencad"

#: OpenCAD operation aliases -> canonical names (kernel_adapter.py rules).
_ALIASES = {
    "add_sketch": "create_sketch",
    "fillet": "fillet_edges",
    "add_cylinder": "create_cylinder",
    "add_box": "create_box",
    "add_sphere": "create_sphere",
}

_SKIP_OPS = {
    "seed", "root",
    "assembly_mate", "create_assembly_mate", "delete_assembly_mate",
    "list_assembly_mates",
    "import_step", "export_step",
}

_AXIS_PLANES = {
    (1.0, 0.0, 0.0): "YZ",
    (0.0, 1.0, 0.0): "XZ",
    (0.0, 0.0, 1.0): "XY",
}


class OpenCadIngestError(ValueError):
    """The document is not an ingestible OpenCAD feature tree."""


@dataclass
class OpenCadIngestResult:
    """The lowered op stream plus an honest account of what did not survive."""

    ops: List[Op] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    sketch_ids: Dict[str, str] = field(default_factory=dict)
    feature_ids: Dict[str, str] = field(default_factory=dict)
    family: str = FAMILY

    def to_dict(self) -> Dict[str, object]:
        return {
            "family": self.family,
            "ops": [op.to_dict() for op in self.ops],
            "notes": list(self.notes),
            "skipped": list(self.skipped),
            "sketch_ids": dict(self.sketch_ids),
            "feature_ids": dict(self.feature_ids),
        }


def _num(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def _pair(value: object) -> Optional[Tuple[float, float]]:
    if isinstance(value, (list, tuple)) and len(value) == 2 \
            and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    for v in value):
        return float(value[0]), float(value[1])
    return None


def _depends_on(node: Mapping[str, object]) -> List[str]:
    """Derive dependencies the way OpenCAD's FeatureNode does."""
    explicit = node.get("depends_on")
    if isinstance(explicit, list):
        return [str(d) for d in explicit]
    deps: List[str] = []
    parent = node.get("parent_id")
    if isinstance(parent, str) and parent:
        deps.append(parent)
    tools = node.get("tool_refs")
    if isinstance(tools, list):
        deps.extend(str(t) for t in tools)
    return deps


def _topological(nodes: Mapping[str, Mapping[str, object]],
                 root_id: str) -> List[str]:
    """Kahn order over the node map, alphabetical tie-breaks; cycles refused."""
    indegree: Dict[str, int] = {nid: 0 for nid in nodes}
    children: Dict[str, Set[str]] = {nid: set() for nid in nodes}
    for nid, node in nodes.items():
        for dep in _depends_on(node):
            if dep == nid:
                raise OpenCadIngestError(
                    "Node '%s' depends on itself." % nid)
            if dep not in nodes:
                raise OpenCadIngestError(
                    "Node '%s' depends on missing node '%s'." % (nid, dep))
            children[dep].add(nid)
            indegree[nid] += 1
    ready = sorted(nid for nid, deg in indegree.items() if deg == 0)
    ordered: List[str] = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for child in sorted(children[current]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
        ready.sort()
    if len(ordered) != len(nodes):
        raise OpenCadIngestError("Feature tree contains a cycle.")
    # Root (if present) contributes nothing; keep it out of emission.
    return [nid for nid in ordered if nid != root_id]


class _Emitter:
    """Tracks the deterministic sk/f id allocation the backends will perform."""

    def __init__(self, result: OpenCadIngestResult) -> None:
        self.result = result
        self._sk = 0
        self._f = 0

    def sketch(self, node_id: str, ops: Sequence[Op]) -> str:
        self._sk += 1
        sid = "sk%d" % self._sk
        self.result.sketch_ids[node_id] = sid
        self.result.ops.extend(ops)
        return sid

    def feature(self, node_id: str, op: Op) -> str:
        self._f += 1
        fid = "f%d" % self._f
        self.result.feature_ids[node_id] = fid
        self.result.ops.append(op)
        return fid

    def note(self, text: str) -> None:
        self.result.notes.append(text)

    def skip(self, node_id: str, reason: str) -> None:
        self.result.skipped.append(node_id)
        self.result.notes.append("skipped '%s': %s" % (node_id, reason))


def _sketch_entity_ops(sid_placeholder: str,
                       parameters: Mapping[str, object],
                       emit_note) -> List[Op]:
    """Lower an OpenCAD sketch node's entities/segments to CISP entity ops.

    Accepts both the agent-tool form (``entities`` map + ``profile_order``) and
    the kernel form (``segments`` list). ``sid_placeholder`` is patched in by
    the caller once the sketch id is allocated -- entity ops here always use
    that exact string.
    """
    ops: List[Op] = []
    points: List[Tuple[float, float]] = []

    ordered: List[Mapping[str, object]] = []
    entities = parameters.get("entities")
    if isinstance(entities, Mapping):
        order = parameters.get("profile_order")
        seen: Set[str] = set()
        if isinstance(order, list):
            for eid in order:
                entity = entities.get(str(eid))
                if isinstance(entity, Mapping):
                    ordered.append(entity)
                    seen.add(str(eid))
        for eid in sorted(str(k) for k in entities):
            if eid in seen:
                continue
            entity = entities[eid]
            if isinstance(entity, Mapping):
                ordered.append(entity)
    segments = parameters.get("segments")
    if isinstance(segments, list):
        for seg in segments:
            if isinstance(seg, Mapping):
                ordered.append(seg)

    for entity in ordered:
        kind = str(entity.get("type", "")).lower()
        if kind == "line":
            start = _pair(entity.get("start"))
            end = _pair(entity.get("end"))
            if start is None or end is None:
                x1, y1 = _num(entity.get("x1")), _num(entity.get("y1"))
                x2, y2 = _num(entity.get("x2")), _num(entity.get("y2"))
            else:
                (x1, y1), (x2, y2) = start, end
            ops.append(AddLine(sketch=sid_placeholder, x1=x1, y1=y1, x2=x2, y2=y2))
        elif kind == "circle":
            centre = _pair(entity.get("center"))
            cx = centre[0] if centre else _num(entity.get("cx", entity.get("x")))
            cy = centre[1] if centre else _num(entity.get("cy", entity.get("y")))
            r = _num(entity.get("radius"), 1.0)
            if entity.get("subtract"):
                emit_note("circle hole flag ('subtract') has no CISP entity "
                          "equivalent; emitted as a plain circle region")
            ops.append(AddCircle(sketch=sid_placeholder, cx=cx, cy=cy, r=r))
        elif kind == "arc":
            centre = _pair(entity.get("center"))
            cx = centre[0] if centre else _num(entity.get("cx"))
            cy = centre[1] if centre else _num(entity.get("cy"))
            r = _num(entity.get("radius"), 1.0)
            start_a = _num(entity.get("start_angle"), 0.0)
            end_a = _num(entity.get("end_angle"), 90.0)
            ops.append(AddArc(sketch=sid_placeholder, cx=cx, cy=cy, r=r,
                              start=start_a, end=end_a))
        elif kind == "rectangle":
            ops.append(AddRectangle(
                sketch=sid_placeholder,
                x=_num(entity.get("x")), y=_num(entity.get("y")),
                w=_num(entity.get("width"), 1.0),
                h=_num(entity.get("height"), 1.0),
            ))
        elif kind == "point":
            x, y = entity.get("x"), entity.get("y")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                points.append((float(x), float(y)))
        elif kind:
            emit_note("sketch entity type '%s' is not lowered" % kind)

    # Legacy point-only payloads chain into a closed polyline (OpenCAD rule).
    if not ops and len(points) >= 2:
        for idx in range(len(points) - 1):
            (x1, y1), (x2, y2) = points[idx], points[idx + 1]
            ops.append(AddLine(sketch=sid_placeholder, x1=x1, y1=y1, x2=x2, y2=y2))
        if len(points) >= 3:
            (x1, y1), (x2, y2) = points[-1], points[0]
            ops.append(AddLine(sketch=sid_placeholder, x1=x1, y1=y1, x2=x2, y2=y2))

    return ops


def _axis6(origin: object, direction: object) -> Tuple[float, ...]:
    o = origin if isinstance(origin, (list, tuple)) and len(origin) == 3 \
        else (0.0, 0.0, 0.0)
    d = direction if isinstance(direction, (list, tuple)) and len(direction) == 3 \
        else (0.0, 0.0, 1.0)
    ox, oy, oz = (_num(v) for v in o)
    dx, dy, dz = (_num(v) for v in d)
    return (ox, oy, oz, ox + dx, oy + dy, oz + dz)


def ingest_opencad_tree(document: Mapping[str, object]) -> OpenCadIngestResult:
    """Lower an OpenCAD feature-tree JSON document to a CISP op stream.

    The document is either the tree itself (``root_id`` + ``nodes``) or a
    wrapper carrying it under ``"feature_tree"`` (the design-artifact layout).
    An explicit ``"family"`` key must be absent or ``"opencad"``.
    """
    if not isinstance(document, Mapping):
        raise OpenCadIngestError("Document must be a JSON object.")

    family = document.get("family")
    if family is not None and str(family) != FAMILY:
        raise OpenCadIngestError(
            "Document is tagged family '%s' but this is the '%s' adapter; "
            "op vocabularies are never blended." % (family, FAMILY)
        )

    tree = document
    if "nodes" not in tree and isinstance(document.get("feature_tree"), Mapping):
        tree = document["feature_tree"]  # type: ignore[assignment]

    nodes_raw = tree.get("nodes")
    if not isinstance(nodes_raw, Mapping):
        raise OpenCadIngestError("Document has no 'nodes' map.")
    root_id = str(tree.get("root_id", "root"))

    nodes: Dict[str, Mapping[str, object]] = {}
    for nid, node in nodes_raw.items():
        if not isinstance(node, Mapping):
            raise OpenCadIngestError("Node '%s' is not an object." % nid)
        nodes[str(nid)] = node

    result = OpenCadIngestResult()
    emitter = _Emitter(result)
    order = _topological(nodes, root_id)

    for node_id in order:
        node = nodes[node_id]
        if node.get("suppressed"):
            emitter.skip(node_id, "node is suppressed")
            continue
        deps = [d for d in _depends_on(node) if d != root_id]
        blocked = [d for d in deps if d in result.skipped]
        if blocked:
            emitter.skip(node_id, "depends on skipped node(s) %s"
                         % ", ".join(sorted(blocked)))
            continue

        operation = str(node.get("operation", ""))
        operation = _ALIASES.get(operation, operation)
        parameters = node.get("parameters")
        parameters = parameters if isinstance(parameters, Mapping) else {}
        _lower_node(emitter, node_id, operation, parameters, result)

    return result


def _resolve_feature_ref(result: OpenCadIngestResult, ref: object) -> str:
    """Map an OpenCAD feature-node id to the predicted backend body id."""
    key = str(ref) if ref is not None else ""
    return result.feature_ids.get(key, "")


def _lower_node(emitter: _Emitter, node_id: str, operation: str,
                parameters: Mapping[str, object],
                result: OpenCadIngestResult) -> None:
    if operation in _SKIP_OPS:
        if operation in ("seed", "root"):
            return
        emitter.skip(node_id, "operation '%s' is not lowered (different "
                     "vocabulary / not a mutating geometry op)" % operation)
        return

    if operation == "create_sketch":
        placeholder = "sk%d" % (emitter._sk + 1)
        entity_ops = _sketch_entity_ops(placeholder, parameters, emitter.note)
        plane = str(parameters.get("plane", "XY")).upper()
        sketch_ops: List[Op] = [NewSketch(plane=plane)]
        sketch_ops.extend(entity_ops)
        if not entity_ops:
            emitter.note("sketch '%s' produced no entity ops" % node_id)
        emitter.sketch(node_id, sketch_ops)
        return

    if operation == "extrude":
        sketch_ref = parameters.get("sketch_id", "")
        sid = result.sketch_ids.get(str(sketch_ref), "")
        if not sid:
            emitter.skip(node_id, "extrude references unknown sketch '%s'"
                         % sketch_ref)
            return
        distance = _num(parameters.get("depth",
                                       parameters.get("distance")), 1.0)
        if parameters.get("both"):
            emitter.note("extrude '%s': symmetric ('both') extrude has no "
                         "CISP flag; emitted single-sided" % node_id)
        emitter.feature(node_id, Extrude(sketch=sid, distance=distance))
        return

    if operation == "create_box":
        emitter.feature(node_id, Primitive(
            shape="box",
            dx=_num(parameters.get("length"), 1.0),
            dy=_num(parameters.get("width"), 1.0),
            dz=_num(parameters.get("height"), 1.0),
        ))
        return

    if operation == "create_cylinder":
        fid = emitter.feature(node_id, Primitive(
            shape="cylinder",
            r=_num(parameters.get("radius"), 1.0),
            h=_num(parameters.get("height"), 1.0),
        ))
        position = parameters.get("position")
        if isinstance(position, Mapping):
            tx = _num(position.get("x"))
            ty = _num(position.get("y"))
            tz = _num(position.get("z"))
            if tx or ty or tz:
                emitter.feature(
                    "%s@move" % node_id,
                    Transform(feature_or_body=fid, tx=tx, ty=ty, tz=tz),
                )
                # The moved body is what downstream refs should see.
                result.feature_ids[node_id] = result.feature_ids["%s@move" % node_id]
                del result.feature_ids["%s@move" % node_id]
        return

    if operation == "create_sphere":
        emitter.feature(node_id, Primitive(
            shape="sphere", r=_num(parameters.get("radius"), 1.0)))
        return

    if operation == "create_cone":
        emitter.feature(node_id, Primitive(
            shape="cone",
            r=_num(parameters.get("radius1"), 1.0),
            r2=_num(parameters.get("radius2"), 0.0),
            h=_num(parameters.get("height"), 1.0),
        ))
        return

    if operation == "create_torus":
        emitter.feature(node_id, Primitive(
            shape="torus",
            r=_num(parameters.get("major_radius"), 1.0),
            r2=_num(parameters.get("minor_radius"), 0.25),
        ))
        return

    if operation in ("boolean_union", "boolean_cut", "boolean_intersection"):
        kind = {"boolean_union": "union", "boolean_cut": "cut",
                "boolean_intersection": "intersect"}[operation]
        base_ref = parameters.get("base_id", parameters.get("shape_a_id"))
        tool_ref = parameters.get("tool_id", parameters.get("shape_b_id"))
        target = _resolve_feature_ref(result, base_ref)
        tool = _resolve_feature_ref(result, tool_ref)
        if not target or not tool:
            emitter.skip(node_id, "boolean references unresolved feature(s) "
                         "base=%r tool=%r" % (base_ref, tool_ref))
            return
        emitter.feature(node_id, Boolean(kind=kind, target=target, tool=tool))
        return

    if operation == "fillet_edges":
        edges = parameters.get("edge_ids", parameters.get("edge_selection"))
        if edges:
            emitter.note("fillet '%s': OpenCAD edge ids %s have no CISP "
                         "selector equivalent; widened to all edges"
                         % (node_id, edges))
        emitter.feature(node_id, Fillet(
            edges=(), radius=_num(parameters.get("radius"), 1.0)))
        return

    if operation == "chamfer_edges":
        edges = parameters.get("edge_ids")
        if edges:
            emitter.note("chamfer '%s': OpenCAD edge ids widened to all edges"
                         % node_id)
        emitter.feature(node_id, Chamfer(
            edges=(), distance=_num(parameters.get("distance"), 1.0)))
        return

    if operation == "shell":
        faces = parameters.get("face_ids")
        if faces:
            emitter.note("shell '%s': OpenCAD face ids have no CISP selector "
                         "equivalent; default open face used" % node_id)
        emitter.feature(node_id, Shell(
            faces=(), thickness=_num(parameters.get("thickness"), 1.0)))
        return

    if operation == "draft":
        faces = parameters.get("face_ids")
        if faces:
            emitter.note("draft '%s': OpenCAD face ids dropped (no selector "
                         "equivalent)" % node_id)
        emitter.feature(node_id, Draft(
            faces=(), angle=_num(parameters.get("angle"), 0.0)))
        return

    if operation == "offset_shape":
        emitter.note("offset '%s' lowered as Thicken (offset-solid)" % node_id)
        emitter.feature(node_id, Thicken(
            thickness=_num(parameters.get("distance"), 1.0)))
        return

    if operation == "revolve":
        profile_ref = parameters.get("shape_id", parameters.get("sketch_id"))
        sid = result.sketch_ids.get(str(profile_ref), "")
        if not sid:
            emitter.skip(node_id, "revolve profile '%s' is not an ingested "
                         "sketch" % profile_ref)
            return
        emitter.feature(node_id, Revolve(
            sketch=sid,
            axis=_axis6(parameters.get("axis_origin"),
                        parameters.get("axis_direction")),
            angle=_num(parameters.get("angle"), 360.0),
        ))
        return

    if operation == "sweep":
        profile = result.sketch_ids.get(str(parameters.get("profile_id")), "")
        path = result.sketch_ids.get(str(parameters.get("path_id")), "")
        if not profile or not path:
            emitter.skip(node_id, "sweep profile/path are not ingested sketches")
            return
        emitter.feature(node_id, Sweep(sketch=profile, path=path))
        return

    if operation == "loft":
        refs = parameters.get("profile_ids")
        sketches: List[str] = []
        if isinstance(refs, list):
            sketches = [result.sketch_ids.get(str(r), "") for r in refs]
        if not sketches or not all(sketches):
            emitter.skip(node_id, "loft profiles are not ingested sketches")
            return
        emitter.feature(node_id, Loft(
            sketches=tuple(sketches),
            ruled=bool(parameters.get("ruled", False)),
        ))
        return

    if operation == "linear_pattern":
        feature = _resolve_feature_ref(result, parameters.get("shape_id"))
        direction = parameters.get("direction")
        d = tuple(_num(v) for v in direction) \
            if isinstance(direction, (list, tuple)) and len(direction) == 3 \
            else (1.0, 0.0, 0.0)
        emitter.feature(node_id, LinearPattern(
            feature=feature, direction=d,
            count=int(_num(parameters.get("count"), 2)),
            spacing=_num(parameters.get("spacing"), 1.0),
        ))
        return

    if operation == "circular_pattern":
        feature = _resolve_feature_ref(result, parameters.get("shape_id"))
        emitter.feature(node_id, CircularPattern(
            feature=feature,
            axis=_axis6(parameters.get("axis_origin"),
                        parameters.get("axis_direction")),
            count=int(_num(parameters.get("count"), 2)),
            angle=_num(parameters.get("angle"), 360.0),
        ))
        return

    if operation == "mirror":
        normal = parameters.get("plane_normal")
        plane = None
        if isinstance(normal, (list, tuple)) and len(normal) == 3:
            key = tuple(abs(_num(v)) for v in normal)
            total = sum(key)
            if total > 0:
                unit = tuple(round(v / total) * 1.0 for v in key)
                plane = _AXIS_PLANES.get(unit)  # normal axis -> mirror plane
        if plane is None:
            emitter.skip(node_id, "mirror plane normal %r is not axis-aligned"
                         % (normal,))
            return
        feature = _resolve_feature_ref(result, parameters.get("shape_id"))
        origin = parameters.get("plane_origin")
        if isinstance(origin, (list, tuple)) and any(_num(v) for v in origin):
            emitter.note("mirror '%s': non-origin plane offset dropped "
                         "(CISP Mirror planes pass through the origin)"
                         % node_id)
        emitter.feature(node_id, Mirror(feature_or_body=feature, plane=plane))
        return

    emitter.skip(node_id, "unknown operation '%s'" % operation)


def ingest_opencad_oplog(entries: Sequence[Mapping[str, object]]
                         ) -> OpenCadIngestResult:
    """Lower an OpenCAD kernel op-log (or SnapshotV1 ``entries``) to CISP ops.

    Each entry is ``{"operation": name, "params": {...}}`` (extra keys --
    ``id``, ``timestamp``, ``result_shape_id`` -- are ignored; entries with
    ``success`` explicitly false are skipped with a note). Replay order is the
    list order, which IS the kernel's append-only journal order.
    """
    result = OpenCadIngestResult()
    emitter = _Emitter(result)
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise OpenCadIngestError("Log entry %d is not an object." % index)
        if entry.get("success") is False:
            emitter.skip("log[%d]" % index, "entry recorded as failed")
            continue
        operation = _ALIASES.get(str(entry.get("operation", "")),
                                 str(entry.get("operation", "")))
        params = entry.get("params")
        params = params if isinstance(params, Mapping) else {}
        node_id = str(entry.get("result_shape_id") or "log[%d]" % index)
        _lower_node(emitter, node_id, operation, params, result)
    return result


# ── selfcheck ───────────────────────────────────────────────────────


def selfcheck(verbose: bool = False) -> bool:
    """Lower the OpenCAD planner's mounting-bracket-shaped tree and check ids."""
    checks: List[Tuple[str, bool]] = []

    doc = {
        "root_id": "root",
        "nodes": {
            "root": {"operation": "seed", "parameters": {}},
            "sketch-0001": {
                "operation": "add_sketch",
                "parent_id": None,
                "parameters": {
                    "entities": {
                        "l1": {"id": "l1", "type": "line",
                               "start": [0.0, 0.0], "end": [30.0, 0.0]},
                        "l2": {"id": "l2", "type": "line",
                               "start": [30.0, 0.0], "end": [30.0, 20.0]},
                        "l3": {"id": "l3", "type": "line",
                               "start": [30.0, 20.0], "end": [0.0, 20.0]},
                        "l4": {"id": "l4", "type": "line",
                               "start": [0.0, 20.0], "end": [0.0, 0.0]},
                    },
                    "profile_order": ["l1", "l2", "l3", "l4"],
                },
            },
            "feat-0001": {
                "operation": "extrude",
                "parent_id": "sketch-0001",
                "parameters": {"sketch_id": "sketch-0001", "depth": 8.0},
            },
            "sketch-0002": {
                "operation": "add_sketch",
                "parent_id": "feat-0001",
                "parameters": {
                    "entities": {
                        "c1": {"id": "c1", "type": "circle",
                               "cx": 15.0, "cy": 10.0, "radius": 4.0},
                    },
                },
            },
            "feat-0002": {
                "operation": "extrude",
                "parent_id": "sketch-0002",
                "parameters": {"sketch_id": "sketch-0002", "depth": 8.0},
            },
            "feat-0003": {
                "operation": "boolean_cut",
                "parent_id": "feat-0001",
                "tool_refs": ["feat-0002"],
                "parameters": {"base_id": "feat-0001", "tool_id": "feat-0002"},
            },
            "feat-0004": {
                "operation": "fillet",
                "parent_id": "feat-0003",
                "parameters": {"shape_id": "feat-0003",
                               "edge_selection": ["outer_perimeter"],
                               "radius": 1.25},
            },
        },
    }

    result = ingest_opencad_tree(doc)
    kinds = [op.OP for op in result.ops]
    checks.append(("op sequence", kinds == [
        "new_sketch", "add_line", "add_line", "add_line", "add_line",
        "extrude", "new_sketch", "add_circle", "extrude", "boolean", "fillet",
    ]))
    checks.append(("sketch ids", result.sketch_ids == {
        "sketch-0001": "sk1", "sketch-0002": "sk2"}))
    checks.append(("feature ids", result.feature_ids == {
        "feat-0001": "f1", "feat-0002": "f2", "feat-0003": "f3",
        "feat-0004": "f4"}))
    booleans = [op for op in result.ops if isinstance(op, Boolean)]
    checks.append(("boolean refs resolved",
                   booleans and booleans[0].target == "f1"
                   and booleans[0].tool == "f2"
                   and booleans[0].kind == "cut"))
    extrudes = [op for op in result.ops if isinstance(op, Extrude)]
    checks.append(("extrude wired to sketch",
                   [e.sketch for e in extrudes] == ["sk1", "sk2"]))
    checks.append(("fillet widening noted",
                   any("widened to all edges" in n for n in result.notes)))
    checks.append(("nothing skipped", result.skipped == []))
    checks.append(("json round trip",
                   json.loads(json.dumps(result.to_dict()))["family"]
                   == FAMILY))

    # Determinism: same input, same output.
    again = ingest_opencad_tree(doc)
    checks.append(("deterministic",
                   [o.to_dict() for o in again.ops]
                   == [o.to_dict() for o in result.ops]))

    # Suppression: a suppressed tool skips the boolean that needs it.
    doc2 = json.loads(json.dumps(doc))
    doc2["nodes"]["feat-0002"]["suppressed"] = True
    result2 = ingest_opencad_tree(doc2)
    checks.append(("suppressed skipped", "feat-0002" in result2.skipped))
    checks.append(("dependent skipped", "feat-0003" in result2.skipped))
    checks.append(("skip is noted",
                   any("feat-0003" in n and "skipped" in n
                       for n in result2.notes)))

    # Legacy point-only sketch closes into a polyline.
    doc3 = {
        "root_id": "root",
        "nodes": {
            "root": {"operation": "seed"},
            "sk-a": {"operation": "add_sketch", "parameters": {"entities": {
                "p1": {"type": "point", "x": 0.0, "y": 0.0},
                "p2": {"type": "point", "x": 10.0, "y": 0.0},
                "p3": {"type": "point", "x": 10.0, "y": 5.0},
            }, "profile_order": ["p1", "p2", "p3"]}},
        },
    }
    result3 = ingest_opencad_tree(doc3)
    lines = [op for op in result3.ops if isinstance(op, AddLine)]
    checks.append(("point fallback closes loop", len(lines) == 3
                   and lines[-1].x2 == 0.0 and lines[-1].y2 == 0.0))

    # Family discipline.
    try:
        ingest_opencad_tree({"family": "deepcad", "root_id": "r", "nodes": {}})
        checks.append(("foreign family refused", False))
    except OpenCadIngestError:
        checks.append(("foreign family refused", True))

    # Cycles refused.
    try:
        ingest_opencad_tree({"root_id": "root", "nodes": {
            "a": {"operation": "extrude", "parent_id": "b", "parameters": {}},
            "b": {"operation": "extrude", "parent_id": "a", "parameters": {}},
        }})
        checks.append(("cycle refused", False))
    except OpenCadIngestError:
        checks.append(("cycle refused", True))

    # Op-log replay path.
    log = [
        {"operation": "create_box",
         "params": {"length": 20.0, "width": 10.0, "height": 5.0},
         "result_shape_id": "box-0001", "success": True},
        {"operation": "create_cylinder",
         "params": {"radius": 2.0, "height": 5.0},
         "result_shape_id": "cyl-0001", "success": True},
        {"operation": "boolean_cut",
         "params": {"shape_a_id": "box-0001", "shape_b_id": "cyl-0001"},
         "result_shape_id": "cut-0001", "success": True},
    ]
    result4 = ingest_opencad_oplog(log)
    checks.append(("oplog lowered", [op.OP for op in result4.ops]
                   == ["primitive", "primitive", "boolean"]))
    booleans4 = [op for op in result4.ops if isinstance(op, Boolean)]
    checks.append(("oplog refs resolved",
                   booleans4[0].target == "f1" and booleans4[0].tool == "f2"))

    ok = all(passed for _, passed in checks)
    if verbose:
        for name, passed in checks:
            print("  %-28s %s" % (name, "ok" if passed else "FAIL"))
        print("opencad_feature_tree selfcheck: %s" % ("ok" if ok else "FAILED"))
    return ok


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.io.ingest.opencad_feature_tree",
        description="Ingest an OpenCAD feature-tree JSON (family: opencad) "
                    "into CISP ops.",
    )
    parser.add_argument("source", nargs="?",
                        help="path to an OpenCAD feature-tree JSON document")
    parser.add_argument("--oplog", action="store_true",
                        help="treat the source as a kernel op-log / snapshot "
                             "entries array")
    parser.add_argument("--selfcheck", action="store_true",
                        help="run the synthetic ingest self-check (no real data)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.selfcheck:
        return 0 if selfcheck(verbose=True) else 1

    if not args.source:
        parser.print_help()
        return 0

    with open(args.source, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if args.oplog:
        result = ingest_opencad_oplog(payload)
    else:
        result = ingest_opencad_tree(payload)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
