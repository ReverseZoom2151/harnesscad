"""Graph-CAD multi-layer knowledge graph (FORMAT v4) parser and validator.

Graph-CAD (Gong et al., ICLR 2026) mediates text-to-CAD generation through a
hierarchical, geometry-aware decomposition graph that is serialised as
structured text. Its stage-1 model emits two blocks: a MATERIAL LIBRARY and a
KNOWLEDGE GRAPH delimited by ``BEGIN_GRAPH`` / ``END_GRAPH`` markers, one node
per line::

    -- MATERIAL LIBRARY --
    mat_wood | diffuse_color=(0.6,0.4,0.2,1.0)
    #END_MATERIALS
    # ----------  BEGIN_GRAPH  ----------
    Lk: id=table | parent=- | type=assembly | create_method=composite
        | assembly_order=[top], [leg_a, leg_b]
    # ----------  END_GRAPH  ----------

The serialisation itself is fully deterministic, and so is everything that can
be checked about it before any geometry is built. This module implements that
half: a tolerant line parser for the pipe-separated ``key=value`` field syntax
(a node may wrap over several continuation lines), a material-library parser,
a canonical re-serialiser, and a structural validator that catches exactly the
failure modes the format's own rules forbid -- duplicate ids, dangling
``parent`` / ``depends_on`` / ``after`` / ``tool_id`` / ``target_id``
references, parent cycles, dependency cycles, unknown materials, leaves with no
``create_method``, boolean nodes missing their tool or target, and
``assembly_order`` groups that do not partition the node's children.

Only the deterministic representation layer is reimplemented here; the trained
stage-1/2/3 language models are external.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "Material",
    "GraphNode",
    "KnowledgeGraph",
    "BOOLEAN_METHODS",
    "parse_material_library",
    "parse_knowledge_graph",
    "parse_document",
    "serialize_material_library",
    "serialize_knowledge_graph",
    "validate_graph",
    "build_waves",
]

BEGIN_GRAPH = "BEGIN_GRAPH"
END_GRAPH = "END_GRAPH"
END_MATERIALS = "#END_MATERIALS"
MATERIAL_HEADER = "-- MATERIAL LIBRARY --"

#: ``create_method`` values that consume a tool shape and a target shape.
BOOLEAN_METHODS = ("boolean_subtract", "boolean_union", "boolean_intersect")

_NODE_START = re.compile(r"^\s*L(?P<layer>\d+)\s*:\s*(?P<body>.*)$")
_FIELD = re.compile(r"^(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?P<value>.*)$")
_MATERIAL = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\|\s*diffuse_color\s*=\s*"
    r"\((?P<rgba>[^)]*)\)\s*$"
)
_GROUP = re.compile(r"\[([^\]]*)\]")

_EMPTY = {"", "-", "none", "None"}


def _is_empty(value: Optional[str]) -> bool:
    return value is None or value.strip() in _EMPTY


def _clean(value: Optional[str]) -> Optional[str]:
    """Normalise a raw field value, mapping the ``-`` placeholder to ``None``."""
    if _is_empty(value):
        return None
    return " ".join(value.split())


def _id_list(value: Optional[str]) -> Tuple[str, ...]:
    text = _clean(value)
    if text is None:
        return ()
    return tuple(
        item.strip() for item in text.replace(";", ",").split(",") if item.strip()
    )


@dataclass(frozen=True)
class Material:
    """One MATERIAL LIBRARY entry: a snake_case name plus an RGBA colour."""

    name: str
    diffuse_color: Tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("material name is required")
        if len(self.diffuse_color) != 4:
            raise ValueError("diffuse_color must have four components")
        for component in self.diffuse_color:
            if not 0.0 <= component <= 1.0:
                raise ValueError("diffuse_color components must lie in [0, 1]")

    def serialize(self) -> str:
        body = ",".join(f"{value:g}" for value in self.diffuse_color)
        return f"{self.name} | diffuse_color=({body})"


@dataclass(frozen=True)
class GraphNode:
    """A single ``Lk:`` line of the decomposition graph."""

    node_id: str
    layer: int
    parent: Optional[str] = None
    type: Optional[str] = None
    size: Optional[str] = None
    align: Optional[str] = None
    anchor: Optional[str] = None
    pos: Optional[str] = None
    connect: Optional[str] = None
    orientation: Optional[str] = None
    rotation: Optional[str] = None
    pattern: Optional[str] = None
    mat: Optional[str] = None
    create_method: Optional[str] = None
    assembly_order: Optional[str] = None
    constraint: Optional[str] = None
    after: Tuple[str, ...] = ()
    depends_on: Tuple[str, ...] = ()
    tool_id: Optional[str] = None
    target_id: Optional[str] = None
    extra: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node id is required")
        if self.layer < 0:
            raise ValueError("layer must be non-negative")

    @property
    def is_boolean(self) -> bool:
        return self.create_method in BOOLEAN_METHODS

    def assembly_groups(self) -> Tuple[Tuple[str, ...], ...]:
        """Parse ``assembly_order=[a, b], [c]`` into ordered parallel groups."""
        if _is_empty(self.assembly_order):
            return ()
        text = self.assembly_order or ""
        groups: List[Tuple[str, ...]] = []
        matches = _GROUP.findall(text)
        if not matches:
            members = _id_list(text)
            return (members,) if members else ()
        for raw in matches:
            members = _id_list(raw)
            if members:
                groups.append(members)
        return tuple(groups)

    def serialize(self) -> str:
        parts = [f"id={self.node_id}", f"parent={self.parent or '-'}"]
        for key in (
            "type",
            "size",
            "align",
            "anchor",
            "pos",
            "connect",
            "orientation",
            "rotation",
            "pattern",
            "mat",
            "create_method",
            "assembly_order",
            "constraint",
        ):
            value = getattr(self, key)
            if value is not None:
                parts.append(f"{key}={value}")
        if self.after:
            parts.append("after=" + ", ".join(self.after))
        if self.depends_on:
            parts.append("depends_on=" + ", ".join(self.depends_on))
        for key in ("tool_id", "target_id"):
            value = getattr(self, key)
            if value is not None:
                parts.append(f"{key}={value}")
        for key in sorted(self.extra):
            parts.append(f"{key}={self.extra[key]}")
        return f"L{self.layer}: " + " | ".join(parts)


@dataclass(frozen=True)
class KnowledgeGraph:
    """A parsed FORMAT v4 graph: ordered nodes plus their material library."""

    nodes: Tuple[GraphNode, ...]
    materials: Tuple[Material, ...] = ()

    def __post_init__(self) -> None:
        seen: Dict[str, int] = {}
        for index, node in enumerate(self.nodes):
            if node.node_id in seen:
                raise ValueError(f"duplicate node id: {node.node_id}")
            seen[node.node_id] = index

    def by_id(self) -> Dict[str, GraphNode]:
        return {node.node_id: node for node in self.nodes}

    def children_of(self, node_id: Optional[str]) -> Tuple[GraphNode, ...]:
        return tuple(node for node in self.nodes if node.parent == node_id)

    def roots(self) -> Tuple[GraphNode, ...]:
        return self.children_of(None)

    def leaves(self) -> Tuple[GraphNode, ...]:
        parents = {node.parent for node in self.nodes if node.parent}
        return tuple(node for node in self.nodes if node.node_id not in parents)

    def material_names(self) -> Tuple[str, ...]:
        return tuple(material.name for material in self.materials)


_KNOWN_FIELDS = {
    "type",
    "size",
    "align",
    "anchor",
    "pos",
    "connect",
    "orientation",
    "rotation",
    "pattern",
    "mat",
    "create_method",
    "assembly_order",
    "constraint",
    "tool_id",
    "target_id",
}


def _split_fields(body: str) -> List[str]:
    """Split a node body on ``|`` while keeping bracketed groups intact."""
    parts: List[str] = []
    current: List[str] = []
    depth = 0
    for char in body:
        if char in "([":
            depth += 1
        elif char in ")]":
            depth = max(0, depth - 1)
        if char == "|" and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return [part.strip() for part in parts if part.strip()]


def _node_from_body(layer: int, body: str) -> GraphNode:
    fields: Dict[str, str] = {}
    for chunk in _split_fields(body):
        match = _FIELD.match(chunk)
        if not match:
            continue
        key = match.group("key").lower()
        value = match.group("value").strip()
        if key not in fields:
            fields[key] = value
    node_id = _clean(fields.pop("id", None))
    if node_id is None:
        raise ValueError(f"node line without an id: L{layer}: {body}")

    parent = _clean(fields.pop("parent", None))
    known = {key: _clean(fields.pop(key, None)) for key in _KNOWN_FIELDS}
    after = _id_list(fields.pop("after", None))
    depends_on = _id_list(fields.pop("depends_on", None))
    extra = {key: value for key, value in fields.items() if not _is_empty(value)}
    return GraphNode(
        node_id=node_id,
        layer=layer,
        parent=parent,
        after=after,
        depends_on=depends_on,
        extra=extra,
        **known,
    )


def parse_knowledge_graph(text: str) -> Tuple[GraphNode, ...]:
    """Parse every ``Lk:`` node between the BEGIN_GRAPH / END_GRAPH markers.

    If no markers are present the whole text is scanned, so partial model
    output can still be inspected. Node bodies may wrap onto continuation
    lines that start with ``|``.
    """
    lines = text.splitlines()
    begin = None
    end = None
    for index, line in enumerate(lines):
        if begin is None and BEGIN_GRAPH in line:
            begin = index + 1
        elif begin is not None and END_GRAPH in line:
            end = index
            break
    region = lines[begin:end] if begin is not None else lines
    if begin is not None and end is None:
        region = lines[begin:]

    nodes: List[GraphNode] = []
    pending_layer: Optional[int] = None
    pending_body: List[str] = []

    def flush() -> None:
        if pending_layer is not None and pending_body:
            nodes.append(_node_from_body(pending_layer, " ".join(pending_body)))

    for raw in region:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        match = _NODE_START.match(line)
        if match:
            flush()
            pending_layer = int(match.group("layer"))
            pending_body = [match.group("body").strip()]
            continue
        if pending_layer is not None and stripped.startswith("|"):
            pending_body.append(stripped)
            continue
        if stripped.startswith("#"):
            continue
    flush()
    return tuple(nodes)


def parse_material_library(text: str) -> Tuple[Material, ...]:
    """Parse the MATERIAL LIBRARY block (terminated by ``#END_MATERIALS``)."""
    materials: List[Material] = []
    started = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if MATERIAL_HEADER in line:
            started = True
            continue
        if line.startswith(END_MATERIALS):
            break
        if not started:
            continue
        match = _MATERIAL.match(line)
        if not match:
            continue
        components = [item.strip() for item in match.group("rgba").split(",")]
        if len(components) == 3:
            components.append("1.0")
        if len(components) != 4:
            raise ValueError(f"material {match.group('name')!r} needs 3 or 4 channels")
        rgba = tuple(float(item) for item in components)
        materials.append(Material(match.group("name"), rgba))  # type: ignore[arg-type]
    return tuple(materials)


def parse_document(text: str) -> KnowledgeGraph:
    """Parse a full stage-1 output: material library plus knowledge graph."""
    return KnowledgeGraph(
        nodes=parse_knowledge_graph(text),
        materials=parse_material_library(text),
    )


def serialize_material_library(materials: Sequence[Material]) -> str:
    """Emit the MATERIAL LIBRARY block, alphabetically sorted as the spec asks."""
    lines = [MATERIAL_HEADER]
    lines.extend(
        material.serialize() for material in sorted(materials, key=lambda m: m.name)
    )
    lines.append(END_MATERIALS)
    return "\n".join(lines)


def serialize_knowledge_graph(nodes: Sequence[GraphNode]) -> str:
    """Emit the delimited graph block in the given node order."""
    lines = [f"# ----------  {BEGIN_GRAPH}  ----------"]
    lines.extend(node.serialize() for node in nodes)
    lines.append(f"# ----------  {END_GRAPH}  ----------")
    return "\n".join(lines)


def _parent_cycles(nodes: Sequence[GraphNode]) -> List[str]:
    index = {node.node_id: node for node in nodes}
    cyclic: List[str] = []
    for node in nodes:
        seen = {node.node_id}
        current = node.parent
        while current is not None and current in index:
            if current in seen:
                cyclic.append(node.node_id)
                break
            seen.add(current)
            current = index[current].parent
    return cyclic


def _dependency_cycle(edges: Mapping[str, Sequence[str]]) -> bool:
    colour: Dict[str, int] = {}

    def visit(node: str) -> bool:
        state = colour.get(node, 0)
        if state == 1:
            return True
        if state == 2:
            return False
        colour[node] = 1
        for successor in edges.get(node, ()):  # deterministic: input order
            if successor in edges and visit(successor):
                return True
        colour[node] = 2
        return False

    return any(visit(node) for node in edges)


def validate_graph(graph: KnowledgeGraph) -> Tuple[str, ...]:
    """Return every structural violation of the FORMAT v4 rules, in order.

    An empty tuple means the graph is well formed: references resolve, the
    parent tree is acyclic, dependency edges are acyclic, leaves are buildable,
    boolean nodes name a tool and a target, materials exist, and each
    ``assembly_order`` exactly partitions its node's children.
    """
    errors: List[str] = []
    index = graph.by_id()
    materials = set(graph.material_names())

    if not graph.nodes:
        errors.append("graph has no nodes")
        return tuple(errors)
    if not graph.roots():
        errors.append("graph has no root node (every node declares a parent)")

    for node in graph.nodes:
        if node.parent is not None and node.parent not in index:
            errors.append(f"{node.node_id}: unknown parent {node.parent!r}")
        if node.parent == node.node_id:
            errors.append(f"{node.node_id}: node is its own parent")
        for reference in node.depends_on:
            if reference not in index:
                errors.append(f"{node.node_id}: unknown depends_on {reference!r}")
        for reference in node.after:
            if reference not in index:
                errors.append(f"{node.node_id}: unknown after {reference!r}")
            elif index[reference].parent != node.parent:
                errors.append(f"{node.node_id}: after {reference!r} is not a sibling")
        for key in ("tool_id", "target_id"):
            reference = getattr(node, key)
            if reference is not None and reference not in index:
                errors.append(f"{node.node_id}: unknown {key} {reference!r}")
        if node.mat is not None and materials and node.mat not in materials:
            errors.append(f"{node.node_id}: material {node.mat!r} is not in the library")
        if node.is_boolean:
            if node.tool_id is None:
                errors.append(f"{node.node_id}: {node.create_method} needs a tool_id")
            if node.target_id is None:
                errors.append(f"{node.node_id}: {node.create_method} needs a target_id")

    for node_id in _parent_cycles(graph.nodes):
        errors.append(f"{node_id}: parent chain forms a cycle")

    edges = {
        node.node_id: tuple(node.depends_on) + tuple(node.after)
        for node in graph.nodes
    }
    if _dependency_cycle(edges):
        errors.append("dependency edges (after/depends_on) form a cycle")

    for node in graph.leaves():
        if node.create_method is None:
            errors.append(f"{node.node_id}: leaf node has no create_method")

    for node in graph.nodes:
        groups = node.assembly_groups()
        if not groups:
            continue
        children = {child.node_id for child in graph.children_of(node.node_id)}
        listed: List[str] = []
        for group in groups:
            listed.extend(group)
        duplicates = sorted({item for item in listed if listed.count(item) > 1})
        for item in duplicates:
            errors.append(f"{node.node_id}: assembly_order repeats {item!r}")
        for item in sorted(set(listed) - children):
            errors.append(f"{node.node_id}: assembly_order lists non-child {item!r}")
        for item in sorted(children - set(listed)):
            errors.append(f"{node.node_id}: assembly_order omits child {item!r}")

    return tuple(errors)


def build_waves(graph: KnowledgeGraph) -> Tuple[Tuple[str, ...], ...]:
    """Schedule the graph into parallel build waves (children before parents).

    A node becomes ready once its children, its ``depends_on`` / ``after``
    predecessors, and its boolean ``tool_id`` / ``target_id`` operands are all
    built. Within a wave, ids keep their declaration order; a parent's
    ``assembly_order`` further constrains its children so that group *i* is
    fully built before any member of group *i + 1*. Raises ``ValueError`` if
    the constraints cannot be satisfied (a cycle).
    """
    index = graph.by_id()
    prerequisites: Dict[str, set] = {node.node_id: set() for node in graph.nodes}

    for node in graph.nodes:
        for child in graph.children_of(node.node_id):
            prerequisites[node.node_id].add(child.node_id)
        for reference in tuple(node.depends_on) + tuple(node.after):
            if reference in index:
                prerequisites[node.node_id].add(reference)
        for key in ("tool_id", "target_id"):
            reference = getattr(node, key)
            if reference is not None and reference in index and reference != node.node_id:
                prerequisites[node.node_id].add(reference)

    for node in graph.nodes:
        groups = node.assembly_groups()
        earlier: List[str] = []
        for group in groups:
            for member in group:
                if member in index:
                    prerequisites[member].update(
                        item for item in earlier if item != member
                    )
            earlier.extend(item for item in group if item in index)

    order = [node.node_id for node in graph.nodes]
    done: set = set()
    waves: List[Tuple[str, ...]] = []
    while len(done) < len(order):
        ready = tuple(
            node_id
            for node_id in order
            if node_id not in done and prerequisites[node_id] <= done
        )
        if not ready:
            remaining = sorted(set(order) - done)
            raise ValueError(f"cyclic build constraints among {remaining}")
        waves.append(ready)
        done.update(ready)
    return tuple(waves)


def iter_subtree(graph: KnowledgeGraph, root_id: str) -> Iterable[GraphNode]:
    """Yield ``root_id`` and its descendants in deterministic pre-order."""
    if root_id not in graph.by_id():
        raise KeyError(root_id)
    stack = [root_id]
    while stack:
        current = stack.pop(0)
        yield graph.by_id()[current]
        stack = [child.node_id for child in graph.children_of(current)] + stack
