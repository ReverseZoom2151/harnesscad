"""Deterministic, framework-neutral debug views for CAD graph state.

The view model combines operation history (including branches), semantic
feature relationships, and verifier diagnostics.  It is deliberately plain
data so the SSE surface, a CLI, or a static report can consume the same output.
"""

from __future__ import annotations

import hashlib
import html
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional


def _plain(value: Any) -> Any:
    """Convert model values to stable JSON-compatible data."""
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "value"):
        return _plain(value.value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _canonical(value: Any) -> str:
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class ViewNode:
    id: str
    kind: str
    label: str
    data: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "label": self.label, "data": _plain(self.data)}


@dataclass(frozen=True)
class ViewEdge:
    source: str
    target: str
    relation: str

    def to_dict(self) -> dict:
        return {"source": self.source, "target": self.target, "relation": self.relation}


@dataclass(frozen=True)
class GraphView:
    nodes: tuple[ViewNode, ...]
    edges: tuple[ViewEdge, ...]
    diagnostics: tuple[Mapping[str, Any], ...] = ()
    active_branch: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "active_branch": self.active_branch,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "diagnostics": [_plain(d) for d in self.diagnostics],
        }

    def to_json(self, *, indent: Optional[int] = None) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, ensure_ascii=False, separators=None if indent else (",", ":"),
            indent=indent,
        )

    def to_svg(self) -> str:
        """Render a compact static debug diagram without script or raw markup."""
        width, row = 960, 34
        height = max(80, 42 + row * (len(self.nodes) + len(self.diagnostics)))
        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="CAD graph debug view">',
            '<rect width="100%" height="100%" fill="#10151d"/>',
            '<g font-family="monospace" font-size="13" fill="#dce7f3">',
        ]
        for i, node in enumerate(self.nodes):
            y = 28 + i * row
            color = "#58a6ff" if node.kind == "operation" else "#7ee787"
            text = html.escape(f"{node.kind}: {node.label} [{node.id}]", quote=True)
            lines.extend([
                f'<circle cx="18" cy="{y - 4}" r="5" fill="{color}"/>',
                f'<text x="32" y="{y}">{text}</text>',
            ])
        base = 28 + len(self.nodes) * row
        for i, diagnostic in enumerate(self.diagnostics):
            y = base + i * row
            severity = str(diagnostic.get("severity", "info"))
            text = html.escape(
                f"{severity}: {diagnostic.get('code', '')} — {diagnostic.get('message', '')}",
                quote=True,
            )
            lines.append(f'<text x="18" y="{y}" fill="#f2cc60">{text}</text>')
        lines.extend(["</g>", "</svg>"])
        return "".join(lines)


def build_graph_view(opdag=None, feature_graph=None, diagnostics: Iterable[Any] = ()) -> GraphView:
    """Build a deterministic combined history/feature/diagnostic view."""
    nodes: dict[str, ViewNode] = {}
    edges: set[tuple[str, str, str]] = set()

    if opdag is not None:
        branches = opdag.branches() if hasattr(opdag, "branches") else ["main"]
        previous_by_signature: dict[tuple[str, ...], str] = {}
        for branch in sorted(branches):
            ops = opdag.branch_ops(branch) if hasattr(opdag, "branch_ops") else opdag.ops()
            signatures: list[str] = []
            parent = f"branch:{branch}"
            nodes[parent] = ViewNode(parent, "branch", branch, {"head": len(ops)})
            for index, op in enumerate(ops):
                op_data = _plain(op)
                signature = hashlib.sha256(_canonical(op_data).encode("utf-8")).hexdigest()[:12]
                signatures.append(signature)
                prefix = tuple(signatures)
                node_id = previous_by_signature.get(prefix)
                if node_id is None:
                    node_id = f"op:{signature}:{index}"
                    previous_by_signature[prefix] = node_id
                    label = str(op_data.get("op", type(op).__name__)) if isinstance(op_data, dict) else type(op).__name__
                    nodes[node_id] = ViewNode(node_id, "operation", label, {"index": index, "op": op_data})
                edges.add((parent, node_id, "contains" if index == 0 else "next"))
                parent = node_id

    if feature_graph is not None:
        for node in feature_graph.nodes:
            data = _plain(node)
            raw_id = str(data.get("id", ""))
            node_id = f"feature:{raw_id}"
            nodes[node_id] = ViewNode(
                node_id, "feature", str(data.get("type", "feature")), data.get("params", {})
            )
        for edge in feature_graph.edges:
            data = _plain(edge)
            edges.add((
                f"feature:{data['source']}", f"feature:{data['target']}", str(data["relation"])
            ))

    diag_data = tuple(sorted((_plain(d) for d in diagnostics), key=_canonical))
    return GraphView(
        nodes=tuple(sorted(nodes.values(), key=lambda n: (n.kind, n.id))),
        edges=tuple(ViewEdge(*edge) for edge in sorted(edges)),
        diagnostics=diag_data,
        active_branch=getattr(opdag, "current_branch", None),
    )
