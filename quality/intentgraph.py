"""Typed causal, spatial and functional relations over CAD intent nodes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Mapping, Optional, Tuple


class RelationKind(str, Enum):
    CAUSAL = "causal"
    SPATIAL = "spatial"
    FUNCTIONAL = "functional"
    REFERENCE = "reference"


@dataclass(frozen=True)
class IntentNode:
    id: str
    intent: str
    feature_id: Optional[str] = None
    attributes: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.intent:
            raise ValueError("intent nodes require id and intent")


@dataclass(frozen=True)
class IntentRelation:
    source: str
    target: str
    kind: RelationKind
    label: str = ""


class IntentGraph:
    def __init__(self, nodes: Iterable[IntentNode] = (),
                 relations: Iterable[IntentRelation] = ()) -> None:
        self.nodes: Dict[str, IntentNode] = {}
        self.relations: List[IntentRelation] = []
        for node in nodes:
            self.add_node(node)
        for relation in relations:
            self.add_relation(relation)

    def add_node(self, node: IntentNode) -> None:
        if node.id in self.nodes:
            raise ValueError(f"duplicate intent node {node.id!r}")
        self.nodes[node.id] = node

    def add_relation(self, relation: IntentRelation) -> None:
        if relation.source not in self.nodes or relation.target not in self.nodes:
            raise ValueError("relation endpoints must exist")
        if relation.source == relation.target:
            raise ValueError("self-relations are not allowed")
        if relation not in self.relations:
            self.relations.append(relation)

    def adjacent(self, node_id: str,
                 kind: Optional[RelationKind] = None) -> Tuple[IntentRelation, ...]:
        return tuple(sorted(
            (
                relation for relation in self.relations
                if (relation.source == node_id or relation.target == node_id)
                and (kind is None or relation.kind is kind)
            ),
            key=lambda relation: (
                relation.kind.value, relation.source, relation.target, relation.label
            ),
        ))

    def causal_order(self) -> Tuple[str, ...]:
        """Return stable causal order; reject cyclic design dependencies."""
        incoming = {node_id: 0 for node_id in self.nodes}
        outgoing = {node_id: [] for node_id in self.nodes}
        for relation in self.relations:
            if relation.kind is RelationKind.CAUSAL:
                incoming[relation.target] += 1
                outgoing[relation.source].append(relation.target)
        ready = sorted(node_id for node_id, count in incoming.items() if count == 0)
        result = []
        while ready:
            node_id = ready.pop(0)
            result.append(node_id)
            for target in sorted(outgoing[node_id]):
                incoming[target] -= 1
                if incoming[target] == 0:
                    ready.append(target)
                    ready.sort()
        if len(result) != len(self.nodes):
            raise ValueError("causal intent relations contain a cycle")
        return tuple(result)

    def to_dict(self) -> dict:
        return {
            "nodes": [
                {
                    "id": node.id,
                    "intent": node.intent,
                    "feature_id": node.feature_id,
                    "attributes": dict(sorted(node.attributes.items())),
                }
                for node in sorted(self.nodes.values(), key=lambda item: item.id)
            ],
            "relations": [
                {
                    "source": relation.source,
                    "target": relation.target,
                    "kind": relation.kind.value,
                    "label": relation.label,
                }
                for relation in sorted(
                    self.relations,
                    key=lambda item: (
                        item.kind.value, item.source, item.target, item.label
                    ),
                )
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
