"""Conversation / message branch tree (deterministic, stdlib-only).

Ported from ``shared/Tree.ts`` in CADAM (Adam-CAD's open-source text-to-CAD web
app). CADAM stores chat turns as flat rows carrying a ``parent_message_id`` and
reconstructs the branching conversation tree client-side: a message may have
several children (regenerations / edits create sibling branches), and the UI
walks from a leaf back to the root to render the active thread.

The harness has a multi-agent idea2cad workflow and a blackboard, but no
generic parent-pointer conversation tree with cycle-safe root-path walking, so
this supplies it. Useful anywhere turns/branches form a DAG-of-parents forest
(agent dialogue trees, edit histories, regeneration branches).

Design notes carried over from the original:
  * Two-pass build: create every node first, then wire parent/child links, so
    forward references (a child listed before its parent) resolve.
  * A row whose ``parent_message_id`` points to a missing id becomes a root
    (orphans are not dropped).
  * ``path_to_root`` guards against cycles in the (untrusted) input with a
    visited set, so a self-referential or looped row truncates instead of
    spinning forever.

Deterministic: insertion order is preserved; no clock, no randomness.
"""

from __future__ import annotations

from typing import Dict, Generic, Hashable, Iterable, List, Optional, TypeVar

T = TypeVar("T")


class MessageNode(Generic[T]):
    """A node wrapping a payload with parent/child links."""

    __slots__ = ("id", "parent_id", "payload", "children", "parent", "_roots")

    def __init__(self, node_id: Hashable, parent_id: Optional[Hashable], payload: T):
        self.id = node_id
        self.parent_id = parent_id
        self.payload = payload
        self.children: List["MessageNode[T]"] = []
        self.parent: Optional["MessageNode[T]"] = None
        # Injected by the owning tree so a root node can report its siblings.
        self._roots: Optional[List["MessageNode[T]"]] = None

    @property
    def siblings(self) -> List["MessageNode[T]"]:
        """Nodes sharing this node's parent (or the root list if a root)."""
        if self.parent is not None:
            return self.parent.children
        return self._roots if self._roots is not None else []

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"MessageNode(id={self.id!r}, children={len(self.children)})"


class MessageTree(Generic[T]):
    """Build a parent-pointer forest from flat ``(id, parent_id, payload)`` rows.

    ``elements`` is any iterable of objects exposing ``id`` and
    ``parent_message_id`` attributes, or ``(id, parent_id, payload)`` tuples.
    """

    def __init__(self, elements: Iterable, key: str = "id",
                 parent_key: str = "parent_message_id"):
        self._nodes: "Dict[Hashable, MessageNode[T]]" = {}
        self._roots: List["MessageNode[T]"] = []
        self._key = key
        self._parent_key = parent_key

        rows = [self._normalize(e) for e in elements]

        # Pass 1: create all nodes (so forward references resolve).
        for node_id, parent_id, payload in rows:
            node = MessageNode(node_id, parent_id, payload)
            node._roots = self._roots
            self._nodes[node_id] = node

        # Pass 2: wire parent/child relationships.
        for node_id, parent_id, _payload in rows:
            node = self._nodes[node_id]
            parent = self._nodes.get(parent_id) if parent_id is not None else None
            if parent is not None:
                parent.children.append(node)
                node.parent = parent
            else:
                self._roots.append(node)

    def _normalize(self, element):
        if isinstance(element, tuple):
            if len(element) == 3:
                return element
            if len(element) == 2:
                return element[0], element[1], None
            raise ValueError("tuple rows must be (id, parent_id[, payload])")
        node_id = getattr(element, self._key)
        parent_id = getattr(element, self._parent_key)
        return node_id, parent_id, element

    @property
    def roots(self) -> List["MessageNode[T]"]:
        return self._roots

    @property
    def nodes(self) -> "Dict[Hashable, MessageNode[T]]":
        return self._nodes

    def get(self, node_id: Hashable) -> Optional["MessageNode[T]"]:
        return self._nodes.get(node_id)

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: Hashable) -> bool:
        return node_id in self._nodes

    def path_to_root(self, node_id: Hashable) -> List["MessageNode[T]"]:
        """Root -> node path. Cycle-safe: a looped chain truncates cleanly.

        Returns an empty list if ``node_id`` is unknown.
        """
        path: List["MessageNode[T]"] = []
        visited: set = set()
        current = self._nodes.get(node_id)
        while current is not None:
            if current.id in visited:
                break  # cycle: stop rather than loop forever
            visited.add(current.id)
            path.insert(0, current)
            current = current.parent
        return path

    def depth(self, node_id: Hashable) -> int:
        """Number of edges from ``node_id`` up to its root (root == 0).

        ``-1`` if the node is unknown.
        """
        if node_id not in self._nodes:
            return -1
        return len(self.path_to_root(node_id)) - 1

    def leaves(self) -> List["MessageNode[T]"]:
        """All nodes with no children, in insertion order."""
        return [n for n in self._nodes.values() if not n.children]
