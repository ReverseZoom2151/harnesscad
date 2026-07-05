"""Ops-DAG — the event-sourced operation history ("git for CAD").

The op stream is append-only and content-hashed: each node's hash chains its
parent hash with the canonical JSON of its op, so an identical op sequence
always produces an identical `head_hash`. This is the substrate for checkpoint,
rollback, bisect, and deterministic replay. v0 keeps a linear history; branching
(true DAG) is a later addition.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from cisp.ops import Op, canonical_json

_GENESIS = hashlib.sha256(b"harnesscad-genesis-v0").hexdigest()


@dataclass
class OpNode:
    index: int
    op: Op
    parent_hash: str
    node_hash: str


class OpDAG:
    def __init__(self) -> None:
        self._nodes: List[OpNode] = []
        self._checkpoints: Dict[str, int] = {}

    @property
    def head_hash(self) -> str:
        return self._nodes[-1].node_hash if self._nodes else _GENESIS

    def append(self, op: Op) -> OpNode:
        parent = self.head_hash
        h = hashlib.sha256((parent + "|" + canonical_json(op)).encode()).hexdigest()
        node = OpNode(len(self._nodes), op, parent, h)
        self._nodes.append(node)
        return node

    def ops(self) -> List[Op]:
        return [n.op for n in self._nodes]

    def __len__(self) -> int:
        return len(self._nodes)

    def checkpoint(self, label: str) -> None:
        self._checkpoints[label] = len(self._nodes)

    def index_of(self, label: str) -> int:
        return self._checkpoints[label]

    def truncate(self, count: int) -> None:
        """Keep the first `count` nodes; drop the rest, plus stale checkpoints."""
        del self._nodes[count:]
        self._checkpoints = {k: v for k, v in self._checkpoints.items() if v <= count}

    def rollback(self, label: str) -> None:
        self.truncate(self._checkpoints[label])

    def bisect(self, predicate: Callable[[List[Op]], bool]) -> Optional[int]:
        """Binary-search the op history for the first op that flips ``predicate``.

        ``predicate(ops_prefix)`` receives the op list *up to and including* an
        index and returns True while the history is still "good" and False once
        it has gone "bad". Assuming the prefixes are monotone (good then bad),
        this returns the index of the **first bad** op — the earliest op after
        whose inclusion the predicate first returns False — or ``None`` when the
        predicate never flips (every prefix is good, or the history is empty).

        Pure and deterministic: it only reads the recorded op list (no replay,
        no mutation) and evaluates ``predicate`` O(log n) times. Mirrors
        ``git bisect``: predicate True == "good", the returned index == first
        "bad" commit.
        """
        ops = self.ops()
        n = len(ops)
        # `lo` is always a known-good boundary (prefixes < lo are good), `hi` a
        # known/assumed-bad boundary. Prefixes are ops[: i + 1].
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if predicate(ops[: mid + 1]):
                lo = mid + 1     # this prefix is good -> first bad is later
            else:
                hi = mid         # this prefix is bad -> first bad is at/earlier
        return lo if lo < n else None
