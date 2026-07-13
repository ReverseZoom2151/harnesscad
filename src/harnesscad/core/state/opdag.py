"""Ops-DAG — the event-sourced operation history ("git for CAD").

The op stream is append-only and content-hashed: each node's hash chains its
parent hash with the canonical JSON of its op, so an identical op sequence
always produces an identical `head_hash`. This is the substrate for checkpoint,
rollback, bisect, and deterministic replay.

On top of the linear core sits an *additive* branching layer: named branches
each hold their own node list over the shared genesis root, so the same op
sequence on any branch reproduces the same content hashes. `branch`/`checkout`/
`merge` give version-control semantics (named branches, merge requests) — a
3-way `merge` finds the common ancestor from the content-hash history, replays
non-conflicting ops from both sides, and *flags* (never silently clobbers)
edits where both branches touched the same feature/parameter.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from harnesscad.core.cisp.ops import Op, SetParam, canonical_json

_GENESIS = hashlib.sha256(b"harnesscad-genesis-v0").hexdigest()

DEFAULT_BRANCH = "main"


@dataclass
class OpNode:
    index: int
    op: Op
    parent_hash: str
    node_hash: str


@dataclass
class MergeResult:
    """Outcome of a 3-way branch merge.

    ``merged_ops`` is the replayed, non-conflicting op sequence (base ancestor
    ops followed by each side's new, deduplicated ops). ``conflicts`` lists the
    conflicting op pairs — ``{"a": <source op>, "b": <target op>, "reason": str}``
    — left for human/agent review rather than auto-resolved. ``clean`` is True
    iff no conflicts were detected.
    """

    merged_ops: List[Op] = field(default_factory=list)
    conflicts: List[dict] = field(default_factory=list)
    clean: bool = True


class OpDAG:
    def __init__(self) -> None:
        # Branching is additive: the linear core operates on ``self._nodes``,
        # which is always an alias for the current branch's node list, so every
        # pre-existing linear behaviour (append/head_hash/truncate/...) is
        # unchanged when only the default branch is used.
        self._branches: Dict[str, List[OpNode]] = {DEFAULT_BRANCH: []}
        self._current: str = DEFAULT_BRANCH
        self._nodes: List[OpNode] = self._branches[DEFAULT_BRANCH]
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

    # --- branching layer ---------------------------------------------------
    @property
    def current_branch(self) -> str:
        """Name of the branch currently checked out (default ``main``)."""
        return self._current

    def branches(self) -> List[str]:
        """All branch names, sorted for deterministic iteration."""
        return sorted(self._branches)

    def _resolve_at(self, at: Optional[object]) -> int:
        """Turn a branch-point spec into a node count of the current branch.

        ``at`` is ``None`` (current head), an int index, or a checkpoint label.
        """
        if at is None:
            return len(self._nodes)
        if isinstance(at, bool):  # guard: bool is an int subclass
            raise TypeError(f"invalid branch point {at!r}")
        if isinstance(at, int):
            if at < 0 or at > len(self._nodes):
                raise IndexError(f"branch point {at} out of range")
            return at
        return self._checkpoints[at]  # checkpoint label

    def branch(self, name: str, at: Optional[object] = None) -> None:
        """Create branch ``name`` off the current branch at ``at``.

        ``at`` defaults to the current head; it may also be an int index or a
        checkpoint label. The new branch copies the prefix of nodes up to that
        point — a prefix of a content-hash chain is itself a valid chain, so the
        shared history keeps identical hashes and subsequent appends diverge
        deterministically.
        """
        if name in self._branches:
            raise ValueError(f"branch {name!r} already exists")
        count = self._resolve_at(at)
        self._branches[name] = list(self._nodes[:count])

    def checkout(self, name: str) -> None:
        """Switch the current branch; the linear API now operates on it."""
        if name not in self._branches:
            raise KeyError(name)
        self._current = name
        self._nodes = self._branches[name]

    def branch_ops(self, name: str) -> List[Op]:
        """The op list recorded on branch ``name`` (application order)."""
        return [n.op for n in self._branches[name]]

    def _common_ancestor(self, a: List[OpNode], b: List[OpNode]) -> int:
        """Length of the longest shared content-hash prefix of two branches.

        Because branches copy a prefix from their parent, the shared history has
        identical ``node_hash`` values; the first differing hash marks where the
        branches diverged from their common ancestor.
        """
        i = 0
        while i < len(a) and i < len(b) and a[i].node_hash == b[i].node_hash:
            i += 1
        return i

    @staticmethod
    def _write_keys(op: Op) -> set:
        """Feature/parameter identities an op *edits* (its conflict surface).

        A conflict is two branches writing the same key. Purely *additive*
        geometry (new sketches/points/lines/circles) claims nothing and so never
        conflicts; ops that mutate an existing parameter or transform a named
        feature/body/edge do.
        """
        keys: set = set()
        if isinstance(op, SetParam):
            keys.add(("param", int(op.target), str(op.param)))
            return keys
        d = op.to_dict()
        # Ops naming an existing entity they transform (Boolean.target,
        # Mirror.feature_or_body, patterns' feature, Hole.face_or_sketch, ...).
        for f in ("target", "feature", "feature_or_body", "face_or_sketch"):
            v = d.get(f)
            if isinstance(v, str) and v:
                keys.add(("entity", v))
        # Ops naming existing edges/faces they modify (fillet/chamfer/shell/...).
        for f in ("edges", "faces"):
            v = d.get(f)
            if isinstance(v, (list, tuple)):
                for e in v:
                    keys.add(("entity", e))
        return keys

    def merge(self, source: str, target: Optional[str] = None) -> MergeResult:
        """3-way merge ``source`` into ``target`` (default: current branch).

        The common ancestor is derived from the shared content-hash prefix. Ops
        each side added after the ancestor are replayed; where both sides wrote
        the same feature/parameter key with differing ops, the pair is reported
        as a conflict (not auto-resolved). Non-mutating: it returns a
        :class:`MergeResult`; the caller decides whether to commit ``merged_ops``.
        """
        if source not in self._branches:
            raise KeyError(source)
        target = self._current if target is None else target
        if target not in self._branches:
            raise KeyError(target)

        src_nodes = self._branches[source]
        tgt_nodes = self._branches[target]
        base = self._common_ancestor(src_nodes, tgt_nodes)
        base_ops = [n.op for n in tgt_nodes[:base]]
        src_new = [n.op for n in src_nodes[base:]]
        tgt_new = [n.op for n in tgt_nodes[base:]]

        conflicts: List[dict] = []
        for a in src_new:
            ka = self._write_keys(a)
            if not ka:
                continue
            for b in tgt_new:
                if a == b:
                    continue  # convergent identical edit — not a conflict
                shared = ka & self._write_keys(b)
                if shared:
                    conflicts.append(
                        {"a": a, "b": b, "reason": self._conflict_reason(shared)}
                    )

        # Replay non-conflicting ops: the target's new ops, then source-only new
        # ops (dedup identical convergent edits), on top of the shared ancestor.
        conflicting = {id(c["a"]) for c in conflicts}
        merged = list(base_ops) + list(tgt_new)
        for a in src_new:
            if a in tgt_new or id(a) in conflicting:
                continue
            merged.append(a)

        return MergeResult(merged_ops=merged, conflicts=conflicts,
                           clean=not conflicts)

    @staticmethod
    def _conflict_reason(shared: set) -> str:
        parts = []
        for key in sorted(shared, key=repr):
            if key[0] == "param":
                parts.append(f"both edited param '{key[2]}' of op #{key[1]}")
            else:
                parts.append(f"both modified feature '{key[1]}'")
        return "; ".join(parts)
