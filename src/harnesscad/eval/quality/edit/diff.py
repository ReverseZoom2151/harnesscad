"""Semantic diff layer — "git diff" for the ops-DAG.

Two complementary diffs:

  * :func:`op_diff` — an LCS/sequence diff over two op streams. It classifies
    every change as added / removed / modified. A *modified* op is one that
    keeps its tag and position but changes parameters (e.g. a hole whose
    diameter went 6 -> 5.4); the diff records a field-level delta for it. The
    result is a JSON-serialisable :class:`OpDiff` with a human-readable
    ``render()`` ("+1 fillet, hole Ø6->5.4"). Pure stdlib (``difflib``).

  * :func:`geom_diff` — an optional OCCT face-level diff. It boolean-cuts the
    two solids both ways to measure added / removed material volume and the
    change in face count. It is guarded on cadquery/OCCT and on both sides
    having a solid; when either is unavailable it degrades to a metrics delta
    from ``query('metrics')`` / ``query('measure')`` and never crashes.

  * :func:`diff_checkpoints` — a convenience that diffs two labelled
    checkpoints of an :class:`state.opdag.OpDAG`.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from harnesscad.core.cisp.ops import Op


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_ops(ops) -> List[Op]:
    """Accept an OpDAG (anything exposing ``ops()``) or a plain op list."""
    if hasattr(ops, "ops") and callable(ops.ops):
        return list(ops.ops())
    return list(ops)


def _params(op: Op) -> Dict[str, Any]:
    """The op's fields (its ``to_dict`` without the ``op`` tag)."""
    d = dict(op.to_dict())
    d.pop("op", None)
    return d


def _key(op: Op) -> str:
    """A hashable identity for sequence matching (tag + sorted params)."""
    import json

    return json.dumps(op.to_dict(), sort_keys=True, separators=(",", ":"))


def _fmt(v: Any) -> str:
    """Compact human formatting: 6.0 -> '6', 5.4 -> '5.4'."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return "{:g}".format(v)
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_fmt(x) for x in v) + "]"
    return str(v)


def _field_phrase(tag: str, name: str, before: Any, after: Any) -> str:
    b, a = _fmt(before), _fmt(after)
    if name == "diameter":
        return f"{tag} Ø{b}->{a}"
    if name == "radius":
        return f"{tag} r{b}->{a}"
    return f"{tag} {name} {b}->{a}"


def _param_delta(op_a: Op, op_b: Op) -> List[Dict[str, Any]]:
    """Field-level delta between two same-tag ops."""
    pa, pb = _params(op_a), _params(op_b)
    delta: List[Dict[str, Any]] = []
    for name in sorted(set(pa) | set(pb)):
        before, after = pa.get(name), pb.get(name)
        if before != after:
            delta.append({"field": name, "before": before, "after": after})
    return delta


# ---------------------------------------------------------------------------
# op-level diff
# ---------------------------------------------------------------------------
@dataclass
class OpDiff:
    """A structured, JSON-serialisable diff of two op streams."""

    added: List[Dict[str, Any]] = field(default_factory=list)
    removed: List[Dict[str, Any]] = field(default_factory=list)
    modified: List[Dict[str, Any]] = field(default_factory=list)
    unchanged_count: int = 0

    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.modified)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "added": self.added,
            "removed": self.removed,
            "modified": self.modified,
            "unchanged_count": self.unchanged_count,
        }

    def render(self) -> str:
        """A one-line human summary, e.g. '+1 fillet, hole Ø6->5.4'."""
        tokens: List[str] = []

        # additions, grouped by tag: '+1 fillet'
        tokens += _grouped_counts(self.added, "+")
        # removals, grouped by tag: '-1 fillet'
        tokens += _grouped_counts(self.removed, "-")
        # modifications, per changed field: 'hole Ø6->5.4'
        for m in self.modified:
            for d in m["params"]:
                tokens.append(
                    _field_phrase(m["tag"], d["field"], d["before"], d["after"]))

        return ", ".join(tokens) if tokens else "no changes"


def _grouped_counts(entries: List[Dict[str, Any]], sign: str) -> List[str]:
    counts: Dict[str, int] = {}
    order: List[str] = []
    for e in entries:
        tag = e["tag"]
        if tag not in counts:
            order.append(tag)
        counts[tag] = counts.get(tag, 0) + 1
    return [f"{sign}{counts[t]} {t}" for t in order]


def _added_entry(index: int, op: Op) -> Dict[str, Any]:
    return {"index": index, "tag": op.OP, "op": op.to_dict()}


def _removed_entry(index: int, op: Op) -> Dict[str, Any]:
    return {"index": index, "tag": op.OP, "op": op.to_dict()}


def _modified_entry(index: int, op_a: Op, op_b: Op,
                    delta: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "index": index,
        "tag": op_a.OP,
        "before": op_a.to_dict(),
        "after": op_b.to_dict(),
        "params": delta,
    }


def op_diff(ops_a, ops_b) -> OpDiff:
    """LCS/sequence diff between two op streams (OpDAGs or op lists).

    Uses :class:`difflib.SequenceMatcher` (an LCS-style matcher) over a
    canonical key of each op. Within a replaced region, ops are paired by
    position: a same-tag pair is a *modified* op (with its field delta), an
    unmatched left op is *removed*, an unmatched right op is *added*.
    """
    a = _as_ops(ops_a)
    b = _as_ops(ops_b)
    keys_a = [_key(op) for op in a]
    keys_b = [_key(op) for op in b]

    diff = OpDiff()
    sm = difflib.SequenceMatcher(a=keys_a, b=keys_b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            diff.unchanged_count += (i2 - i1)
        elif tag == "delete":
            for i in range(i1, i2):
                diff.removed.append(_removed_entry(i, a[i]))
        elif tag == "insert":
            for j in range(j1, j2):
                diff.added.append(_added_entry(j, b[j]))
        elif tag == "replace":
            _diff_replace(diff, a, b, i1, i2, j1, j2)
    return diff


def _diff_replace(diff: OpDiff, a: List[Op], b: List[Op],
                  i1: int, i2: int, j1: int, j2: int) -> None:
    """Classify a replaced region by pairing same-tag ops as modifications.

    Left (a) and right (b) ops in the region are matched greedily by tag, in
    order: the k-th ``fillet`` on the left pairs with the k-th ``fillet`` on
    the right as a *modified* op (with its field delta). Unmatched left ops are
    *removed*; unmatched right ops are *added*. This keeps a same-tag param edit
    (hole 6 -> 5.4) legible even when adds/removes of other ops sit alongside.
    """
    from collections import defaultdict, deque

    right_by_tag: Dict[str, "deque"] = defaultdict(deque)
    for j in range(j1, j2):
        right_by_tag[b[j].OP].append(j)

    matched: set = set()
    for i in range(i1, i2):
        oa = a[i]
        bucket = right_by_tag.get(oa.OP)
        if bucket:
            j = bucket.popleft()
            matched.add(j)
            diff.modified.append(_modified_entry(j, oa, b[j], _param_delta(oa, b[j])))
        else:
            diff.removed.append(_removed_entry(i, oa))

    for j in range(j1, j2):
        if j not in matched:
            diff.added.append(_added_entry(j, b[j]))


def diff_checkpoints(opdag, label_a: str, label_b: str) -> OpDiff:
    """Diff the op streams at two labelled checkpoints of an OpDAG."""
    ops = opdag.ops()
    ia = opdag.index_of(label_a)
    ib = opdag.index_of(label_b)
    return op_diff(ops[:ia], ops[:ib])


# ---------------------------------------------------------------------------
# geometric (face-level) diff
# ---------------------------------------------------------------------------
@dataclass
class GeomDiff:
    """A geometric diff between two backends' solids.

    ``mode`` is 'boolean' when a real OCCT face-level diff ran, or 'metrics'
    when it degraded to a metrics/measure delta (OCCT unavailable, or one side
    has no solid). ``reason`` explains a degrade. All numeric fields are plain
    floats/ints so the object is JSON-serialisable.
    """

    mode: str = "metrics"
    available: bool = False
    reason: Optional[str] = None
    volume_a: float = 0.0
    volume_b: float = 0.0
    volume_delta: float = 0.0
    added_volume: Optional[float] = None
    removed_volume: Optional[float] = None
    face_count_a: Optional[int] = None
    face_count_b: Optional[int] = None
    face_count_delta: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "available": self.available,
            "reason": self.reason,
            "volume_a": self.volume_a,
            "volume_b": self.volume_b,
            "volume_delta": self.volume_delta,
            "added_volume": self.added_volume,
            "removed_volume": self.removed_volume,
            "face_count_a": self.face_count_a,
            "face_count_b": self.face_count_b,
            "face_count_delta": self.face_count_delta,
        }

    def render(self) -> str:
        parts = [f"ΔV {_fmt(self.volume_delta)} ({_fmt(self.volume_a)}"
                 f"->{_fmt(self.volume_b)})"]
        if self.mode == "boolean":
            if self.added_volume is not None:
                parts.append(f"+{_fmt(self.added_volume)} material")
            if self.removed_volume is not None:
                parts.append(f"-{_fmt(self.removed_volume)} material")
            if self.face_count_delta is not None:
                parts.append(
                    f"faces {self.face_count_a}->{self.face_count_b}")
        else:
            parts.append(f"[metrics: {self.reason or 'degraded'}]")
        return ", ".join(parts)


def _cadquery_available() -> bool:
    try:
        import cadquery  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _safe_query(backend, q: str) -> Dict[str, Any]:
    query = getattr(backend, "query", None)
    if not callable(query):
        return {}
    try:
        out = query(q)
        return out if isinstance(out, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _volume_of(backend) -> float:
    """Best-effort scalar volume from metrics/measure (0.0 if none)."""
    m = _safe_query(backend, "metrics")
    if isinstance(m.get("volume"), (int, float)):
        return float(m["volume"])
    meas = _safe_query(backend, "measure")
    if isinstance(meas.get("volume"), (int, float)):
        return float(meas["volume"])
    return 0.0


def _combined_shape(backend):
    """The backend's combined OCCT shape, or None (never raises)."""
    fn = getattr(backend, "_combined", None)
    if not callable(fn):
        return None
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return None


def _metrics_geom_diff(backend_a, backend_b, reason: str) -> GeomDiff:
    va, vb = _volume_of(backend_a), _volume_of(backend_b)
    return GeomDiff(
        mode="metrics",
        available=False,
        reason=reason,
        volume_a=va,
        volume_b=vb,
        volume_delta=vb - va,
    )


def geom_diff(backend_a, backend_b) -> GeomDiff:
    """Face-level geometric diff of two backends; degrades safely.

    When cadquery/OCCT is importable and both backends expose a solid, this
    boolean-cuts both ways: ``B cut A`` is added material, ``A cut B`` is
    removed material, and face counts are compared directly. Otherwise it
    degrades to a volume delta from the backends' metrics queries.
    """
    if not _cadquery_available():
        return _metrics_geom_diff(backend_a, backend_b, "cadquery/OCCT unavailable")

    shape_a = _combined_shape(backend_a)
    shape_b = _combined_shape(backend_b)
    if shape_a is None or shape_b is None:
        return _metrics_geom_diff(backend_a, backend_b, "one side has no solid")

    try:
        import cadquery as cq

        wp_a = cq.Workplane("XY").add(shape_a)
        wp_b = cq.Workplane("XY").add(shape_b)

        vol_a = float(shape_a.Volume())
        vol_b = float(shape_b.Volume())

        added = wp_b.cut(cq.Workplane("XY").add(shape_a))
        removed = wp_a.cut(cq.Workplane("XY").add(shape_b))
        added_volume = _wp_volume(added)
        removed_volume = _wp_volume(removed)

        faces_a = len(shape_a.Faces())
        faces_b = len(shape_b.Faces())
    except Exception as exc:  # noqa: BLE001 - never crash; degrade instead
        return _metrics_geom_diff(
            backend_a, backend_b, f"boolean diff failed: {exc}")

    return GeomDiff(
        mode="boolean",
        available=True,
        reason=None,
        volume_a=vol_a,
        volume_b=vol_b,
        volume_delta=vol_b - vol_a,
        added_volume=added_volume,
        removed_volume=removed_volume,
        face_count_a=faces_a,
        face_count_b=faces_b,
        face_count_delta=faces_b - faces_a,
    )


def _wp_volume(wp) -> float:
    """Volume of a cq Workplane result, 0.0 when the cut is empty."""
    try:
        solids = wp.solids().vals()
        if not solids:
            return 0.0
        return float(sum(s.Volume() for s in solids))
    except Exception:  # noqa: BLE001
        return 0.0
