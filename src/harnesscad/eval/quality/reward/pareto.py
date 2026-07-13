"""Multi-objective Pareto trade-off surfacer for the exploration layer.

:mod:`quality.fitness` collapses several competing metrics (mass, cost,
constraint violations, ...) into a *single* scalar score by weighting them, which
is exactly what a tournament / best-of-N loop wants when it must pick one winner.
But a designer often wants the opposite: not a pre-committed weighting, but the
*set of designs that embody a genuine trade-off* — the ones where you cannot do
better on one objective without doing worse on another. That set is the **Pareto
front**, and surfacing it (plus the fronts behind it, and a readable trade-off
table) is what this module adds on top of the metric vectors
:class:`quality.fitness.Objective` already produces.

The dominance test is the same partial order :func:`quality.fitness.dominates`
defines (minimise on every axis; ``a`` dominates ``b`` iff it is no worse
everywhere and strictly better somewhere) — reused verbatim so the ranking here
and the Pareto tuple there never disagree.

Three entry points:
  * :func:`pareto_front`  — the non-dominated set of ``items``.
  * :func:`pareto_rank`   — non-dominated *sorting*: items bucketed into
    successive fronts (front 0 = Pareto-optimal, front 1 = optimal once front 0
    is removed, ...).
  * :func:`trade_off_matrix` — a table of each surfaced item's raw objective
    values, with a text ``render``.

Items are flexible: a :class:`exploration.tournament.Variant`, a plain ``dict``,
any object, or an ``(item, metric_vector)`` pair. An :class:`Objective` says, per
metric, which direction is better (``min`` / ``max``) and how to read the value
off an item. Everything is deterministic: dominance never depends on order, and
tie-breaks fall back to a stable key, so the same input always yields the same
front in the same order (input order preserved).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from harnesscad.eval.quality.reward.fitness import dominates as _dominates

Number = Union[int, float]
Key = Union[str, int, Callable[[Any], Number], None]


# --------------------------------------------------------------------------- #
# Objective specification
# --------------------------------------------------------------------------- #
@dataclass
class Objective:
    """One optimisation axis: a name, a direction, and how to read the value.

    * ``goal``  — ``"min"`` (smaller is better) or ``"max"`` (larger is better).
    * ``key``   — how to extract this metric's raw value from an item:
        - ``None``     : positional — use this objective's index into the item's
          metric vector (item must be an ``(item, vector)`` pair or a bare vector).
        - ``str``      : mapping/attribute lookup (``item[key]`` then
          ``getattr(item, key)``).
        - ``int``      : index into the item's metric vector (or the item itself
          if it is a sequence).
        - ``callable`` : ``key(item)`` returns the value.
    """

    name: str
    goal: str = "min"
    key: Key = None

    def __post_init__(self) -> None:
        if self.goal not in ("min", "max"):
            raise ValueError(f"unknown goal {self.goal!r} (use 'min'/'max')")


def _split(item: Any) -> Tuple[Any, Optional[Sequence[Number]]]:
    """Split an item into (payload, metric_vector-or-None).

    An ``(payload, vector)`` pair is recognised when ``item`` is a 2-tuple/2-list
    whose second element is a sequence of numbers. A bare numeric sequence is
    treated as its own vector (payload is the sequence itself)."""
    if isinstance(item, tuple) and len(item) == 2 and _is_number_seq(item[1]):
        return item[0], list(item[1])
    if _is_number_seq(item):
        return item, list(item)
    return item, None


def _is_number_seq(v: Any) -> bool:
    if isinstance(v, (str, bytes, dict)):
        return False
    if not isinstance(v, (list, tuple)):
        return False
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)


def _raw_value(item: Any, obj: Objective, index: int) -> Number:
    """Read objective ``obj``'s raw (un-oriented) value off ``item``."""
    payload, vector = _split(item)
    key = obj.key
    if callable(key):
        return float(key(payload))
    if isinstance(key, int) and not isinstance(key, bool):
        src = vector if vector is not None else payload
        return float(src[key])
    if isinstance(key, str):
        if isinstance(payload, dict):
            return float(payload[key])
        return float(getattr(payload, key))
    # key is None -> positional into the metric vector.
    if vector is None:
        raise ValueError(
            f"objective {obj.name!r} has no key and item {payload!r} carries no "
            "metric vector; pass items as (item, vector) or give the objective a key")
    return float(vector[index])


def _vector(item: Any, objectives: Sequence[Objective]) -> Tuple[float, ...]:
    """Minimise-oriented metric tuple for an item (max axes negated)."""
    out: List[float] = []
    for i, obj in enumerate(objectives):
        v = _raw_value(item, obj, i)
        out.append(v if obj.goal == "min" else -v)
    return tuple(out)


def raw_values(item: Any, objectives: Sequence[Objective]) -> Tuple[float, ...]:
    """The item's raw (un-oriented) objective values, in objective order."""
    return tuple(_raw_value(item, obj, i) for i, obj in enumerate(objectives))


# --------------------------------------------------------------------------- #
# Pareto front + non-dominated sorting
# --------------------------------------------------------------------------- #
def pareto_front(items: Sequence[Any],
                 objectives: Sequence[Objective]) -> List[Any]:
    """Return the non-dominated (Pareto-optimal) subset of ``items``.

    An item survives iff no *other* item dominates it (:func:`fitness.dominates`,
    minimise on every oriented axis). Input order is preserved among survivors,
    so the result is deterministic. Duplicated metric vectors are mutually
    non-dominating, so all copies are kept."""
    if not objectives:
        raise ValueError("at least one objective is required")
    vecs = [_vector(it, objectives) for it in items]
    front: List[Any] = []
    for i, it in enumerate(items):
        if not any(_dominates(vecs[j], vecs[i]) for j in range(len(items)) if j != i):
            front.append(it)
    return front


def pareto_rank(items: Sequence[Any],
                objectives: Sequence[Objective]) -> List[List[Any]]:
    """Non-dominated sorting: partition ``items`` into successive Pareto fronts.

    Front 0 is the Pareto-optimal set; front 1 is Pareto-optimal once front 0 is
    removed; and so on. Deterministic (input order preserved within each front).
    Returns a list of fronts; an empty ``items`` yields ``[]``."""
    if not objectives:
        raise ValueError("at least one objective is required")
    remaining = list(range(len(items)))
    vecs = [_vector(it, objectives) for it in items]
    fronts: List[List[Any]] = []
    while remaining:
        current: List[int] = []
        for i in remaining:
            if not any(_dominates(vecs[j], vecs[i])
                       for j in remaining if j != i):
                current.append(i)
        if not current:  # safety: cycles are impossible, but never loop forever.
            current = list(remaining)
        fronts.append([items[i] for i in current])
        current_set = set(current)
        remaining = [i for i in remaining if i not in current_set]
    return fronts


# --------------------------------------------------------------------------- #
# Trade-off matrix
# --------------------------------------------------------------------------- #
@dataclass
class TradeOffMatrix:
    """A table of surfaced items' raw objective values.

    ``labels`` names the rows; ``columns`` are the objective names; ``rows`` are
    parallel lists of raw (un-oriented) values. ``goals`` records each column's
    direction so :meth:`render` can annotate it."""

    labels: List[str] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    rows: List[List[float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "columns": list(self.columns),
            "goals": list(self.goals),
            "labels": list(self.labels),
            "rows": [list(r) for r in self.rows],
        }

    def render(self) -> str:
        """A fixed-width text table of the trade-off front."""
        if not self.columns:
            return "trade-off: (no objectives)"
        headers = ["item"] + [f"{c} ({g})"
                              for c, g in zip(self.columns, self.goals)]
        body = [[lbl] + [_fmt(v) for v in row]
                for lbl, row in zip(self.labels, self.rows)]
        widths = [len(h) for h in headers]
        for r in body:
            for i, cell in enumerate(r):
                widths[i] = max(widths[i], len(cell))
        def fmt_row(cells: List[str]) -> str:
            return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))
        lines = [fmt_row(headers), fmt_row(["-" * w for w in widths])]
        lines += [fmt_row(r) for r in body]
        return "\n".join(lines)


def trade_off_matrix(items: Sequence[Any],
                     objectives: Sequence[Objective],
                     labels: Optional[Sequence[str]] = None) -> TradeOffMatrix:
    """Build a :class:`TradeOffMatrix` of each item's raw objective values.

    ``items`` is typically a :func:`pareto_front` result. ``labels`` names the
    rows; when omitted, each item's ``id``/``name`` (attr or dict key) is used,
    falling back to ``item{index}``. Deterministic."""
    cols = [o.name for o in objectives]
    goals = [o.goal for o in objectives]
    lbls: List[str] = []
    rows: List[List[float]] = []
    for idx, it in enumerate(items):
        lbls.append(str(labels[idx]) if labels is not None and idx < len(labels)
                    else _label_of(it, idx))
        rows.append([_raw_value(it, o, i) for i, o in enumerate(objectives)])
    return TradeOffMatrix(labels=lbls, columns=cols, goals=goals, rows=rows)


def _label_of(item: Any, index: int) -> str:
    payload, _ = _split(item)
    for attr in ("id", "name"):
        val = getattr(payload, attr, None)
        if val:
            return str(val)
        if isinstance(payload, dict) and payload.get(attr):
            return str(payload[attr])
    return f"item{index}"


def _fmt(v: float) -> str:
    if v == int(v):
        return str(int(v))
    return f"{v:.6g}"
