"""Text2CAD CAD-sequence evaluation: primitive/extrusion F1 via loop matching.

Text2CAD evaluates a generated CAD sequence against the ground-truth sequence by
the protocol of CAD-SIGNet (Khan et al. [19], reused in this paper, Sec. 5.1 A):

  * predicted loops are aligned to ground-truth loops *within the same sketch* using
    the **Hungarian matching algorithm** (Kuhn, 1955);
  * once loops are matched, **F1 scores** are computed for each primitive type
    (line, arc, circle) and for extrusions;
  * the **Invalidity Ratio** (IR) is the fraction of predicted sequences that fail
    to build a model.

This module provides the deterministic, geometry-free core of that protocol: a
compact Hungarian assignment over a loop-to-loop cost matrix (cost = primitive
multiset disagreement), aggregation of true/false positives/negatives per primitive
type across the matched loops, and F1 computation. The learned generator, the OCC
build (which decides validity) and Chamfer distance over meshes are out of scope;
IR here is computed from an explicit per-sample validity flag.

Pure stdlib, deterministic (ties in matching break to the lowest index).
"""

from __future__ import annotations

from dataclasses import dataclass

from harnesscad.domain.reconstruction.deepcad_command_spec import ARC, CIRCLE, EXT, LINE, SOL, Command

_PRIMITIVE_TYPES = (LINE, ARC, CIRCLE)


class SequenceF1Error(ValueError):
    """Raised for malformed evaluation inputs."""


# --- loop extraction --------------------------------------------------------
def loop_primitive_counts(commands: list[Command]) -> tuple[dict[str, int], ...]:
    """Split a command sequence into loops, each a {primitive_type: count} map.

    A new loop opens on ``SOL`` (or on the first primitive if none preceded it);
    ``EXT`` closes the current loop. Empty loops are dropped.
    """
    loops: list[dict[str, int]] = []
    current: dict[str, int] | None = None

    def _open() -> dict[str, int]:
        return {LINE: 0, ARC: 0, CIRCLE: 0}

    for cmd in commands:
        if cmd.type == SOL:
            if current is not None and sum(current.values()) > 0:
                loops.append(current)
            current = _open()
        elif cmd.type in _PRIMITIVE_TYPES:
            if current is None:
                current = _open()
            current[cmd.type] += 1
        elif cmd.type == EXT:
            if current is not None and sum(current.values()) > 0:
                loops.append(current)
            current = None
    if current is not None and sum(current.values()) > 0:
        loops.append(current)
    return tuple(loops)


def _loop_cost(a: dict[str, int], b: dict[str, int]) -> int:
    """Assignment cost between two loops: total primitive-multiset disagreement."""
    return sum(abs(a.get(t, 0) - b.get(t, 0)) for t in _PRIMITIVE_TYPES)


# --- Hungarian assignment (square, minimisation) ----------------------------
def hungarian_assignment(cost: list[list[int]]) -> list[int]:
    """Optimal min-cost assignment for a square cost matrix (Kuhn-Munkres).

    Returns ``assign`` where row ``i`` is matched to column ``assign[i]``. Pure
    integer/float arithmetic; deterministic. Small matrices only (loops per sketch).
    """
    n = len(cost)
    if n == 0:
        return []
    if any(len(row) != n for row in cost):
        raise SequenceF1Error("cost matrix must be square")

    # O(n^3) Hungarian via potentials (Jonker-Volgenant style augmentation).
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)   # p[j] = row assigned to column j (1-based)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    assign = [0] * n
    for j in range(1, n + 1):
        if p[j] != 0:
            assign[p[j] - 1] = j - 1
    return assign


def _pad_square(pred: tuple[dict[str, int], ...], gt: tuple[dict[str, int], ...]):
    """Build a square cost matrix; padded rows/cols carry a high dummy cost."""
    n = max(len(pred), len(gt))
    # A dummy (unmatched) loop is an empty loop; its cost against a real loop is the
    # real loop's primitive count (all counted as misses).
    empty = {LINE: 0, ARC: 0, CIRCLE: 0}
    pred_l = list(pred) + [empty] * (n - len(pred))
    gt_l = list(gt) + [empty] * (n - len(gt))
    cost = [[_loop_cost(pred_l[i], gt_l[j]) for j in range(n)] for i in range(n)]
    return pred_l, gt_l, cost


@dataclass(frozen=True)
class PrimitiveCounts:
    """TP/FP/FN tallies for one primitive type (or extrusion)."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: "PrimitiveCounts") -> "PrimitiveCounts":
        return PrimitiveCounts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def _match_counts(pred_loop: dict[str, int], gt_loop: dict[str, int]) -> dict[str, PrimitiveCounts]:
    """TP/FP/FN per primitive type between one matched loop pair."""
    out: dict[str, PrimitiveCounts] = {}
    for t in _PRIMITIVE_TYPES:
        pv, gv = pred_loop.get(t, 0), gt_loop.get(t, 0)
        tp = min(pv, gv)
        out[t] = PrimitiveCounts(tp=tp, fp=max(0, pv - gv), fn=max(0, gv - pv))
    return out


@dataclass(frozen=True)
class SequenceEvaluation:
    """Per-type F1 counts for a predicted-vs-ground-truth sequence pair."""

    line: PrimitiveCounts
    arc: PrimitiveCounts
    circle: PrimitiveCounts
    extrusion: PrimitiveCounts

    def counts_for(self, primitive: str) -> PrimitiveCounts:
        return {LINE: self.line, ARC: self.arc, CIRCLE: self.circle,
                EXT: self.extrusion}[primitive]


def evaluate_sequence(pred: list[Command], gt: list[Command]) -> SequenceEvaluation:
    """F1 counts for a predicted vs ground-truth DeepCAD command sequence.

    Loops are matched with Hungarian assignment before counting, so a correct loop
    predicted in a different order still scores as a match.
    """
    pred_loops = loop_primitive_counts(pred)
    gt_loops = loop_primitive_counts(gt)
    tallies = {t: PrimitiveCounts() for t in _PRIMITIVE_TYPES}

    if pred_loops or gt_loops:
        pred_l, gt_l, cost = _pad_square(pred_loops, gt_loops)
        assign = hungarian_assignment(cost)
        for i, j in enumerate(assign):
            for t, c in _match_counts(pred_l[i], gt_l[j]).items():
                tallies[t] = tallies[t] + c

    n_pred_ext = sum(1 for c in pred if c.type == EXT)
    n_gt_ext = sum(1 for c in gt if c.type == EXT)
    ext = PrimitiveCounts(
        tp=min(n_pred_ext, n_gt_ext),
        fp=max(0, n_pred_ext - n_gt_ext),
        fn=max(0, n_gt_ext - n_pred_ext),
    )
    return SequenceEvaluation(
        line=tallies[LINE], arc=tallies[ARC], circle=tallies[CIRCLE], extrusion=ext)


def aggregate_f1(evals: list[SequenceEvaluation]) -> dict[str, float]:
    """Micro-averaged F1 per type across a batch (paper Table 1 layout)."""
    totals = {
        "line": PrimitiveCounts(), "arc": PrimitiveCounts(),
        "circle": PrimitiveCounts(), "extrusion": PrimitiveCounts(),
    }
    for e in evals:
        totals["line"] = totals["line"] + e.line
        totals["arc"] = totals["arc"] + e.arc
        totals["circle"] = totals["circle"] + e.circle
        totals["extrusion"] = totals["extrusion"] + e.extrusion
    return {name: c.f1 for name, c in totals.items()}


def invalidity_ratio(validity_flags: list[bool]) -> float:
    """IR = (#invalid) / (#total): fraction of sequences that fail to build (Sec. 5.1)."""
    if not validity_flags:
        raise SequenceF1Error("need at least one sample")
    invalid = sum(1 for ok in validity_flags if not ok)
    return invalid / len(validity_flags)
