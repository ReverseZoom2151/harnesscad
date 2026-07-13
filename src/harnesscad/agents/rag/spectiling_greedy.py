"""Greedy submodular exemplar selection (DST Algorithm 1).

Implements the exemplar-retrieval core of *Design-Specification Tiling for
ICL-based CAD Code Generation* (Sec. 3.3, Algorithm 1).

Given a query component set ``C_query`` and per-exemplar component sets
``{C_i}``, we select ``k`` exemplars that maximise the *weighted tiling* of the
query::

    S* = argmax_{|S|=k}  w( C(S) & C_query )                         (Eq. 7)

Maximising this objective is NP-hard, but the objective is non-negative,
monotone and submodular (paper Proposition 1 + Appendix A), so the greedy
algorithm that repeatedly adds the exemplar with the largest *marginal
weighted-tiling gain* attains a ``(1 - 1/e)`` approximation. Selection stops
early once no candidate provides positive gain (Algorithm 1, lines 15-17).

The implementation tracks the covered-component accumulator ``T`` incrementally
so each marginal-gain evaluation only pays for the new exemplar's contribution,
exactly as in the pseudocode. Ties are broken by lowest index for determinism.

stdlib-only. Import the component representation by full path.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Sequence

from harnesscad.agents.context.spectiling_components import (
    ComponentSet,
    tiling_ratio,
    weighted_size,
)


class SelectionStep(NamedTuple):
    """One greedy pick: which exemplar, its marginal gain, ratio-so-far."""

    index: int
    marginal_gain: int
    tiling_ratio_after: float


class DSTSelection(NamedTuple):
    """Result of a DST greedy selection."""

    indices: List[int]           # selected exemplar indices, in pick order
    steps: List[SelectionStep]   # per-iteration trace
    tiling_ratio: float          # final f_suff(S; query)

    def ordered_indices(self) -> List[int]:
        return list(self.indices)


def marginal_gain(
    covered_query: ComponentSet,
    candidate: ComponentSet,
    query: ComponentSet,
) -> int:
    """Weighted marginal tiling gain of adding ``candidate`` (Alg.1 line 8).

    ``covered_query`` is ``T`` -- the query components already tiled. The gain
    is ``w(T u (C_j & C_query)) - w(T)``. Because ``T`` is already a subset of
    the query, this equals the added weight of newly-tiled query components.
    """
    contribution = candidate.intersection(query)
    new_T = covered_query.union(contribution)
    return weighted_size(new_T) - weighted_size(covered_query)


def dst_select(
    query: ComponentSet,
    exemplars: Sequence[ComponentSet],
    k: int,
) -> DSTSelection:
    """Greedy DST exemplar selection -- Algorithm 1.

    Args:
        query: ``C_query`` multi-granular component set.
        exemplars: ``[C_0, C_1, ...]`` one component set per database exemplar.
        k: selection capacity (``k << n``). Clamped to ``len(exemplars)``.

    Returns:
        A :class:`DSTSelection` with the chosen indices (in the order the
        greedy picked them), a per-step trace, and the final tiling ratio.

    Determinism: candidates are scanned in ascending index order and the first
    exemplar achieving the maximum positive gain wins ties -- so the output is a
    pure function of the inputs.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    n = len(exemplars)
    cap = min(k, n)

    selected: List[int] = []
    steps: List[SelectionStep] = []
    chosen = set()
    covered_query = ComponentSet.empty(query.granularities)  # T

    for _ in range(cap):
        best_idx = -1
        best_gain = 0
        for j in range(n):
            if j in chosen:
                continue
            g = marginal_gain(covered_query, exemplars[j], query)
            if g > best_gain:
                best_gain = g
                best_idx = j
        if best_idx == -1:  # no positive marginal gain -> terminate early
            break
        contribution = exemplars[best_idx].intersection(query)
        covered_query = covered_query.union(contribution)
        chosen.add(best_idx)
        selected.append(best_idx)
        steps.append(
            SelectionStep(
                index=best_idx,
                marginal_gain=best_gain,
                tiling_ratio_after=tiling_ratio(covered_query, query),
            )
        )

    return DSTSelection(
        indices=selected,
        steps=steps,
        tiling_ratio=tiling_ratio(covered_query, query),
    )


def uncovered_components(
    query: ComponentSet,
    selected: Sequence[ComponentSet],
) -> Dict[int, List[str]]:
    """Query components NOT tiled by the selected exemplars, per granularity.

    Useful for diagnosing *why* a selection is insufficient (which design
    features remain uncovered). Returns ``{n: ["a b", ...]}`` with components
    rendered as space-joined phrases, sorted for determinism.
    """
    if selected:
        union = selected[0]
        for cs in selected[1:]:
            union = union.union(cs)
    else:
        union = ComponentSet.empty(query.granularities)
    out: Dict[int, List[str]] = {}
    for n in query.granularities:
        missing = query.at(n) - union.at(n)
        if missing:
            out[n] = sorted(" ".join(c) for c in missing)
    return out
