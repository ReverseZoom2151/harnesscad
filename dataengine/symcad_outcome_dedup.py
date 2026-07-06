"""De-duplication of benchmark instances by identical per-option outcome vector.

Transferable data-curation step from del Rio & England, *Lessons on Datasets and
Paradigms in Machine Learning for Symbolic Computation: A Case Study on CAD*
(Section 3.1, "Elimination of duplicates"). Each problem carried a vector of six
CAD cell-counts (one per variable ordering); when two problems had *identical*
outcome vectors they were treated as effectively the same problem and one was
dropped. This collapsed 5599 raw problems to 1019 -- and the paper reports (7.1.3)
that a model trained on the deduplicated 1019 matched one trained on 6x more raw
data, because "it is the number of qualitatively different problems that matters"
and, critically, dedup "should avoid data leakage between testing and training".

This is distinct from ``dataengine/datacon_lowdata_dedup`` (which dedups by a
*scale-invariant feature* signature of the input geometry). Here we dedup by the
*outcome/label vector* -- the per-option costs an instance produces -- which is
exactly the signal a leakage audit needs: two instances that behave identically
under every option are indistinguishable to any choice model and must not straddle
a train/test split. Applicable to any mechanical-CAD benchmark where each item has
a vector of per-strategy outcomes (build times, cell/face counts, pass/fail).

Provided functions:
  * :func:`outcome_signature` -- canonical, rounded, timeout-aware fingerprint.
  * :func:`deduplicate`       -- first-seen-wins collapse to unique outcomes.
  * :func:`duplicate_groups`  -- groups of indices sharing an outcome vector.
  * :func:`dedup_report`      -- before/after counts and reduction ratio.

Stdlib-only, deterministic (sorted iteration; no wall clock, no randomness).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

Cost = Optional[float]


def outcome_signature(
    outcomes: Sequence[Cost],
    *,
    decimals: int = 6,
    timeout_token: str = "T",
) -> Tuple[Hashable, ...]:
    """Canonical fingerprint of a per-option outcome vector.

    ``None`` entries (timeouts / failures) become ``timeout_token`` so they match
    each other but not a finite cost. Finite costs are rounded to ``decimals`` to
    absorb floating-point / measurement noise, matching the paper's intent that
    "problems that differ only by a single coefficient" collapse together.
    """
    sig: List[Hashable] = []
    for value in outcomes:
        if value is None:
            sig.append(timeout_token)
        else:
            sig.append(round(float(value), decimals))
    return tuple(sig)


def duplicate_groups(
    instances: Sequence[Sequence[Cost]],
    *,
    decimals: int = 6,
) -> List[List[int]]:
    """Return index groups (size >= 2) that share an identical outcome vector.

    Groups are ordered by their first-seen index; indices within a group are
    ascending. Deterministic.
    """
    buckets: Dict[Tuple[Hashable, ...], List[int]] = {}
    order: List[Tuple[Hashable, ...]] = []
    for idx, outcomes in enumerate(instances):
        sig = outcome_signature(outcomes, decimals=decimals)
        if sig not in buckets:
            buckets[sig] = []
            order.append(sig)
        buckets[sig].append(idx)
    return [buckets[sig] for sig in order if len(buckets[sig]) > 1]


def deduplicate(
    instances: Sequence[Sequence[Cost]],
    *,
    decimals: int = 6,
    return_indices: bool = False,
):
    """Collapse instances to one representative per unique outcome vector.

    First occurrence wins (stable order preserved). With ``return_indices`` the
    kept indices are returned instead of the instances themselves.
    """
    seen: set = set()
    kept_indices: List[int] = []
    for idx, outcomes in enumerate(instances):
        sig = outcome_signature(outcomes, decimals=decimals)
        if sig in seen:
            continue
        seen.add(sig)
        kept_indices.append(idx)
    if return_indices:
        return kept_indices
    return [instances[i] for i in kept_indices]


@dataclass(frozen=True)
class DedupReport:
    n_before: int
    n_after: int
    n_removed: int
    n_duplicate_groups: int
    reduction_ratio: float


def dedup_report(
    instances: Sequence[Sequence[Cost]],
    *,
    decimals: int = 6,
) -> DedupReport:
    """Summarise the effect of :func:`deduplicate` without mutating input."""
    before = len(instances)
    kept = deduplicate(instances, decimals=decimals, return_indices=True)
    after = len(kept)
    groups = duplicate_groups(instances, decimals=decimals)
    return DedupReport(
        n_before=before,
        n_after=after,
        n_removed=before - after,
        n_duplicate_groups=len(groups),
        reduction_ratio=(before - after) / before if before else 0.0,
    )
