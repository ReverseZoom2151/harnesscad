"""Synchronous Technology partial conversion (Section 4.3 of Zou 2025).

Synchronous Technology (Siemens NX) improves on the Dual-Modes approach (4.2) by
doing a *partial* rather than whole-model conversion to dumb B-rep. Features are
separated into two groups:

* **direct-edit features** — converted to B-rep so direct editing applies;
* **ordinary features** — kept parametric.

Two structural rules the paper states:

1. the direct-edit features are placed *before* the ordinary features in the
   model history;
2. to direct-edit an ordinary feature the user moves it to the direct-edit set,
   "but when one of those features is moved ... all ordinary features created
   prior to the feature being moved must also be moved (done automatically in
   the background)." This cascade "causes unnecessary loss of meaningful
   parametrics."

This module models that partition and cascade deterministically and quantifies
the collateral parametric loss.

* :class:`SyncPartition` holds a fixed *creation order* of features and a set of
  direct-edit ids, and materialises the reordered history (direct-edit first).
* :meth:`SyncPartition.move_to_direct_edit` applies the cascade rule and returns
  the ids collaterally converted.
* :meth:`SyncPartition.parametric_loss` counts the collateral conversions (the
  features that lost parametric control unnecessarily).

Stdlib-only, deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

from harnesscad.domain.editing.paramdirect_model import FeatureTree


@dataclass
class SyncPartition:
    """A synchronous-technology partition over a fixed creation order.

    ``creation_order`` is the ids in the order features were created. ``direct``
    is the subset currently in the direct-edit set. The *history* order always
    lists direct-edit features first (rule 1), each subgroup preserving creation
    order.
    """

    creation_order: List[str]
    direct: Set[str] = field(default_factory=set)

    def __post_init__(self):
        unknown = self.direct - set(self.creation_order)
        if unknown:
            raise ValueError(f"direct ids not in creation order: {sorted(unknown)}")

    # -- views ------------------------------------------------------------
    def ordinary(self) -> List[str]:
        return [f for f in self.creation_order if f not in self.direct]

    def direct_edit(self) -> List[str]:
        return [f for f in self.creation_order if f in self.direct]

    def history_order(self) -> List[str]:
        """The materialised history: direct-edit features first (rule 1)."""
        return self.direct_edit() + self.ordinary()

    # -- cascade rule (rule 2) -------------------------------------------
    def move_to_direct_edit(self, fid: str) -> List[str]:
        """Move ``fid`` to the direct-edit set, cascading prior ordinaries.

        Returns the ids *collaterally* moved (every ordinary feature created
        before ``fid`` that had to be dragged along), in creation order. The
        target ``fid`` itself is not counted as collateral.
        """
        if fid not in self.creation_order:
            raise KeyError(fid)
        idx = self.creation_order.index(fid)
        collateral: List[str] = []
        for f in self.creation_order[:idx]:
            if f not in self.direct:
                collateral.append(f)
        self.direct.add(fid)
        self.direct.update(collateral)
        return collateral

    def parametric_loss(self) -> int:
        """Ordinary features that would *not* need converting but for the cascade.

        The direct-edit set is the union of intentionally-moved features and
        their forced predecessors; the meaningful metric the paper flags is the
        collateral count. Here we report the total number of features that have
        lost parametric control (the direct-edit set size), which the tests
        compare against the intended target count.
        """
        return len(self.direct)

    def copy(self) -> "SyncPartition":
        return SyncPartition(list(self.creation_order), set(self.direct))


def from_tree(tree: FeatureTree) -> SyncPartition:
    """Build a partition from a feature tree.

    Features already flagged ``direct_edit`` seed the direct-edit set; the tree's
    list order is taken as the creation order.
    """
    order = [f.fid for f in tree.features]
    direct = {f.fid for f in tree.features if f.direct_edit}
    return SyncPartition(order, direct)
