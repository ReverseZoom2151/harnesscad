"""skeleton — the top-down master-sketch / layout-first front-of-pipeline.

A :class:`~skeleton.layout.Skeleton` is a *master layout*: a named datum
reference frame (origin, primary axes, principal planes), a bounding envelope, a
master sketch of key reference geometry (envelope boundary + feature reference
points such as hole centres), and an editable driving-dimension parameter table.

It emits CISP ops (:mod:`cisp.ops`) so a :class:`loop.HarnessSession` can apply
the layout first; downstream part features then reference these datums instead of
re-deriving near-final geometry. This is the layout-first / rough-sizing /
constraint-reasoning wedge (see the sizing package for the numbers that fill the
parameter table).
"""

from __future__ import annotations

from skeleton.layout import Datum, Envelope, Skeleton, build_skeleton

__all__ = ["Datum", "Envelope", "Skeleton", "build_skeleton"]
