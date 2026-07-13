"""LoopDetector — sliding-window oscillation detector for the harness loop.

Per HARNESS_BLUEPRINT.md sec.3: CAD agents oscillate (re-extruding, retrying a
failing boolean). This hashes ``(op_tag, sorted-args)`` over a sliding window and
flags a loop when the same signature recurs ``threshold`` times inside the window.

It is a PRE-apply advisory layer that complements HarnessSession's block-and-correct
(loop.py): the session stops a *bad* op, this stops a *repeated* op the agent keeps
retrying unchanged. Fully deterministic — no wall clock, no randomness — so the same
op stream always produces the same detections (replayable, like the rest of the spine).
"""

from __future__ import annotations

from collections import deque
from typing import Deque

from harnesscad.core.cisp.ops import Op, canonical_json


def signature(op: Op) -> str:
    """Content signature of an op: its tag plus sorted args, deterministically.

    Two ops with the same tag and identical field values share a signature; any
    differing field (a new distance, a different sketch, a bumped radius) yields a
    different signature — so an agent that *adjusts* params is not flagged, only one
    repeating the exact same op. ``canonical_json`` already sorts keys, giving the
    "(op_tag, sorted-args)" hash the blueprint calls for.
    """
    return canonical_json(op)


class LoopDetector:
    """Flag oscillation when an op signature repeats ``threshold`` times in ``window``.

    ``observe(op)`` returns True the moment a signature's count within the current
    sliding window reaches ``threshold``. The window bounds how far back a repeat
    still "counts", so unrelated distinct ops in between do not mask a genuine loop
    but a long-ago identical op eventually ages out.
    """

    def __init__(self, window: int = 6, threshold: int = 3) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        if threshold < 2:
            raise ValueError("threshold must be >= 2 (a single op is never a loop)")
        self.window = window
        self.threshold = threshold
        self._recent: Deque[str] = deque(maxlen=window)

    def signature(self, op: Op) -> str:
        """Expose the signature helper as a method for callers/logging."""
        return signature(op)

    def observe(self, op: Op) -> bool:
        """Record ``op`` and report whether it completes a loop.

        Returns True when this op's signature now occurs at least ``threshold``
        times within the sliding window (i.e. the agent is stuck repeating it).
        """
        sig = signature(op)
        self._recent.append(sig)  # deque(maxlen) evicts the oldest automatically
        return self._recent.count(sig) >= self.threshold

    def reset(self) -> None:
        """Clear the window (e.g. after a successful checkpoint / new step)."""
        self._recent.clear()
