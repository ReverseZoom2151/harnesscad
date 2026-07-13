"""dataengine — the data-engine / training-trace layer (HARNESS_BLUEPRINT.md
sec.17 + sec.21).

The harness IS the data flywheel: every session already logs
``prompt -> plan -> ops -> geometry -> tests`` through the trace stream
(``trace.py``). This package folds that stream into the one canonical training
record the blueprint calls for and exports it to the named training formats.

  * :mod:`dataengine.trajectory` — ``Step`` = (S_t, A_t=[reasoning, tool_call],
    R_t, S_{t+1}), ``Trajectory`` (ordered steps + final scalar reward + sub-goal
    labels + prompt/plan metadata), ``from_events`` (build one from a JsonlTracer
    stream, reward per step from the verifier's outcome), and
    ``Trajectory.trajectory_slice(to_first_divergence=True)`` for dense signal.
  * :mod:`dataengine.export` — ``to_grpo`` / ``to_dpo`` / ``to_star`` exporters,
    ``write_jsonl``, and ``flywheel_metrics`` (human-corrections-per-plan +
    success/efficiency aggregates).
"""

from __future__ import annotations

from harnesscad.data.dataengine.trace.trajectory import (
    Action,
    REWARD_FAIL,
    REWARD_NEUTRAL,
    REWARD_PASS,
    Step,
    SubGoal,
    Trajectory,
    from_events,
)
from harnesscad.data.dataengine.trace.export import (
    flywheel_metrics,
    to_dpo,
    to_grpo,
    to_star,
    write_jsonl,
)
from harnesscad.data.dataengine.edits.edit_pairs import (
    EditPair,
    EditPairStore,
    capture_edit_pair,
    to_preference,
)

__all__ = [
    "Action",
    "Step",
    "SubGoal",
    "Trajectory",
    "from_events",
    "REWARD_PASS",
    "REWARD_FAIL",
    "REWARD_NEUTRAL",
    "to_grpo",
    "to_dpo",
    "to_star",
    "flywheel_metrics",
    "write_jsonl",
    "EditPair",
    "EditPairStore",
    "capture_edit_pair",
    "to_preference",
]
