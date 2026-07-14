"""The graded trajectory -- a versioned, stable on-disk format, and the hook.

The pressure run produced 223 real, model-generated, graded attempts. They live
in a cache keyed for *reproducibility* and in a results file keyed for a *report*.
Neither is keyed for *learning*: the cache is keyed by prompt hash and carries no
verdict, and the report's records carry the v1 envelope verdict and nothing else.

This module defines the thing that should have been persisted all along::

    (brief, prompt, op_stream, oracle_verdict, measurements, per_step_rewards)

one JSON object per line, ``schema`` stamped, sorted keys, no wall clock. A
trajectory written today reads identically in a year, and a corpus built from two
runs is the concatenation of their files.

THE HOOK I NEED IN THE LOOP
===========================
``core/loop.py`` and ``core/pipeline.py`` are owned by another agent (a
loop-collapse is in progress) and ``eval/pressure/`` is owned by a third, so this
module defines the INTERFACE and does not wire it. What is needed is one optional
callback on the loop, invoked once per model call, after the attempt is graded
and before the next turn is composed:

    ``on_attempt(record: AttemptCapture) -> None``

with ``AttemptCapture`` carrying exactly what the loop already has in hand at that
moment and nothing it would have to compute: the brief id, the rendered prompt,
the raw completion, the parsed ops, the ApplyOpsResult diagnostics, and the
attempt index. **Everything else in this package is derived offline**, from that
tuple plus the brief -- deliberately, so the hook cannot slow the loop down, and
so a trajectory can be re-graded when the oracle improves without re-running a
single model.

:func:`capture_from_pressure_record` is the same interface, applied retroactively
to the results.json the pressure run already wrote. It is why the corpus exists
today with no new inference.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from harnesscad.agents.selftrain import SCHEMA_VERSION

__all__ = [
    "AttemptCapture",
    "StepReward",
    "Trajectory",
    "write_jsonl",
    "read_jsonl",
    "capture_from_pressure_record",
    "TrajectorySink",
]


@dataclass(frozen=True)
class AttemptCapture:
    """Exactly what the loop must hand over, and nothing it must compute.

    This is the payload of the ``on_attempt`` hook. Every field is already a
    local variable inside ``core/loop.py``'s attempt body at the moment the hook
    would fire.
    """

    brief_id: str
    prompt: str                     # the rendered user turn, verbatim
    raw: str                        # the model's completion, verbatim
    ops: List[dict]                 # parsed op stream ([] when it did not parse)
    parse_ok: bool
    parse_error: Optional[str]
    attempt: int                    # 1-based
    diagnostics: List[dict]         # the fleet's output; RECORDED, never a label
    feedback: Optional[str]         # what this arm handed back to the model
    model: str = ""
    loop: str = ""
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StepReward:
    """The process reward for one op. See :mod:`selftrain.divergence`."""

    index: int                      # 0-based op index
    op: str                         # the op tag
    applied: bool                   # the kernel accepted it
    reward: float                   # +1 correct prefix, 0 after divergence, -1 AT it
    divergent: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Trajectory:
    """One graded rollout. The unit of the training corpus.

    ``verdict`` is the ORACLE's, from :func:`selftrain.ledger.certify` -- the
    conjunction of gate + envelope + shape. ``diagnostics`` is the FLEET's, and it
    is data, not a label: the fleet's false-positive rate is the reason the
    harness lost its own experiment.
    """

    schema: str = SCHEMA_VERSION
    trajectory_id: str = ""
    brief_id: str = ""
    brief_text: str = ""
    model: str = ""
    loop: str = ""
    seed: int = 0
    attempt: int = 1
    prompt: str = ""
    raw: str = ""
    ops: List[dict] = field(default_factory=list)
    parse_ok: bool = False
    parse_error: Optional[str] = None

    # --- the oracle ------------------------------------------------------- #
    verdict: Dict[str, Any] = field(default_factory=dict)     # Certificate
    measurements: Dict[str, Any] = field(default_factory=dict)

    # --- the process reward ----------------------------------------------- #
    step_rewards: List[dict] = field(default_factory=list)    # [StepReward]
    first_divergence: Optional[int] = None                    # 0-based op index
    divergence_detail: str = ""

    # --- the aggregate (tool_reward.aggregate_reward) ---------------------- #
    reward_total: float = 0.0
    reward_outcome: float = 0.0
    reward_step_mean: float = 0.0
    reward_format: float = 0.0

    # --- the fleet, as data -------------------------------------------------#
    diagnostics: List[dict] = field(default_factory=list)
    feedback: Optional[str] = None

    @property
    def accepted(self) -> bool:
        return bool(self.verdict.get("accepted"))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Trajectory":
        known = {f for f in cls.__dataclass_fields__}      # noqa: SLF001
        return cls(**{k: v for k, v in d.items() if k in known})


def trajectory_id(model: str, loop: str, brief_id: str, attempt: int,
                  seed: int) -> str:
    """A stable, human-readable key. Deterministic; no hashing, no clock."""
    return "%s|%s|%s|a%d|s%d" % (model, loop, brief_id, attempt, seed)


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def write_jsonl(path: str, trajectories: Iterable[Trajectory]) -> int:
    """Write trajectories one JSON object per line. Sorted keys. Returns count."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for t in trajectories:
            fh.write(json.dumps(t.to_dict(), sort_keys=True))
            fh.write("\n")
            n += 1
    return n


def read_jsonl(path: str) -> List[Trajectory]:
    """Read a corpus back. Rejects a file whose schema is not this schema."""
    out: List[Trajectory] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            got = d.get("schema")
            if got != SCHEMA_VERSION:
                raise ValueError(
                    "%s:%d has schema %r, this reader is %r -- migrate it, do "
                    "not guess" % (path, lineno, got, SCHEMA_VERSION))
            out.append(Trajectory.from_dict(d))
    return out


class TrajectorySink:
    """The object the loop would be handed. ``sink.on_attempt(capture)``.

    Buffers in memory and flushes to JSONL. Grading is NOT done here -- the loop
    must not pay for it. The sink stores the raw capture; the corpus builder
    grades offline.
    """

    def __init__(self) -> None:
        self._captures: List[AttemptCapture] = []

    def on_attempt(self, capture: AttemptCapture) -> None:
        self._captures.append(capture)

    @property
    def captures(self) -> List[AttemptCapture]:
        return list(self._captures)

    def __len__(self) -> int:
        return len(self._captures)


# --------------------------------------------------------------------------- #
# retroactive capture: the pressure run already produced the data
# --------------------------------------------------------------------------- #
def capture_from_pressure_record(cell: Dict[str, Any],
                                 record: Dict[str, Any]) -> AttemptCapture:
    """Rebuild an :class:`AttemptCapture` from one ``results.json`` attempt.

    The pressure results file stores everything the hook would have captured
    except the rendered prompt (both arms share one system+user prompt built from
    ``brief.text``, and the per-turn feedback IS stored), so the prompt is
    reconstructed from the brief and the previous turn's feedback rather than
    invented.
    """
    grade = record.get("grade") or {}
    return AttemptCapture(
        brief_id=str(cell["brief"]),
        prompt="",                       # filled by the corpus builder
        raw=str(record.get("raw") or ""),
        ops=[dict(o) for o in (record.get("ops") or [])],
        parse_ok=bool(record.get("parse_ok")),
        parse_error=record.get("parse_error"),
        attempt=int(record.get("attempt") or 1),
        diagnostics=[dict(d) for d in (grade.get("diagnostics") or [])],
        feedback=record.get("feedback"),
        model=str(cell.get("model") or ""),
        loop=str(cell.get("loop") or ""),
        seed=int(cell.get("seed") or 0),
    )


def captures_from_results(results: Dict[str, Any]
                          ) -> Iterator[AttemptCapture]:
    """Every attempt in a pressure ``results.json``, in file order."""
    for cell in results.get("results", []):
        for record in cell.get("records", []):
            yield capture_from_pressure_record(cell, record)
