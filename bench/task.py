"""CADBench-Verified task schema (HARNESS_BLUEPRINT.md sec.16).

A Task is the SWE-bench-for-CAD unit: a natural-language spec plus a reference
CISP op stream plus a programmatic acceptance spec. The runner builds the part
(from the reference ops, or from a pluggable NL->ops solver) and a geometric
checker scores it against the acceptance spec.

The acceptance spec deliberately reuses the backend's own read-only query keys
(`summary`, `validity`, `measure`) so the checker asserts against exactly the
data the kernel exposes — no bespoke measurement path that could drift from what
the agent sees.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List

from cisp.ops import Op, parse_op

# The difficulty band (sec.16: report per-difficulty; curate the 30-70% solve band).
DIFFICULTIES = ("easy", "medium", "hard")


@dataclass(frozen=True)
class Task:
    """A single CADBench-Verified task.

    Fields
    ------
    id:          stable task identifier.
    difficulty:  one of DIFFICULTIES ("easy" | "medium" | "hard").
    brief:       the natural-language spec handed to the agent.
    ops:         the reference/ground-truth CISP op stream, as a list of dicts
                 (each the ``to_dict()`` form of a cisp.ops.Op). This is both the
                 gold trajectory (its length is the optimal L* for trajectory
                 efficiency) and the default solver.
    acceptance:  the programmatic checker spec. A dict keyed by backend query
                 family, plus an optional numeric tolerance::

                     {
                       "summary":  {"feature_count": 1, "solid_present": true},
                       "validity": {"is_valid": true},
                       "measure":  {"bbox": [20, 10, 5], "volume": 1000},
                       "tolerance": 0.02
                     }

                 Fields under a family are matched against ``backend.query(family)``.
                 A field the running backend cannot measure (e.g. `measure` on the
                 dependency-free StubBackend) is skipped, not failed.
    """

    id: str
    difficulty: str
    brief: str
    ops: List[dict] = field(default_factory=list)
    acceptance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.difficulty not in DIFFICULTIES:
            raise ValueError(
                f"task {self.id!r}: difficulty {self.difficulty!r} "
                f"not in {DIFFICULTIES}")

    def reference_ops(self) -> List[Op]:
        """The reference op stream as parsed cisp.ops.Op objects."""
        return [parse_op(d) for d in self.ops]

    def optimal_len(self) -> int:
        """L* — the optimal trajectory length (the reference op count)."""
        return len(self.ops)

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        return cls(
            id=d["id"],
            difficulty=d["difficulty"],
            brief=d.get("brief", ""),
            ops=list(d.get("ops", [])),
            acceptance=dict(d.get("acceptance", {})),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "difficulty": self.difficulty,
            "brief": self.brief,
            "ops": self.ops,
            "acceptance": self.acceptance,
        }


def load_task(path: str) -> Task:
    """Load a single task from a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        return Task.from_dict(json.load(fh))


def load_tasks(directory: str) -> List[Task]:
    """Load every ``*.json`` task file in ``directory`` (sorted by filename)."""
    tasks: List[Task] = []
    for name in sorted(os.listdir(directory)):
        if name.endswith(".json"):
            tasks.append(load_task(os.path.join(directory, name)))
    return tasks
