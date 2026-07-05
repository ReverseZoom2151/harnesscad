"""dataengine.edit_pairs — human-edit-delta preference capture.

The harness IS the data flywheel (sec.17/21): the richest, cheapest preference
signal it produces is the *edit a human made to the harness's own proposal*. When
the model proposes an op stream and the user commits a different final one, the
diff between them is a ready-made preference pair — the user's committed edit is
the *chosen* response and the harness's proposal is the *rejected* one. A heavily
edited output is, by construction, a weak proposal; a barely touched one is a
strong proposal.

This module captures that:

  * :func:`capture_edit_pair` records ``(proposed_ops, human_final_ops)`` plus the
    brief/outcome, using ``quality.diff.op_diff`` to record exactly *what changed*
    (added / removed / modified ops, with field-level deltas) and how large the
    edit was.
  * :func:`to_preference` emits a DPO-shaped record (``chosen`` = human final,
    ``rejected`` = proposed) carrying the diff summary — the same row shape as
    ``dataengine.export.to_dpo`` so it drops straight into the training set.
  * :class:`EditPairStore` accumulates pairs and round-trips them to/from JSON.

Absolute imports, stdlib + the in-repo ``quality.diff`` primitive only.
Deterministic; no wall-clock.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from quality.diff import op_diff


# A proposal edited beyond this fraction of its ops is a "weak" proposal.
_HEAVY_EDIT_RATIO = 0.5


class _OpShim:
    """Adapt a serialised op dict to the ``op_diff`` protocol (``OP`` + ``to_dict``).

    ``quality.diff.op_diff`` reads ``op.to_dict()`` and ``op.OP``; callers that
    already hold op *dicts* (e.g. from a trace / memory) are wrapped so the diff
    runs without reconstructing typed ``Op`` instances.
    """

    def __init__(self, d: Dict[str, Any]) -> None:
        self._d = dict(d)
        self.OP = self._d.get("op", "op")

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._d)


def _as_op_objs(ops: Sequence[Any]) -> List[Any]:
    """Coerce a list of typed ops or op dicts into diff-compatible objects."""
    out: List[Any] = []
    for op in ops or []:
        if hasattr(op, "to_dict") and hasattr(op, "OP"):
            out.append(op)
        elif isinstance(op, dict):
            out.append(_OpShim(op))
        else:
            raise TypeError(f"cannot treat {type(op).__name__} as an op")
    return out


def _to_dicts(ops: Sequence[Any]) -> List[Dict[str, Any]]:
    """Serialise ops (typed or dict) to plain JSON-able dicts."""
    out: List[Dict[str, Any]] = []
    for op in ops or []:
        if hasattr(op, "to_dict"):
            out.append(dict(op.to_dict()))
        elif isinstance(op, dict):
            out.append(dict(op))
        else:
            raise TypeError(f"cannot serialise {type(op).__name__} as an op")
    return out


@dataclass
class EditPair:
    """One captured (proposed -> human-final) edit, scored as a preference.

    - ``proposed`` / ``human_final`` : the two op streams as serialised dicts.
    - ``diff`` / ``diff_summary``    : ``op_diff`` result + its one-line render.
    - ``n_changes``                  : added + removed + modified op count.
    - ``edit_ratio``                 : changes / (changes + unchanged) in [0, 1].
    - ``heavily_edited``             : True when the edit was large (weak proposal).
    """

    brief: str
    proposed: List[Dict[str, Any]]
    human_final: List[Dict[str, Any]]
    diff: Dict[str, Any]
    diff_summary: str
    n_changes: int
    edit_ratio: float
    heavily_edited: bool
    outcome: str = "committed"
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_signal(self) -> bool:
        """True when the human actually changed something (a usable preference)."""
        return self.n_changes > 0

    def to_dict(self) -> dict:
        return {
            "brief": self.brief,
            "proposed": self.proposed,
            "human_final": self.human_final,
            "diff": self.diff,
            "diff_summary": self.diff_summary,
            "n_changes": self.n_changes,
            "edit_ratio": self.edit_ratio,
            "heavily_edited": self.heavily_edited,
            "outcome": self.outcome,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EditPair":
        return cls(
            brief=d.get("brief", ""),
            proposed=list(d.get("proposed", [])),
            human_final=list(d.get("human_final", [])),
            diff=dict(d.get("diff", {})),
            diff_summary=d.get("diff_summary", ""),
            n_changes=int(d.get("n_changes", 0)),
            edit_ratio=float(d.get("edit_ratio", 0.0)),
            heavily_edited=bool(d.get("heavily_edited", False)),
            outcome=d.get("outcome", "committed"),
            meta=dict(d.get("meta", {})),
        )


def capture_edit_pair(proposed_ops: Sequence[Any],
                      human_final_ops: Sequence[Any],
                      brief: str = "",
                      outcome: str = "committed",
                      heavy_edit_ratio: float = _HEAVY_EDIT_RATIO,
                      **meta: Any) -> EditPair:
    """Capture the diff between a proposed and a human-committed op stream.

    Reuses ``quality.diff.op_diff`` to record what changed (added/removed/modified
    ops with field-level deltas). The edit magnitude drives ``heavily_edited`` —
    the flag that marks the proposal as weak training signal.
    """
    diff = op_diff(_as_op_objs(proposed_ops), _as_op_objs(human_final_ops))
    n_changes = len(diff.added) + len(diff.removed) + len(diff.modified)
    total = n_changes + diff.unchanged_count
    edit_ratio = (n_changes / total) if total else 0.0
    return EditPair(
        brief=brief,
        proposed=_to_dicts(proposed_ops),
        human_final=_to_dicts(human_final_ops),
        diff=diff.to_dict(),
        diff_summary=diff.render(),
        n_changes=n_changes,
        edit_ratio=edit_ratio,
        heavily_edited=edit_ratio >= heavy_edit_ratio and n_changes > 0,
        outcome=outcome,
        meta=dict(meta),
    )


def to_preference(pair: EditPair) -> dict:
    """Emit a DPO-shaped preference record from an :class:`EditPair`.

    ``chosen`` is the human's committed final stream, ``rejected`` is the
    harness's proposal — mirroring ``dataengine.export.to_dpo``'s row shape so the
    record drops straight into the preference-training set. The diff summary rides
    along so the learning signal (what the human had to change) is inspectable.
    The larger the edit, the stronger the (chosen > rejected) preference.
    """
    return {
        "prompt": pair.brief,
        "plan": None,
        "reward_kind": "human_edit_delta",
        "chosen": {
            "response": [{"tool_call": op} for op in pair.human_final],
            "source": "human_final",
        },
        "rejected": {
            "response": [{"tool_call": op} for op in pair.proposed],
            "source": "proposed",
        },
        "diff": pair.diff,
        "diff_summary": pair.diff_summary,
        "edit_magnitude": pair.n_changes,
        "edit_ratio": pair.edit_ratio,
        "heavily_edited": pair.heavily_edited,
        "has_signal": pair.is_signal,
    }


class EditPairStore:
    """An append-only store of captured edit pairs, JSON-persistable."""

    def __init__(self) -> None:
        self.pairs: List[EditPair] = []

    def __len__(self) -> int:
        return len(self.pairs)

    def add(self, pair: EditPair) -> EditPair:
        self.pairs.append(pair)
        return pair

    def capture(self, proposed_ops: Sequence[Any], human_final_ops: Sequence[Any],
                brief: str = "", outcome: str = "committed", **meta: Any) -> EditPair:
        """Convenience: capture a pair and add it to the store in one call."""
        return self.add(capture_edit_pair(
            proposed_ops, human_final_ops, brief=brief, outcome=outcome, **meta))

    def to_preferences(self) -> List[dict]:
        """Every stored pair as a DPO-shaped preference record."""
        return [to_preference(p) for p in self.pairs]

    def to_dict(self) -> dict:
        return {"version": 1, "pairs": [p.to_dict() for p in self.pairs]}

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "EditPairStore":
        store = cls()
        store.pairs = [EditPair.from_dict(p) for p in d.get("pairs", [])]
        return store

    @classmethod
    def load(cls, path: str) -> "EditPairStore":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
