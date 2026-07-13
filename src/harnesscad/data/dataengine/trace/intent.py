"""Design-intent capture (the differentiator the corpus is missing).

A raw op stream records *what* the agent did; it throws away *why*. The mined
corpus argument is that the scarce, valuable signal in CAD data is the design
**intent** behind each op — the rationale ("this rib resists the bending load")
and the manufacturing constraints ("keep wall >= 2 mm for injection moulding",
"this hole is a press-fit, H7") that a human engineer holds in their head while
they model. Capturing that alongside the tool call turns a bare trajectory into
*intent-labeled* data.

:class:`IntentAnnotation` is the JSON-serialisable record of one such intent, and
:func:`attach_intent` binds it to either a single :class:`~dataengine.trajectory.Step`
(per-op intent) or a whole :class:`~dataengine.trajectory.Trajectory` (design-level
intent), without mutating those classes' schemas — the annotations live on an
``intents`` side-channel (an attribute on a Step, ``metadata['intents']`` on a
Trajectory). Round-trips through ``to_dict``/``from_dict``.

Absolute imports, stdlib only, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class IntentAnnotation:
    """The design intent behind one op (or a whole design).

    ``rationale`` is the free-text 'why'; ``constraints`` captures the structured
    manufacturing/engineering tweaks (a dict such as
    ``{"process": "cnc", "min_wall_mm": 2.0, "fit": "H7"}`` — or a list of such
    notes). ``op`` / ``index`` locate the annotated step when the intent is
    per-op; both are ``None`` for a design-level annotation.
    """

    rationale: str
    constraints: Union[Dict[str, Any], List[Any]] = field(default_factory=dict)
    op: Optional[str] = None            # op tag of the annotated step, if per-op
    index: Optional[int] = None         # step index of the annotated step, if per-op

    def to_dict(self) -> dict:
        constraints = (list(self.constraints) if isinstance(self.constraints, list)
                       else dict(self.constraints))
        return {
            "rationale": self.rationale,
            "constraints": constraints,
            "op": self.op,
            "index": self.index,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "IntentAnnotation":
        return cls(
            rationale=d.get("rationale", ""),
            constraints=d.get("constraints", {}),
            op=d.get("op"),
            index=d.get("index"),
        )


def _is_trajectory(target: Any) -> bool:
    return hasattr(target, "steps") and hasattr(target, "metadata")


def _is_step(target: Any) -> bool:
    return hasattr(target, "action") and hasattr(target, "index")


def attach_intent(step_or_trajectory: Any,
                  rationale: str,
                  constraints: Optional[Union[Dict[str, Any], List[Any]]] = None,
                  *,
                  op: Optional[str] = None,
                  index: Optional[int] = None) -> IntentAnnotation:
    """Attach a design-intent annotation to a Step or a Trajectory.

    For a :class:`~dataengine.trajectory.Step` the op tag and index are inferred
    from the step (unless overridden) and the annotation is appended to a
    ``step.intents`` list. For a :class:`~dataengine.trajectory.Trajectory` the
    annotation is recorded under ``trajectory.metadata['intents']`` as a dict.
    A plain dict target is supported too (``target['intents']``). Returns the
    created :class:`IntentAnnotation`.
    """
    if constraints is None:
        constraints = {}

    if _is_trajectory(step_or_trajectory):
        ann = IntentAnnotation(rationale=rationale, constraints=constraints,
                               op=op, index=index)
        bucket = step_or_trajectory.metadata.setdefault("intents", [])
        bucket.append(ann.to_dict())
        return ann

    if _is_step(step_or_trajectory):
        inferred_op = op
        if inferred_op is None:
            call = getattr(step_or_trajectory.action, "tool_call", None) or {}
            if isinstance(call, dict):
                inferred_op = call.get("op")
        inferred_index = index if index is not None else getattr(step_or_trajectory, "index", None)
        ann = IntentAnnotation(rationale=rationale, constraints=constraints,
                               op=inferred_op, index=inferred_index)
        existing = getattr(step_or_trajectory, "intents", None)
        if existing is None:
            existing = []
            setattr(step_or_trajectory, "intents", existing)
        existing.append(ann)
        return ann

    if isinstance(step_or_trajectory, dict):
        ann = IntentAnnotation(rationale=rationale, constraints=constraints,
                               op=op, index=index)
        step_or_trajectory.setdefault("intents", []).append(ann.to_dict())
        return ann

    raise TypeError(
        "attach_intent expects a Step, a Trajectory, or a dict; "
        f"got {type(step_or_trajectory).__name__}")


def intents_of(step_or_trajectory: Any) -> List[IntentAnnotation]:
    """Read back the intents attached to a Step / Trajectory / dict."""
    if _is_trajectory(step_or_trajectory):
        raw = step_or_trajectory.metadata.get("intents", [])
        return [IntentAnnotation.from_dict(d) for d in raw]
    if _is_step(step_or_trajectory):
        return list(getattr(step_or_trajectory, "intents", []) or [])
    if isinstance(step_or_trajectory, dict):
        return [IntentAnnotation.from_dict(d)
                for d in step_or_trajectory.get("intents", [])]
    return []
