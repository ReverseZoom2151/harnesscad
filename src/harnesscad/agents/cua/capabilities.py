"""capabilities — the model-agnostic CUA loop, as a deterministic capability router.

Two very different model abilities are kept as separate capabilities:

* ``predict_step`` — a full computer-use model that, given a screenshot and an
  instruction, emits the next ACTION directly (OpenAI computer-use, Claude
  computer-use, UI-TARS). It plans and grounds in one call.
* ``predict_click`` — a GROUNDING-only model that, given an element description,
  returns only WHERE to click (OmniParser + a locator, ShowUI, a Set-of-Marks
  picker). It cannot plan; it can only point.

A model that advertises ``predict_step`` runs a native loop; a model that only
advertises ``predict_click`` runs a composed
loop where a separate planner decides WHAT to do and the click model decides WHERE.
The loop is model-agnostic precisely because it routes over the capability SET
rather than the model's identity.

This module is that router, extracted as a pure function of (objective, capability
set) — never a model call. Given what a configured agent can do, :func:`route`
returns the ordered list of capabilities to invoke to satisfy an objective, or an
explicit "unroutable, missing X" when the set is insufficient.
:class:`CapabilityRouter` binds injected handlers to those capabilities and runs
the route, so the whole planning/grounding composition is unit-testable with
fakes and contains no model, screenshot, or GUI.

How this sits beside the rest of the CUA surface
------------------------------------------------
:mod:`harnesscad.io.cua.wire` routes a COMMAND to a driver handler (the transport
envelope). This routes an OBJECTIVE to a sequence of model capabilities (the
cognition side). :mod:`harnesscad.eval.grounding.som` supplies the ``id2xy`` a
``predict_click`` grounder answers against; a Set-of-Marks picker is the canonical
click-only capability this router composes with a planner.

Pure stdlib, deterministic, import-safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


class Capability(str, Enum):
    """The model abilities a CUA loop composes over."""

    PREDICT_STEP = "predict_step"    # instruction + screenshot -> next action (plan+ground)
    PREDICT_CLICK = "predict_click"  # element description -> click point (ground only)
    PLAN = "plan"                    # instruction + observation -> element description
    TYPE_TEXT = "type_text"          # emit literal text to enter


class RouteError(ValueError):
    """An objective cannot be served by the available capabilities."""


class Objective(str, Enum):
    """What the loop is asked to produce this turn."""

    NEXT_ACTION = "next_action"   # decide and locate the next thing to do
    CLICK_TARGET = "click_target" # locate a described element only


@dataclass(frozen=True)
class Route:
    """An ordered plan of capabilities to invoke for an objective.

    ``kind`` labels the loop cua-main would have assembled: ``native`` when one
    ``predict_step`` model does everything, ``composed`` when a planner and a click
    grounder are chained. ``steps`` is the invocation order.
    """

    objective: Objective
    steps: Tuple[Capability, ...]
    kind: str

    def to_dict(self) -> dict:
        return {"objective": self.objective.value,
                "steps": [c.value for c in self.steps], "kind": self.kind}


def route(objective: Objective, capabilities: Sequence[Capability]) -> Route:
    """The capability sequence that serves ``objective``, or raise :class:`RouteError`.

    The routing rule IS the model-agnostic insight, stated once:

    * A ``NEXT_ACTION`` is served natively by a single ``predict_step`` model; if
      there is none, it is COMPOSED from ``plan`` (what to do) + ``predict_click``
      (where). A click-only model is thus usable for full autonomy the moment a
      planner is added — and a step model needs no planner at all.
    * A ``CLICK_TARGET`` is served by ``predict_click`` directly, or, failing that,
      by a ``predict_step`` model constrained to a pointing action.

    Deterministic and total over the enum; the preference order (native before
    composed) is fixed so the same capability set always routes the same way.
    """
    caps = set(capabilities)
    if objective is Objective.NEXT_ACTION:
        if Capability.PREDICT_STEP in caps:
            return Route(objective, (Capability.PREDICT_STEP,), kind="native")
        if Capability.PLAN in caps and Capability.PREDICT_CLICK in caps:
            return Route(objective, (Capability.PLAN, Capability.PREDICT_CLICK),
                         kind="composed")
        missing = ("predict_step, OR (plan AND predict_click)")
        raise RouteError("cannot serve next_action; have %s, need %s"
                         % (sorted(c.value for c in caps), missing))
    if objective is Objective.CLICK_TARGET:
        if Capability.PREDICT_CLICK in caps:
            return Route(objective, (Capability.PREDICT_CLICK,), kind="grounding")
        if Capability.PREDICT_STEP in caps:
            return Route(objective, (Capability.PREDICT_STEP,), kind="native")
        raise RouteError("cannot serve click_target; have %s, need predict_click "
                         "or predict_step" % sorted(c.value for c in caps))
    raise RouteError("unknown objective %r" % (objective,))


def can_serve(objective: Objective, capabilities: Sequence[Capability]) -> bool:
    """True iff ``objective`` is routable with these capabilities (never raises)."""
    try:
        route(objective, capabilities)
        return True
    except RouteError:
        return False


@dataclass(frozen=True)
class Action:
    """The router's output: what to do, expressed uniformly regardless of loop.

    ``kind`` is ``click`` / ``type`` / ``done`` / ``other``; ``point`` is a click
    pixel when known; ``text`` carries typed text; ``description`` is the planner's
    target phrase (present in the composed loop). This is the single shape a driver
    consumes whether it came from one step model or a planner+grounder pair.
    """

    kind: str
    point: Optional[Tuple[int, int]] = None
    text: str = ""
    description: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "point": list(self.point) if self.point else None,
                "text": self.text, "description": self.description,
                "meta": dict(self.meta)}


class CapabilityRouter:
    """Bind handlers to capabilities and execute a :class:`Route`.

    Handlers are injected callables — real model adapters in production, fakes in a
    test. Each has a fixed signature keyed by capability:

    * ``PREDICT_STEP(observation) -> Action`` (already a full action)
    * ``PLAN(observation) -> str`` (an element description)
    * ``PREDICT_CLICK(observation, description) -> (x, y)``
    * ``TYPE_TEXT(observation) -> str``

    :meth:`next_action` routes the ``NEXT_ACTION`` objective and runs the chosen
    capabilities in order, returning the uniform :class:`Action`. The composition
    (planner feeds its description to the grounder) lives HERE, once, deterministic
    — not scattered across per-model loops.
    """

    def __init__(self, handlers: Dict[Capability, Callable[..., Any]]) -> None:
        self.handlers = dict(handlers)

    @property
    def capabilities(self) -> List[Capability]:
        return sorted(self.handlers, key=lambda c: c.value)

    def _require(self, cap: Capability) -> Callable[..., Any]:
        h = self.handlers.get(cap)
        if h is None:
            raise RouteError("no handler bound for capability %s" % cap.value)
        return h

    def plan_route(self, objective: Objective) -> Route:
        return route(objective, self.capabilities)

    def next_action(self, observation: Any) -> Action:
        """Produce the next uniform :class:`Action` from an observation.

        Runs whichever loop the capability set implies. A ``native`` route calls the
        step model and returns its Action verbatim. A ``composed`` route calls the
        planner for a target description, then the click grounder for the pixel, and
        assembles the Action — the click-only model made autonomous by the planner.
        """
        r = self.plan_route(Objective.NEXT_ACTION)
        if r.kind == "native":
            action = self._require(Capability.PREDICT_STEP)(observation)
            return self._as_action(action)
        # composed: plan -> click
        description = self._require(Capability.PLAN)(observation)
        if not description:
            return Action(kind="done", meta={"reason": "planner produced no target"})
        point = self._require(Capability.PREDICT_CLICK)(observation, description)
        return Action(kind="click", point=_as_point(point), description=str(description),
                      meta={"loop": "composed"})

    def click_target(self, observation: Any, description: str) -> Action:
        """Locate a described element, using whichever pointing capability exists."""
        r = self.plan_route(Objective.CLICK_TARGET)
        if r.steps[0] is Capability.PREDICT_CLICK:
            point = self._require(Capability.PREDICT_CLICK)(observation, description)
            return Action(kind="click", point=_as_point(point),
                          description=str(description), meta={"loop": r.kind})
        # a step model constrained to pointing
        action = self._require(Capability.PREDICT_STEP)(observation)
        return self._as_action(action)

    @staticmethod
    def _as_action(value: Any) -> Action:
        """Normalise a step model's return into an :class:`Action`.

        Accepts an Action verbatim, or a dict from a model adapter with ``kind`` and
        optional ``point``/``text``/``description``. Anything else is a bug and
        raises, because a driver must never receive an ambiguous action.
        """
        if isinstance(value, Action):
            return value
        if isinstance(value, dict):
            return Action(kind=str(value.get("kind", "other")),
                          point=_as_point(value.get("point")),
                          text=str(value.get("text", "")),
                          description=str(value.get("description", "")),
                          meta=dict(value.get("meta", {})))
        raise RouteError("predict_step handler returned %r; expected Action or dict"
                         % type(value))


def _as_point(value: Any) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    x, y = value
    return (int(x), int(y))
