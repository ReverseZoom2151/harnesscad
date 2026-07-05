"""Classify-then-route cost-control layer (HARNESS_BLUEPRINT.md sec.11, sec.13).

The harness talks to a single `LLM` (llm.base). This module is a *drop-in* `LLM`
that sits in front of several concrete models and decides, per request, which one
to spend on: cheap models for param edits / unit conversion / boilerplate, an
expensive reasoning model for spatial planning / constraint solving / assembly
(sec.13's "classify-then-route"). On error it walks a **sequential fallback**
chain (OpenRouter-style) to the next model and keeps going, and it keeps a
running cost/usage tally so spend is observable.

Everything here is provider-agnostic and lazy: a "route" is *any* object that
satisfies the `LLM` protocol, so nothing here needs a network or API keys — a
`MockLLM` is a perfectly valid route. Stdlib only; absolute imports.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

from llm.base import LLM, CompletionResult, Message, ToolSpec


# --- task taxonomy ---------------------------------------------------------
class TaskClass(Enum):
    """How much reasoning a request is expected to need — the routing key.

    CHEAP    param edits, unit conversion, boilerplate (fast/cheap model).
    STANDARD the ordinary case (a mid-tier model).
    HARD     spatial planning, constraint solving, assembly (reasoning model).
    """

    CHEAP = "cheap"
    STANDARD = "standard"
    HARD = "hard"


@runtime_checkable
class Classifier(Protocol):
    """Decides a `TaskClass` for a request from its messages and optional hints.

    `hints` is a free-form mapping the caller may pass (e.g. an explicit
    ``{"task_class": TaskClass.HARD}`` override, or upstream signals). Cheap and
    synchronous by contract — classification must never cost more than the call
    it is trying to save.
    """

    def classify(
        self, messages: List[Message], hints: Optional[Mapping[str, Any]] = None
    ) -> TaskClass:
        ...


# Cheap lexical signals. These are deliberately coarse: the classifier is a
# front door, not a planner. Order of evaluation is HARD-before-CHEAP so an
# ambiguous brief errs toward more capability (cheaper to over-provision once
# than to fail a hard task on a weak model and fall back).
_HARD_SIGNALS = (
    "spatial",
    "plan",
    "layout",
    "arrange",
    "constraint",
    "constrain",
    "solve",
    "assembly",
    "assemble",
    "mate",
    "interference",
    "clearance",
    "kinematic",
    "tolerance stack",
    "load path",
    "fea",
    "optimi",  # optimise / optimize / optimisation
    "pack",
)
_CHEAP_SIGNALS = (
    "param",
    "parameter",
    "unit conversion",
    "convert",
    "mm to",
    "cm to",
    "to inch",
    "inches",
    "boilerplate",
    "rename",
    "relabel",
    "tweak",
    "bump",
    "change the value",
    "set the value",
    "increase the",
    "decrease the",
    "adjust the",
    "resize",
    "rescale",
    "scale the",
)


def _text_of(messages: List[Message], hints: Optional[Mapping[str, Any]]) -> str:
    parts: List[str] = []
    for m in messages:
        content = getattr(m, "content", None)
        if content:
            parts.append(str(content))
    if hints:
        note = hints.get("note") or hints.get("text")
        if note:
            parts.append(str(note))
    return "\n".join(parts).lower()


class HeuristicClassifier(Classifier):
    """Default `Classifier`: bucket by cheap lexical signals in the request.

    Precedence: an explicit ``hints['task_class']`` override wins; otherwise HARD
    signals, then CHEAP signals, then STANDARD as the fallthrough.
    """

    def __init__(
        self,
        hard_signals: Tuple[str, ...] = _HARD_SIGNALS,
        cheap_signals: Tuple[str, ...] = _CHEAP_SIGNALS,
    ) -> None:
        self.hard_signals = tuple(hard_signals)
        self.cheap_signals = tuple(cheap_signals)

    def classify(
        self, messages: List[Message], hints: Optional[Mapping[str, Any]] = None
    ) -> TaskClass:
        if hints:
            override = hints.get("task_class")
            if isinstance(override, TaskClass):
                return override
            if isinstance(override, str):
                try:
                    return TaskClass(override.lower())
                except ValueError:
                    pass
        text = _text_of(messages, hints)
        if any(sig in text for sig in self.hard_signals):
            return TaskClass.HARD
        if any(sig in text for sig in self.cheap_signals):
            return TaskClass.CHEAP
        return TaskClass.STANDARD


# --- cost accounting -------------------------------------------------------
@dataclass(frozen=True)
class ModelPrice:
    """Per-1k-token prices for one model (input = prompt, output = completion)."""

    input_per_1k: float = 0.0
    output_per_1k: float = 0.0


@dataclass
class Usage:
    """Token usage for one call. Zeros when the provider reports nothing."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def count_tokens(text: str) -> int:
    """Rough token count (~4 chars/token). Monotonic in length — enough for a
    pre-flight estimate; real usage comes back from the provider."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def messages_tokens(messages: List[Message]) -> int:
    """Estimated prompt tokens for a message list."""
    total = 0
    for m in messages:
        content = getattr(m, "content", None)
        if content:
            total += count_tokens(str(content))
    return total


def usage_from_result(result: CompletionResult) -> Usage:
    """Best-effort usage extraction from a `CompletionResult`.

    Reads a provider ``usage`` block off ``result.raw`` when present (dict or
    object with ``prompt_tokens`` / ``completion_tokens``); otherwise falls back
    to estimating completion tokens from the returned text. Never raises.
    """
    raw = getattr(result, "raw", None)
    usage_obj: Any = None
    if isinstance(raw, Mapping):
        usage_obj = raw.get("usage")
    elif raw is not None:
        usage_obj = getattr(raw, "usage", None)

    def _pick(obj: Any, key: str) -> Optional[int]:
        if obj is None:
            return None
        val = obj.get(key) if isinstance(obj, Mapping) else getattr(obj, key, None)
        try:
            return int(val) if val is not None else None
        except (TypeError, ValueError):
            return None

    prompt = _pick(usage_obj, "prompt_tokens")
    completion = _pick(usage_obj, "completion_tokens")
    if completion is None:
        completion = count_tokens(getattr(result, "text", "") or "")
    return Usage(prompt_tokens=prompt or 0, completion_tokens=completion)


class CostTable:
    """Pluggable per-model price book with cost + pre-flight estimate helpers.

    Unknown models cost **0** and surface a note (never a crash) so an
    unpriced route degrades to "free but flagged" rather than breaking routing.
    """

    def __init__(
        self,
        prices: Optional[Mapping[str, Any]] = None,
        default: Optional[ModelPrice] = None,
    ) -> None:
        self._prices: Dict[str, ModelPrice] = {}
        for model, price in (prices or {}).items():
            self._prices[model] = self._coerce(price)
        self.default = default

    @staticmethod
    def _coerce(price: Any) -> ModelPrice:
        if isinstance(price, ModelPrice):
            return price
        if isinstance(price, Mapping):
            return ModelPrice(
                float(price.get("input_per_1k", 0.0)),
                float(price.get("output_per_1k", 0.0)),
            )
        if isinstance(price, (tuple, list)) and len(price) == 2:
            return ModelPrice(float(price[0]), float(price[1]))
        raise TypeError(f"unsupported price spec for a model: {price!r}")

    def price(self, model: str) -> Optional[ModelPrice]:
        return self._prices.get(model, self.default)

    def cost_of(self, model: str, usage: Usage) -> Tuple[float, Optional[str]]:
        """Return ``(dollar_cost, note)``; note is set only when unpriced."""
        price = self.price(model)
        if price is None:
            return 0.0, f"no price for model '{model}'; counted as $0"
        cost = (
            usage.prompt_tokens / 1000.0 * price.input_per_1k
            + usage.completion_tokens / 1000.0 * price.output_per_1k
        )
        return cost, None

    def estimate(self, model: str, messages: List[Message]) -> float:
        """Pre-flight (input-only) dollar estimate for sending `messages`."""
        price = self.price(model)
        if price is None:
            return 0.0
        return messages_tokens(messages) / 1000.0 * price.input_per_1k


# --- routing decisions -----------------------------------------------------
@dataclass
class RouteDecision:
    """A record of one routed call: what class, which model actually answered,
    how many fallbacks it took, and what it cost."""

    task_class: TaskClass
    model: str
    fallbacks_taken: int = 0
    cost: float = 0.0
    usage: Usage = field(default_factory=Usage)
    note: Optional[str] = None
    error: Optional[str] = None  # set when every model in the chain failed


class AllRoutesFailed(RuntimeError):
    """Raised when the primary and every fallback model raised."""


# --- the router (itself an LLM) --------------------------------------------
class RoutingLLM(LLM):
    """A drop-in `LLM` that classifies each request and dispatches to a mapped
    model, with a sequential fallback chain and a running cost/usage tally.

    Args:
        routes: ``{TaskClass: LLM}`` — the model to try first for each class.
        classifier: a `Classifier` (defaults to `HeuristicClassifier`).
        fallbacks: ordered `LLM`s tried, in turn, when the mapped model errors
            (OpenRouter-style sequential fallback). Applies to every class.
        cost_table: a `CostTable` for the spend tally (defaults to empty → $0).
        default_class: class used when a route for the classified class is
            missing (defaults to STANDARD, then any available route).

    Pass a per-call ``hints=`` mapping (popped before dispatch) to steer the
    classifier, e.g. ``complete(msgs, hints={"task_class": TaskClass.HARD})``.
    """

    def __init__(
        self,
        routes: Mapping[TaskClass, LLM],
        classifier: Optional[Classifier] = None,
        fallbacks: Optional[List[LLM]] = None,
        cost_table: Optional[CostTable] = None,
        default_class: TaskClass = TaskClass.STANDARD,
    ) -> None:
        if not routes:
            raise ValueError("RoutingLLM needs at least one route")
        self.routes: Dict[TaskClass, LLM] = dict(routes)
        self.classifier: Classifier = classifier or HeuristicClassifier()
        self.fallbacks: List[LLM] = list(fallbacks or [])
        self.cost_table: CostTable = cost_table or CostTable()
        self.default_class = default_class

        # Running tallies.
        self.decisions: List[RouteDecision] = []
        self._calls_per_class: Counter = Counter()
        self._fallbacks_taken = 0
        self._spend = 0.0

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def model_name(llm: LLM) -> str:
        """Best-effort human name for a route (its ``.model`` or its type)."""
        name = getattr(llm, "model", None)
        return str(name) if name else type(llm).__name__

    def _route_for(self, task_class: TaskClass) -> LLM:
        if task_class in self.routes:
            return self.routes[task_class]
        if self.default_class in self.routes:
            return self.routes[self.default_class]
        # Deterministic last resort: the lowest-cost class that exists.
        for tc in (TaskClass.CHEAP, TaskClass.STANDARD, TaskClass.HARD):
            if tc in self.routes:
                return self.routes[tc]
        # routes is non-empty (checked in __init__), so this is unreachable.
        return next(iter(self.routes.values()))

    def _chain(self, task_class: TaskClass) -> List[LLM]:
        primary = self._route_for(task_class)
        chain = [primary]
        for fb in self.fallbacks:
            if fb is not primary:
                chain.append(fb)
        return chain

    def classify(
        self, messages: List[Message], hints: Optional[Mapping[str, Any]] = None
    ) -> TaskClass:
        return self.classifier.classify(messages, hints)

    def estimate(
        self, messages: List[Message], hints: Optional[Mapping[str, Any]] = None
    ) -> float:
        """Pre-flight cost estimate for the model this request would route to.

        Monotonic in message size (more/larger messages ⇒ ≥ estimate)."""
        task_class = self.classify(messages, hints)
        model = self.model_name(self._route_for(task_class))
        return self.cost_table.estimate(model, messages)

    def _record(self, decision: RouteDecision) -> None:
        self.decisions.append(decision)
        self._calls_per_class[decision.task_class] += 1
        self._fallbacks_taken += decision.fallbacks_taken
        self._spend += decision.cost

    # -- LLM protocol ----------------------------------------------------
    def complete(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSpec]] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        **opts: Any,
    ) -> CompletionResult:
        hints = opts.pop("hints", None)
        task_class = self.classify(messages, hints)
        chain = self._chain(task_class)

        last_error: Optional[Exception] = None
        for idx, llm in enumerate(chain):
            try:
                result = llm.complete(messages, tools, response_schema, **opts)
            except Exception as e:  # sequential fallback: try the next model
                last_error = e
                continue
            usage = usage_from_result(result)
            cost, note = self.cost_table.cost_of(self.model_name(llm), usage)
            self._record(
                RouteDecision(
                    task_class=task_class,
                    model=self.model_name(llm),
                    fallbacks_taken=idx,
                    cost=cost,
                    usage=usage,
                    note=note,
                )
            )
            return result

        # Whole chain failed — record the miss and raise.
        self._record(
            RouteDecision(
                task_class=task_class,
                model=self.model_name(chain[-1]),
                fallbacks_taken=len(chain) - 1,
                error=str(last_error),
            )
        )
        raise AllRoutesFailed(
            f"all {len(chain)} models failed for {task_class.value} request; "
            f"last error: {last_error}"
        ) from last_error

    def stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSpec]] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        **opts: Any,
    ) -> Iterator[str]:
        hints = opts.pop("hints", None)
        task_class = self.classify(messages, hints)
        chain = self._chain(task_class)

        last_error: Optional[Exception] = None
        for idx, llm in enumerate(chain):
            try:
                it = iter(llm.stream(messages, tools, response_schema, **opts))
                first = next(it)  # force the first chunk so early errors fall back
            except StopIteration:
                # Empty but successful stream.
                self._record(
                    RouteDecision(task_class=task_class, model=self.model_name(llm), fallbacks_taken=idx)
                )
                return
            except Exception as e:
                last_error = e
                continue
            # Success: usage isn't known for streams, so cost is recorded as $0.
            self._record(
                RouteDecision(task_class=task_class, model=self.model_name(llm), fallbacks_taken=idx)
            )
            yield first
            for chunk in it:
                yield chunk
            return

        self._record(
            RouteDecision(
                task_class=task_class,
                model=self.model_name(chain[-1]),
                fallbacks_taken=len(chain) - 1,
                error=str(last_error),
            )
        )
        raise AllRoutesFailed(
            f"all {len(chain)} models failed for {task_class.value} stream; "
            f"last error: {last_error}"
        ) from last_error

    # -- observability ---------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """Routing + spend summary: calls per class, fallbacks taken, spend."""
        return {
            "total_calls": len(self.decisions),
            "calls_per_class": {
                tc.value: self._calls_per_class.get(tc, 0) for tc in TaskClass
            },
            "fallbacks_taken": self._fallbacks_taken,
            "estimated_spend": self._spend,
            "notes": [d.note for d in self.decisions if d.note],
            "errors": [d.error for d in self.decisions if d.error],
        }
