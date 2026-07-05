"""Deterministic tool knowledge and task-specific dispatch.

The module captures the useful 3D-GPT pattern without requiring an LLM:
tools carry compact documentation, contextual prerequisites, and worked
examples; a catalog retrieves only the smallest useful set for a task.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


_WORDS = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class ToolExample:
    intent: str
    arguments: Mapping[str, object]
    outcome: str

    def __post_init__(self) -> None:
        if not self.intent.strip() or not self.outcome.strip():
            raise ValueError("tool examples require intent and outcome")


@dataclass(frozen=True)
class ToolKnowledgeCard:
    """Agent-facing knowledge for one tool, normally named after a CISP op."""

    name: str
    summary: str
    required_context: tuple[str, ...] = ()
    questions: Mapping[str, str] = field(default_factory=dict)
    examples: tuple[ToolExample, ...] = ()
    keywords: tuple[str, ...] = ()
    prerequisites: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.summary.strip():
            raise ValueError("tool cards require a name and summary")
        missing_questions = set(self.required_context) - set(self.questions)
        if missing_questions:
            raise ValueError(
                "missing contextual questions for: " + ", ".join(sorted(missing_questions))
            )

    def missing_context(self, context: Mapping[str, object]) -> tuple[str, ...]:
        return tuple(
            key for key in self.required_context
            if key not in context or context[key] is None or context[key] == ""
        )

    def conceptualize(self, context: Mapping[str, object]) -> "ToolConcept":
        missing = self.missing_context(context)
        return ToolConcept(
            tool=self.name,
            ready=not missing,
            rationale=self.summary,
            missing=missing,
            questions=tuple(self.questions[key] for key in missing),
            prerequisites=self.prerequisites,
        )


@dataclass(frozen=True)
class ToolConcept:
    tool: str
    ready: bool
    rationale: str
    missing: tuple[str, ...]
    questions: tuple[str, ...]
    prerequisites: tuple[str, ...]


@dataclass(frozen=True)
class DispatchPlan:
    task: str
    cards: tuple[ToolKnowledgeCard, ...]
    concepts: tuple[ToolConcept, ...]

    @property
    def ready_tools(self) -> tuple[str, ...]:
        return tuple(c.tool for c in self.concepts if c.ready)

    @property
    def questions(self) -> tuple[str, ...]:
        return tuple(q for concept in self.concepts for q in concept.questions)


class ToolKnowledgeCatalog:
    """Registry and deterministic minimal-set retriever."""

    def __init__(self, cards: Iterable[ToolKnowledgeCard] = ()) -> None:
        self._cards: dict[str, ToolKnowledgeCard] = {}
        for card in cards:
            self.register(card)

    def register(self, card: ToolKnowledgeCard) -> None:
        if card.name in self._cards:
            raise ValueError(f"duplicate tool card: {card.name}")
        self._cards[card.name] = card

    def get(self, name: str) -> ToolKnowledgeCard:
        return self._cards[name]

    def retrieve(
        self, task: str, *, limit: int = 4, required_tools: Sequence[str] = ()
    ) -> tuple[ToolKnowledgeCard, ...]:
        """Return a stable, relevance-ranked, bounded set of cards.

        Explicitly required tools are retained first. Remaining cards must
        overlap the task lexically, preventing an irrelevant full-catalog dump.
        """
        if limit < 1:
            raise ValueError("limit must be positive")
        unknown = set(required_tools) - self._cards.keys()
        if unknown:
            raise KeyError(f"unknown required tools: {', '.join(sorted(unknown))}")
        task_words = set(_WORDS.findall(task.lower()))
        chosen = list(dict.fromkeys(required_tools))
        ranked: list[tuple[int, str]] = []
        for name, card in self._cards.items():
            if name in chosen:
                continue
            terms = set(_WORDS.findall(" ".join((name, card.summary, *card.keywords)).lower()))
            score = len(task_words & terms)
            if score:
                ranked.append((score, name))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        chosen.extend(name for _, name in ranked[: max(0, limit - len(chosen))])
        return tuple(self._cards[name] for name in chosen[:limit])

    def dispatch(
        self,
        task: str,
        context: Mapping[str, object],
        *,
        limit: int = 4,
        required_tools: Sequence[str] = (),
    ) -> DispatchPlan:
        cards = self.retrieve(task, limit=limit, required_tools=required_tools)
        return DispatchPlan(
            task=task,
            cards=cards,
            concepts=tuple(card.conceptualize(context) for card in cards),
        )


def default_cisp_cards() -> tuple[ToolKnowledgeCard, ...]:
    """Small reference catalog demonstrating cards over stable CISP op names."""
    return (
        ToolKnowledgeCard(
            name="new_sketch",
            summary="Create a sketch on a datum plane.",
            required_context=("plane",),
            questions={"plane": "Which datum plane should contain the sketch?"},
            keywords=("profile", "draw", "sketch"),
            examples=(ToolExample("start an XY profile", {"plane": "XY"}, "empty XY sketch"),),
        ),
        ToolKnowledgeCard(
            name="extrude",
            summary="Turn a closed sketch profile into a prismatic solid.",
            required_context=("sketch", "distance"),
            questions={
                "sketch": "Which closed sketch should be extruded?",
                "distance": "What extrusion distance and direction are required?",
            },
            keywords=("plate", "prismatic", "thickness", "solid"),
            prerequisites=("new_sketch",),
            examples=(
                ToolExample(
                    "make a 10 mm plate", {"sketch": "plate_profile", "distance": 10},
                    "prismatic plate solid",
                ),
            ),
        ),
        ToolKnowledgeCard(
            name="hole",
            summary="Cut a semantic through or blind hole into a solid.",
            required_context=("diameter", "location"),
            questions={
                "diameter": "What hole diameter is required?",
                "location": "Where is the hole center located?",
            },
            keywords=("drill", "bore", "fastener", "counterbore", "countersink"),
            prerequisites=("extrude",),
            examples=(
                ToolExample(
                    "add an M6 clearance hole",
                    {"diameter": 6.6, "location": (20, 20), "through": True},
                    "semantic through hole",
                ),
            ),
        ),
        ToolKnowledgeCard(
            name="fillet",
            summary="Round selected solid edges to a specified radius.",
            required_context=("edges", "radius"),
            questions={
                "edges": "Which edges should be rounded?",
                "radius": "What fillet radius is required?",
            },
            keywords=("round", "radius", "edge"),
            prerequisites=("extrude",),
            examples=(
                ToolExample("round outer edges", {"edges": ("e1",), "radius": 2}, "rounded edge"),
            ),
        ),
    )
