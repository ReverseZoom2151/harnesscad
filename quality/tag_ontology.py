"""Hierarchical CAD tags, heterogeneous graphs, and deterministic retrieval."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Tag:
    id: str
    parent: str | None = None
    aliases: tuple[str, ...] = ()
    label: str | None = None


class TagOntology:
    """Validated tag hierarchy with unambiguous alias resolution."""

    def __init__(self, tags: Iterable[Tag]) -> None:
        values = tuple(tags)
        self._tags = {tag.id: tag for tag in values}
        if len(self._tags) != len(values):
            raise ValueError("duplicate tag id")
        for tag in values:
            if not tag.id:
                raise ValueError("tag id cannot be empty")
            if tag.parent is not None and tag.parent not in self._tags:
                raise ValueError(f"unknown parent {tag.parent!r} for tag {tag.id!r}")

        self._names: dict[str, str] = {}
        for tag in values:
            for name in (tag.id, *tag.aliases):
                key = self._key(name)
                owner = self._names.get(key)
                if owner is not None and owner != tag.id:
                    raise ValueError(f"ambiguous alias {name!r}: {owner!r}, {tag.id!r}")
                self._names[key] = tag.id

        for tag in values:
            seen: set[str] = set()
            current: str | None = tag.id
            while current is not None:
                if current in seen:
                    raise ValueError(f"cycle in tag hierarchy at {current!r}")
                seen.add(current)
                current = self._tags[current].parent

    @staticmethod
    def _key(value: str) -> str:
        return " ".join(value.strip().casefold().replace("_", " ").split())

    @property
    def tags(self) -> tuple[Tag, ...]:
        return tuple(sorted(self._tags.values(), key=lambda tag: tag.id))

    def resolve(self, name: str) -> str:
        try:
            return self._names[self._key(name)]
        except KeyError as exc:
            raise KeyError(f"unknown tag: {name}") from exc

    def ancestors(self, name: str, *, include_self: bool = False) -> tuple[str, ...]:
        current = self.resolve(name)
        result = [current] if include_self else []
        current = self._tags[current].parent
        while current is not None:
            result.append(current)
            current = self._tags[current].parent
        return tuple(result)

    def children(self, name: str, *, recursive: bool = False) -> tuple[str, ...]:
        root = self.resolve(name)
        direct = sorted(tag.id for tag in self._tags.values() if tag.parent == root)
        if not recursive:
            return tuple(direct)
        result: list[str] = []
        pending = list(direct)
        while pending:
            child = pending.pop(0)
            result.append(child)
            pending.extend(self.children(child))
        return tuple(result)

    def expand(self, names: Iterable[str]) -> frozenset[str]:
        expanded: set[str] = set()
        for name in names:
            expanded.update(self.ancestors(name, include_self=True))
        return frozenset(expanded)


@dataclass(frozen=True, order=True)
class GraphEdge:
    source_type: str
    source: str
    relation: str
    target_type: str
    target: str


@dataclass(frozen=True)
class RetrievalResult:
    model_id: str
    score: float
    exact_tags: tuple[str, ...]
    hierarchy_matches: tuple[str, ...]
    missing_query_tags: tuple[str, ...]
    explanation: str


@dataclass(frozen=True)
class TagMotif:
    tags: tuple[str, ...]
    support: int
    model_ids: tuple[str, ...]
    explanation: str


class ModelTagGraph:
    """Heterogeneous model-tag-category graph with multi-label assignments."""

    def __init__(self, ontology: TagOntology) -> None:
        self.ontology = ontology
        self._model_tags: dict[str, set[str]] = {}
        self._model_categories: dict[str, set[str]] = {}
        self._tag_categories: dict[str, set[str]] = {}

    def add_model(
        self,
        model_id: str,
        *,
        tags: Iterable[str] = (),
        categories: Iterable[str] = (),
    ) -> None:
        if not model_id:
            raise ValueError("model id cannot be empty")
        resolved = {self.ontology.resolve(tag) for tag in tags}
        self._model_tags.setdefault(model_id, set()).update(resolved)
        self._model_categories.setdefault(model_id, set()).update(
            str(category) for category in categories
        )

    def assign_tags(self, model_id: str, tags: Iterable[str]) -> None:
        if model_id not in self._model_tags:
            raise KeyError(f"unknown model: {model_id}")
        self._model_tags[model_id].update(self.ontology.resolve(tag) for tag in tags)

    def categorize_tag(self, tag: str, category: str) -> None:
        tag_id = self.ontology.resolve(tag)
        self._tag_categories.setdefault(tag_id, set()).add(str(category))

    @property
    def edges(self) -> tuple[GraphEdge, ...]:
        result: list[GraphEdge] = []
        for model, tags in self._model_tags.items():
            result.extend(GraphEdge("model", model, "has-tag", "tag", tag) for tag in tags)
        for model, categories in self._model_categories.items():
            result.extend(
                GraphEdge("model", model, "in-category", "category", category)
                for category in categories
            )
        for tag, categories in self._tag_categories.items():
            result.extend(
                GraphEdge("tag", tag, "in-category", "category", category)
                for category in categories
            )
        for tag in self.ontology.tags:
            if tag.parent:
                result.append(GraphEdge("tag", tag.id, "is-a", "tag", tag.parent))
        return tuple(sorted(result))

    def retrieve(
        self,
        query_tags: Iterable[str],
        *,
        limit: int | None = None,
        category: str | None = None,
    ) -> tuple[RetrievalResult, ...]:
        query = {self.ontology.resolve(tag) for tag in query_tags}
        if not query:
            raise ValueError("at least one query tag is required")
        expanded_query = self.ontology.expand(query)
        results: list[RetrievalResult] = []
        for model_id, tags in self._model_tags.items():
            if category is not None and category not in self._model_categories.get(model_id, set()):
                continue
            expanded_model = self.ontology.expand(tags)
            exact = query & tags
            shared = expanded_query & expanded_model
            hierarchy = shared - exact
            union = expanded_query | expanded_model
            score = (len(exact) + 0.5 * len(hierarchy)) / max(1, len(union))
            missing = query - expanded_model
            explanation = (
                f"{len(exact)} exact and {len(hierarchy)} ancestor-level matches; "
                f"{len(missing)} query tags unmatched"
            )
            results.append(
                RetrievalResult(
                    model_id,
                    round(score, 6),
                    tuple(sorted(exact)),
                    tuple(sorted(hierarchy)),
                    tuple(sorted(missing)),
                    explanation,
                )
            )
        results.sort(key=lambda item: (-item.score, item.model_id))
        return tuple(results[:limit] if limit is not None else results)

    def frequent_motifs(
        self,
        *,
        min_support: int | float = 2,
        min_size: int = 2,
        max_size: int = 3,
    ) -> tuple[TagMotif, ...]:
        if min_size < 1 or max_size < min_size:
            raise ValueError("invalid motif size range")
        model_count = len(self._model_tags)
        if isinstance(min_support, float):
            if not 0 < min_support <= 1:
                raise ValueError("fractional support must be in (0, 1]")
            threshold = max(1, int(model_count * min_support + 0.999999))
        else:
            if min_support < 1:
                raise ValueError("support must be positive")
            threshold = min_support

        owners: dict[tuple[str, ...], list[str]] = {}
        for model_id, tags in sorted(self._model_tags.items()):
            ordered = sorted(tags)
            for size in range(min_size, min(max_size, len(ordered)) + 1):
                for motif in combinations(ordered, size):
                    owners.setdefault(motif, []).append(model_id)
        motifs = [
            TagMotif(
                tags,
                len(models),
                tuple(models),
                f"tags {', '.join(tags)} co-occur in {len(models)}/{model_count} models",
            )
            for tags, models in owners.items()
            if len(models) >= threshold
        ]
        motifs.sort(key=lambda item: (-item.support, -len(item.tags), item.tags))
        return tuple(motifs)
