"""Source-aware synthetic/wild benchmark manifests and leakage audits."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import re
from typing import Iterable, Mapping


_SPACE = re.compile(r"\s+")


def normalized_prompt(prompt: str) -> str:
    return _SPACE.sub(" ", prompt.casefold()).strip()


def prompt_digest(prompt: str) -> str:
    return hashlib.sha256(normalized_prompt(prompt).encode()).hexdigest()


@dataclass(frozen=True)
class SplitEntry:
    id: str
    prompt: str
    split: str
    source: str
    category: str
    instruction_type: str
    length_bin: str

    def __post_init__(self) -> None:
        if self.split not in {"sim", "wild"}:
            raise ValueError("split must be sim or wild")
        if not all((self.id, self.prompt.strip(), self.source)):
            raise ValueError("id, prompt and source are required")

    @property
    def digest(self) -> str:
        return prompt_digest(self.prompt)


@dataclass(frozen=True)
class SplitAudit:
    duplicate_ids: tuple[str, ...]
    duplicate_prompts: tuple[str, ...]
    cross_split_leakage: tuple[str, ...]
    quota_shortfalls: Mapping[str, int]

    @property
    def ok(self) -> bool:
        return not (
            self.duplicate_ids or self.duplicate_prompts
            or self.cross_split_leakage or self.quota_shortfalls
        )


def audit_splits(
    entries: Iterable[SplitEntry],
    *,
    quotas: Mapping[str, int] | None = None,
) -> SplitAudit:
    items = tuple(entries)
    ids = Counter(item.id for item in items)
    digests = Counter(item.digest for item in items)
    digest_splits: dict[str, set[str]] = {}
    for item in items:
        digest_splits.setdefault(item.digest, set()).add(item.split)
    counts = Counter(item.split for item in items)
    shortfalls = {
        split: required - counts[split]
        for split, required in sorted((quotas or {}).items())
        if counts[split] < required
    }
    return SplitAudit(
        tuple(sorted(key for key, count in ids.items() if count > 1)),
        tuple(sorted(key for key, count in digests.items() if count > 1)),
        tuple(sorted(key for key, splits in digest_splits.items() if len(splits) > 1)),
        shortfalls,
    )
