"""errornotebook_store — a corrective "Error Notebook" memory for part retrieval.

Paper: "Error Notebook-Guided, Training-Free Part Retrieval in 3D CAD
Assemblies via Vision-Language Models" (ICLR 2026). The Error Notebook is an
inference-time adaptation mechanism: instead of fine-tuning weights, past
retrieval *mistakes* are recorded as corrected reasoning trajectories and,
for a new query, the most specification-similar past errors are retrieved and
supplied as few-shot exemplars to steer the model away from repeating them.

This module implements the deterministic, locally-buildable core of that idea:

  - :class:`ErrorNotebookEntry` — the {specification, part_descriptions,
    wrong_answer, ground_truth, corrected_cot, insight} triplet-plus schema.
  - :class:`ErrorNotebook` — an append-only store with JSON persistence and
    specification-similarity retrieval (top-n, with leak-safe exclusion of the
    current query, per Eq. 5 in the paper).

The VLM that *produces* the corrected chain-of-thought is external/research
(skipped here per the campaign rules). Everything below — the schema, the
persistence, the similarity retrieval, the known-wrong index — is deterministic
and dependency-free (stdlib only).
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


def char_similarity(a: str, b: str) -> float:
    """Normalised character-level similarity in [0, 1].

    The paper's default retriever is a "character-level similarity retriever"
    (App. A.2). difflib's ratio is a deterministic, stdlib character-level
    matching score — exactly that family.
    """
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def jaccard_similarity(a: str, b: str) -> float:
    """Token-level Jaccard overlap in [0, 1] (the paper's App. A.2 variant)."""
    qa, da = set(_tokens(a)), set(_tokens(b))
    if not qa and not da:
        return 1.0
    if not qa or not da:
        return 0.0
    inter = len(qa & da)
    union = len(qa | da)
    return inter / union if union else 0.0


def _normalize_answer(ans: Sequence[str]) -> Tuple[str, ...]:
    """Order-insensitive, de-duplicated tuple of filenames for set comparison."""
    return tuple(sorted({str(x).strip() for x in ans if str(x).strip()}))


@dataclass
class ErrorNotebookEntry:
    """One corrective record: a past retrieval mistake and its correction.

    Fields mirror the paper's specification-CoT-answer triplet, extended with
    the *wrong* answer and a distilled *insight* so the notebook doubles as a
    known-wrong index for re-ranking.
    """

    specification: str                       # S — the query spec sentence
    ground_truth: Sequence[str]              # P*(gt) — correct filenames
    wrong_answer: Sequence[str] = ()         # a_b — the earlier wrong prediction
    part_descriptions: Dict[str, str] = field(default_factory=dict)  # D
    corrected_cot: str = ""                  # R_corr — corrected trajectory text
    insight: str = ""                        # human/derived one-line takeaway
    entry_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "specification": self.specification,
            "ground_truth": list(self.ground_truth),
            "wrong_answer": list(self.wrong_answer),
            "part_descriptions": dict(self.part_descriptions),
            "corrected_cot": self.corrected_cot,
            "insight": self.insight,
            "entry_id": self.entry_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ErrorNotebookEntry":
        return cls(
            specification=d["specification"],
            ground_truth=list(d.get("ground_truth", [])),
            wrong_answer=list(d.get("wrong_answer", [])),
            part_descriptions=dict(d.get("part_descriptions", {})),
            corrected_cot=d.get("corrected_cot", ""),
            insight=d.get("insight", ""),
            entry_id=d.get("entry_id"),
        )

    def few_shot_block(self, include_cot: bool = True) -> str:
        """Render this entry as a few-shot exemplar prompt block.

        ``include_cot`` toggles the paper's CoT vs Non-CoT exemplar groups
        (Table 2): with CoT the corrected reasoning is shown; without, only the
        final corrected answer is shown.
        """
        lines = ["Specification: " + self.specification]
        if self.part_descriptions:
            lines.append("Part descriptions:")
            for fn, desc in self.part_descriptions.items():
                lines.append(f"  {fn}: {desc}")
        if include_cot and self.corrected_cot:
            lines.append(self.corrected_cot.rstrip())
        answer = ";".join(_normalize_answer(self.ground_truth))
        lines.append("Final Answer: " + answer)
        return "\n".join(lines)

    def known_wrong(self) -> Tuple[str, ...]:
        """Filenames this entry recorded as an incorrect answer for its spec."""
        return _normalize_answer(self.wrong_answer)


class ErrorNotebook:
    """Append-only corrective memory with similarity retrieval + persistence."""

    def __init__(self, scorer: str = "char") -> None:
        if scorer not in ("char", "jaccard"):
            raise ValueError("scorer must be 'char' or 'jaccard'")
        self.scorer = scorer
        self._entries: List[ErrorNotebookEntry] = []
        self._auto = 0

    # -- construction ------------------------------------------------------
    def add(self, entry: ErrorNotebookEntry) -> ErrorNotebookEntry:
        """Add one corrected trajectory. Assigns a stable id if missing."""
        if entry.entry_id is None:
            entry.entry_id = f"en{self._auto}"
        self._auto += 1
        self._entries.append(entry)
        return entry

    def record_mistake(
        self,
        specification: str,
        wrong_answer: Sequence[str],
        ground_truth: Sequence[str],
        part_descriptions: Optional[Dict[str, str]] = None,
        corrected_cot: str = "",
        insight: str = "",
    ) -> ErrorNotebookEntry:
        """Convenience: build + add an entry from a corrected mistake."""
        return self.add(ErrorNotebookEntry(
            specification=specification,
            ground_truth=list(ground_truth),
            wrong_answer=list(wrong_answer),
            part_descriptions=dict(part_descriptions or {}),
            corrected_cot=corrected_cot,
            insight=insight,
        ))

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> List[ErrorNotebookEntry]:
        return list(self._entries)

    def _score(self, a: str, b: str) -> float:
        return char_similarity(a, b) if self.scorer == "char" else jaccard_similarity(a, b)

    # -- retrieval ---------------------------------------------------------
    def retrieve(
        self,
        specification: str,
        n: int = 2,
        exclude_id: Optional[str] = None,
        exclude_spec_exact: bool = True,
    ) -> List[Tuple[ErrorNotebookEntry, float]]:
        """Top-``n`` most specification-similar entries (Eq. 5).

        Leak-safe: the current query instance is never returned. We exclude by
        ``exclude_id`` and, when ``exclude_spec_exact`` is set, any entry whose
        specification is byte-identical to the query (the same instance under a
        different id). Ties break deterministically by ``entry_id``.
        """
        scored: List[Tuple[ErrorNotebookEntry, float]] = []
        for e in self._entries:
            if exclude_id is not None and e.entry_id == exclude_id:
                continue
            if exclude_spec_exact and e.specification == specification:
                continue
            scored.append((e, self._score(specification, e.specification)))
        scored.sort(key=lambda t: (-t[1], str(t[0].entry_id)))
        if n < 0:
            return scored
        return scored[:n]

    def few_shot_prompt(
        self,
        specification: str,
        n: int = 2,
        include_cot: bool = True,
        exclude_id: Optional[str] = None,
    ) -> str:
        """Concatenated few-shot exemplar blocks for the top-``n`` entries."""
        hits = self.retrieve(specification, n=n, exclude_id=exclude_id)
        blocks = [e.few_shot_block(include_cot=include_cot) for e, _ in hits]
        return "\n\n".join(blocks)

    def known_wrong_for(
        self,
        specification: str,
        n: int = 5,
        min_similarity: float = 0.0,
        exclude_id: Optional[str] = None,
    ) -> Dict[Tuple[str, ...], float]:
        """Map each known-wrong answer-set to the max spec-similarity that flagged it.

        Consulted by the re-ranker to down-weight answers that similar past
        queries got wrong. Similarity acts as the confidence of the flag.
        """
        out: Dict[Tuple[str, ...], float] = {}
        for e, sim in self.retrieve(specification, n=n, exclude_id=exclude_id):
            if sim < min_similarity:
                continue
            kw = e.known_wrong()
            if not kw:
                continue
            if kw not in out or sim > out[kw]:
                out[kw] = sim
        return out

    # -- persistence -------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(
            {"scorer": self.scorer, "entries": [e.to_dict() for e in self._entries]},
            indent=2, sort_keys=True,
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())

    @classmethod
    def from_json(cls, text: str) -> "ErrorNotebook":
        data = json.loads(text)
        nb = cls(scorer=data.get("scorer", "char"))
        for d in data.get("entries", []):
            nb.add(ErrorNotebookEntry.from_dict(d))
        return nb

    @classmethod
    def load(cls, path: str) -> "ErrorNotebook":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_json(fh.read())
