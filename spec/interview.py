"""interview — plan the clarifying questions a brief still needs answered.

The blueprint's "requirements interview -> defined specs": given a brief (or an
already-formalised :class:`spec.formalize.RequirementSet`), work out which
high-value fields are *under-specified* — material, quantity, tolerance,
envelope (overall dimensions), load — and produce a ranked list of targeted
questions to close those gaps.

This is pure planning: no side effects, no I/O. An injected
:class:`llm.base.LLM` is used only to *rephrase* the heuristic questions more
naturally; without one, the deterministic default templates are returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

from spec.formalize import RequirementSet, formalize


@dataclass
class Question:
    """One clarifying question. Lower ``priority`` sorts first (more urgent)."""

    field: str
    text: str
    priority: int = 100

    def __str__(self) -> str:
        return self.text


# Ordered most- to least-urgent. Each entry: (field, priority, template).
_TEMPLATES = [
    ("material", 1,
     "What material should the part be made from (e.g. aluminium, ABS, steel)?"),
    ("quantity", 2,
     "How many of each feature are required (e.g. number of mounting holes), "
     "and of what kind?"),
    ("tolerance", 3,
     "What dimensional tolerance is acceptable (e.g. +/- 0.1 mm)?"),
    ("envelope", 4,
     "What are the overall dimensions or bounding envelope "
     "(length x width x height)?"),
    ("load", 5,
     "What loads, forces or stresses must the part withstand in use?"),
]


class RequirementsInterview:
    """Plan the targeted clarifying questions for an under-specified brief."""

    def __init__(self, llm=None) -> None:
        self.llm = llm

    # -- gap analysis ------------------------------------------------------- #
    def missing_fields(self, brief_or_reqset: Union[str, RequirementSet]
                       ) -> List[str]:
        """Return the under-specified field names, ranked most-urgent first."""
        reqset, brief_text = self._coerce(brief_or_reqset)
        low = brief_text.lower()

        present = {
            "material": reqset.has_kind("material"),
            "quantity": reqset.has_kind("count"),
            "tolerance": (reqset.has_kind("tolerance")
                          or any(r.tolerance is not None
                                 for r in reqset.by_kind("dimension"))),
            "envelope": (reqset.has_kind("dimension")
                         or reqset.has_kind("envelope")),
            # A brief that never mentions a load/force implies the gap.
            "load": any(w in low for w in
                        ("load", "force", "stress", "weight", "pressure",
                         "torque", "structural")),
        }
        return [f for f, _p, _t in _TEMPLATES if not present.get(f, False)]

    # -- question planning -------------------------------------------------- #
    def next_questions(self, brief_or_reqset: Union[str, RequirementSet],
                       k: int = 3) -> List[Question]:
        """Return up to ``k`` ranked :class:`Question` objects for the gaps."""
        missing = set(self.missing_fields(brief_or_reqset))
        questions = [
            Question(field=f, text=t, priority=p)
            for f, p, t in _TEMPLATES if f in missing
        ]
        questions.sort(key=lambda q: q.priority)
        if k is not None and k >= 0:
            questions = questions[:k]
        if self.llm is not None and questions:
            self._rephrase(brief_or_reqset, questions)
        return questions

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def _coerce(brief_or_reqset: Union[str, RequirementSet]):
        if isinstance(brief_or_reqset, RequirementSet):
            return brief_or_reqset, brief_or_reqset.description or ""
        brief = str(brief_or_reqset or "")
        return formalize(brief), brief

    def _rephrase(self, brief_or_reqset, questions: List[Question]) -> None:
        """Optionally sharpen question wording with the LLM. Best-effort: any
        failure leaves the deterministic templates untouched."""
        from llm.base import Message  # local import: llm layer is optional

        _, brief_text = self._coerce(brief_or_reqset)
        for q in questions:
            messages = [
                Message("system",
                        "Rephrase the clarifying question to be specific to the "
                        "brief. Return only the question, one line."),
                Message("user", f"Brief: {brief_text}\nField: {q.field}\n"
                                f"Question: {q.text}"),
            ]
            try:
                result = self.llm.complete(messages)
                text = (getattr(result, "text", "") or "").strip()
            except Exception:  # noqa: BLE001 - keep the template on any failure
                text = ""
            if text:
                q.text = text.splitlines()[0].strip()
