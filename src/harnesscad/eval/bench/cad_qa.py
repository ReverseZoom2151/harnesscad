"""Evidence-grounded CAD question-answer grading."""

from __future__ import annotations


def grade_answer(answer, expected, *, observation_fields=()):
    value = str(answer.get("answer", "")).strip().casefold()
    target = str(expected).strip().casefold()
    citations = tuple(answer.get("evidence", ()))
    unknown = tuple(sorted(set(citations) - set(observation_fields)))
    abstained = value in {"", "unknown", "abstain"}
    return {"correct": not abstained and value == target and not unknown,
            "abstained": abstained, "unknown_evidence": unknown,
            "grounded": bool(citations) and not unknown}


def qa_accuracy(rows):
    rows = tuple(rows)
    return sum(bool(row.get("correct")) for row in rows) / len(rows) if rows else None
