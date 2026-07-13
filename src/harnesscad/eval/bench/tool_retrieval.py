"""Required-tool recall and context-cost evaluation."""

from __future__ import annotations


def evaluate_tool_retrieval(cases, retrieve, *, k=4, token_count=lambda text: len(text.split())):
    rows = []
    for case in cases:
        returned = tuple(retrieve(case["task"], k))
        required = set(case.get("required", ()))
        names = [getattr(item, "name", str(item)) for item in returned]
        hit = required & set(names)
        costs = [token_count(getattr(item, "summary", str(item))) for item in returned]
        rows.append({"task": case["task"], "returned": tuple(names),
                     "recall": len(hit) / len(required) if required else 1.0,
                     "irrelevant": len(set(names) - required),
                     "tokens": sum(costs)})
    return {"rows": tuple(rows),
            "recall_at_k": sum(row["recall"] for row in rows) / len(rows) if rows else 1.0}
