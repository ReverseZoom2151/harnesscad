"""Pre/post fine-tuning capability retention and plasticity matrix."""

from __future__ import annotations


def capability_retention(cases, before, after):
    rows = []
    for case in cases:
        pre, post = float(before(case)), float(after(case))
        rows.append({"id": case["id"], "operation": case["operation"],
                     "prompt_kind": case.get("prompt_kind", "abstract"),
                     "before": pre, "after": post, "delta": post - pre,
                     "retained": post >= pre})
    retention = sum(row["retained"] for row in rows) / len(rows) if rows else None
    return {"rows": tuple(rows), "retention_rate": retention,
            "mean_delta": (sum(row["delta"] for row in rows) / len(rows)
                           if rows else None)}
