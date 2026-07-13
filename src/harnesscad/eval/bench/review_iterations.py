"""Review-loop marginal utility and plateau analysis."""

from __future__ import annotations


def review_iteration_report(traces, *, plateau_tolerance=1e-9):
    grouped = {}
    first_valid = []
    for trace in traces:
        for step in trace:
            grouped.setdefault(step["iteration"], []).append(step)
        valid = next((step["iteration"] for step in trace if step["valid"]), None)
        first_valid.append(valid)
    rows, previous = [], None
    for iteration, items in sorted(grouped.items()):
        validity = sum(item["valid"] for item in items) / len(items)
        distances = [item["distance"] for item in items if item.get("distance") is not None]
        mean_distance = sum(distances)/len(distances) if distances else None
        marginal = None if previous is None else validity - previous
        rows.append({"iteration": iteration, "validity": validity,
                     "invalidity": 1-validity, "mean_distance": mean_distance,
                     "marginal_validity": marginal,
                     "cost": sum(float(item.get("cost", 0)) for item in items)})
        previous = validity
    plateau = next((row["iteration"] for row in rows[1:]
                    if row["marginal_validity"] is not None
                    and row["marginal_validity"] <= plateau_tolerance), None)
    return {"rows": tuple(rows), "first_valid": tuple(first_valid), "plateau": plateau}
