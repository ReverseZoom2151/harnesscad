"""Prefix candidate-count quality, invalidity, cost and plateau curves."""

from __future__ import annotations


def candidate_scaling(attempts, *, tolerance=1e-9):
    rows, best, cumulative_cost = [], None, 0.
    for index, item in enumerate(attempts, 1):
        cumulative_cost += float(item.get("cost", 0))
        if item.get("valid") and item.get("distance") is not None:
            best = min(float(item["distance"]), best) if best is not None else float(item["distance"])
        prefix = attempts[:index]
        rows.append({"k": index, "best_distance": best,
                     "invalidity": sum(not value.get("valid") for value in prefix)/index,
                     "cost": cumulative_cost,
                     "marginal": None if index == 1 or rows[-1]["best_distance"] is None
                     or best is None else rows[-1]["best_distance"]-best})
    plateau = next((row["k"] for row in rows[1:]
                    if row["marginal"] is not None and row["marginal"] <= tolerance), None)
    return {"rows": tuple(rows), "plateau": plateau}
