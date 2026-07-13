"""Failure-aware morphology aggregates without hiding invalid generations."""

from __future__ import annotations

from statistics import mean, median


def morphology_report(results, *, invalid_penalty=None):
    items = tuple(results)
    distances = [float(item["distance"]) for item in items
                 if item.get("valid") and item.get("distance") is not None]
    invalid = sum(not item.get("valid", False) for item in items)
    penalized = None
    if invalid_penalty is not None and items:
        penalized = mean(float(item["distance"]) if item.get("valid")
                         and item.get("distance") is not None else invalid_penalty
                         for item in items)
    ordered = sorted(distances)
    quantile = lambda p: ordered[min(len(ordered)-1, round((len(ordered)-1)*p))] \
        if ordered else None
    return {"total": len(items), "valid": len(distances), "invalid": invalid,
            "invalidity_ratio": invalid/len(items) if items else None,
            "mean_valid_distance": mean(distances) if distances else None,
            "median_valid_distance": median(distances) if distances else None,
            "p90_valid_distance": quantile(.9),
            "failure_penalized_distance": penalized}
