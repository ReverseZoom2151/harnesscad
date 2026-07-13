"""Comparable timing/cost profiles for compiler and visual judges."""

from __future__ import annotations


def judge_efficiency(records):
    rows = []
    for name, items in sorted(records.items()):
        values = tuple(items)
        rows.append({
            "judge": name, "samples": len(values),
            "mean_latency": (sum(float(item.get("latency", 0)) for item in values)
                             / len(values) if values else None),
            "total_cost": sum(float(item.get("cost", 0)) for item in values),
            "cache_hit_rate": (sum(bool(item.get("cache_hit")) for item in values)
                               / len(values) if values else None),
            "error_rate": (sum(bool(item.get("error")) for item in values)
                           / len(values) if values else None),
        })
    return tuple(rows)
