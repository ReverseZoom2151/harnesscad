"""Paired point-count quality/resource frontier."""

from __future__ import annotations


def point_budget_report(rows):
    groups = {}
    for row in rows:
        groups.setdefault(int(row["points"]), []).append(row)
    reports = []
    for count, items in sorted(groups.items()):
        valid = [item for item in items if item.get("valid")]
        distances = [float(item["distance"]) for item in valid
                     if item.get("distance") is not None]
        reports.append({"points": count, "samples": len(items),
                        "invalidity": 1-len(valid)/len(items),
                        "mean_distance": sum(distances)/len(distances)
                        if distances else None,
                        "mean_latency": sum(float(item.get("latency", 0))
                                            for item in items)/len(items),
                        "peak_memory": max(int(item.get("memory", 0))
                                           for item in items)})
    return tuple(reports)
