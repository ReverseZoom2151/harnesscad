"""Candidate-controlled ablation of preference-judge pipelines."""

from __future__ import annotations


def judge_ablation(rows):
    """Rows share candidate IDs and contain method/f1/distance/invalid/time/cost."""
    methods, candidate_sets = {}, {}
    for row in rows:
        methods.setdefault(row["method"], []).append(row)
        candidate_sets.setdefault(row["method"], set()).add(row["candidate_id"])
    sets = list(candidate_sets.values())
    controlled = not sets or all(value == sets[0] for value in sets[1:])
    reports = []
    for method, items in sorted(methods.items()):
        valid_distances = [float(item["distance"]) for item in items
                           if not item.get("invalid") and item.get("distance") is not None]
        reports.append({
            "method": method, "samples": len(items),
            "mean_f1": sum(float(item.get("f1", 0)) for item in items)/len(items),
            "mean_distance": (sum(valid_distances)/len(valid_distances)
                              if valid_distances else None),
            "invalidity_ratio": sum(bool(item.get("invalid")) for item in items)/len(items),
            "time": sum(float(item.get("time", 0)) for item in items),
            "cost": sum(float(item.get("cost", 0)) for item in items),
        })
    return {"candidate_controlled": controlled, "reports": tuple(reports)}
