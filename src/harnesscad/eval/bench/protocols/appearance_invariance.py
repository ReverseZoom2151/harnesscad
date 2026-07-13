"""Counterfactual geometry-vs-appearance shortcut audit."""

from __future__ import annotations


def appearance_invariance(cases, predict, compare):
    outputs = {case["id"]: predict(case) for case in cases}
    within, between = [], []
    for i, left in enumerate(cases):
        for right in cases[i+1:]:
            score = float(compare(outputs[left["id"]], outputs[right["id"]]))
            (within if left["geometry"] == right["geometry"] else between).append(score)
    return {"within_geometry": sum(within)/len(within) if within else None,
            "between_geometry": sum(between)/len(between) if between else None,
            "shortcut_suspected": bool(within and between
                                       and sum(within)/len(within) <= sum(between)/len(between))}
