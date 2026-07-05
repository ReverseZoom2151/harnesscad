"""Deterministic image-domain perturbation manifests and consistency scoring."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageCondition:
    id: str
    camera: str
    material: str
    background: str
    domain: str = "rendered"


def evaluate_conditions(conditions, generate, compare):
    values = tuple(conditions)
    if not values:
        return {"cases": (), "consistency": None}
    outputs = {item.id: generate(item) for item in values}
    baseline = outputs[values[0].id]
    cases = tuple({"id": item.id, "domain": item.domain,
                   "score": float(compare(outputs[item.id], baseline))}
                  for item in values)
    return {"cases": cases,
            "consistency": sum(item["score"] for item in cases) / len(cases)}
