"""Hierarchical corpus cross-tabs, rarity, and target-ratio checks."""

from __future__ import annotations
from collections import Counter
from typing import Iterable, Mapping
from dataengine.anomaly_schema import AnomalyAsset


def audit_anomaly_distribution(assets: Iterable[AnomalyAsset], rarity=50,
                               targets: Mapping[str,float]|None=None) -> dict:
    values=tuple(assets); cube=Counter()
    anomalies=Counter(a.anomaly for a in values if not a.normal)
    sources=Counter(a.source_kind for a in values)
    for a in values:
        for task in a.tasks:
            cube[(a.domain,a.system,a.part,a.anomaly,task.value,a.source_kind)] += 1
    total=len(values)
    actual={k:sources[k]/total for k in sources} if total else {}
    gaps={k:actual.get(k,0)-v for k,v in sorted((targets or {}).items())}
    return {"n":total,"cube":dict(sorted(cube.items())),"source_ratios":actual,
            "target_gaps":gaps,
            "rare_anomalies":tuple(sorted(k for k,v in anomalies.items() if v<rarity))}
