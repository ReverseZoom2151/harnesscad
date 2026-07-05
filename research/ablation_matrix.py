"""Paired, stratified ablation summaries."""

from __future__ import annotations
from collections import defaultdict


def compare_ablation(rows, *, metric: str):
    """Rows carry stratum, pair_id, variant, and numeric metric."""
    groups=defaultdict(dict)
    for row in rows:groups[(row["stratum"],row["pair_id"])][row["variant"]]=float(row[metric])
    strata=defaultdict(list)
    for (stratum,_),pair in groups.items():
        if "control" in pair and "treatment" in pair:
            strata[stratum].append(pair["treatment"]-pair["control"])
    return {key:{"n":len(vals),"mean_delta":sum(vals)/len(vals),
                 "wins":sum(v>0 for v in vals),"losses":sum(v<0 for v in vals)}
            for key,vals in sorted(strata.items())}
