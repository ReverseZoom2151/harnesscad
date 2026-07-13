"""Model-independent STL/MTL interaction and Pareto analysis."""

from __future__ import annotations
from typing import Mapping


def interaction_report(stl: Mapping[str,float], mtl: Mapping[str,float],
                       higher_is_better: Mapping[str,bool]|None=None) -> dict:
    if set(stl)!=set(mtl): raise ValueError("task sets differ")
    directions=higher_is_better or {k:True for k in stl}
    deltas={k:(mtl[k]-stl[k])*(1 if directions.get(k,True) else -1) for k in sorted(stl)}
    return {"deltas":deltas,
            "negative_transfer":tuple(k for k,v in deltas.items() if v<0),
            "improved":tuple(k for k,v in deltas.items() if v>0),
            "pareto_dominates":all(v>=0 for v in deltas.values()) and any(v>0 for v in deltas.values())}


def efficiency(parameters: float, tasks: int) -> float:
    if parameters<=0 or tasks<=0: raise ValueError("positive parameters and tasks required")
    return tasks/parameters
