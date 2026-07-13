"""Injected adapter matrix; records evidence without making platform claims."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class PlatformResult:
    platform: str; version: str; opened: bool; reexported: bool
    geometry_fidelity: float|None; annotation_retention: float|None; error: str=""


def evaluate_platforms(payload, adapters):
    out=[]
    for name,version,adapter in sorted(adapters,key=lambda x:(x[0],x[1])):
        try:
            result=adapter(payload)
            out.append(PlatformResult(name,version,bool(result.get("opened")),
                bool(result.get("reexported")),result.get("geometry_fidelity"),
                result.get("annotation_retention"),str(result.get("error",""))))
        except Exception as exc:
            out.append(PlatformResult(name,version,False,False,None,None,
                                      f"{type(exc).__name__}: {exc}"))
    return tuple(out)
