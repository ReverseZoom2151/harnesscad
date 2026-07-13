"""Ordered admission contract for evolutionary CAD candidates."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Admission:
    accepted: bool
    stage: str
    evidence: tuple[dict, ...]
    repair_packet: dict | None = None

def validate_candidate(candidate, *, execute, integrity, render, semantic):
    evidence=[]
    stages=(("execution",execute),("integrity",integrity),
            ("render",render),("semantic",semantic))
    value=candidate
    for name,fn in stages:
        try: result=dict(fn(value))
        except Exception as exc:
            result={"accepted":False,"reason":f"{type(exc).__name__}: {exc}"}
        result={"stage":name,**result}; evidence.append(result)
        if not result.get("accepted"):
            return Admission(False,name,tuple(evidence),
                             {"stage":name,"reason":result.get("reason","rejected"),
                              "candidate_id":getattr(candidate,"id",None)})
        value=result.get("output",value)
    return Admission(True,"accepted",tuple(evidence))

def canonical_seven_views():
    return ("isometric","+x","-x","+y","-y","+z","-z")
