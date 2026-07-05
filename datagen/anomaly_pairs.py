"""Injected anomaly compositor producing paired, localized provenance."""

from __future__ import annotations
from dataclasses import dataclass
import hashlib
from typing import Callable


@dataclass(frozen=True)
class AnomalyPair:
    normal: object; anomalous: object; mask: frozenset[tuple[int,int]]
    box: tuple[int,int,int,int]; seed: int; provenance: str


def compose_pair(normal, anomaly, seed, compositor: Callable) -> AnomalyPair:
    changed, pixels = compositor(normal, anomaly, seed)
    mask=frozenset((int(x),int(y)) for x,y in pixels)
    if not mask: raise ValueError("compositor produced an empty anomaly")
    xs=[x for x,_ in mask]; ys=[y for _,y in mask]
    box=(min(xs),min(ys),max(xs)+1,max(ys)+1)
    digest=hashlib.sha256(f"{normal!r}\0{anomaly!r}\0{seed}\0{sorted(mask)!r}".encode()).hexdigest()
    return AnomalyPair(normal,changed,mask,box,seed,digest)
