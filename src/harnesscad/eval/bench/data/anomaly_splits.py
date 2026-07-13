"""Leakage-safe manifests for anomaly learning regimes."""

from __future__ import annotations
from dataclasses import dataclass
import hashlib
from typing import Iterable
from harnesscad.data.dataengine.schemas.anomaly_schema import AnomalyAsset


@dataclass(frozen=True)
class SplitManifest:
    train: tuple[str,...]; validation: tuple[str,...]; test: tuple[str,...]
    held_out: tuple[str,...] = ()


def group_safe_split(assets: Iterable[AnomalyAsset], train=.7, validation=.15) -> SplitManifest:
    groups={}
    for a in assets: groups.setdefault(a.group_id or a.id,[]).append(a)
    buckets=[[],[],[]]
    for key in sorted(groups, key=lambda x: hashlib.sha256(x.encode()).hexdigest()):
        v=int(hashlib.sha256(key.encode()).hexdigest(),16)/2**256
        bucket=0 if v<train else (1 if v<train+validation else 2)
        buckets[bucket].extend(x.id for x in groups[key])
    return SplitManifest(*(tuple(sorted(x)) for x in buckets))


def real_only_test(assets): return tuple(sorted(a.id for a in assets if a.source_kind=="real"))
def synthetic_transfer(assets):
    values=tuple(assets)
    return SplitManifest(tuple(sorted(a.id for a in values if a.source_kind!="real")),(),
                         tuple(sorted(a.id for a in values if a.source_kind=="real")))
def normal_only_train(assets): return tuple(sorted(a.id for a in assets if a.normal))
def few_shot(assets, n):
    groups={}
    for a in assets: groups.setdefault(a.anomaly,[]).append(a.id)
    return tuple(x for key in sorted(groups) for x in sorted(groups[key])[:n])
def open_set(assets, held_out):
    held=frozenset(held_out); values=tuple(assets)
    return SplitManifest(tuple(sorted(a.id for a in values if a.anomaly not in held)),(),
                         tuple(sorted(a.id for a in values if a.anomaly in held)),
                         tuple(sorted(held)))
