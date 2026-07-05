"""Visual asset QC records independent of image-decoding libraries."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class QCResult:
    asset_id: str; passed: bool; reasons: tuple[str,...]
    reviewers: tuple[str,...] = (); adjudicated: bool = False


def inspect_visual(asset_id, payload, *, decode: Callable, minimum_size=(1,1),
                   ambiguous=False, reviewers: Iterable[str]=(), adjudicated=False):
    reasons=[]
    try: width,height=decode(payload)
    except Exception: return QCResult(asset_id,False,("decode_failed",),tuple(reviewers),adjudicated)
    if width<minimum_size[0] or height<minimum_size[1]: reasons.append("undersized")
    if ambiguous: reasons.append("ambiguous_label")
    names=tuple(reviewers)
    if len(names)<2: reasons.append("insufficient_review")
    return QCResult(asset_id,not reasons,tuple(reasons),names,adjudicated)
