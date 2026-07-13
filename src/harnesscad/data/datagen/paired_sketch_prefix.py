"""Generate paired primitive/render prefixes without target leakage."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class PairedPrefix:
    ratio: float; prefix: tuple; target: tuple; partial_render: object; full_render: object
def paired_prefixes(entities,renderer,ratios=(.2,.4,.6,.8)):
    values=tuple(entities);full=renderer(values);out=[]
    for ratio in ratios:
        if not 0<ratio<1:raise ValueError("ratios must be strictly between 0 and 1")
        cut=max(1,min(len(values)-1,round(len(values)*ratio))) if len(values)>1 else len(values)
        prefix=values[:cut]
        out.append(PairedPrefix(ratio,prefix,values[cut:],renderer(prefix),full))
    return tuple(out)
