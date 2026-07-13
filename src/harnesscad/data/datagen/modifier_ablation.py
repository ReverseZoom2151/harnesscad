"""One-at-a-time modifier removal through injected topology verification."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Ablation:
    index: int; operation: str; commands: tuple[object,...]; retained: bool; reason: str

def ablate_modifiers(commands,*,kind=lambda x:x["op"],rebuild,topology_complete,
                     removable=("fillet","chamfer")):
    values=tuple(commands); out=[]
    for i,item in enumerate(values):
        op=kind(item)
        if op not in removable:continue
        candidate=values[:i]+values[i+1:]
        try:shape=rebuild(candidate); ok=bool(topology_complete(shape))
        except Exception:shape=None;ok=False
        out.append(Ablation(i,op,candidate,ok,"ok" if ok else "invalid-topology"))
    return tuple(out)
