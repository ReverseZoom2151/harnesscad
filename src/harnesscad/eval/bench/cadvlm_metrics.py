"""Exact/tolerant CadVLM Entity, Sketch, and macro CAD-F1 metrics."""
from __future__ import annotations
from math import dist

def _match(actual,expected,tolerance):
    remaining=list(range(len(expected)));matched=0
    for item in actual:
        choices=[]
        for i in remaining:
            other=expected[i]
            if item[0]!=other[0] or len(item[1:])!=len(other[1:]):continue
            ok=item==other if tolerance is None else dist(item[1:],other[1:])<=tolerance
            if ok:choices.append(i)
        if choices:remaining.remove(min(choices));matched+=1
    return matched
def cadvlm_metrics(actual,expected,*,tolerance=None):
    a,e=tuple(actual),tuple(expected)
    if len(a)!=len(e):raise ValueError("sample count mismatch")
    any_hit=strict=0;f1s=[]
    for left,right in zip(a,e):
        m=_match(tuple(left),tuple(right),tolerance);any_hit+=m>0;strict+=m==len(left)==len(right)
        p=m/len(left) if left else float(not right);r=m/len(right) if right else float(not left)
        f1s.append(2*p*r/(p+r) if p+r else 0)
    n=len(a)
    return {"n":n,"entity_accuracy":any_hit/n if n else None,
            "sketch_accuracy":strict/n if n else None,
            "cad_f1":sum(f1s)/n if n else None}
def sliced_metrics(rows,*,tolerance=None):
    groups={}
    for row in rows:groups.setdefault((row["ratio"],row["condition"]),[]).append(row)
    return {key:cadvlm_metrics([x["actual"] for x in values],
                               [x["expected"] for x in values],tolerance=tolerance)
            for key,values in sorted(groups.items())}
