"""Progressive modality curricula and balanced combination manifests."""
from __future__ import annotations
from itertools import combinations

ORDER=("text","point","image")
def modality_curriculum():
    return tuple((stage,tuple(ORDER[:stage+1])) for stage in range(len(ORDER)))
def modality_combinations(available):
    values=tuple(x for x in ORDER if x in set(available))
    return tuple(combo for n in range(1,len(values)+1) for combo in combinations(values,n))
def combination_balance(rows):
    counts={}
    for row in rows:
        key=tuple(sorted(row));counts[key]=counts.get(key,0)+1
    return dict(sorted(counts.items()))
