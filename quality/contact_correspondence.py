"""Stable minimum-cost rectangular assignment and ambiguity reporting."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache

@dataclass(frozen=True)
class Correspondence:
    pairs: tuple[tuple[int,int,float],...]; unmatched_left: tuple[int,...]
    unmatched_right: tuple[int,...]; total_cost: float; ambiguous: bool

def assign(costs):
    rows=tuple(tuple(map(float,row)) for row in costs)
    if not rows:return Correspondence((),(),(),0,False)
    width=len(rows[0])
    if any(len(r)!=width for r in rows):raise ValueError("ragged costs")
    transposed=len(rows)>width
    matrix=tuple(zip(*rows)) if transposed else rows
    n,m=len(matrix),len(matrix[0]) if matrix else 0
    @lru_cache(None)
    def solve(i,used):
        if i==n:return 0.0,()
        choices=[]
        for j in range(m):
            if not used>>j&1:
                cost,pairs=solve(i+1,used|1<<j)
                choices.append((matrix[i][j]+cost,((i,j),)+pairs))
        return min(choices,key=lambda x:(x[0],x[1]))
    total,pairs=solve(0,0)
    mapped=tuple((j,i,rows[j][i]) if transposed else (i,j,rows[i][j]) for i,j in pairs)
    left={i for i,_,_ in mapped};right={j for _,j,_ in mapped}
    ul=tuple(i for i in range(len(rows)) if i not in left)
    ur=tuple(j for j in range(width) if j not in right)
    return Correspondence(tuple(sorted(mapped)),ul,ur,total,bool(ul or ur))
