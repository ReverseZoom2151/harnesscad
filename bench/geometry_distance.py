"""Injected surface sampling and symmetric Chamfer distance."""

from __future__ import annotations
from math import dist


def symmetric_chamfer(a,b):
    x,y=tuple(a),tuple(b)
    if not x or not y:return None
    directed=lambda p,q:sum(min(dist(i,j) for j in q) for i in p)/len(p)
    return (directed(x,y)+directed(y,x))/2


def sampled_distance(expected,actual,sampler,*,count=1024,seed=0,scale=1.0):
    if count<=0 or scale<=0:raise ValueError("count and scale must be positive")
    a=[tuple(v*scale for v in p) for p in sampler(expected,count,seed)]
    b=[tuple(v*scale for v in p) for p in sampler(actual,count,seed)]
    return symmetric_chamfer(a,b)
