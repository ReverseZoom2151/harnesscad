"""Contact position/shape and normalized bounding-box objectives."""
from __future__ import annotations
from math import dist

def position_cost(points,reference,distance):
    values=tuple(points)
    return sum(distance(p,reference) for p in values)/len(values) if values else None
def edge_shape_cost(edge,reference,*,length_weight=1,angle_weight=1):
    if len(edge)!=len(reference) or len(edge)<2:return None
    length=angle=0
    for (a,b),(c,d) in zip(zip(edge,edge[1:]),zip(reference,reference[1:])):
        la,lb=dist(a,b),dist(c,d);length+=(la-lb)**2
        if la and lb:
            ua=tuple((y-x)/la for x,y in zip(a,b));ub=tuple((y-x)/lb for x,y in zip(c,d))
            angle+=1-sum(x*y for x,y in zip(ua,ub))
    n=len(edge)-1
    return length_weight*length/n+angle_weight*angle/n
def bbox_geometry_cost(candidate,guide):
    def norm(boxes):
        centers=[b["center"] for b in boxes];scale=max((abs(x) for c in centers for x in c),default=1) or 1
        return [(tuple(x/scale for x in b["center"]),tuple(b["dimensions"])) for b in boxes]
    a,b=norm(candidate),norm(guide)
    return sum(min(dist(c,gc)+dist(d,gd) for gc,gd in b) for c,d in a)/len(a) if a and b else None
def scheduled_weights(t,threshold=.7):
    return (1.0,1.0) if t>threshold else (0.0,0.0)
