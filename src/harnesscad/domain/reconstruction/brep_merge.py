"""Two-signal deterministic geometry clustering."""
from dataclasses import dataclass
from math import dist

@dataclass(frozen=True)
class GeometryNode:
    id:str; bbox:tuple[float,...]; samples:tuple[tuple[float,float,float],...]

def duplicates(a,b,bbox_tolerance=.08,shape_tolerance=.2):
    if len(a.bbox)!=len(b.bbox) or len(a.samples)!=len(b.samples): return False
    bd=sum((x-y)**2 for x,y in zip(a.bbox,b.bbox))**.5
    sd=sum(dist(x,y) for x,y in zip(a.samples,b.samples))/len(a.samples) if a.samples else 0
    return bd<=bbox_tolerance and sd<=shape_tolerance

def cluster(nodes, **kwargs):
    groups=[]
    for n in sorted(nodes,key=lambda x:x.id):
        hits=[g for g in groups if any(duplicates(n,x,**kwargs) for x in g)]
        if not hits: groups.append([n])
        else:
            merged=[n]
            for g in hits: merged+=g; groups.remove(g)
            groups.append(sorted(merged,key=lambda x:x.id))
    return tuple(tuple(g) for g in sorted(groups,key=lambda g:g[0].id))
