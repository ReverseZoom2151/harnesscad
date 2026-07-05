"""Geometry reconciliation primitives independent of a CAD kernel."""
from math import dist

def average_vertices(points):
    if not points: raise ValueError("points required")
    return tuple(sum(p[i] for p in points)/len(points) for i in range(3))

def align_edge(samples,start,end):
    if len(samples)<2: raise ValueError("two samples required")
    if dist(samples[0],start)+dist(samples[-1],end)>dist(samples[-1],start)+dist(samples[0],end):
        samples=tuple(reversed(samples))
    a,b=samples[0],samples[-1]; length=dist(a,b)
    if not length: raise ValueError("degenerate edge")
    target=dist(start,end); scale=target/length
    return tuple(tuple(start[i]+(p[i]-a[i])*scale for i in range(3)) for p in samples[:-1])+(end,)

def consistency(edge_samples,surface_distance):
    vals=[surface_distance(p) for edge in edge_samples for p in edge]
    return max(vals,default=0.0), sum(vals)/len(vals) if vals else 0.0
