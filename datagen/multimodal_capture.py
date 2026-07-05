"""Stable fixed-view and point-normal capture contracts."""
from __future__ import annotations
from dataclasses import dataclass
import random

@dataclass(frozen=True)
class Camera:
    name: str; direction: tuple[float,float,float]

DEFAULT_CAMERAS=tuple(Camera(n,d) for n,d in (
 ("front",(0,-1,0)),("back",(0,1,0)),("left",(-1,0,0)),("right",(1,0,0)),
 ("top",(0,0,1)),("bottom",(0,0,-1)),("iso_ne",(1,-1,1)),("iso_sw",(-1,1,1))))

def choose_views(k,seed,cameras=DEFAULT_CAMERAS):
    if not 0<=k<=len(cameras):raise ValueError("invalid view count")
    return tuple(sorted(random.Random(seed).sample(tuple(cameras),k),key=lambda x:x.name))

def capture_manifest(shape,*,renderer,point_sampler,count,seed,cameras=DEFAULT_CAMERAS):
    views=tuple((c.name,renderer(shape,c)) for c in cameras)
    points=tuple(point_sampler(shape,count,seed))
    if any(len(item)!=2 for item in points):raise ValueError("point sampler must return point-normal pairs")
    return {"views":views,"points":points,"seed":seed,"count":count}
