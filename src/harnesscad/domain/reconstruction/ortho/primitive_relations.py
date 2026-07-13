from dataclasses import dataclass
from math import sqrt
@dataclass(frozen=True)
class Primitive: id:str; axis:tuple[float,float,float]
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def infer(a,b,tolerance=.01):
 d=abs(dot(a.axis,b.axis));return "parallel" if abs(d-1)<=tolerance else ("perpendicular" if d<=tolerance else None)
def project(a,b,relation):
 if relation=="parallel":return a,Primitive(b.id,a.axis)
 if relation=="perpendicular":
  d=dot(a.axis,b.axis);v=tuple(b.axis[i]-d*a.axis[i] for i in range(3));n=sqrt(dot(v,v))
  if not n:raise ValueError("contradictory")
  return a,Primitive(b.id,tuple(x/n for x in v))
 raise ValueError("unknown relation")
