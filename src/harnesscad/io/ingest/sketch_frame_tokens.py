from dataclasses import dataclass
from math import cos,sin
@dataclass(frozen=True)
class SketchFrame:
 origin:tuple[float,float,float]; angle:float
 def local_to_world(self,p):
  c,s=cos(self.angle),sin(self.angle); return (self.origin[0]+c*p[0]-s*p[1],self.origin[1]+s*p[0]+c*p[1],self.origin[2])
 def world_to_local(self,p):
  x,y=p[0]-self.origin[0],p[1]-self.origin[1]; c,s=cos(self.angle),sin(self.angle)
  return (c*x+s*y,-s*x+c*y)
def quantize(values,bins,low=-1,high=1):
 if bins<2:raise ValueError("bins")
 tokens=tuple(round((min(high,max(low,x))-low)/(high-low)*(bins-1)) for x in values)
 decoded=tuple(low+t*(high-low)/(bins-1) for t in tokens)
 return tokens,decoded,max((abs(a-b) for a,b in zip(values,decoded)),default=0)
