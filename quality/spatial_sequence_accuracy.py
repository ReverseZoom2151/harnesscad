from math import acos,cos,pi
def angle_error(a,b): return acos(max(-1,min(1,cos(a-b))))
def score(actual,expected,tolerance=1e-6):
 n=max(len(expected),1); cmds=sum(a["op"]==e["op"] for a,e in zip(actual,expected))/n
 origins=[]; angles=[]
 for a,e in zip(actual,expected):
  origins.append(max(abs(x-y) for x,y in zip(a.get("origin",(0,0,0)),e.get("origin",(0,0,0))))<=tolerance)
  angles.append(angle_error(a.get("angle",0),e.get("angle",0))<=tolerance)
 return {"command":cmds,"origin":sum(origins)/len(origins) if origins else 0,"orientation":sum(angles)/len(angles) if angles else 0}
