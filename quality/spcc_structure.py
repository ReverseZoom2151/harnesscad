from dataclasses import dataclass
@dataclass(frozen=True)
class Component: ops:tuple
@dataclass(frozen=True)
class Macro: component:Component; count:int
def collapse(components,threshold=4):
 out=[];i=0
 while i<len(components):
  j=i+1
  while j<len(components) and components[j]==components[i]:j+=1
  n=j-i;out.append(Macro(components[i],n) if n>=threshold else components[i:j]);i=j
 return tuple(x for group in out for x in (group if isinstance(group,list) else (group,)))
def expand(items):return tuple(c for x in items for c in ((x.component,)*x.count if isinstance(x,Macro) else (x,)))
def stats(original,collapsed):return {"components":len(original),"nodes":len(collapsed),"reduction":len(original)-len(collapsed)}
