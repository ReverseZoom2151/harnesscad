from dataclasses import dataclass
from math import isfinite
@dataclass(frozen=True)
class CameraSample: id:str; angle:float; geometry:float; semantic:float
def prune(samples,geometry_min,semantic_min,min_views=0):
 accepted=[]; rejected=[]
 for x in sorted(samples,key=lambda x:x.id):
  reasons=[]
  if not isfinite(x.geometry) or x.geometry<geometry_min: reasons.append("geometry")
  if not isfinite(x.semantic) or x.semantic<semantic_min: reasons.append("semantic")
  (rejected if reasons else accepted).append((x,tuple(reasons)))
 if len(accepted)<min_views:
  ranked=sorted(rejected,key=lambda y:(-(y[0].geometry+y[0].semantic),y[0].id))
  chosen=ranked[:min_views-len(accepted)]
  accepted += [(x,("fallback",)) for x,_ in chosen]
  chosen_ids={x.id for x,_ in chosen}
  rejected=[x for x in rejected if x[0].id not in chosen_ids]
 return tuple(accepted),tuple(rejected)
