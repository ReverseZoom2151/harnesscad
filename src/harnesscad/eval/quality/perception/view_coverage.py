def audit(entities,views,thin_features=(),pixel_threshold=1):
 covered=set().union(*(set(x["visible"]) for x in views)) if views else set()
 missing=set(entities)-covered
 thin=tuple(sorted(x["id"] for x in thin_features if x["pixels"]<pixel_threshold))
 recommendation=None
 candidates=[x for x in views if set(x.get("potential",()))&missing]
 if candidates:recommendation=max(candidates,key=lambda x:(len(set(x.get("potential",()))&missing),-x.get("angle",0)))["id"]
 return {"uncovered":tuple(sorted(missing)),"thin":thin,"recommendation":recommendation}
