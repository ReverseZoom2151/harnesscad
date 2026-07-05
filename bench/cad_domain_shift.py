from math import log
def audit(source,target):
 s=set().union(*(set(x) for x in source)) if source else set(); t=set().union(*(set(x) for x in target)) if target else set()
 return {"unseen":tuple(sorted(t-s)),"unsupported_rate":len(t-s)/len(t) if t else 0.0,"source_vocab":len(s),"target_vocab":len(t)}
