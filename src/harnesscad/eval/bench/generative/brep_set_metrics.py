"""Deterministic generated-set B-rep metrics."""
from math import log

def ratios(generated, training, valid=lambda x: True, signature=lambda x:x):
    sig=[signature(x) for x in generated if valid(x)]; train={signature(x) for x in training}
    return {"valid":len(sig)/len(generated) if generated else 0.0,
            "unique":len(set(sig))/len(sig) if sig else 0.0,
            "novel":sum(x not in train for x in sig)/len(sig) if sig else 0.0}

def coverage_mmd(generated, reference, distance):
    if not generated or not reference: return {"coverage":0.0,"mmd":None}
    nearest=[min((distance(r,g),i) for i,g in enumerate(generated)) for r in reference]
    return {"coverage":len({i for _,i in nearest})/len(reference),
            "mmd":sum(d for d,_ in nearest)/len(nearest)}

def jsd(a,b):
    keys=set(a)|set(b); sa=sum(a.values()); sb=sum(b.values())
    if not sa or not sb:return None
    p={k:a.get(k,0)/sa for k in keys}; q={k:b.get(k,0)/sb for k in keys}
    m={k:(p[k]+q[k])/2 for k in keys}
    kl=lambda x:sum(v*log(v/m[k],2) for k,v in x.items() if v)
    return (kl(p)+kl(q))/2
