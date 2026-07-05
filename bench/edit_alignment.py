"""Occupancy JSD, directional cosine, validity/CD and complexity slices."""
from __future__ import annotations
from collections import defaultdict
from math import log2,sqrt

def occupancy_jsd(a,b):
    keys=set(a)|set(b); sa=sum(a.values());sb=sum(b.values())
    if not sa or not sb:return None
    p={k:a.get(k,0)/sa for k in keys};q={k:b.get(k,0)/sb for k in keys};m={k:(p[k]+q[k])/2 for k in keys}
    kl=lambda x:sum(v*log2(v/m[k]) for k,v in x.items() if v)
    return (kl(p)+kl(q))/2

def directional_cosine(original_image,edited_image,neutral_text,instruction_text,*,image_embed,text_embed):
    oi,ei=image_embed(original_image),image_embed(edited_image)
    nt,it=text_embed(neutral_text),text_embed(instruction_text)
    di=[b-a for a,b in zip(oi,ei)];dt=[b-a for a,b in zip(nt,it)]
    denom=sqrt(sum(x*x for x in di))*sqrt(sum(x*x for x in dt))
    return sum(a*b for a,b in zip(di,dt))/denom if denom else None

def aggregate_edits(rows):
    values=tuple(rows); cds=[r["cd"] for r in values if r.get("valid") and r.get("cd") is not None]
    return {"valid_ratio":sum(bool(r.get("valid")) for r in values)/len(values) if values else None,
            "mean_cd":sum(cds)/len(cds) if cds else None}

def slice_by_se(rows):
    groups=defaultdict(list)
    for row in rows:groups[row["se_count"]].append(row)
    return {key:aggregate_edits(groups[key]) for key in sorted(groups)}
