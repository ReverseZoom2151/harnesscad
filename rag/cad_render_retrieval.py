"""Verified CAD-render retrieval through an injected joint embedding."""
from __future__ import annotations
from math import sqrt

def _cos(a,b):
    d=sqrt(sum(x*x for x in a))*sqrt(sum(y*y for y in b))
    return sum(x*y for x,y in zip(a,b))/d if d else -1.0
def retrieve_render(text,records,*,embed_text,embed_image,k=1):
    query=embed_text(text);scored=[]
    for record in records:
        if not record.get("verified_feasible"):continue
        scored.append((_cos(query,embed_image(record["image"])),str(record["id"]),record))
    return tuple(item for _,_,item in sorted(scored,key=lambda x:(-x[0],x[1]))[:k])
