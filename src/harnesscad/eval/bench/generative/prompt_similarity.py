"""Cross-product cosine similarity matrices between generation settings."""
from math import sqrt
def _cos(a,b):
    d=sqrt(sum(x*x for x in a))*sqrt(sum(y*y for y in b))
    return sum(x*y for x,y in zip(a,b))/d if d else 0.0
def similarity_matrix(groups,embed):
    names=sorted(groups);out={}
    for a in names:
        for b in names:
            pairs=[(x,y) for i,x in enumerate(groups[a]) for j,y in enumerate(groups[b])
                   if a!=b or i!=j]
            out[(a,b)]=sum(_cos(embed(x),embed(y)) for x,y in pairs)/len(pairs) if pairs else None
    return out
