"""Blinded rating QC, rank tests, and feasibility/novelty Pareto analysis."""
from __future__ import annotations
from collections import defaultdict
from math import sqrt

def _ranks(values):
    order=sorted(range(len(values)),key=lambda i:values[i]);out=[0.0]*len(values);i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]:j+=1
        rank=(i+j+1)/2
        for k in order[i:j]:out[k]=rank
        i=j
    return out
def spearman(x,y):
    if len(x)!=len(y) or len(x)<2:raise ValueError("aligned samples of size >=2 required")
    a,b=_ranks(x),_ranks(y);am=sum(a)/len(a);bm=sum(b)/len(b)
    num=sum((u-am)*(v-bm) for u,v in zip(a,b))
    den=sqrt(sum((u-am)**2 for u in a)*sum((v-bm)**2 for v in b))
    return num/den if den else 0.0
def mann_whitney(a,b):
    values=list(a)+list(b)
    if not a or not b:raise ValueError("both samples required")
    ranks=_ranks(values);u1=sum(ranks[:len(a)])-len(a)*(len(a)+1)/2
    return {"u":u1,"u_other":len(a)*len(b)-u1}
def rating_qc(ratings,min_raters=2):
    groups=defaultdict(list)
    for row in ratings:
        if not 1<=row["feasibility"]<=7 or not 1<=row["novelty"]<=7:continue
        groups[row["item_id"]].append(row)
    return {k:{"n":len(v),"accepted":len({x["rater_id"] for x in v})>=min_raters,
               "feasibility":sum(x["feasibility"] for x in v)/len(v),
               "novelty":sum(x["novelty"] for x in v)/len(v)}
            for k,v in sorted(groups.items())}
def pareto_items(items):
    values=tuple(items)
    return tuple(x for x in values if not any(
        y["feasibility"]>=x["feasibility"] and y["novelty"]>=x["novelty"] and
        (y["feasibility"]>x["feasibility"] or y["novelty"]>x["novelty"])
        for y in values if y is not x))
