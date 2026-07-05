"""Lineage-safe split auditing and exact SE-length balancing."""
from __future__ import annotations
from collections import defaultdict

def split_leakage(records):
    seen={}; leaks=[]
    for r in records:
        for key in (r["source_id"],r["lineage"]):
            old=seen.setdefault(key,r["split"])
            if old!=r["split"]:leaks.append(key)
    return tuple(sorted(set(leaks)))

def balance_se(records,n_per_length,lengths=range(1,6),key=lambda x:x["id"]):
    groups=defaultdict(list)
    for r in records:groups[r["se_count"]].append(r)
    if any(len(groups[n])<n_per_length for n in lengths):raise ValueError("insufficient examples")
    return tuple(x for n in lengths for x in sorted(groups[n],key=key)[:n_per_length])
