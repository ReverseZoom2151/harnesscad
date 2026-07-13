"""CD/IV/PR/VR aggregation and contact-condition slicing."""
from __future__ import annotations
from collections import defaultdict

def evaluate_sample(*,valid,cd=None,intersection_volume=None,condition_volume=None,proximity=None):
    iv=(intersection_volume/condition_volume if valid and condition_volume and
        intersection_volume is not None else None)
    return {"valid":bool(valid),"cd":cd if valid else None,"iv":iv,"pr":proximity if valid else None}
def aggregate(samples):
    values=tuple(samples)
    def mean(key):
        x=[r[key] for r in values if r.get(key) is not None]
        return sum(x)/len(x) if x else None
    return {"n":len(values),"vr":sum(r["valid"] for r in values)/len(values) if values else None,
            "cd":mean("cd"),"iv":mean("iv"),"pr":mean("pr")}
def slice_metrics(rows):
    groups=defaultdict(list)
    for row in rows:groups[(row["contact_count"],row["mapping_kind"])].append(row["metrics"])
    return {key:aggregate(groups[key]) for key in sorted(groups)}
