"""Task-specific modality/objective ablation with resource-aware promotion."""
from __future__ import annotations
def ablation_matrix(rows):
    out={}
    for row in rows:
        key=(row["task"],tuple(sorted(row["modalities"])),tuple(sorted(row["objectives"])))
        out[key]={"quality":float(row["quality"]),"memory":int(row.get("memory",0)),
                  "latency":float(row.get("latency",0))}
    return dict(sorted(out.items()))
def promotion(candidate,baseline,*,min_gain=0,max_memory_increase=0):
    gain=candidate["quality"]-baseline["quality"];memory=candidate["memory"]-baseline["memory"]
    reasons=[]
    if gain<min_gain:reasons.append("quality")
    if memory>max_memory_increase:reasons.append("memory")
    return {"promoted":not reasons,"gain":gain,"memory_increase":memory,"reasons":tuple(reasons)}
