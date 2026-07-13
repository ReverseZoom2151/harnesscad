"""Round-level validity, novelty and lineage-growth summaries."""
from __future__ import annotations

def evolution_dynamics(rounds):
    rows=[]
    for index,row in enumerate(rounds):
        proposed=int(row["proposed"]); accepted=int(row.get("accepted",0))
        invalid=int(row.get("invalid",0)); novel=int(row.get("novel",0))
        rows.append({"round":index,"proposed":proposed,
                     "invalid_ratio":invalid/proposed if proposed else 0.,
                     "novelty_ratio":novel/proposed if proposed else 0.,
                     "acceptance_ratio":accepted/proposed if proposed else 0.})
    slopes=tuple(rows[i]["novelty_ratio"]-rows[i-1]["novelty_ratio"]
                 for i in range(1,len(rows)))
    return {"rounds":tuple(rows),"novelty_slopes":slopes,
            "diminishing":bool(slopes and sum(slopes)/len(slopes)<0)}

def lineage_stats(records):
    parents={r.id:r.parent_ids for r in records}; memo={}
    def depth(node,seen=()):
        if node in memo:return memo[node]
        if node in seen:return 0
        memo[node]=0 if not parents.get(node) else 1+max(
            depth(p,seen+(node,)) for p in parents[node] if p in parents)
        return memo[node]
    children={node:0 for node in parents}
    for values in parents.values():
        for parent in values:
            if parent in children: children[parent]+=1
    return {"max_depth":max((depth(x) for x in parents),default=0),
            "branching":tuple(sorted(children.items()))}
