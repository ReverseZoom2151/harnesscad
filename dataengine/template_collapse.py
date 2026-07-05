"""Family-level CAD script template-collapse diagnostics."""
from __future__ import annotations
import ast, math
from collections import Counter, defaultdict

class _Skeleton(ast.NodeTransformer):
    def visit_Name(self,node): return ast.copy_location(ast.Name(id="_name",ctx=node.ctx),node)
    def visit_Constant(self,node):
        if isinstance(node.value,(int,float,str)): return ast.copy_location(ast.Constant("_lit"),node)
        return node

def skeleton(source):
    return ast.dump(_Skeleton().visit(ast.parse(source)), annotate_fields=False)

def template_collapse(records):
    groups=defaultdict(list)
    for row in records: groups[row["family"]].append(skeleton(row["code"]))
    rows=[]; all_skeletons=[]
    for family, values in sorted(groups.items()):
        counts=Counter(values); all_skeletons.extend(values); total=len(values)
        probs=[n/total for n in counts.values()]
        rows.append({"family":family,"count":total,"unique":len(counts),
                     "concentration":max(probs),
                     "entropy":-sum(p*math.log2(p) for p in probs)})
    global_counts=Counter(all_skeletons)
    return {"families":tuple(rows),"global_unique":len(global_counts),
            "global_concentration":max(global_counts.values())/len(all_skeletons)
            if all_skeletons else None}

def identifier_leakage(records):
    owners=defaultdict(set)
    for row in records:
        for node in ast.walk(ast.parse(row["code"])):
            if isinstance(node,ast.Name): owners[node.id].add(row["family"])
    return tuple(sorted(name for name,families in owners.items() if len(families)==1))
