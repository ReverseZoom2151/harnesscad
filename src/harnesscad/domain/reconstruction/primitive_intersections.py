def assemble(primitives,adjacency,pair_intersection,triple_intersection):
 ids={p.id:p for p in primitives};edges=[];vertices=[]
 for a,b in sorted(tuple(sorted(x)) for x in adjacency):
  value=pair_intersection(ids[a],ids[b])
  if value is not None:edges.append((a,b,value))
 for a,b,c in sorted({tuple(sorted((a,b,c))) for a,b in adjacency for c,d in adjacency if b==c and tuple(sorted((a,d))) in {tuple(sorted(x)) for x in adjacency}}):
  value=triple_intersection(ids[a],ids[b],ids[c])
  if value is not None:vertices.append((a,b,c,value))
 return {"edges":tuple(edges),"vertices":tuple(vertices)}
