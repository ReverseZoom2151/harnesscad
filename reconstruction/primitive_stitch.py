def stitch(value,step,residual,max_iterations=20,tolerance=1e-6):
 best=value;best_r=residual(value);trace=[best_r]
 for _ in range(max_iterations):
  candidate=step(best);r=residual(candidate);trace.append(r)
  if r>best_r:return {"value":best,"residual":best_r,"trace":tuple(trace),"rolled_back":True}
  best,best_r=candidate,r
  if r<=tolerance:break
 return {"value":best,"residual":best_r,"trace":tuple(trace),"rolled_back":False}
