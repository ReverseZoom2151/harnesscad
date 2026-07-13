from dataclasses import dataclass
import random
@dataclass(frozen=True)
class State: transform:tuple[float,...]=(0,0,0); material:str=""
@dataclass(frozen=True)
class Production: source:str; targets:tuple[str,...]; weight:float=1
def derive(start, productions, terminals, *, seed=0, max_depth=20, max_nodes=1000):
 r=random.Random(seed); queue=[(start,State(),0)]; out=[]; trace=[]; diagnostics=[]
 while queue and len(trace)<max_nodes:
  symbol,state,depth=queue.pop(0)
  if symbol in terminals: out.append((symbol,state)); continue
  choices=[p for p in productions if p.source==symbol and p.weight>0]
  if not choices: diagnostics.append(f"unproductive:{symbol}"); continue
  if depth>=max_depth: diagnostics.append(f"depth_budget:{symbol}"); continue
  p=r.choices(choices,weights=[x.weight for x in choices],k=1)[0]
  trace.append((symbol,p.targets))
  queue[0:0]=[(x,state,depth+1) for x in p.targets]
 if queue: diagnostics.append("node_budget")
 return tuple(out),tuple(trace),tuple(sorted(set(diagnostics)))
