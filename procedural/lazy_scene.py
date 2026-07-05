def expand(nodes, visible, children, terminal_key):
 out=[]; visited=culled=0
 stack=list(reversed(nodes))
 while stack:
  n=stack.pop(); visited+=1
  if not visible(n): culled+=1; continue
  kids=tuple(children(n))
  if kids: stack.extend(reversed(kids))
  else: out.append(n)
 batches={}
 for n in out:batches.setdefault(terminal_key(n),[]).append(n)
 return tuple(out),{k:tuple(v) for k,v in sorted(batches.items())},{"visited":visited,"culled":culled}
