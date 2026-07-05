class IterativeEditPolicy:
 def __init__(self,max_rounds=5): self.max_rounds=max_rounds
 def choose(self,current,candidate,history):
  if len(history)>=self.max_rounds:return current,"budget"
  sig=candidate["digest"]
  if sig in history:return current,"loop"
  if not candidate["valid"] or candidate["alignment"]<current["alignment"]:return current,"rollback"
  return candidate,"accept"
