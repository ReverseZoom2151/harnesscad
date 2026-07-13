from dataclasses import dataclass
@dataclass(frozen=True)
class Handle: name:str; kind:str; generation:int
class Context:
 def __init__(self):self.bindings={};self.generation=0
 def bind(self,name,kind): h=Handle(name,kind,self.generation);self.bindings[name]=h;return h
 def require(self,h,kind):
  if h.generation!=self.generation or self.bindings.get(h.name)!=h:raise ValueError("stale_handle")
  if h.kind!=kind:raise TypeError(f"expected_{kind}")
 def snapshot(self):return self.generation,dict(self.bindings)
 def rollback(self,snapshot):self.generation+=1;self.bindings={k:Handle(k,v.kind,self.generation) for k,v in snapshot[1].items()}
