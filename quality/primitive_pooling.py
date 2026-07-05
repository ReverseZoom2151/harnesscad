from dataclasses import dataclass
@dataclass(frozen=True)
class Span: kind:str; start:int; end:int
def spans(tokens,max_primitives=12):
 out=[];start=0
 for i,t in enumerate(tokens):
  if t in {"curve_end","loop_end","face_end","sketch_end","extrusion_end"}:
   out.append(Span("curve" if t=="curve_end" else t[:-4],start,i+1));start=i+1
 if start<len(tokens):out.append(Span("tail",start,len(tokens)))
 covered=tuple(i for x in out for i in range(x.start,x.end))
 return tuple(out),{"exact_coverage":covered==tuple(range(len(tokens))),"overflow":max(0,len(out)-max_primitives)}
