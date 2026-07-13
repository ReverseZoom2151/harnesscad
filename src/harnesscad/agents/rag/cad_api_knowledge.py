from dataclasses import dataclass
@dataclass(frozen=True)
class API: name:str; signature:str; returns:str; example:tuple[str,...]
def validate(apis):
 names={x.name for x in apis};issues=[]
 for x in apis:
  if not x.signature:issues.append(f"signature:{x.name}")
  if not x.returns:issues.append(f"returns:{x.name}")
  issues += [f"stale:{x.name}:{call}" for call in x.example if call not in names]
 return tuple(issues)
def chunks(apis):return tuple(f"{x.name}\n{x.signature}\nreturns {x.returns}\nexamples {','.join(x.example)}" for x in sorted(apis,key=lambda x:x.name))
