from dataclasses import dataclass
@dataclass(frozen=True)
class Challenge: id:str; category:str; frames:tuple
def fixtures():
 return (Challenge("perpendicular","orientation",((0,0,0),(1.5707963267948966,0,0))),
 Challenge("offset-repeat","position",((0,0,0),(10,5,0))),
 Challenge("asymmetric-mirror","symmetry",((-1,0,0),(1,0,0))))
def stratify(results):
 out={}
 for challenge,value in results:out.setdefault(challenge.category,[]).append(value)
 return {k:sum(v)/len(v) for k,v in sorted(out.items())}
