from dataclasses import dataclass
import hashlib,json
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()
@dataclass(frozen=True)
class EditTriplet: instruction:str; original_digest:str; edited_digest:str; original_render_digest:str; edited_render_digest:str; edits:int
def build(instruction,original,edited,original_render,edited_render,*,executable=True,max_edits=3):
 edits=sum(a!=b for a,b in zip(original,edited))+abs(len(original)-len(edited))
 if not executable or not 1<=edits<=max_edits or original_render==edited_render:return None
 return EditTriplet(instruction,digest(original),digest(edited),digest(original_render),digest(edited_render),edits)
