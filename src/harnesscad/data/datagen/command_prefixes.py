"""Split-safe post-solid command checkpoint augmentation."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib

@dataclass(frozen=True)
class CommandPrefix:
    id: str; parent_id: str; split: str; endpoint: int; commands: tuple[object,...]

def post_solid_prefixes(parent_id,split,commands,*,is_checkpoint):
    values=tuple(commands); out=[]
    for endpoint in range(1,len(values)+1):
        prefix=values[:endpoint]
        if is_checkpoint(prefix):
            digest=hashlib.sha256(f"{parent_id}\0{endpoint}\0{prefix!r}".encode()).hexdigest()
            out.append(CommandPrefix(digest,parent_id,split,endpoint,prefix))
    return tuple(out)

def assert_split_before_expand(prefixes):
    owners={}
    for item in prefixes:
        old=owners.setdefault(item.parent_id,item.split)
        if old!=item.split:raise ValueError(f"parent split leakage: {item.parent_id}")
