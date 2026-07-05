"""Enumerate base↔variant and variant↔variant edit pairs with lineage."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib

@dataclass(frozen=True)
class EditPair:
    original: object; edited: object; original_id: str; edited_id: str; lineage: str; direction: str

def enumerate_pairs(base,variants,*,identity=repr):
    items=[("base",base)]+[(f"v{i}",v) for i,v in enumerate(variants)]
    out=[]; seen=set()
    for i,(aid,a) in enumerate(items):
        for j,(bid,b) in enumerate(items):
            if i==j or identity(a)==identity(b):continue
            key=(identity(a),identity(b))
            if key in seen:continue
            seen.add(key); lineage=hashlib.sha256(f"{identity(base)}\0{aid}\0{bid}".encode()).hexdigest()
            direction="base-forward" if i==0 else ("base-reverse" if j==0 else "cross")
            out.append(EditPair(a,b,aid,bid,lineage,direction))
    return tuple(out)
