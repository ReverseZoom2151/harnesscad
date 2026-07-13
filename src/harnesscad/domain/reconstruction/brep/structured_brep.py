"""Fixed-width face-edge-vertex hierarchy encoding."""
from dataclasses import dataclass
import random

@dataclass(frozen=True)
class Node:
    id: str
    kind: str
    samples: tuple[tuple[float,float,float], ...]
    children: tuple["Node", ...] = ()

def pad_children(children, width, seed=0):
    items=list(children)
    if width < len(items) or (width and not items): raise ValueError("invalid width")
    rng=random.Random(seed)
    while len(items)<width: items.append(items[rng.randrange(len(items))])
    return tuple(items)

def unique_children(nodes):
    return tuple({n.id:n for n in nodes}.values())

def validate_tree(root):
    issues=[]
    def walk(n):
        if n.kind=="edge" and len(unique_children(n.children)) != 2:
            issues.append(f"edge_vertices:{n.id}")
        for c in n.children: walk(c)
    walk(root); return tuple(sorted(set(issues)))
