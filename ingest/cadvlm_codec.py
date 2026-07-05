"""Versioned CadVLM-compatible sketch token codec."""
from __future__ import annotations
from dataclasses import dataclass
from math import dist

VERSION="cadvlm-sketch-v1"; LOW=1; HIGH=64
CONSTRAINT_TOKENS={"coincident":65,"concentric":66,"equal":67,"fix":68,
 "horizontal":69,"midpoint":70,"normal":71,"offset":72,"parallel":73,
 "perpendicular":74,"quadrant":75,"tangent":76,"vertical":77}
TOKEN_CONSTRAINTS={v:k for k,v in CONSTRAINT_TOKENS.items()}

@dataclass(frozen=True)
class SketchFrame:
    center: tuple[float,float]; width: float
    def __post_init__(self):
        if self.width<=0:raise ValueError("frame width must be positive")

def fit_frame(points):
    values=tuple(points)
    if not values:return SketchFrame((0,0),1)
    xs,ys=zip(*values);width=max(max(xs)-min(xs),max(ys)-min(ys),1e-12)
    return SketchFrame(((max(xs)+min(xs))/2,(max(ys)+min(ys))/2),width)
def _q(v):return min(HIGH,max(LOW,round(LOW+(v+.5)*(HIGH-LOW))))
def _dq(t):
    if not LOW<=t<=HIGH:raise ValueError("coordinate token outside [1,64]")
    return (t-LOW)/(HIGH-LOW)-.5
def encode_point(point,frame):
    return (_q((point[0]-frame.center[0])/frame.width),
            _q((point[1]-frame.center[1])/frame.width))
def decode_point(tokens,frame):
    return (frame.center[0]+_dq(tokens[0])*frame.width,
            frame.center[1]+_dq(tokens[1])*frame.width)
def encode_entity(entity,frame):
    kind=entity["type"]
    if kind=="line":points=(entity["start"],entity["end"])
    elif kind=="arc":points=(entity["start"],entity["mid"],entity["end"])
    elif kind=="circle":points=tuple(entity["points"])
    else:points=None
    if points is None or (kind=="circle" and len(points)!=4):raise ValueError("invalid entity")
    return (kind,)+tuple(v for p in points for v in encode_point(p,frame))
def decode_entity(tokens,frame):
    kind=tokens[0];counts={"line":2,"arc":3,"circle":4}
    if kind not in counts or len(tokens)!=1+2*counts[kind]:raise ValueError("invalid entity tokens")
    pts=tuple(decode_point(tokens[i:i+2],frame) for i in range(1,len(tokens),2))
    keys={"line":("start","end"),"arc":("start","mid","end")}
    if kind=="circle":
        center=(sum(p[0] for p in pts)/4,sum(p[1] for p in pts)/4)
        return {"type":kind,"points":pts,"center":center,
                "radius":sum(dist(center,p) for p in pts)/4}
    return {"type":kind,**dict(zip(keys[kind],pts))}
def encode_constraint(kind,references):
    if kind not in CONSTRAINT_TOKENS or not references:raise ValueError("bad constraint")
    return (CONSTRAINT_TOKENS[kind],)+tuple(int(x) for x in references)
def decode_constraint(tokens):
    if not tokens or tokens[0] not in TOKEN_CONSTRAINTS:return None
    if len(tokens)<2:raise ValueError("constraint requires references")
    return TOKEN_CONSTRAINTS[tokens[0]],tuple(tokens[1:])
