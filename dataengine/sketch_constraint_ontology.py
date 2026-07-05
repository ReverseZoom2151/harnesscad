"""Canonical CadVLM constraint ontology and conversion aliases."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ConstraintKind:
    name: str; token: int; minimum_references: int; aliases: tuple[str,...]=()
_DATA=(("coincident",65,2),("concentric",66,2),("equal",67,2),("fix",68,1),
("horizontal",69,1),("midpoint",70,2),("normal",71,2),("offset",72,2),
("parallel",73,2),("perpendicular",74,2),("quadrant",75,2),("tangent",76,2),
("vertical",77,1))
KINDS=tuple(ConstraintKind(*x) for x in _DATA)
BY_NAME={x.name:x for x in KINDS};BY_TOKEN={x.token:x for x in KINDS}
ALIASES={"coincident":"coincident","coincidence":"coincident","perp":"perpendicular",
         "normal":"normal","fixed":"fix"}
def resolve(value):
    if isinstance(value,int):
        if value not in BY_TOKEN:raise KeyError(value)
        return BY_TOKEN[value]
    key=ALIASES.get(str(value).casefold(),str(value).casefold())
    if key not in BY_NAME:raise KeyError(value)
    return BY_NAME[key]
def validate_constraint(kind,references):
    item=resolve(kind)
    return () if len(tuple(references))>=item.minimum_references else ("insufficient-references",)
