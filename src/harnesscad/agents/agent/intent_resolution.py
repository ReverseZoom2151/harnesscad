"""Host-neutral resolution of deterministic natural-language CAD intent."""
from __future__ import annotations
from dataclasses import dataclass
import re

@dataclass(frozen=True)
class IntentEnvelope:
    text: str
    dimensions: tuple[tuple[float,str], ...]
    operations: tuple[str, ...]
    assumptions: tuple[str, ...]
    unresolved: tuple[str, ...]
    seed: int | None
    @property
    def needs_clarification(self): return bool(self.unresolved)

def resolve_intent(text, *, seed=None, default_unit="mm"):
    lower=text.lower()
    dimensions=tuple((float(number),unit or default_unit) for number,unit in
                     re.findall(r"(-?\d+(?:\.\d+)?)\s*(mm|cm|m|in)?",lower))
    operations=tuple(op for op in ("union","cut","intersect","extrude","revolve",
                                    "loft","sweep","bake","simulate")
                     if re.search(rf"\b{op}\w*",lower))
    unresolved=[]; assumptions=[]
    if "random" in lower:
        if seed is None: unresolved.append("random-choice-requires-seed")
        else: assumptions.append(f"random-choice-seed:{seed}")
    if dimensions and any(not unit for _,unit in dimensions):
        assumptions.append(f"default-unit:{default_unit}")
    if not operations: unresolved.append("operation-unspecified")
    return IntentEnvelope(text,dimensions,operations,tuple(assumptions),
                          tuple(unresolved),seed)
