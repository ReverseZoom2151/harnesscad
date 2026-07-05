"""Seeded, legality-aware parent-script family expansion."""

from __future__ import annotations
from dataclasses import dataclass
import hashlib, random
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class ParentTemplate:
    family: str
    imports: str
    construction: str
    main: str
    axes: Mapping[str, Sequence[object]]

    def render(self, parameters: Mapping[str, object]) -> str:
        values=dict(parameters)
        return "\n\n".join((self.imports.strip(),
                            self.construction.format(**values).strip(),
                            self.main.format(**values).strip())) + "\n"


@dataclass(frozen=True)
class ScriptVariant:
    index: int; parameters: Mapping[str, object]; script: str
    digest: str; rejected_before: int


@dataclass(frozen=True)
class Expansion:
    variants: tuple[ScriptVariant,...]; attempts: int; rejection_reasons: Mapping[str,int]


def expand(template: ParentTemplate, n: int, seed: int,
           legal: Callable[[Mapping[str,object]], bool|tuple[bool,str]],
           *, max_attempts=1000) -> Expansion:
    rng=random.Random(seed); out=[]; reasons={}; attempts=0; seen=set(); since=0
    names=tuple(sorted(template.axes))
    while len(out)<n and attempts<max_attempts:
        attempts+=1
        params={name:rng.choice(tuple(template.axes[name])) for name in names}
        key=tuple((name,repr(params[name])) for name in names)
        verdict=legal(params); ok,reason=(verdict if isinstance(verdict,tuple) else (verdict,"illegal"))
        if key in seen: ok,reason=False,"duplicate"
        if not ok:
            reasons[reason]=reasons.get(reason,0)+1; since+=1; continue
        seen.add(key); script=template.render(params)
        out.append(ScriptVariant(len(out),params,script,
                                 hashlib.sha256(script.encode()).hexdigest(),since))
        since=0
    if len(out)<n: raise RuntimeError(f"generated {len(out)} of {n} variants")
    return Expansion(tuple(out),attempts,dict(sorted(reasons.items())))
