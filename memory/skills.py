"""SkillLibrary — a Voyager-style, execution-verified library of parametric
CAD skills that grows monotonically (blueprint sec.8).

A **skill** is a named, parameterised op-template: given params it expands to a
``list[cisp.ops.Op]`` — the exact same op stream the agent would emit by hand.
That makes a skill trivially *verifiable*: run its expanded ops through a
``HarnessSession`` and, per Voyager, admit it to the library **only if the batch
verifies (``ok == True``)**. The library therefore only ever contains skills
whose geometry actually builds — it is monotonic and trustworthy, and it
improves the harness with zero training.

Design notes:
  - ``register`` adds a skill unconditionally (useful for building blocks /
    composition). ``add_verified`` is the gated, Voyager path.
  - ``find`` routes by lightweight text similarity over name + description
    (pluggable; a real embedder is the future upgrade, same as MemoryStore).
  - Persistence stores names + descriptions + param schemas (JSON). The expander
    *functions* live in code and are re-attached on load via an ``expanders``
    map — you cannot serialise executable geometry rules, only their metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from cisp.ops import (
    Op, NewSketch, AddCircle, AddRectangle, Constrain, Extrude, Boolean,
)

# expander: (**params) -> list[Op]
Expander = Callable[..., List[Op]]

# similarity strategy is shared with the memory store
from memory.store import Similarity, TokenOverlapSimilarity


@dataclass
class Skill:
    """A named, parameterised op-template.

    ``params`` is a lightweight schema: name -> {"type": ..., "default": ...,
    "doc": ...}. ``sample_params`` are concrete values used to EXECUTE-verify the
    skill (defaults are derived from the schema when not given).
    """

    name: str
    description: str
    template: Expander
    params: Dict[str, dict] = field(default_factory=dict)
    sample_params: Dict[str, Any] = field(default_factory=dict)
    verified: bool = False

    def defaults(self) -> Dict[str, Any]:
        return {k: spec.get("default") for k, spec in self.params.items()
                if "default" in spec}

    def verify_params(self) -> Dict[str, Any]:
        merged = self.defaults()
        merged.update(self.sample_params)
        return merged

    def expand(self, **params: Any) -> List[Op]:
        merged = self.defaults()
        merged.update(params)
        return list(self.template(**merged))

    # --- metadata (de)serialisation (expander stays in code) --------------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "params": self.params,
            "sample_params": self.sample_params,
            "verified": self.verified,
        }


class SkillLibrary:
    def __init__(self, similarity: Optional[Similarity] = None) -> None:
        self.similarity: Similarity = similarity or TokenOverlapSimilarity()
        self._skills: Dict[str, Skill] = {}

    # --- registration -----------------------------------------------------
    def register(self, skill: Skill) -> Skill:
        """Add a skill unconditionally (no execution gate)."""
        self._skills[skill.name] = skill
        return skill

    def add_verified(self, skill: Skill, session_factory: Callable[[], Any],
                     params: Optional[Dict[str, Any]] = None) -> bool:
        """Voyager gate: EXECUTE the skill's expanded ops through a fresh
        HarnessSession and admit it ONLY if the batch verifies (ok == True).

        ``session_factory`` returns a fresh session (e.g.
        ``lambda: HarnessSession(StubBackend())``) — injected so this module
        stays free of a hard dependency on loop.py / a specific backend.

        Returns True if admitted (now in the library, verified), False if the
        skill's ops fail to apply/verify (rejected, library unchanged — the
        monotonic-trust invariant).
        """
        use = params if params is not None else skill.verify_params()
        try:
            ops = skill.expand(**use)
            session = session_factory()
            result = session.apply_ops(ops)
        except Exception:
            return False
        if not getattr(result, "ok", False):
            return False
        skill.verified = True
        self._skills[skill.name] = skill
        return True

    # --- lookup -----------------------------------------------------------
    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def get(self, name: str) -> Skill:
        return self._skills[name]

    def names(self) -> List[str]:
        return list(self._skills)

    def expand(self, name: str, **params: Any) -> List[Op]:
        return self._skills[name].expand(**params)

    def find(self, query: str, k: int = 3) -> List[Skill]:
        """Return the k skills most similar to `query` (name + description)."""
        scored: List[Tuple[float, int, Skill]] = []
        for i, sk in enumerate(self._skills.values()):
            doc = f"{sk.name} {sk.description}"
            scored.append((self.similarity.score(query, doc), i, sk))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [sk for _, _, sk in scored[:k]]

    # --- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        return {"version": 1, "skills": [s.to_dict() for s in self._skills.values()]}

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str, expanders: Dict[str, Expander],
             similarity: Optional[Similarity] = None) -> "SkillLibrary":
        """Rebuild a library from JSON metadata, re-attaching expander functions
        from `expanders` (name -> callable). A skill whose expander is missing is
        loaded as metadata only and cannot expand until re-registered in code.
        """
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        lib = cls(similarity=similarity)
        for s in d.get("skills", []):
            name = s["name"]
            template = expanders.get(name, _missing_expander(name))
            lib._skills[name] = Skill(
                name=name,
                description=s.get("description", ""),
                template=template,
                params=s.get("params", {}),
                sample_params=s.get("sample_params", {}),
                verified=s.get("verified", False),
            )
        return lib


def _missing_expander(name: str) -> Expander:
    def _raise(**_params: Any) -> List[Op]:
        raise RuntimeError(
            f"skill '{name}' was loaded from metadata only; re-register its "
            f"expander in code before expanding")
    return _raise


# ---------------------------------------------------------------------------
# Example skills (ship 1-2, blueprint sec.8). Ids follow StubBackend's scheme:
# the first NewSketch -> 'sk1', its first primitive -> 'e1', first feature 'f1'.
# ---------------------------------------------------------------------------
def plate_ops(w: float = 10.0, h: float = 10.0, thickness: float = 2.0) -> List[Op]:
    """A fully-constrained rectangular plate: sketch a rectangle, pin its 4 DOF,
    extrude to `thickness`. Verifies clean (dof 0, solid present)."""
    return [
        NewSketch(plane="XY"),
        AddRectangle(sketch="sk1", x=0.0, y=0.0, w=w, h=h),
        Constrain(kind="horizontal", a="e1"),
        Constrain(kind="vertical", a="e1"),
        Constrain(kind="distance", a="e1", value=w),
        Constrain(kind="distance", a="e1", value=h),
        Extrude(sketch="sk1", distance=thickness),
    ]


def bracket_ops(w: float = 20.0, h: float = 20.0, thickness: float = 3.0,
                hole_r: float = 3.0) -> List[Op]:
    """A plate with a through-hole cut: build the plate solid (f1), then a
    circular boss (f2) and boolean-cut it out. Verifies (two features -> cut)."""
    ops = plate_ops(w=w, h=h, thickness=thickness)          # sk1, e1, f1
    ops += [
        NewSketch(plane="XY"),                              # sk2
        AddCircle(sketch="sk2", cx=w / 2.0, cy=h / 2.0, r=hole_r),  # e2
        Constrain(kind="distance", a="e2", value=w / 2.0),
        Constrain(kind="distance", a="e2", value=h / 2.0),
        Constrain(kind="radius", a="e2", value=hole_r),
        Extrude(sketch="sk2", distance=thickness),          # f2
        Boolean(kind="cut", target="f1", tool="f2"),
    ]
    return ops


def plate_skill() -> Skill:
    return Skill(
        name="plate",
        description="A flat rectangular plate of width w, height h and given "
                    "thickness; a fully-constrained base feature.",
        template=plate_ops,
        params={
            "w": {"type": "float", "default": 10.0, "doc": "plate width (x)"},
            "h": {"type": "float", "default": 10.0, "doc": "plate height (y)"},
            "thickness": {"type": "float", "default": 2.0, "doc": "extrude depth"},
        },
        sample_params={"w": 10.0, "h": 10.0, "thickness": 2.0},
    )


def bracket_skill() -> Skill:
    return Skill(
        name="bracket",
        description="A mounting bracket: a rectangular plate with a central "
                    "through-hole of radius hole_r, cut with a boolean.",
        template=bracket_ops,
        params={
            "w": {"type": "float", "default": 20.0, "doc": "plate width (x)"},
            "h": {"type": "float", "default": 20.0, "doc": "plate height (y)"},
            "thickness": {"type": "float", "default": 3.0, "doc": "extrude depth"},
            "hole_r": {"type": "float", "default": 3.0, "doc": "hole radius"},
        },
        sample_params={"w": 20.0, "h": 20.0, "thickness": 3.0, "hole_r": 3.0},
    )


def default_expanders() -> Dict[str, Expander]:
    """name -> expander, for SkillLibrary.load re-attachment."""
    return {"plate": plate_ops, "bracket": bracket_ops}


def build_default_library(session_factory: Callable[[], Any],
                          similarity: Optional[Similarity] = None) -> SkillLibrary:
    """A SkillLibrary seeded with the example skills, each execution-verified."""
    lib = SkillLibrary(similarity=similarity)
    for sk in (plate_skill(), bracket_skill()):
        lib.add_verified(sk, session_factory)
    return lib
