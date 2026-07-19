"""SkillLibrary — an execution-verified library of parametric
CAD skills that grows monotonically (blueprint sec.8).

A **skill** is a named, parameterised op-template: given params it expands to a
``list[cisp.ops.Op]`` — the exact same op stream the agent would emit by hand.
That makes a skill trivially *verifiable*: run its expanded ops through a
``HarnessSession`` and admit it to the library **only if the batch
verifies (``ok == True``)**. The library therefore only ever contains skills
whose geometry actually builds — it is monotonic and trustworthy, and it
improves the harness with zero training.

Design notes:
  - ``register`` adds a skill unconditionally (useful for building blocks /
    composition). ``add_verified`` is the gated, verification path.
  - ``find`` routes by lightweight text similarity over name + description
    (pluggable; a real embedder is the future upgrade, same as MemoryStore).
  - Persistence stores names + descriptions + param schemas (JSON). The expander
    *functions* live in code and are re-attached on load via an ``expanders``
    map — you cannot serialise executable geometry rules, only their metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from harnesscad.core.cisp.ops import (
    Op, NewSketch, AddCircle, AddRectangle, Constrain, Extrude, Boolean, parse_op,
)

# expander: (**params) -> list[Op]
Expander = Callable[..., List[Op]]

# similarity strategy is shared with the memory store
import harnesscad.agents.memory.persistence as persistence
from harnesscad.agents.memory.similarity import default_similarity
from harnesscad.agents.memory.store import Similarity


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
        self.similarity: Similarity = similarity or default_similarity()
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

    def learn_from_correction(
        self,
        name: str,
        corrected_ops: Sequence[Any],
        session_factory: Callable[[], Any],
        *,
        description: str = "",
        params: Optional[Dict[str, dict]] = None,
        sample_params: Optional[Dict[str, Any]] = None,
        oracle: Optional[Callable[[Any, List[Op]], Any]] = None,
        overwrite: bool = False,
    ) -> bool:
        """Generalise a HUMAN correction of a produced part into a verified skill.

        The corrective loop (blueprint 26.7.4): a person fixes a bad part -- by
        editing its op stream or supplying a corrected trajectory from a
        diagnostic -- and that fix becomes a reusable, named construction pattern
        instead of a one-off. ``corrected_ops`` is the fixed op stream (CISP ``Op``
        objects OR their serialised dicts, e.g. straight out of an
        :class:`~harnesscad.agents.memory.store.Episode`); it is frozen into a
        skill template.

        This is deliberately the SAME Voyager gate as :meth:`add_verified`: the
        corrected ops are EXECUTED on a fresh session and the skill is admitted
        ONLY if they actually build (``ok == True``). A human saying "this is the
        fix" is not enough -- a correction that does not itself verify is
        unverified plausible garbage and is refused, keeping the library
        monotonically trustworthy. Pass a stronger ``oracle`` (e.g. the harness's
        measured output gate, ``callable(session, ops) -> verdict`` with an ``ok``
        attribute) to additionally require the built geometry pass that gate.

        Returns True if the corrected skill was admitted (now in the library,
        verified), False otherwise (library unchanged). ``overwrite`` guards an
        existing name: without it, a name already present is left untouched.
        """
        if name in self._skills and not overwrite:
            return False
        ops = _freeze_ops(corrected_ops)
        if not ops:
            return False
        skill = Skill(
            name=name,
            description=description or f"correction-derived skill '{name}'",
            template=_const_expander(ops),
            params=dict(params or {}),
            sample_params=dict(sample_params or {}),
        )
        try:
            expanded = skill.expand(**skill.verify_params())
            session = session_factory()
            result = session.apply_ops(expanded)
        except Exception:
            return False
        if not getattr(result, "ok", False):
            return False
        if oracle is not None:
            try:
                verdict = oracle(session, expanded)
            except Exception:
                return False
            if not bool(getattr(verdict, "ok", verdict)):
                return False
        skill.verified = True
        self._skills[name] = skill
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
        """Persist skill metadata deterministically and atomically."""
        persistence.dump_json(self.to_dict(), path)

    @classmethod
    def load(cls, path: str, expanders: Dict[str, Expander],
             similarity: Optional[Similarity] = None) -> "SkillLibrary":
        """Rebuild a library from JSON metadata, re-attaching expander functions
        from `expanders` (name -> callable). A skill whose expander is missing is
        loaded as metadata only and cannot expand until re-registered in code.
        """
        d = persistence.load_json(path)
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


def _freeze_ops(items: Sequence[Any]) -> List[Op]:
    """Coerce a corrected op stream to concrete ``Op`` objects.

    Accepts CISP ``Op`` instances (kept as-is) or their serialised dict form
    (rebuilt via :func:`~harnesscad.core.cisp.ops.parse_op`), so a correction can
    be supplied straight from an :class:`Episode`'s stored ``ops`` list or from a
    live op stream. Anything else is skipped; an all-unparseable stream yields
    ``[]`` and the learn call refuses.
    """
    out: List[Op] = []
    for it in items:
        if isinstance(it, Op):
            out.append(it)
        elif isinstance(it, dict):
            try:
                out.append(parse_op(it))
            except Exception:
                continue
    return out


def _const_expander(ops: List[Op]) -> Expander:
    """A skill template that ignores params and re-emits a fixed op stream.

    The corrected geometry is captured literally: a copy is returned each call so
    a caller cannot mutate the stored fix. Correction-derived skills are exact by
    construction; a later revision can register a parameterised expander in code.
    """
    frozen = list(ops)

    def _expand(**_params: Any) -> List[Op]:
        return list(frozen)

    return _expand


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
