"""CAD intermediate representation (CAD-IR) -- dependency resolution and skill
instantiation (Xu et al., 2026, "ArtisanCAD: An Industrial-Level CAD Agent with
Expert-Grounded Knowledge Distillation").

ArtisanCAD's core is CAD-IR: an executable procedural representation that encodes
"parameters, ordered operations, MCP tool bindings, dependencies, generated
entities, and verification rules." CAD-IR serves two roles -- a carrier for
distilling expert procedures into reusable *parameterized skills*, and a
procedural scaffold that turns vague prompts into a complete, ordered, executable
operation list. The model/CATIA execution is out of scope, but the IR's
book-keeping is deterministic and testable:

* An **operation** declares the entities it *produces* and the entities it
  *consumes* (its dependencies). A valid CAD-IR program is a DAG: every consumed
  entity must be produced by an earlier operation, with no cycles and no
  redefinition of an entity.
* :func:`topological_order` returns a stable execution order (or raises on a
  missing dependency / cycle), reproducing the "always well-posed" scaffold.
* A **skill** is a parameterized CAD-IR fragment with named parameters and
  defaults; :func:`instantiate_skill` binds arguments to produce a concrete
  operation list, the "distill expert procedure into reusable skill" step.

Deterministic, stdlib-only. Operations and skills are plain dicts / dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence

__all__ = [
    "CadIRError",
    "Operation",
    "Skill",
    "validate_program",
    "topological_order",
    "instantiate_skill",
]


class CadIRError(ValueError):
    """Raised on a malformed CAD-IR program or skill instantiation."""


@dataclass(frozen=True)
class Operation:
    name: str
    op: str
    produces: tuple = ()
    consumes: tuple = ()
    params: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.op:
            raise CadIRError("operation needs a name and an op type")


def _as_ops(ops: Sequence[object]) -> List[Operation]:
    out: List[Operation] = []
    for o in ops:
        if isinstance(o, Operation):
            out.append(o)
        elif isinstance(o, Mapping):
            out.append(
                Operation(
                    name=o["name"],
                    op=o["op"],
                    produces=tuple(o.get("produces", ())),
                    consumes=tuple(o.get("consumes", ())),
                    params=dict(o.get("params", {})),
                )
            )
        else:
            raise CadIRError("operation must be an Operation or mapping")
    return out


def validate_program(ops: Sequence[object]) -> None:
    """Validate CAD-IR structural invariants (raises :class:`CadIRError`).

    Enforces: unique operation names, no entity produced twice, and every
    consumed entity produced somewhere in the program.
    """
    parsed = _as_ops(ops)
    names = [o.name for o in parsed]
    if len(set(names)) != len(names):
        raise CadIRError("duplicate operation names")
    produced: Dict[str, str] = {}
    for o in parsed:
        for e in o.produces:
            if e in produced:
                raise CadIRError(f"entity {e!r} produced by two operations")
            produced[e] = o.name
    for o in parsed:
        for e in o.consumes:
            if e not in produced:
                raise CadIRError(f"operation {o.name!r} consumes undefined entity {e!r}")


def topological_order(ops: Sequence[object]) -> List[str]:
    """Return a stable execution order of operation names.

    Kahn's algorithm over the entity-dependency graph; ties are broken by the
    operation's original index so the order is deterministic. Raises on a cycle.
    """
    parsed = _as_ops(ops)
    validate_program(parsed)
    producer = {e: o.name for o in parsed for e in o.produces}
    by_name = {o.name: o for o in parsed}
    order_index = {o.name: i for i, o in enumerate(parsed)}
    # dependency edges: op depends on producer of each consumed entity.
    deps: Dict[str, set] = {o.name: set() for o in parsed}
    for o in parsed:
        for e in o.consumes:
            dep = producer[e]
            if dep != o.name:
                deps[o.name].add(dep)
    ready = sorted((n for n, d in deps.items() if not d), key=lambda n: order_index[n])
    result: List[str] = []
    remaining = dict(deps)
    while ready:
        n = ready.pop(0)
        result.append(n)
        del remaining[n]
        newly = []
        for m, d in remaining.items():
            if n in d:
                d.discard(n)
                if not d:
                    newly.append(m)
        for m in sorted(newly, key=lambda x: order_index[x]):
            # insert keeping ready sorted by original index
            ready.append(m)
        ready.sort(key=lambda x: order_index[x])
    if remaining:
        raise CadIRError(f"dependency cycle among {sorted(remaining)}")
    return result


@dataclass(frozen=True)
class Skill:
    name: str
    parameters: Mapping[str, object]  # param name -> default (or None if required)
    template: Sequence[Mapping[str, object]]  # op templates; values may be "$param"

    def required(self) -> List[str]:
        return [k for k, v in self.parameters.items() if v is None]


def instantiate_skill(skill: Skill, arguments: Mapping[str, object]) -> List[Operation]:
    """Bind ``arguments`` into a skill's template, producing concrete operations.

    Missing required parameters raise :class:`CadIRError`; unknown arguments are
    rejected. A template value of the form ``"$param"`` is substituted by the
    bound value. The resulting program is validated before return.
    """
    unknown = set(arguments) - set(skill.parameters)
    if unknown:
        raise CadIRError(f"unknown arguments: {sorted(unknown)}")
    bindings = dict(skill.parameters)
    bindings.update(arguments)
    missing = [k for k in skill.required() if arguments.get(k) is None]
    if missing:
        raise CadIRError(f"missing required parameters: {sorted(missing)}")

    def sub(value):
        if isinstance(value, str) and value.startswith("$"):
            key = value[1:]
            if key not in bindings:
                raise CadIRError(f"template references unbound parameter {key!r}")
            return bindings[key]
        return value

    ops: List[Operation] = []
    for tmpl in skill.template:
        params = {k: sub(v) for k, v in dict(tmpl.get("params", {})).items()}
        ops.append(
            Operation(
                name=str(sub(tmpl["name"])),
                op=str(tmpl["op"]),
                produces=tuple(sub(e) for e in tmpl.get("produces", ())),
                consumes=tuple(sub(e) for e in tmpl.get("consumes", ())),
                params=params,
            )
        )
    validate_program(ops)
    return ops
