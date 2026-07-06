"""Knowledge inference and design-rule synthesis for CAD sequences
(Li & Sha, *Image2CADSeq*, 2024, Sec. 4.4, 5.1 & 6.1).

Beyond reconstructing geometry, the paper's motivation is that a CAD *sequence*
"provides access to the historical construction process and associated design
knowledge". This module makes the deterministic knowledge pieces concrete:

1. **Shape-category (template) inference from the op-type signature.** Different
   template shapes correspond to distinct op-type sequences (paper Table 5): a
   cylinder is ``[sketch, circle, extrude]``, a triangular prism is
   ``[sketch, line, line, line, extrude]``, etc. :func:`op_type_signature` and
   :func:`infer_template` recover the shape category (the "shape category
   determined by the sequence of CAD operation types", Sec. 6.4).

2. **Part-attribute inference.** :func:`part_attributes` counts sketches,
   curves, extrudes, and boolean roles - the structured "knowledge" a designer
   would read off a construction history.

3. **Design-rule schema + synthesis.** The paper contrasts two data-synthesis
   strategies (Sec. 5.1): *without rules* (parameters sampled independently)
   and *with rules* (parameters derived from earlier ones, e.g. "the extrusion
   depth of a circle is determined by the coordinates of its center point").
   :class:`DesignRule` captures such a parametric relation; :func:`apply_rules`
   makes a sequence rule-consistent, :func:`check_rules` verifies whether an
   inferred sequence honours the embedded design intent (the knowledge-recovery
   test), and :func:`synthesize_program` deterministically generates a random-
   or rule-based program from a template.

Op-type constants mirror the Sim-Gallery DSL. Pure and deterministic; any
randomness is seeded via :class:`random.Random`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# Op-type constants (kept local; identical values to the gallery DSL module).
ADD_SKETCH = 0
ADD_LINE = 1
ADD_ARC = 2
ADD_CIRCLE = 3
ADD_EXTRUDE = 4
SOP = 5
EOP = 6

_CURVE_TYPES = frozenset({ADD_LINE, ADD_ARC, ADD_CIRCLE})

# Canonical template shapes keyed by op-type signature (paper Table 5).
# Values give a human-readable category name.
TEMPLATE_SIGNATURES: dict[tuple[int, ...], str] = {
    (ADD_SKETCH, ADD_CIRCLE, ADD_EXTRUDE): "cylinder",
    (ADD_SKETCH, ADD_LINE, ADD_LINE, ADD_LINE, ADD_EXTRUDE): "triangular_prism",
    (ADD_SKETCH, ADD_LINE, ADD_LINE, ADD_LINE, ADD_LINE, ADD_EXTRUDE): "rectangular_prism",
    (ADD_SKETCH, ADD_ARC, ADD_ARC, ADD_EXTRUDE): "lens_prism",
    (ADD_SKETCH, ADD_LINE, ADD_ARC, ADD_LINE, ADD_EXTRUDE): "rounded_prism",
}


# --- signature / template inference ----------------------------------------
def op_type_signature(op_types) -> tuple[int, ...]:
    """The CAD-op signature of a program: op types with SOP/EOP markers removed."""
    return tuple(t for t in op_types if t not in (SOP, EOP))


def infer_template(op_types) -> str | None:
    """Infer the template shape category from an op-type sequence, or ``None``
    if the signature matches no known template."""
    return TEMPLATE_SIGNATURES.get(op_type_signature(op_types))


# --- part-attribute inference ----------------------------------------------
@dataclass(frozen=True)
class PartAttributes:
    """Structured design knowledge read off a CAD construction history."""

    n_sketches: int
    n_lines: int
    n_arcs: int
    n_circles: int
    n_extrudes: int
    n_curves: int
    template: str | None


def part_attributes(op_types) -> PartAttributes:
    """Infer part attributes (counts + template) from an op-type sequence."""
    sig = op_type_signature(op_types)
    n_lines = sig.count(ADD_LINE)
    n_arcs = sig.count(ADD_ARC)
    n_circles = sig.count(ADD_CIRCLE)
    return PartAttributes(
        n_sketches=sig.count(ADD_SKETCH),
        n_lines=n_lines,
        n_arcs=n_arcs,
        n_circles=n_circles,
        n_extrudes=sig.count(ADD_EXTRUDE),
        n_curves=n_lines + n_arcs + n_circles,
        template=infer_template(op_types),
    )


# --- design-rule schema ----------------------------------------------------
@dataclass(frozen=True)
class DesignRule:
    """A parametric design rule linking a *target* parameter to a *source*.

    ``target`` and ``source`` are ``(op_index, slot)`` addresses into a program
    (``slot`` being one of ``"x", "y", "alpha", "r", "d"``). The rule asserts
    ``target = scale * source + offset`` - e.g. "extrude depth = 2 * circle
    center-x" encodes the paper's example that "the extrusion depth of a circle
    is determined by the coordinates of its center point".
    """

    target: tuple[int, str]
    source: tuple[int, str]
    scale: float = 1.0
    offset: float = 0.0
    name: str = ""

    def evaluate(self, program) -> float:
        """Value the target *should* take given the current source parameter."""
        si, ss = self.source
        return self.scale * _get(program[si], ss) + self.offset

    def holds(self, program, tol: float = 1e-6) -> bool:
        """Whether the program currently satisfies the rule within ``tol``."""
        ti, ts = self.target
        return abs(_get(program[ti], ts) - self.evaluate(program)) <= tol


# Programs here are lists of mutable parameter dicts, one per CAD op:
#   {"t": <op type>, "x":..., "y":..., "alpha":..., "r":..., "d":...}
_PARAM_KEYS = ("x", "y", "alpha", "r", "d")


def _get(op: dict, slot: str) -> float:
    return op.get(slot, 0.0)


def apply_rules(program, rules) -> list[dict]:
    """Return a copy of ``program`` with every design rule enforced.

    Rules are applied in order; a rule's target is overwritten with its
    evaluated value (this embeds the design knowledge into the data, paper's
    "dataset with rules").
    """
    out = [dict(op) for op in program]
    for rule in rules:
        ti, ts = rule.target
        out[ti][ts] = rule.evaluate(out)
    return out


def check_rules(program, rules, tol: float = 1e-6) -> dict:
    """Verify which design rules a (possibly predicted) program honours.

    This is the knowledge-recovery test: given an inferred CAD sequence, does it
    preserve the design intent embedded at synthesis time? Returns per-rule
    satisfaction plus an overall ``fidelity`` fraction.
    """
    results = {}
    for i, rule in enumerate(rules):
        key = rule.name or f"rule_{i}"
        results[key] = rule.holds(program, tol)
    satisfied = sum(results.values())
    return {"per_rule": results,
            "satisfied": satisfied,
            "total": len(rules),
            "fidelity": satisfied / len(rules) if rules else 1.0}


# --- deterministic synthesis (with / without rules) ------------------------
def _random_op(op_type: int, rng: random.Random) -> dict:
    """Sample a single op's continuous parameters within the paper's ranges."""
    op = {"t": op_type}
    if op_type == ADD_LINE or op_type == ADD_ARC:
        op["x"] = rng.uniform(-1.0, 1.0)
        op["y"] = rng.uniform(-1.0, 1.0)
        if op_type == ADD_ARC:
            op["alpha"] = rng.uniform(-1.0, 1.0)
    elif op_type == ADD_CIRCLE:
        op["x"] = rng.uniform(-1.0, 1.0)
        op["y"] = rng.uniform(-1.0, 1.0)
        op["r"] = rng.uniform(1e-3, 1.0)
    elif op_type == ADD_EXTRUDE:
        op["d"] = rng.uniform(-1.0, 1.0)
    return op


def synthesize_program(signature, seed: int, rules=None) -> list[dict]:
    """Deterministically synthesise one CAD program from an op-type signature.

    With ``rules=None`` this is the paper's *dataset-without-rules* strategy
    (independent random parameters); passing ``rules`` additionally enforces
    them (the *dataset-with-rules* strategy that embeds design knowledge).
    """
    rng = random.Random(seed)
    program = [_random_op(t, rng) for t in signature]
    if rules:
        program = apply_rules(program, rules)
    return program


# --- knowledge record ------------------------------------------------------
@dataclass(frozen=True)
class KnowledgeRecord:
    """A compact bundle of inferred design knowledge for one CAD sequence."""

    template: str | None
    attributes: PartAttributes
    rule_fidelity: float
    rules_satisfied: int
    rules_total: int


def infer_knowledge(op_types, program=None, rules=None) -> KnowledgeRecord:
    """Infer a :class:`KnowledgeRecord` from an op-type sequence and optional
    parameterised program + design rules."""
    attrs = part_attributes(op_types)
    if program is not None and rules:
        rc = check_rules(program, rules)
        fidelity, sat, total = rc["fidelity"], rc["satisfied"], rc["total"]
    else:
        fidelity, sat, total = 1.0, 0, 0
    return KnowledgeRecord(
        template=attrs.template,
        attributes=attrs,
        rule_fidelity=fidelity,
        rules_satisfied=sat,
        rules_total=total,
    )
