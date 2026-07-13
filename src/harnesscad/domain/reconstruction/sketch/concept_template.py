"""Parameterised sub-sketch *concept* templates (Yang & Pan, "Discovering Design
Concepts for CAD Sketches", NeurIPS 2022).

The paper's central object is a **concept**: a reusable, *parameterised* modular
template over a sub-sketch. A concept is not a fixed piece of geometry -- it is a
structure of primitives and constraints in which the numeric attributes are left
*free* (slots), so a single concept explains many concrete sub-sketches once its
slots are bound. Concepts also expose a typed **interface** so they can be wired
into a larger sketch:

  * **members** -- the primitives the concept owns (line / circle / arc / point),
    each parameter of each member being either a *constant* or a reference to a
    concept **slot** (the free parameters);
  * **constraints** -- relations among the members, and among members and the
    concept's *external inputs*;
  * **input references** (``in_arity``) -- the paper's ``ref_in_argument_num``:
    primitives owned by the *enclosing* context that this concept constrains
    against but does not own;
  * **output references** (``out_refs``) -- the paper's ``ref_out_argument_num``:
    the ordered subset of owned members exported as the concept's interface, i.e.
    the handles a parent may constrain against.

Concepts are **hierarchical** (the paper's abstraction levels,
``max_abstruction_decompose_query``): a concept may instantiate other concepts as
sub-instances (:class:`SubInstance`), binding their slots to its own slots and
their inputs to its own members / inputs. Flattening those hierarchies lives in
:mod:`library.sketchconcept_library`; this module provides the representation,
validation, deterministic *instantiation* (bind slots + inputs -> concrete
primitives and constraints) and a **canonical signature** used for library
deduplication (two concepts that differ only in member naming, member ordering or
slot naming get the same signature).

Everything here is deterministic and pure stdlib. The trained network of the paper
(which *proposes* the concepts) is external; what is reimplemented is the concept
algebra it operates on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations, product
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Flat sketch data model (shared by the whole sketchconcept_* family)
# ---------------------------------------------------------------------------

#: primitive types and their parameter names (SketchGraphs' four types)
PRIMITIVE_PARAMS: Dict[str, Tuple[str, ...]] = {
    "line": ("x1", "y1", "x2", "y2"),
    "circle": ("x", "y", "r"),
    "arc": ("x1", "y1", "x2", "y2", "x3", "y3"),
    "point": ("x", "y"),
}


@dataclass(frozen=True)
class Primitive:
    """A concrete sketch primitive."""

    pid: str
    ptype: str
    params: Tuple[Tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if self.ptype not in PRIMITIVE_PARAMS:
            raise ValueError("unknown primitive type: %s" % (self.ptype,))

    @staticmethod
    def make(pid: str, ptype: str, params: Mapping[str, float]) -> "Primitive":
        return Primitive(pid, ptype, tuple(sorted((str(k), float(v)) for k, v in params.items())))

    def param_map(self) -> Dict[str, float]:
        return dict(self.params)


@dataclass(frozen=True)
class Constraint:
    """A constraint among primitives, referenced by primitive id."""

    ctype: str
    refs: Tuple[str, ...]


@dataclass(frozen=True)
class Sketch:
    """A flat sketch: primitives plus constraints over them."""

    primitives: Tuple[Primitive, ...]
    constraints: Tuple[Constraint, ...] = ()

    def by_id(self) -> Dict[str, Primitive]:
        return {p.pid: p for p in self.primitives}

    def validate(self) -> List[str]:
        errs: List[str] = []
        seen = set()
        for p in self.primitives:
            if p.pid in seen:
                errs.append("duplicate primitive id: %s" % p.pid)
            seen.add(p.pid)
        for i, c in enumerate(self.constraints):
            if not c.refs:
                errs.append("constraint %d has no refs" % i)
            for r in c.refs:
                if r not in seen:
                    errs.append("constraint %d references unknown primitive %s" % (i, r))
        return errs


# ---------------------------------------------------------------------------
# Parameter references: a member parameter is a constant or a free slot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Slot:
    """Reference to a free parameter of the concept."""

    name: str


@dataclass(frozen=True)
class Const:
    """A baked-in constant parameter value."""

    value: float


ParamRef = object  # Slot | Const


def resolve_param(ref: ParamRef, bindings: Mapping[str, float]) -> float:
    """Resolve a :class:`Slot` / :class:`Const` to a number."""
    if isinstance(ref, Const):
        return float(ref.value)
    if isinstance(ref, Slot):
        if ref.name not in bindings:
            raise KeyError("unbound slot: %s" % ref.name)
        return float(bindings[ref.name])
    raise TypeError("not a parameter reference: %r" % (ref,))


# ---------------------------------------------------------------------------
# Concept structure
# ---------------------------------------------------------------------------

#: a member reference inside a concept is one of
#:   "<local_id>"        -- an owned member
#:   "in:<k>"            -- the k-th external input reference
#:   "<sub_id>#<j>"      -- the j-th output of a sub-instance
def input_ref(k: int) -> str:
    return "in:%d" % int(k)


def sub_out_ref(sub_id: str, j: int) -> str:
    return "%s#%d" % (sub_id, int(j))


def parse_ref(ref: str) -> Tuple[str, object]:
    """Classify a reference: ``("input", k)`` / ``("sub", (id, j))`` / ``("member", id)``."""
    if ref.startswith("in:"):
        return "input", int(ref[3:])
    if "#" in ref:
        sid, _, j = ref.partition("#")
        return "sub", (sid, int(j))
    return "member", ref


@dataclass(frozen=True)
class Member:
    """A primitive owned by the concept; parameters are slots or constants."""

    local_id: str
    ptype: str
    params: Tuple[Tuple[str, ParamRef], ...] = ()

    @staticmethod
    def make(local_id: str, ptype: str, params: Mapping[str, ParamRef]) -> "Member":
        if ptype not in PRIMITIVE_PARAMS:
            raise ValueError("unknown primitive type: %s" % (ptype,))
        return Member(local_id, ptype, tuple(sorted(params.items(), key=lambda kv: kv[0])))

    def param_map(self) -> Dict[str, ParamRef]:
        return dict(self.params)


@dataclass(frozen=True)
class ConstraintSpec:
    """A constraint inside a concept; refs are member / input / sub-output refs."""

    ctype: str
    refs: Tuple[str, ...]


@dataclass(frozen=True)
class SubInstance:
    """A nested concept instance (one abstraction level down).

    ``bindings`` maps the *sub-concept's* slot names to parameter references in the
    *enclosing* concept (a :class:`Slot` of the parent, or a :class:`Const`).
    ``inputs`` supplies, in order, the parent-level reference bound to each of the
    sub-concept's input references.
    """

    local_id: str
    concept: str
    bindings: Tuple[Tuple[str, ParamRef], ...] = ()
    inputs: Tuple[str, ...] = ()

    @staticmethod
    def make(local_id: str, concept: str,
             bindings: Optional[Mapping[str, ParamRef]] = None,
             inputs: Sequence[str] = ()) -> "SubInstance":
        b = tuple(sorted((bindings or {}).items(), key=lambda kv: kv[0]))
        return SubInstance(local_id, concept, b, tuple(inputs))

    def binding_map(self) -> Dict[str, ParamRef]:
        return dict(self.bindings)


@dataclass(frozen=True)
class Concept:
    """A parameterised sub-sketch template."""

    name: str
    slots: Tuple[str, ...] = ()
    members: Tuple[Member, ...] = ()
    constraints: Tuple[ConstraintSpec, ...] = ()
    subs: Tuple[SubInstance, ...] = ()
    in_arity: int = 0
    out_refs: Tuple[str, ...] = ()
    defaults: Tuple[Tuple[str, float], ...] = ()

    # --- basic queries ----------------------------------------------------
    @property
    def is_flat(self) -> bool:
        return not self.subs

    def member_ids(self) -> Tuple[str, ...]:
        return tuple(m.local_id for m in self.members)

    def member(self, local_id: str) -> Member:
        for m in self.members:
            if m.local_id == local_id:
                return m
        raise KeyError("no such member: %s" % local_id)

    def default_map(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.defaults}

    def size(self) -> int:
        """Number of owned members plus sub-instances (a proxy for concept size)."""
        return len(self.members) + len(self.subs)

    # --- validation -------------------------------------------------------
    def validate(self) -> List[str]:
        errs: List[str] = []
        ids: List[str] = []
        for m in self.members:
            ids.append(m.local_id)
        for s in self.subs:
            ids.append(s.local_id)
        for i in ids:
            if ids.count(i) > 1:
                errs.append("duplicate local id: %s" % i)
                break
        member_ids = set(self.member_ids())
        sub_ids = {s.local_id for s in self.subs}
        slots = set(self.slots)

        for m in self.members:
            expected = set(PRIMITIVE_PARAMS[m.ptype])
            got = {k for k, _ in m.params}
            if got != expected:
                errs.append("member %s: expected params %s, got %s"
                            % (m.local_id, sorted(expected), sorted(got)))
            for k, ref in m.params:
                if isinstance(ref, Slot) and ref.name not in slots:
                    errs.append("member %s: unknown slot %s" % (m.local_id, ref.name))
                elif not isinstance(ref, (Slot, Const)):
                    errs.append("member %s: bad parameter ref for %s" % (m.local_id, k))

        for s in self.subs:
            for _, ref in s.bindings:
                if isinstance(ref, Slot) and ref.name not in slots:
                    errs.append("sub %s: unknown slot %s" % (s.local_id, ref.name))
                elif not isinstance(ref, (Slot, Const)):
                    errs.append("sub %s: bad binding" % s.local_id)
            for r in s.inputs:
                errs.extend(self._check_ref(r, member_ids, sub_ids, "sub %s input" % s.local_id))

        for i, c in enumerate(self.constraints):
            if not c.refs:
                errs.append("constraint %d has no refs" % i)
            for r in c.refs:
                errs.extend(self._check_ref(r, member_ids, sub_ids, "constraint %d" % i))

        for r in self.out_refs:
            errs.extend(self._check_ref(r, member_ids, sub_ids, "out_ref"))

        for k, _ in self.defaults:
            if k not in slots:
                errs.append("default for unknown slot: %s" % k)
        return errs

    def _check_ref(self, ref: str, member_ids, sub_ids, where: str) -> List[str]:
        kind, payload = parse_ref(ref)
        if kind == "input":
            if not (0 <= int(payload) < self.in_arity):
                return ["%s: input index %s out of range" % (where, payload)]
            return []
        if kind == "sub":
            sid, _j = payload  # type: ignore[misc]
            if sid not in sub_ids:
                return ["%s: unknown sub-instance %s" % (where, sid)]
            return []
        if ref not in member_ids:
            return ["%s: unknown member %s" % (where, ref)]
        return []

    def free_slots(self) -> Tuple[str, ...]:
        """Slots without a default value (must be supplied at instantiation)."""
        d = self.default_map()
        return tuple(s for s in self.slots if s not in d)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Instantiation:
    """The result of applying a concept: concrete primitives + constraints."""

    primitives: Tuple[Primitive, ...]
    constraints: Tuple[Constraint, ...]
    outputs: Tuple[str, ...]  # concrete primitive ids exposed by the interface

    def as_sketch(self) -> Sketch:
        return Sketch(self.primitives, self.constraints)


def instantiate(concept: Concept,
                bindings: Optional[Mapping[str, float]] = None,
                inputs: Sequence[str] = (),
                prefix: str = "") -> Instantiation:
    """Apply a *flat* concept: bind slots and inputs, emit concrete geometry.

    ``prefix`` namespaces the produced primitive ids (``"<prefix>/<local_id>"``),
    so several instances of the same concept coexist in one sketch.
    """
    errs = concept.validate()
    if errs:
        raise ValueError("invalid concept %s: %s" % (concept.name, errs[0]))
    if not concept.is_flat:
        raise ValueError("concept %s is hierarchical; flatten it first" % concept.name)
    if len(inputs) != concept.in_arity:
        raise ValueError("concept %s expects %d input refs, got %d"
                         % (concept.name, concept.in_arity, len(inputs)))

    env = concept.default_map()
    env.update({k: float(v) for k, v in (bindings or {}).items()})
    missing = [s for s in concept.slots if s not in env]
    if missing:
        raise KeyError("unbound slots for %s: %s" % (concept.name, sorted(missing)))
    extra = [k for k in (bindings or {}) if k not in set(concept.slots)]
    if extra:
        raise KeyError("unknown slots for %s: %s" % (concept.name, sorted(extra)))

    def cid(local_id: str) -> str:
        return "%s/%s" % (prefix, local_id) if prefix else local_id

    prims = tuple(
        Primitive.make(cid(m.local_id), m.ptype,
                       {k: resolve_param(ref, env) for k, ref in m.params})
        for m in concept.members
    )

    def deref(ref: str) -> str:
        kind, payload = parse_ref(ref)
        if kind == "input":
            return inputs[int(payload)]
        if kind == "sub":
            raise ValueError("sub reference in flat concept: %s" % ref)
        return cid(ref)

    cons = tuple(Constraint(c.ctype, tuple(deref(r) for r in c.refs))
                 for c in concept.constraints)
    outs = tuple(deref(r) for r in concept.out_refs)
    return Instantiation(prims, cons, outs)


@dataclass(frozen=True)
class ConceptInstance:
    """A *use* of a concept inside a sketch decomposition."""

    concept: str
    prefix: str
    bindings: Tuple[Tuple[str, float], ...] = ()
    inputs: Tuple[str, ...] = ()

    @staticmethod
    def make(concept: str, prefix: str,
             bindings: Optional[Mapping[str, float]] = None,
             inputs: Sequence[str] = ()) -> "ConceptInstance":
        b = tuple(sorted(((k, float(v)) for k, v in (bindings or {}).items()),
                         key=lambda kv: kv[0]))
        return ConceptInstance(concept, prefix, b, tuple(inputs))

    def binding_map(self) -> Dict[str, float]:
        return dict(self.bindings)


def realise(concept: Concept, instance: ConceptInstance) -> Instantiation:
    """Instantiate ``concept`` according to ``instance``."""
    if instance.concept != concept.name:
        raise ValueError("instance names concept %s, got %s" % (instance.concept, concept.name))
    return instantiate(concept, instance.binding_map(), instance.inputs, instance.prefix)


# ---------------------------------------------------------------------------
# Canonical signature (library deduplication)
# ---------------------------------------------------------------------------

_MAX_CANONICAL_PERMUTATIONS = 20000


def _member_invariant(concept: Concept, m: Member) -> str:
    """Cheap isomorphism-invariant used to partition members before search."""
    inc: List[str] = []
    for c in concept.constraints:
        n = sum(1 for r in c.refs if r == m.local_id)
        if n:
            inc.append("%s*%d/%d" % (c.ctype, n, len(c.refs)))
    outs = [str(i) for i, r in enumerate(concept.out_refs) if r == m.local_id]
    consts = sorted("%s=%r" % (k, ref.value) for k, ref in m.params if isinstance(ref, Const))
    return "|".join([m.ptype, ",".join(sorted(inc)), ",".join(outs), ",".join(consts)])


def _encode(concept: Concept, order: Sequence[Member]) -> str:
    index = {m.local_id: i for i, m in enumerate(order)}
    slot_num: Dict[str, int] = {}

    def enc_ref(ref: ParamRef) -> str:
        if isinstance(ref, Const):
            return "c%r" % (float(ref.value),)
        if ref.name not in slot_num:
            slot_num[ref.name] = len(slot_num)
        return "s%d" % slot_num[ref.name]

    parts: List[str] = ["arity=%d" % concept.in_arity]
    for i, m in enumerate(order):
        body = ",".join("%s=%s" % (k, enc_ref(r)) for k, r in m.params)
        parts.append("m%d:%s(%s)" % (i, m.ptype, body))

    def enc_target(r: str) -> str:
        kind, payload = parse_ref(r)
        if kind == "input":
            return "in%d" % int(payload)
        if kind == "sub":
            sid, j = payload  # type: ignore[misc]
            return "sub:%s#%d" % (sid, j)
        return "m%d" % index[r]

    cons = sorted("%s(%s)" % (c.ctype, ",".join(enc_target(r) for r in c.refs))
                  for c in concept.constraints)
    parts.append("C[" + ";".join(cons) + "]")
    parts.append("O[" + ";".join(enc_target(r) for r in concept.out_refs) + "]")
    return "|".join(parts)


def canonical_signature(concept: Concept) -> str:
    """A naming/ordering-independent signature of a *flat* concept's structure.

    Two concepts get the same signature iff they are identical up to renaming of
    members and slots and reordering of members and constraints. Constant
    parameters, the interface order and the input arity are all significant. Used
    by :mod:`library.sketchconcept_library` to deduplicate.
    """
    if not concept.is_flat:
        raise ValueError("canonical_signature requires a flat concept")
    if concept.validate():
        raise ValueError("invalid concept: %s" % concept.validate()[0])

    blocks: Dict[str, List[Member]] = {}
    for m in concept.members:
        blocks.setdefault(_member_invariant(concept, m), []).append(m)

    total = 1
    for key in blocks:
        n = len(blocks[key])
        for i in range(2, n + 1):
            total *= i
    if total > _MAX_CANONICAL_PERMUTATIONS:
        raise ValueError("concept %s too large / too symmetric to canonicalise" % concept.name)

    keys = sorted(blocks)
    per_block = [list(permutations(sorted(blocks[k], key=lambda m: m.local_id))) for k in keys]
    best: Optional[str] = None
    for combo in product(*per_block):
        order: List[Member] = []
        for chunk in combo:
            order.extend(chunk)
        enc = _encode(concept, order)
        if best is None or enc < best:
            best = enc
    return best or _encode(concept, [])


def structural_key(concept: Concept) -> str:
    """Short deterministic hash-free key: the canonical signature itself."""
    return canonical_signature(concept)
