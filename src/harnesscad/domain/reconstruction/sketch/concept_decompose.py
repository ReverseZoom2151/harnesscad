"""Decomposing a sketch into a **composition of concept instances** (Yang & Pan,
"Discovering Design Concepts for CAD Sketches", NeurIPS 2022).

Given a concept library, the paper's decoder explains a sketch as a set of concept
instances plus their parameter bindings and cross-references. This module is the
deterministic, search-based counterpart of that step: no network, an explicit
sub-structure matcher.

**Matching.** A *flat* concept matches a sketch under an injective assignment of
its members to sketch primitives such that

  * types agree (line / circle / arc / point),
  * every constant member parameter equals the primitive's value (within ``tol``),
  * every **slot** is bound consistently -- a slot reused across members (or across
    parameters of one member) forces those primitive values to be equal, which is
    what makes a *parameterised* concept genuinely restrictive and not just a type
    pattern,
  * every concept constraint is *present* in the sketch between the mapped
    primitives (constraint refs compared as a multiset by default, since sketch
    constraints are largely symmetric),
  * external **input references** are mapped to sketch primitives that the concept
    does not own (they are constrained against, not claimed).

**Decomposition.** :func:`decompose` enumerates matches for every concept, then
greedily selects a set of non-overlapping instances preferring larger coverage
(more members, then more explained constraints, then a stable name/id order). The
result records, per instance, its bindings, its input wiring and the original
primitive ids it covers, so :func:`reconstruct` can rebuild the sketch exactly and
:func:`is_exact` can certify the decomposition is lossless.

Deterministic and pure stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from harnesscad.domain.reconstruction.sketch.concept_template import (
    Concept,
    ConceptInstance,
    Const,
    Constraint,
    Primitive,
    Sketch,
    Slot,
    instantiate,
    parse_ref,
)

DEFAULT_TOL = 1e-9


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Match:
    """One embedding of a concept into a sketch."""

    concept: str
    member_map: Tuple[Tuple[str, str], ...]     # local member id -> sketch pid
    inputs: Tuple[str, ...]                     # sketch pids bound to input refs
    bindings: Tuple[Tuple[str, float], ...]     # slot -> value
    covered_constraints: Tuple[int, ...]        # indices into sketch.constraints

    def owned(self) -> Tuple[str, ...]:
        return tuple(pid for _, pid in self.member_map)

    def binding_map(self) -> Dict[str, float]:
        return dict(self.bindings)

    def pid_of(self, local_id: str) -> str:
        return dict(self.member_map)[local_id]


def _constraint_key(ctype: str, refs: Sequence[str], ordered: bool) -> Tuple:
    return (ctype, tuple(refs) if ordered else tuple(sorted(refs)))


def find_matches(concept: Concept, sketch: Sketch, tol: float = DEFAULT_TOL,
                 ordered_refs: bool = False, limit: Optional[int] = None) -> List[Match]:
    """All embeddings of a *flat* ``concept`` into ``sketch``, in deterministic order."""
    if not concept.is_flat:
        raise ValueError("find_matches requires a flat concept (flatten it first)")
    errs = concept.validate()
    if errs:
        raise ValueError("invalid concept %s: %s" % (concept.name, errs[0]))

    prims = list(sketch.primitives)
    pindex = {p.pid: p for p in prims}

    # sketch constraint index: key -> list of constraint indices
    cindex: Dict[Tuple, List[int]] = {}
    for i, c in enumerate(sketch.constraints):
        cindex.setdefault(_constraint_key(c.ctype, c.refs, ordered_refs), []).append(i)

    members = list(concept.members)
    results: List[Match] = []

    def check_params(m, prim: Primitive, bindings: Dict[str, float]) -> Optional[Dict[str, float]]:
        pv = prim.param_map()
        new = dict(bindings)
        for k, ref in m.params:
            if k not in pv:
                return None
            v = pv[k]
            if isinstance(ref, Const):
                if abs(v - float(ref.value)) > tol:
                    return None
            else:
                name = ref.name
                if name in new:
                    if abs(new[name] - v) > tol:
                        return None
                else:
                    new[name] = v
        return new

    def finish(member_map: Dict[str, str], inputs: List[str],
               bindings: Dict[str, float]) -> None:
        used: List[int] = []
        for c in concept.constraints:
            refs = []
            for r in c.refs:
                kind, payload = parse_ref(r)
                if kind == "input":
                    refs.append(inputs[int(payload)])
                elif kind == "member":
                    refs.append(member_map[r])
                else:
                    raise ValueError("sub reference in flat concept")
            key = _constraint_key(c.ctype, refs, ordered_refs)
            hits = [i for i in cindex.get(key, []) if i not in used]
            if not hits:
                return
            used.append(hits[0])
        # slots with defaults may remain unbound if unused by any member
        defaults = concept.default_map()
        for s in concept.slots:
            if s not in bindings and s in defaults:
                bindings[s] = defaults[s]
        results.append(Match(
            concept=concept.name,
            member_map=tuple((k, member_map[k]) for k in (m.local_id for m in members)),
            inputs=tuple(inputs),
            bindings=tuple(sorted(bindings.items())),
            covered_constraints=tuple(sorted(used)),
        ))

    def assign_inputs(member_map: Dict[str, str], inputs: List[str],
                      bindings: Dict[str, float]) -> None:
        if limit is not None and len(results) >= limit:
            return
        k = len(inputs)
        if k == concept.in_arity:
            finish(member_map, inputs, bindings)
            return
        taken = set(member_map.values()) | set(inputs)
        for p in prims:
            if p.pid in taken:
                continue
            assign_inputs(member_map, inputs + [p.pid], bindings)
            if limit is not None and len(results) >= limit:
                return

    def assign_members(i: int, member_map: Dict[str, str],
                       bindings: Dict[str, float]) -> None:
        if limit is not None and len(results) >= limit:
            return
        if i == len(members):
            assign_inputs(member_map, [], bindings)
            return
        m = members[i]
        used = set(member_map.values())
        for p in prims:
            if p.ptype != m.ptype or p.pid in used:
                continue
            nb = check_params(m, p, bindings)
            if nb is None:
                continue
            member_map[m.local_id] = p.pid
            assign_members(i + 1, member_map, nb)
            del member_map[m.local_id]
            if limit is not None and len(results) >= limit:
                return

    del pindex
    assign_members(0, {}, {})
    return results


# ---------------------------------------------------------------------------
# Greedy decomposition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Placement:
    """A selected concept instance inside a decomposition."""

    instance: ConceptInstance
    match: Match

    def covered_primitives(self) -> Tuple[str, ...]:
        return self.match.owned()


@dataclass(frozen=True)
class Decomposition:
    """A sketch explained as concept instances plus an unexplained residual."""

    sketch: Sketch
    placements: Tuple[Placement, ...]
    residual_primitives: Tuple[str, ...]
    residual_constraints: Tuple[int, ...]

    def covered_primitives(self) -> Tuple[str, ...]:
        out: List[str] = []
        for p in self.placements:
            out.extend(p.covered_primitives())
        return tuple(out)

    def primitive_coverage(self) -> float:
        n = len(self.sketch.primitives)
        return 1.0 if n == 0 else len(self.covered_primitives()) / n

    def constraint_coverage(self) -> float:
        n = len(self.sketch.constraints)
        if n == 0:
            return 1.0
        return (n - len(self.residual_constraints)) / n

    def concept_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for p in self.placements:
            counts[p.instance.concept] = counts.get(p.instance.concept, 0) + 1
        return counts


def _match_rank(m: Match) -> Tuple:
    return (-len(m.member_map), -len(m.covered_constraints), m.concept, m.owned())


def decompose(sketch: Sketch, library, concepts: Optional[Sequence[str]] = None,
              tol: float = DEFAULT_TOL, ordered_refs: bool = False,
              limit_per_concept: Optional[int] = 200) -> Decomposition:
    """Explain ``sketch`` as a maximal, non-overlapping set of concept instances.

    ``library`` is a :class:`library.sketchconcept_library.ConceptLibrary` (any
    object exposing ``names()`` and ``flatten(name)``).
    """
    errs = sketch.validate()
    if errs:
        raise ValueError("invalid sketch: %s" % errs[0])
    names = list(concepts) if concepts is not None else list(library.names())

    all_matches: List[Match] = []
    for name in names:
        flat = library.flatten(name)
        all_matches.extend(find_matches(flat, sketch, tol=tol, ordered_refs=ordered_refs,
                                        limit=limit_per_concept))
    all_matches.sort(key=_match_rank)

    used_prims: Set[str] = set()
    used_cons: Set[int] = set()
    placements: List[Placement] = []
    for m in all_matches:
        owned = set(m.owned())
        if owned & used_prims:
            continue
        if set(m.covered_constraints) & used_cons:
            continue
        inst = ConceptInstance.make(m.concept, "i%d" % len(placements),
                                    m.binding_map(), m.inputs)
        placements.append(Placement(inst, m))
        used_prims |= owned
        used_cons |= set(m.covered_constraints)

    residual_p = tuple(p.pid for p in sketch.primitives if p.pid not in used_prims)
    residual_c = tuple(i for i in range(len(sketch.constraints)) if i not in used_cons)
    return Decomposition(sketch, tuple(placements), residual_p, residual_c)


# ---------------------------------------------------------------------------
# Reconstruction / verification
# ---------------------------------------------------------------------------


def reconstruct(decomp: Decomposition, library) -> Sketch:
    """Rebuild a sketch from a decomposition (instances + residual), in original ids."""
    prims: Dict[str, Primitive] = {}
    cons: List[Constraint] = []
    for pl in decomp.placements:
        flat = library.flatten(pl.instance.concept)
        inst = instantiate(flat, pl.instance.binding_map(), pl.instance.inputs,
                           prefix=pl.instance.prefix)
        rename = {}
        for local, pid in pl.match.member_map:
            rename["%s/%s" % (pl.instance.prefix, local)] = pid
        for p in inst.primitives:
            pid = rename[p.pid]
            prims[pid] = Primitive(pid, p.ptype, p.params)
        for c in inst.constraints:
            cons.append(Constraint(c.ctype, tuple(rename.get(r, r) for r in c.refs)))

    by_id = decomp.sketch.by_id()
    for pid in decomp.residual_primitives:
        prims[pid] = by_id[pid]
    for i in decomp.residual_constraints:
        cons.append(decomp.sketch.constraints[i])

    ordered = tuple(prims[p.pid] for p in decomp.sketch.primitives if p.pid in prims)
    return Sketch(ordered, tuple(cons))


def is_exact(decomp: Decomposition, library, tol: float = DEFAULT_TOL,
             ordered_refs: bool = False) -> bool:
    """True iff reconstruction reproduces every primitive and constraint of the sketch."""
    rebuilt = reconstruct(decomp, library)
    orig = decomp.sketch
    if len(rebuilt.primitives) != len(orig.primitives):
        return False
    rb = rebuilt.by_id()
    for p in orig.primitives:
        q = rb.get(p.pid)
        if q is None or q.ptype != p.ptype:
            return False
        pm, qm = p.param_map(), q.param_map()
        if set(pm) != set(qm):
            return False
        for k in pm:
            if abs(pm[k] - qm[k]) > tol:
                return False

    def keys(s: Sketch):
        out: Dict[Tuple, int] = {}
        for c in s.constraints:
            k = _constraint_key(c.ctype, c.refs, ordered_refs)
            out[k] = out.get(k, 0) + 1
        return out

    return keys(rebuilt) == keys(orig)
