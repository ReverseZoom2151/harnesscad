"""Deterministic **concept induction** from a sketch corpus (the search-based
counterpart of Yang & Pan, "Discovering Design Concepts for CAD Sketches",
NeurIPS 2022).

The paper *learns* a concept library end-to-end; the discovery signal it exploits
is nevertheless explicit and combinatorial: a good concept is a sub-sketch
structure that (a) recurs across the corpus and (b) is cheap to name relative to
spelling its primitives and constraints out every time. This module implements
exactly that objective without any learning:

1. **Enumerate** every connected sub-structure of every sketch, up to
   ``max_size`` primitives (connectivity = "share a constraint").
2. **Abstract** each sub-structure into a :class:`Concept`: primitives become
   members, every numeric parameter becomes a free slot, and the induced
   constraints are kept. With ``share_equal_params=True`` parameters that happen
   to be *equal* inside the region share one slot, which is how equality-style
   design intent (two circles of the same radius, two lines at the same height)
   is captured as a genuinely restrictive template rather than a type pattern.
3. **Count** occurrences by canonical signature
   (:func:`reconstruction.sketchconcept_template.canonical_signature`), so
   naming / ordering variants collapse together.
4. **Score** each candidate by a description-length gain::

       region_cost   = sum(1 + |params|) over members + sum(1 + arity) over constraints
       instance_cost = 1 + |slots|                      (a call plus its bindings)
       gain          = occurrences * (region_cost - instance_cost) - (region_cost + 1)

   i.e. how many tokens the corpus saves by naming this structure once and
   calling it, minus the cost of storing it in the library.
5. **Select** a compact library greedily by gain (see :func:`build_library`),
   which deduplicates admitted concepts through
   :class:`library.sketchconcept_library.ConceptLibrary`.

Pure stdlib, deterministic; all tie-breaks are by canonical signature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from harnesscad.domain.library.sketchconcept_library import ConceptLibrary
from harnesscad.domain.reconstruction.sketch.sketchconcept_template import (
    Concept,
    ConstraintSpec,
    Member,
    PRIMITIVE_PARAMS,
    Sketch,
    Slot,
    canonical_signature,
)


# ---------------------------------------------------------------------------
# Connected sub-structure enumeration
# ---------------------------------------------------------------------------


def adjacency(sketch: Sketch) -> Dict[str, Set[str]]:
    """Primitive adjacency induced by shared constraints."""
    adj: Dict[str, Set[str]] = {p.pid: set() for p in sketch.primitives}
    for c in sketch.constraints:
        refs = [r for r in c.refs if r in adj]
        for a in refs:
            for b in refs:
                if a != b:
                    adj[a].add(b)
    return adj


def connected_subsets(sketch: Sketch, min_size: int = 2, max_size: int = 3,
                      max_subsets: Optional[int] = 20000) -> List[Tuple[str, ...]]:
    """All connected primitive subsets with ``min_size <= |S| <= max_size``.

    Returned in deterministic order (by size, then by sketch primitive order).
    """
    if min_size < 1 or max_size < min_size:
        raise ValueError("bad size range")
    order = [p.pid for p in sketch.primitives]
    rank = {pid: i for i, pid in enumerate(order)}
    adj = adjacency(sketch)

    level: List[FrozenSet[str]] = [frozenset([pid]) for pid in order]
    seen: Set[FrozenSet[str]] = set(level)
    out: List[FrozenSet[str]] = [s for s in level if min_size <= 1]

    for _ in range(max_size - 1):
        nxt: List[FrozenSet[str]] = []
        for s in level:
            frontier = sorted({n for pid in s for n in adj[pid]} - s, key=lambda p: rank[p])
            for n in frontier:
                cand = s | {n}
                if cand in seen:
                    continue
                seen.add(cand)
                nxt.append(cand)
                if len(cand) >= min_size:
                    out.append(cand)
                if max_subsets is not None and len(out) >= max_subsets:
                    return _sorted_subsets(out, rank)
        if not nxt:
            break
        level = nxt
    return _sorted_subsets(out, rank)


def _sorted_subsets(subsets: Iterable[FrozenSet[str]], rank: Dict[str, int]):
    keyed = [tuple(sorted(s, key=lambda p: rank[p])) for s in subsets]
    keyed.sort(key=lambda t: (len(t), tuple(rank[p] for p in t)))
    return keyed


# ---------------------------------------------------------------------------
# Abstraction: sub-structure -> parameterised concept
# ---------------------------------------------------------------------------


def abstract_region(sketch: Sketch, pids: Sequence[str], name: str = "cand",
                    share_equal_params: bool = True, tol: float = 1e-9) -> Concept:
    """Turn a sub-structure into a parameterised concept (all parameters free)."""
    by_id = sketch.by_id()
    sel = list(pids)
    local = {pid: "m%d" % i for i, pid in enumerate(sel)}

    slots: List[str] = []
    # equality sharing is per parameter *name*: two circles with the same radius
    # share the 'r' slot, but a circle's x and y never collapse into one slot.
    values: List[Tuple[str, str, float]] = []  # (key, slot, value)

    def slot_for(pid: str, key: str, value: float) -> str:
        if share_equal_params:
            for k, s, v in values:
                if k == key and abs(v - value) <= tol:
                    return s
        s = "%s_%s" % (local[pid], key)
        slots.append(s)
        values.append((key, s, value))
        return s

    members: List[Member] = []
    for pid in sel:
        p = by_id[pid]
        pv = p.param_map()
        params = {}
        for key in PRIMITIVE_PARAMS[p.ptype]:
            params[key] = Slot(slot_for(pid, key, pv[key]))
        members.append(Member.make(local[pid], p.ptype, params))

    selset = set(sel)
    cons: List[ConstraintSpec] = []
    for c in sketch.constraints:
        if c.refs and all(r in selset for r in c.refs):
            cons.append(ConstraintSpec(c.ctype, tuple(local[r] for r in c.refs)))

    return Concept(
        name=name,
        slots=tuple(slots),
        members=tuple(members),
        constraints=tuple(cons),
        in_arity=0,
        out_refs=tuple(local[pid] for pid in sel),
    )


# ---------------------------------------------------------------------------
# Candidate mining + scoring
# ---------------------------------------------------------------------------


def region_cost(concept: Concept) -> int:
    """Token cost of spelling the concept's structure out inline."""
    cost = 0
    for m in concept.members:
        cost += 1 + len(m.params)
    for c in concept.constraints:
        cost += 1 + len(c.refs)
    return cost


def instance_cost(concept: Concept) -> int:
    """Token cost of *calling* the concept (name + one value per slot + inputs)."""
    return 1 + len(concept.slots) + concept.in_arity


def compression_gain(concept: Concept, occurrences: int) -> int:
    """Corpus tokens saved by naming this concept once and calling it."""
    saving = region_cost(concept) - instance_cost(concept)
    return occurrences * saving - (region_cost(concept) + 1)


@dataclass(frozen=True)
class ConceptCandidate:
    """A mined candidate concept and its corpus statistics."""

    concept: Concept
    signature: str
    occurrences: int
    sketches: int
    gain: int

    def size(self) -> int:
        return len(self.concept.members)


def induce_concepts(corpus: Sequence[Sketch], min_size: int = 2, max_size: int = 3,
                    min_occurrences: int = 2, share_equal_params: bool = True,
                    tol: float = 1e-9,
                    max_subsets_per_sketch: Optional[int] = 20000) -> List[ConceptCandidate]:
    """Mine ranked concept candidates from a corpus of sketches."""
    counts: Dict[str, int] = {}
    sketch_counts: Dict[str, Set[int]] = {}
    reps: Dict[str, Concept] = {}

    for si, sk in enumerate(corpus):
        errs = sk.validate()
        if errs:
            raise ValueError("invalid sketch %d: %s" % (si, errs[0]))
        for subset in connected_subsets(sk, min_size=min_size, max_size=max_size,
                                        max_subsets=max_subsets_per_sketch):
            cand = abstract_region(sk, subset, share_equal_params=share_equal_params, tol=tol)
            try:
                sig = canonical_signature(cand)
            except ValueError:
                continue  # too symmetric / large to canonicalise: skip
            counts[sig] = counts.get(sig, 0) + 1
            sketch_counts.setdefault(sig, set()).add(si)
            reps.setdefault(sig, cand)

    out: List[ConceptCandidate] = []
    for sig, occ in counts.items():
        if occ < min_occurrences:
            continue
        concept = reps[sig]
        out.append(ConceptCandidate(
            concept=concept,
            signature=sig,
            occurrences=occ,
            sketches=len(sketch_counts[sig]),
            gain=compression_gain(concept, occ),
        ))
    out.sort(key=lambda c: (-c.gain, -c.occurrences, -c.size(), c.signature))
    return out


def build_library(candidates: Sequence[ConceptCandidate], max_concepts: Optional[int] = None,
                  min_gain: int = 1, prefix: str = "c",
                  library: Optional[ConceptLibrary] = None) -> ConceptLibrary:
    """Greedily admit the highest-gain candidates into a deduplicated library."""
    lib = library if library is not None else ConceptLibrary()
    admitted = 0
    for cand in candidates:
        if cand.gain < min_gain:
            break
        if max_concepts is not None and admitted >= max_concepts:
            break
        concept = Concept(
            name="%s%d" % (prefix, admitted),
            slots=cand.concept.slots,
            members=cand.concept.members,
            constraints=cand.concept.constraints,
            in_arity=cand.concept.in_arity,
            out_refs=cand.concept.out_refs,
            defaults=cand.concept.defaults,
        )
        stored = lib.add(concept)
        if stored == concept.name:
            admitted += 1
    return lib


def induce_library(corpus: Sequence[Sketch], max_concepts: Optional[int] = None,
                   **kwargs) -> ConceptLibrary:
    """Convenience: mine a corpus and build a library in one deterministic call."""
    min_gain = kwargs.pop("min_gain", 1)
    prefix = kwargs.pop("prefix", "c")
    cands = induce_concepts(corpus, **kwargs)
    return build_library(cands, max_concepts=max_concepts, min_gain=min_gain, prefix=prefix)
