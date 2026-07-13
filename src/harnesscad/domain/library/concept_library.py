"""Concept **library** management for CAD sketches (Yang & Pan, "Discovering Design
Concepts for CAD Sketches", NeurIPS 2022).

The paper's model learns a *library* of modular concepts and explains each sketch
as a composition of concept instances; the library is explicitly **hierarchical**
(a concept may be built from lower-level concepts, up to
``max_abstruction_decompose_query`` abstraction levels) and is only useful if it
is **compact** -- no two concepts should be structurally the same.

This module is the deterministic library machinery around the representation in
:mod:`reconstruction.sketchconcept_template`:

* **deduplication** -- admission keys a concept by its canonical signature (after
  flattening), so a re-added structure returns the *existing* name and is recorded
  as an alias rather than growing the library;
* **hierarchy** -- :meth:`ConceptLibrary.flatten` recursively expands
  sub-instances into a single flat concept: sub-concept slots are substituted by
  the parent's parameter references (or the sub-concept's defaults), sub-concept
  input references are wired to the parent's members / inputs / other
  sub-instances' outputs, and member ids are namespaced. Sub-instances are
  expanded in dependency order and reference cycles are rejected;
* **abstraction level** -- :meth:`depth` is 1 for a primitive-only concept and
  ``1 + max(depth(children))`` otherwise (the paper's abstraction level);
* **usage accounting** -- :meth:`record_use` / :meth:`usage` / :meth:`unused`
  support library pruning and the reuse statistics in
  :mod:`bench.sketchconcept_metrics`.

Pure stdlib, deterministic (iteration order is insertion order; all derived
sequences are explicitly sorted where order would otherwise be arbitrary).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from harnesscad.domain.reconstruction.sketch.concept_template import (
    Concept,
    Const,
    ConstraintSpec,
    Member,
    ParamRef,
    Slot,
    canonical_signature,
    input_ref,
    parse_ref,
)


class ConceptCycleError(ValueError):
    """A concept (directly or transitively) contains itself."""


class ConceptLibrary:
    """A deduplicated, hierarchical library of sketch concepts."""

    def __init__(self) -> None:
        self._concepts: Dict[str, Concept] = {}
        self._signatures: Dict[str, str] = {}   # signature -> canonical concept name
        self._aliases: Dict[str, str] = {}      # rejected name -> canonical name
        self._usage: Dict[str, int] = {}
        self._flat_cache: Dict[str, Concept] = {}

    # --- admission --------------------------------------------------------
    def add(self, concept: Concept, dedup: bool = True) -> str:
        """Admit ``concept``; return the name under which it is stored.

        If an structurally identical concept is already present (same canonical
        signature) the existing name is returned and ``concept.name`` is recorded
        as an alias of it.
        """
        errs = concept.validate()
        if errs:
            raise ValueError("invalid concept %s: %s" % (concept.name, errs[0]))
        for s in concept.subs:
            if s.concept not in self._concepts:
                raise KeyError("sub-concept not in library: %s" % s.concept)
        if concept.name in self._concepts:
            raise ValueError("concept name already used: %s" % concept.name)
        if concept.name in self._aliases:
            raise ValueError("name already registered as an alias: %s" % concept.name)

        self._concepts[concept.name] = concept
        try:
            flat = self.flatten(concept.name)
        except Exception:
            del self._concepts[concept.name]
            raise
        sig = canonical_signature(flat)
        if dedup and sig in self._signatures:
            del self._concepts[concept.name]
            self._flat_cache.pop(concept.name, None)
            existing = self._signatures[sig]
            self._aliases[concept.name] = existing
            return existing
        self._signatures.setdefault(sig, concept.name)
        self._usage.setdefault(concept.name, 0)
        return concept.name

    # --- lookup -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self._concepts)

    def __contains__(self, name: str) -> bool:
        return name in self._concepts

    def names(self) -> Tuple[str, ...]:
        return tuple(self._concepts)

    def get(self, name: str) -> Concept:
        if name in self._concepts:
            return self._concepts[name]
        if name in self._aliases:
            return self._concepts[self._aliases[name]]
        raise KeyError("no such concept: %s" % name)

    def resolve(self, name: str) -> str:
        """Map a possibly-aliased name to the canonical stored name."""
        if name in self._concepts:
            return name
        if name in self._aliases:
            return self._aliases[name]
        raise KeyError("no such concept: %s" % name)

    def aliases(self) -> Dict[str, str]:
        return dict(self._aliases)

    def signature(self, name: str) -> str:
        return canonical_signature(self.flatten(self.resolve(name)))

    # --- hierarchy --------------------------------------------------------
    def depth(self, name: str) -> int:
        """Abstraction level: 1 for a flat concept, 1 + max(child depth) otherwise."""
        return self._depth(self.resolve(name), [])

    def _depth(self, name: str, stack: List[str]) -> int:
        if name in stack:
            raise ConceptCycleError("concept cycle: %s" % (" -> ".join(stack + [name])))
        c = self._concepts[name]
        if not c.subs:
            return 1
        stack = stack + [name]
        return 1 + max(self._depth(self.resolve(s.concept), stack) for s in c.subs)

    def children(self, name: str) -> Tuple[str, ...]:
        c = self.get(name)
        return tuple(self.resolve(s.concept) for s in c.subs)

    def topological_order(self) -> Tuple[str, ...]:
        """Concept names, dependencies before dependants (insertion order otherwise)."""
        out: List[str] = []
        seen: Set[str] = set()

        def visit(n: str, stack: List[str]) -> None:
            if n in seen:
                return
            if n in stack:
                raise ConceptCycleError("concept cycle: %s" % (" -> ".join(stack + [n])))
            for ch in self.children(n):
                visit(ch, stack + [n])
            seen.add(n)
            out.append(n)

        for n in self._concepts:
            visit(n, [])
        return tuple(out)

    def flatten(self, name: str) -> Concept:
        """Expand all sub-instances, returning an equivalent *flat* concept."""
        name = self.resolve(name)
        if name in self._flat_cache:
            return self._flat_cache[name]
        flat = self._flatten(name, [])
        self._flat_cache[name] = flat
        return flat

    def _flatten(self, name: str, stack: List[str]) -> Concept:
        if name in stack:
            raise ConceptCycleError("concept cycle: %s" % (" -> ".join(stack + [name])))
        c = self._concepts[name]
        if c.is_flat:
            return c

        members: List[Member] = list(c.members)
        constraints: List[ConstraintSpec] = []
        own_ids = set(c.member_ids())
        sub_outs: Dict[str, Tuple[str, ...]] = {}

        def presolve(ref: str) -> str:
            """Resolve a parent-level reference to a flat member id / input ref."""
            kind, payload = parse_ref(ref)
            if kind == "member":
                return ref
            if kind == "input":
                return input_ref(int(payload))
            sid, j = payload  # type: ignore[misc]
            if sid not in sub_outs:
                raise KeyError("sub-instance %s used before it is expanded" % sid)
            outs = sub_outs[sid]
            if not (0 <= j < len(outs)):
                raise IndexError("sub %s has %d outputs, index %d requested" % (sid, len(outs), j))
            return outs[j]

        for s in self._order_subs(c):
            sub_name = self.resolve(s.concept)
            sub_flat = self._flatten(sub_name, stack + [name])
            subst: Dict[str, ParamRef] = {}
            bound = s.binding_map()
            sub_defaults = sub_flat.default_map()
            for slot in sub_flat.slots:
                if slot in bound:
                    subst[slot] = bound[slot]
                elif slot in sub_defaults:
                    subst[slot] = Const(sub_defaults[slot])
                else:
                    raise KeyError("sub %s of %s: slot %s is unbound" % (s.local_id, name, slot))
            extra = [k for k in bound if k not in set(sub_flat.slots)]
            if extra:
                raise KeyError("sub %s of %s: unknown slots %s"
                               % (s.local_id, name, sorted(extra)))
            if len(s.inputs) != sub_flat.in_arity:
                raise ValueError("sub %s of %s: expects %d inputs, got %d"
                                 % (s.local_id, name, sub_flat.in_arity, len(s.inputs)))

            def mid(local: str, sid=s.local_id) -> str:
                return "%s/%s" % (sid, local)

            for m in sub_flat.members:
                params = {}
                for k, ref in m.params:
                    params[k] = subst[ref.name] if isinstance(ref, Slot) else ref
                members.append(Member.make(mid(m.local_id), m.ptype, params))

            def sub_ref(r: str, sid=s.local_id, inputs=s.inputs) -> str:
                kind, payload = parse_ref(r)
                if kind == "member":
                    return "%s/%s" % (sid, r)
                if kind == "input":
                    return presolve(inputs[int(payload)])
                raise ValueError("flattened sub still holds a sub reference: %s" % r)

            for sc in sub_flat.constraints:
                constraints.append(ConstraintSpec(sc.ctype, tuple(sub_ref(r) for r in sc.refs)))
            sub_outs[s.local_id] = tuple(sub_ref(r) for r in sub_flat.out_refs)

        for sc in c.constraints:
            constraints.append(ConstraintSpec(sc.ctype, tuple(presolve(r) for r in sc.refs)))
        out_refs = tuple(presolve(r) for r in c.out_refs)

        flat = Concept(
            name=c.name,
            slots=c.slots,
            members=tuple(members),
            constraints=tuple(constraints),
            subs=(),
            in_arity=c.in_arity,
            out_refs=out_refs,
            defaults=c.defaults,
        )
        errs = flat.validate()
        if errs:
            raise ValueError("flattening %s produced an invalid concept: %s" % (name, errs[0]))
        del own_ids
        return flat

    @staticmethod
    def _order_subs(c: Concept):
        """Order sub-instances so that a sub is expanded after the subs it reads."""
        remaining = list(c.subs)
        done: Set[str] = set()
        ordered = []
        while remaining:
            progressed = False
            for s in list(remaining):
                deps = set()
                for r in s.inputs:
                    kind, payload = parse_ref(r)
                    if kind == "sub":
                        deps.add(payload[0])  # type: ignore[index]
                if deps <= done:
                    ordered.append(s)
                    done.add(s.local_id)
                    remaining.remove(s)
                    progressed = True
            if not progressed:
                raise ConceptCycleError(
                    "sub-instance input cycle in concept %s: %s"
                    % (c.name, sorted(s.local_id for s in remaining)))
        return ordered

    # --- usage ------------------------------------------------------------
    def record_use(self, name: str, count: int = 1) -> None:
        n = self.resolve(name)
        self._usage[n] = self._usage.get(n, 0) + int(count)

    def usage(self) -> Dict[str, int]:
        return {n: self._usage.get(n, 0) for n in self._concepts}

    def unused(self) -> Tuple[str, ...]:
        return tuple(n for n in self._concepts if self._usage.get(n, 0) == 0)

    def most_used(self) -> Tuple[Tuple[str, int], ...]:
        return tuple(sorted(self.usage().items(), key=lambda kv: (-kv[1], kv[0])))
