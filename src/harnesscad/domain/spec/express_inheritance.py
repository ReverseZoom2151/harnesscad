"""Inheritance graph and attribute flattening over a parsed EXPRESS schema.

EXPRESS entities form a multiple-inheritance lattice: ``ENTITY b SUBTYPE OF
(a)`` makes ``b`` inherit every attribute of ``a``, and ``SUPERTYPE OF`` /
``SUBTYPE_CONSTRAINT`` declare the complementary edges.  ruststep's ``ir``
stage ("legalize") resolves these edges and materialises, for each concrete
entity, the full ordered attribute list an instance must supply -- supertype
attributes first, then the entity's own.  This is exactly the information the
part-21 data side needs: a ``#N = FOO(...)`` record lists inherited attributes
before local ones, so validating arity requires the *flattened* attribute list,
not just the entity's own declaration.

This module builds that model from a :class:`~spec.express_schema_parser.Schema`:

  * supertype / subtype adjacency and their transitive closures;
  * per-entity flattened attribute list (depth-first over supertypes, in
    declaration order, de-duplicated by attribute name for diamond inheritance);
  * root and leaf detection, and inheritance-cycle detection;
  * ``SELECT`` type expansion (recursively resolving nested selects to their
    underlying entity/simple leaves).

Pure and deterministic; depends only on :mod:`spec.express_schema_parser`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harnesscad.domain.spec.express_schema_parser import Schema, supertype_leaf_names


class InheritanceError(ValueError):
    """Raised on an inconsistent inheritance graph (e.g. a cycle)."""


@dataclass
class InheritanceGraph:
    """Resolved supertype/subtype edges for a schema's entities."""

    schema: Schema
    supertypes: dict = field(default_factory=dict)   # name -> [direct supertypes]
    subtypes: dict = field(default_factory=dict)     # name -> [direct subtypes]

    def roots(self) -> list:
        """Entities with no supertype (in schema declaration order)."""

        return [n for n in self.schema.entity_order
                if not self.supertypes.get(n)]

    def leaves(self) -> list:
        """Entities that are nobody's supertype (in declaration order)."""

        return [n for n in self.schema.entity_order
                if not self.subtypes.get(n)]

    def all_supertypes(self, name: str) -> list:
        """Transitive supertypes of ``name`` (breadth-first, deduplicated)."""

        seen: list = []
        stack = list(self.supertypes.get(name, []))
        while stack:
            cur = stack.pop(0)
            if cur in seen:
                continue
            seen.append(cur)
            stack.extend(self.supertypes.get(cur, []))
        return seen

    def all_subtypes(self, name: str) -> list:
        seen: list = []
        stack = list(self.subtypes.get(name, []))
        while stack:
            cur = stack.pop(0)
            if cur in seen:
                continue
            seen.append(cur)
            stack.extend(self.subtypes.get(cur, []))
        return seen

    def is_subtype_of(self, sub: str, sup: str) -> bool:
        return sup in self.all_supertypes(sub)


def build_inheritance(schema: Schema) -> InheritanceGraph:
    """Resolve direct supertype/subtype edges from ``SUBTYPE OF`` and
    ``SUPERTYPE OF`` declarations."""

    supertypes: dict = {n: [] for n in schema.entity_order}
    subtypes: dict = {n: [] for n in schema.entity_order}

    def add_edge(sub: str, sup: str) -> None:
        if sup not in supertypes.setdefault(sub, []):
            supertypes[sub].append(sup)
        if sub not in subtypes.setdefault(sup, []):
            subtypes[sup].append(sub)

    for name in schema.entity_order:
        ent = schema.entities[name]
        for sup in ent.supertypes:                 # SUBTYPE OF (sup, ...)
            add_edge(name, sup)
        # SUPERTYPE OF (...) declares this entity as a supertype of the leaves.
        for leaf in supertype_leaf_names(ent.supertype_expr):
            add_edge(leaf, name)

    graph = InheritanceGraph(schema=schema, supertypes=supertypes,
                             subtypes=subtypes)
    _check_acyclic(graph)
    return graph


def _check_acyclic(graph: InheritanceGraph) -> None:
    WHITE, GREY, BLACK = 0, 1, 2
    color: dict = {n: WHITE for n in graph.supertypes}

    def visit(node: str) -> None:
        color[node] = GREY
        for sup in graph.supertypes.get(node, []):
            state = color.get(sup, WHITE)
            if state == GREY:
                raise InheritanceError(
                    f"inheritance cycle involving {sup!r}")
            if state == WHITE:
                visit(sup)
        color[node] = BLACK

    for node in list(color):
        if color[node] == WHITE:
            visit(node)


def flatten_attributes(graph: InheritanceGraph, name: str) -> list:
    """Ordered attributes an instance of ``name`` must supply.

    Supertype attributes come first (depth-first over supertypes in declaration
    order), then the entity's own; duplicates by attribute name are dropped
    keeping the first (topmost) occurrence, matching how part-21 records list
    inherited attributes ahead of local ones.
    """

    schema = graph.schema
    if name not in schema.entities:
        raise KeyError(f"unknown entity {name!r}")

    ordered: list = []
    seen_names: set = set()

    def collect(ent_name: str) -> None:
        ent = schema.entities.get(ent_name)
        if ent is None:
            return
        for sup in graph.supertypes.get(ent_name, []):
            collect(sup)
        for attr in ent.attributes:
            if attr.name not in seen_names:
                seen_names.add(attr.name)
                ordered.append(attr)

    collect(name)
    return ordered


def expand_select(schema: Schema, type_name: str, _seen=None) -> list:
    """Recursively expand a ``SELECT`` type to its non-select leaf type names.

    A ``SELECT`` may list other selects; this returns the flattened set of
    concrete entity / defined-type / simple names actually selectable.
    """

    if _seen is None:
        _seen = set()
    if type_name in _seen:
        return []
    _seen.add(type_name)

    td = schema.types.get(type_name)
    if td is None or td.underlying.kind != "select":
        return [type_name]

    out: list = []
    for member in td.underlying.types:
        for leaf in expand_select(schema, member, _seen):
            if leaf not in out:
                out.append(leaf)
    return out
