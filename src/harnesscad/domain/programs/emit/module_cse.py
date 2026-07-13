"""Content-addressed OpenSCAD *module* extraction / CSE (from ``scad-hs``).

The most distinctive part of ``scad-hs`` (a typed Haskell OpenSCAD EDSL) is its
module machinery in ``Graphics.Scad`` -- the ``smodule`` function and the ``#``
/ ``##`` combinators.  When you wrap a child ``c`` with a group ``f``::

    f # c     -- f :: children -> a model,     one child hole
    f ## c    -- f :: children -> [models],    many models around the hole

scad-hs does *not* inline ``f`` at every use site.  Instead it:

1. builds the group body with the child replaced by a ``children()`` placeholder
   (scad-hs's ``Children`` / ``runChildren``);
2. looks the body up in a memo table ``Map SomeModels Text``;
3. on a miss, assigns the next name ``mdl_<N>`` (``N`` = current table size) and
   inserts it; on a hit, reuses the existing name;
4. emits an *application* ``mdl_N() { c }`` at the use site;
5. at render time, prepends one ``module mdl_N() { <body with children()> }``
   definition per unique body.

That is textbook **common-subexpression elimination**: identical group bodies --
however many times they occur, and regardless of which child they wrap -- are
defined once as a named OpenSCAD ``module`` and *called*, shrinking output and
matching OpenSCAD's own ``module`` reuse idiom.  Nothing else in the harness does
this: ``programs.solidpy_scad_emit`` has SolidPython "holes" (a *subtraction*
rewrite, unrelated), ``programs.angelcad_typed_csg`` type-checks, and the
``scadlm`` pair parse/evaluate -- none extract shared subtrees into modules.

This module reimplements the mechanism deterministically over the shared term
type in :mod:`geometry.scadhs_csg_algebra`:

* :class:`ModuleBuilder` -- the stateful memo table with first-encounter
  ``mdl_N`` naming (faithful port of ``smodule``);
* :func:`wrap` (``#``) and :func:`wrap_multi` (``##``) -- register a body that
  contains :data:`~geometry.scadhs_csg_algebra.CHILDREN` and get back a
  :class:`~geometry.scadhs_csg_algebra.ModuleCall` applied to the child;
* :func:`auto_modularize` -- the *automatic* CSE pass scad-hs's API leaves to
  the caller: walk an arbitrary term, find every subtree that occurs more than
  once, and hoist each into a nullary ``mdl_N`` module (largest, most frequent
  first), rewriting occurrences to calls;
* :func:`render` -- emit ``module`` definitions (in ``mdl_N`` order) followed by
  the body, byte-for-byte reproducibly.

Pure stdlib, deterministic.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from harnesscad.domain.geometry.sdf.csg_algebra import (
    CHILDREN,
    Children,
    Difference,
    Intersection,
    ModuleCall,
    Prim,
    Term,
    Transform,
    Union,
    emit,
)

__all__ = [
    "ModuleDef",
    "ModuleBuilder",
    "wrap",
    "wrap_multi",
    "auto_modularize",
    "render",
    "subtree_counts",
]


class ModuleDef:
    """A named module: ``module name() { body }`` (body may hold ``children()``)."""

    __slots__ = ("name", "body")

    def __init__(self, name: str, body: Tuple[Term, ...]) -> None:
        self.name = name
        self.body = tuple(body)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ModuleDef)
            and other.name == self.name
            and other.body == self.body
        )

    def __repr__(self) -> str:
        return "ModuleDef(%r, %d stmts)" % (self.name, len(self.body))


class ModuleBuilder:
    """Stateful memo table extracting unique bodies into ``mdl_N`` modules.

    Faithful port of scad-hs ``smodule`` + the ``Map SomeModels Text`` state:
    the *first* time a body (a tuple of terms, which may contain ``children()``)
    is seen it is assigned ``mdl_<size>``; repeats reuse the name.
    """

    __slots__ = ("_names", "_order", "prefix")

    def __init__(self, prefix: str = "mdl_") -> None:
        self._names: Dict[Tuple[Term, ...], str] = {}
        self._order: List[Tuple[Term, ...]] = []
        self.prefix = prefix

    def intern(self, body: Sequence[Term]) -> str:
        """Return the module name for ``body``, creating it on first sight."""
        key = tuple(body)
        name = self._names.get(key)
        if name is None:
            name = "%s%d" % (self.prefix, len(self._names))
            self._names[key] = name
            self._order.append(key)
        return name

    def apply(self, body: Sequence[Term], child: Optional[Term]) -> ModuleCall:
        """Register ``body`` and return a call ``mdl_N() { child }``."""
        name = self.intern(body)
        return ModuleCall(name, child)

    def definitions(self) -> List[ModuleDef]:
        """The interned modules, in ``mdl_N`` creation order."""
        return [ModuleDef(self._names[k], k) for k in self._order]


def _plug(term: Term, child: Term) -> Term:
    """Replace every ``Children`` placeholder in ``term`` with ``child``."""
    if isinstance(term, Children):
        return child
    if isinstance(term, Transform):
        return Transform(term.name, _plug(term.child, child), dict(term.params))
    if isinstance(term, Union):
        return Union(tuple(_plug(t, child) for t in term.items))
    if isinstance(term, Intersection):
        return Intersection(tuple(_plug(t, child) for t in term.items))
    if isinstance(term, Difference):
        return Difference(
            _plug(term.minuend, child),
            tuple(_plug(t, child) for t in term.subtrahends),
        )
    if isinstance(term, ModuleCall) and term.child is not None:
        return ModuleCall(term.name, _plug(term.child, child))
    return term


def wrap(
    builder: ModuleBuilder,
    group: Callable[[Term], Term],
    child: Term,
) -> ModuleCall:
    """scad-hs ``#``: ``group`` is a body-with-one-``children()``-hole.

    ``group(CHILDREN)`` is interned as a single-statement module body; the
    returned :class:`ModuleCall` applies it to ``child``.
    """
    body = (group(CHILDREN),)
    return builder.apply(body, child)


def wrap_multi(
    builder: ModuleBuilder,
    group: Callable[[Term], Sequence[Term]],
    child: Term,
) -> ModuleCall:
    """scad-hs ``##``: ``group`` yields a *list* of statements around the hole."""
    body = tuple(group(CHILDREN))
    return builder.apply(body, child)


# ---------------------------------------------------------------------------
# automatic CSE over an arbitrary term
# ---------------------------------------------------------------------------


def subtree_counts(term: Term) -> Dict[Term, int]:
    """Count occurrences of every subtree (excluding bare leaves' trivial reuse).

    A term is counted each time it appears anywhere in the tree.  Deterministic.
    """
    counts: Dict[Term, int] = {}

    def visit(t: Term) -> None:
        counts[t] = counts.get(t, 0) + 1
        for c in t.children():
            visit(c)

    visit(term)
    return counts


def _size(term: Term) -> int:
    """Number of nodes in a subtree (used to rank hoisting candidates)."""
    return 1 + sum(_size(c) for c in term.children())


def auto_modularize(
    term: Term,
    builder: Optional[ModuleBuilder] = None,
    min_size: int = 2,
    min_count: int = 2,
) -> Tuple[Term, ModuleBuilder]:
    """Automatic CSE: hoist repeated subtrees into nullary ``mdl_N`` modules.

    The scad-hs API makes the caller mark shared groups by hand; this performs
    the extraction automatically.  A subtree qualifies when it occurs at least
    ``min_count`` times and has at least ``min_size`` nodes.  Candidates are
    processed largest-first (so a big shared block is captured before its
    shared sub-blocks), each becoming a ``module mdl_N() { subtree }`` that
    every occurrence is rewritten to call.  ``children()`` placeholders make a
    subtree ineligible (a nullary module cannot host a hole).

    Returns the rewritten term and the :class:`ModuleBuilder` holding the new
    module definitions.  Deterministic: candidate order is (size desc, count
    desc, emitted-source asc).
    """
    builder = builder or ModuleBuilder()

    def has_children(t: Term) -> bool:
        if isinstance(t, Children):
            return True
        return any(has_children(c) for c in t.children())

    counts = subtree_counts(term)
    candidates = [
        t
        for t, n in counts.items()
        if n >= min_count
        and _size(t) >= min_size
        and not isinstance(t, ModuleCall)
        and not has_children(t)
    ]
    # largest first, then most frequent, then a stable source-text tie-break
    candidates.sort(key=lambda t: (-_size(t), -counts[t], emit(t)))

    # assign a module name to each chosen subtree (skip nested-in-a-chosen ones
    # only if the outer is chosen -- outer rewrite subsumes the inner occurrence)
    chosen: Dict[Term, str] = {}
    for cand in candidates:
        chosen[cand] = builder.intern((cand,))

    def rewrite(t: Term) -> Term:
        name = chosen.get(t)
        if name is not None:
            return ModuleCall(name, None)
        if isinstance(t, Transform):
            return Transform(t.name, rewrite(t.child), dict(t.params))
        if isinstance(t, Union):
            return Union(tuple(rewrite(c) for c in t.items))
        if isinstance(t, Intersection):
            return Intersection(tuple(rewrite(c) for c in t.items))
        if isinstance(t, Difference):
            return Difference(
                rewrite(t.minuend), tuple(rewrite(c) for c in t.subtrahends)
            )
        if isinstance(t, ModuleCall) and t.child is not None:
            return ModuleCall(t.name, rewrite(t.child))
        return t

    # module bodies themselves must be rewritten in terms of inner modules, but
    # never in terms of themselves -- rewrite each body with its own name removed.
    rebuilt = ModuleBuilder(prefix=builder.prefix)
    for cand in candidates:
        name = chosen[cand]
        others = {k: v for k, v in chosen.items() if k is not cand}
        body = _rewrite_with(cand, others)
        rebuilt._names[(body,)] = name
        rebuilt._order.append((body,))

    root = rewrite(term)
    return root, rebuilt


def _rewrite_with(term: Term, table: Dict[Term, str]) -> Term:
    """Rewrite ``term``'s descendants (not ``term`` itself) using ``table``."""

    def rw(t: Term, is_root: bool) -> Term:
        if not is_root:
            name = table.get(t)
            if name is not None:
                return ModuleCall(name, None)
        if isinstance(t, Transform):
            return Transform(t.name, rw(t.child, False), dict(t.params))
        if isinstance(t, Union):
            return Union(tuple(rw(c, False) for c in t.items))
        if isinstance(t, Intersection):
            return Intersection(tuple(rw(c, False) for c in t.items))
        if isinstance(t, Difference):
            return Difference(
                rw(t.minuend, False), tuple(rw(c, False) for c in t.subtrahends)
            )
        if isinstance(t, ModuleCall) and t.child is not None:
            return ModuleCall(t.name, rw(t.child, False))
        return t

    return rw(term, True)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def render(
    body: Term,
    builder: ModuleBuilder,
    indent: str = "  ",
) -> str:
    """Emit ``module`` definitions (in ``mdl_N`` order) then the body.

    Mirrors scad-hs ``render``: the collected module table is printed first,
    each as ``module name() { ... }``, followed by the top-level model.
    """
    lines: List[str] = []
    for mod in builder.definitions():
        lines.append("module %s() {" % mod.name)
        for stmt in mod.body:
            lines.append(emit(stmt, indent, 1))
        lines.append("}")
    lines.append(emit(body, indent, 0))
    return "\n".join(lines) + "\n"
