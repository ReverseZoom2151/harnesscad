"""SetLike CSG combinator algebra with normalising laws (from ``scad-hs``).

``scad-hs`` (a typed Haskell OpenSCAD EDSL, ``Graphics.Scad``) does something
neither of the harness's existing OpenSCAD front ends does.  ``programs.
angelcad_typed_csg`` type-checks a CSG tree; ``programs.solidpy_scad_emit``
emits OpenSCAD source; ``programs.scadlm_ast`` / ``geometry.scadlm_csg_eval``
parse and evaluate it.  All of them treat ``union``/``intersection``/
``difference`` as *opaque* n-ary nodes -- a ``union`` of two ``union``s stays a
nested pair of ``union``s.

``scad-hs`` instead gives its ``Model`` type a ``SetLike`` instance
(``Graphics/Scad/Class.hs``) that carries three *algebraic normalisation laws*,
applied as smart constructors every time two models are combined::

    -- union (<+>) is associative and flattens:
    Union'   xs <+> Union'   ys = Union'   (xs ++ ys)
    x           <+> Union'   ys = Union'   (x : ys)
    Union'   xs <+> y           = Union'   (xs ++ [y])
    x           <+> y           = Union'   [x, y]

    -- intersection (<#>) flattens identically
    -- difference (<->) *absorbs* later subtrahends into ONE difference:
    Difference x y <-> a = Difference x (y <+> a)
    x              <-> y = Difference x y

with monoid units ``Union' []`` and ``Intersection' []``, so the library-level
``union``/``intersection`` (a ``mconcat`` fold) always yields a single flat
``Union'``/``Intersection'`` node, and a chain of differences ``a - b - c - d``
collapses to ``difference(a, union(b, c, d))`` -- exactly one OpenSCAD
``difference()`` block with the subtrahends unioned, which is both what OpenSCAD
semantically means and far cheaper for the kernel than nested differences.

This module reimplements that algebra as deterministic Python smart
constructors over an immutable, hashable CSG term type, plus a recursive
:func:`normalize` that re-applies the laws bottom-up to an arbitrary tree.  The
term type (and its :func:`emit` OpenSCAD pretty-printer, which understands the
``children()`` placeholder) is shared with :mod:`programs.scadhs_module_cse`.

Pure stdlib, deterministic.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "Term",
    "Prim",
    "Union",
    "Intersection",
    "Difference",
    "Transform",
    "ModuleCall",
    "Children",
    "CHILDREN",
    "union",
    "intersection",
    "difference",
    "union_all",
    "intersection_all",
    "normalize",
    "emit",
    "prim",
    "transform",
]


def _freeze_params(params: Optional[Mapping[str, Any]]) -> Tuple[Tuple[str, Any], ...]:
    """Canonicalise a parameter mapping to a sorted, hashable tuple."""
    if not params:
        return ()
    out: List[Tuple[str, Any]] = []
    for k in sorted(params):
        v = params[k]
        if isinstance(v, (list, tuple)):
            v = tuple(
                tuple(e) if isinstance(e, (list, tuple)) else e for e in v
            )
        out.append((str(k), v))
    return tuple(out)


class Term:
    """Base class for immutable, hashable CSG terms."""

    __slots__ = ()

    def children(self) -> Tuple["Term", ...]:
        return ()


class Prim(Term):
    """A leaf primitive: ``name(params)`` -- e.g. ``cube``, ``circle``."""

    __slots__ = ("name", "params")

    def __init__(self, name: str, params: Optional[Mapping[str, Any]] = None) -> None:
        self.name = str(name)
        self.params = _freeze_params(params)

    def _key(self) -> Tuple[Any, ...]:
        return ("prim", self.name, self.params)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Prim) and other._key() == self._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        return "Prim(%r, %r)" % (self.name, dict(self.params))


class Children(Term):
    """The ``children()`` placeholder inside a module body (scad-hs ``Children``)."""

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Children)

    def __hash__(self) -> int:
        return hash("children")

    def __repr__(self) -> str:
        return "Children()"


CHILDREN = Children()


class ModuleCall(Term):
    """An application of a named module to an optional child (scad-hs ``Apply``)."""

    __slots__ = ("name", "child")

    def __init__(self, name: str, child: Optional[Term] = None) -> None:
        self.name = str(name)
        self.child = child

    def children(self) -> Tuple[Term, ...]:
        return (self.child,) if self.child is not None else ()

    def _key(self) -> Tuple[Any, ...]:
        return ("call", self.name, self.child)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ModuleCall) and other._key() == self._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        return "ModuleCall(%r, %r)" % (self.name, self.child)


class Transform(Term):
    """A unary wrapper op: ``name(params) { child }`` -- translate, rotate, ..."""

    __slots__ = ("name", "params", "child")

    def __init__(
        self, name: str, child: Term, params: Optional[Mapping[str, Any]] = None
    ) -> None:
        if not isinstance(child, Term):
            raise TypeError("Transform child must be a Term")
        self.name = str(name)
        self.params = _freeze_params(params)
        self.child = child

    def children(self) -> Tuple[Term, ...]:
        return (self.child,)

    def _key(self) -> Tuple[Any, ...]:
        return ("xform", self.name, self.params, self.child)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Transform) and other._key() == self._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        return "Transform(%r, %r, %r)" % (self.name, self.child, dict(self.params))


class _Nary(Term):
    """Base for the flattening n-ary set operators."""

    __slots__ = ("items",)
    _tag = ""

    def __init__(self, items: Sequence[Term]) -> None:
        it = tuple(items)
        for t in it:
            if not isinstance(t, Term):
                raise TypeError("set-operator members must be Terms")
        self.items = it

    def children(self) -> Tuple[Term, ...]:
        return self.items

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self) and other.items == self.items

    def __hash__(self) -> int:
        return hash((self._tag, self.items))

    def __repr__(self) -> str:
        return "%s(%r)" % (type(self).__name__, list(self.items))


class Union(_Nary):
    """A flattened n-ary ``union()`` (scad-hs ``Union'``)."""

    __slots__ = ()
    _tag = "union"


class Intersection(_Nary):
    """A flattened n-ary ``intersection()`` (scad-hs ``Intersection'``)."""

    __slots__ = ()
    _tag = "intersection"


class Difference(Term):
    """``difference()``: a minuend minus a tuple of subtrahends (scad-hs ``Difference``)."""

    __slots__ = ("minuend", "subtrahends")

    def __init__(self, minuend: Term, subtrahends: Sequence[Term]) -> None:
        if not isinstance(minuend, Term):
            raise TypeError("Difference minuend must be a Term")
        self.minuend = minuend
        self.subtrahends = tuple(subtrahends)

    def children(self) -> Tuple[Term, ...]:
        return (self.minuend,) + self.subtrahends

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Difference)
            and other.minuend == self.minuend
            and other.subtrahends == self.subtrahends
        )

    def __hash__(self) -> int:
        return hash(("difference", self.minuend, self.subtrahends))

    def __repr__(self) -> str:
        return "Difference(%r, %r)" % (self.minuend, list(self.subtrahends))


# ---------------------------------------------------------------------------
# smart constructors -- the SetLike laws
# ---------------------------------------------------------------------------


def union(a: Term, b: Term) -> Union:
    """``a <+> b`` with associative flattening (scad-hs ``<+>``)."""
    left = a.items if isinstance(a, Union) else (a,)
    right = b.items if isinstance(b, Union) else (b,)
    return Union(left + right)


def intersection(a: Term, b: Term) -> Intersection:
    """``a <#> b`` with associative flattening (scad-hs ``<#>``)."""
    left = a.items if isinstance(a, Intersection) else (a,)
    right = b.items if isinstance(b, Intersection) else (b,)
    return Intersection(left + right)


def difference(a: Term, b: Term) -> Difference:
    """``a <-> b``; if ``a`` is already a difference the new subtrahend is
    *absorbed* into its subtrahend union (scad-hs ``<->``)."""
    if isinstance(a, Difference):
        return Difference(a.minuend, a.subtrahends + (b,))
    return Difference(a, (b,))


def union_all(items: Iterable[Term]) -> Union:
    """Monoid fold of ``union`` over ``items`` (scad-hs library ``union``).

    The unit is ``Union([])``; the result is always a single flat ``Union``.
    """
    acc: Union = Union(())
    for t in items:
        acc = union(acc, t)
    return acc


def intersection_all(items: Iterable[Term]) -> Intersection:
    """Monoid fold of ``intersection`` (scad-hs library ``intersection``)."""
    acc: Intersection = Intersection(())
    for t in items:
        acc = intersection(acc, t)
    return acc


def normalize(term: Term) -> Term:
    """Recursively re-apply the SetLike laws bottom-up.

    Nested ``union``/``intersection`` nodes are flattened, chains of
    ``difference`` collapse to a single minuend/subtrahend-union node, and empty
    single-child set nodes are simplified.  Deterministic and idempotent.
    """
    if isinstance(term, Union):
        acc: Term = Union(())
        for c in term.items:
            acc = union(acc, normalize(c))
        return _simplify_nary(acc)
    if isinstance(term, Intersection):
        acc = Intersection(())
        for c in term.items:
            acc = intersection(acc, normalize(c))
        return _simplify_nary(acc)
    if isinstance(term, Difference):
        m = normalize(term.minuend)
        result: Term = m
        for s in term.subtrahends:
            result = difference(result, normalize(s))
        return result
    if isinstance(term, Transform):
        return Transform(term.name, normalize(term.child), dict(term.params))
    if isinstance(term, ModuleCall) and term.child is not None:
        return ModuleCall(term.name, normalize(term.child))
    return term


def _simplify_nary(term: Term) -> Term:
    """A one-element union/intersection is just that element."""
    if isinstance(term, (Union, Intersection)) and len(term.items) == 1:
        return term.items[0]
    return term


# ---------------------------------------------------------------------------
# OpenSCAD emission (shared with the module-CSE pass)
# ---------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        text = "%.10f" % value
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    return str(value)


def _params_text(params: Sequence[Tuple[str, Any]]) -> str:
    return ", ".join("%s = %s" % (k, _fmt(v)) for k, v in params)


def emit(term: Term, indent: str = "  ", level: int = 0) -> str:
    """Pretty-print ``term`` to OpenSCAD source (deterministic).

    ``Children`` renders as ``children();`` and ``ModuleCall`` as ``name()``.
    """
    pad = indent * level

    def block(head: str, kids: Sequence[Term]) -> str:
        if not kids:
            return pad + head + ";"
        lines = [pad + head + " {"]
        for k in kids:
            lines.append(emit(k, indent, level + 1))
        lines.append(pad + "}")
        return "\n".join(lines)

    if isinstance(term, Prim):
        return pad + "%s(%s);" % (term.name, _params_text(term.params))
    if isinstance(term, Children):
        return pad + "children();"
    if isinstance(term, ModuleCall):
        head = "%s()" % term.name
        return block(head, term.children())
    if isinstance(term, Transform):
        return block("%s(%s)" % (term.name, _params_text(term.params)), (term.child,))
    if isinstance(term, Union):
        return block("union()", term.items)
    if isinstance(term, Intersection):
        return block("intersection()", term.items)
    if isinstance(term, Difference):
        return block("difference()", term.children())
    raise TypeError("cannot emit %r" % type(term).__name__)


# ---------------------------------------------------------------------------
# convenience leaf/transform builders
# ---------------------------------------------------------------------------


def prim(name: str, **params: Any) -> Prim:
    return Prim(name, params)


def transform(name: str, child: Term, **params: Any) -> Transform:
    return Transform(name, child, params)
