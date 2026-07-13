"""Deterministic parser for the EXPRESS schema language (ISO 10303-11).

STEP has two distinct languages: the *data* exchange format (part 21, the
``#N = ENTITY(...);`` text handled by :mod:`formats.stepllm_parser`) and the
*schema definition* language, EXPRESS (part 11), in which the meaning of every
entity, its attributes, and its inheritance are declared.  ``espr`` -- the
EXPRESS compiler shipped inside ruststep -- parses the latter to generate Rust
types.  The harness already models the *data* side and a small hand-written
subset of entity layouts (:mod:`formats.stepllm_schema`), but it has no parser
for the EXPRESS *language* itself.  This module fills that gap: it reads an
``.exp`` schema and builds an entity / type / attribute / inheritance model that
downstream code (inheritance flattening, schema-to-data validation) can query.

The grammar handled follows ISO 10303-11 (production numbers cited from the
document, mirroring espr's ``parser/`` module):

  * ``SCHEMA name; ... END_SCHEMA;`` with ``USE``/``REFERENCE`` interface specs;
  * ``ENTITY`` declarations: comma-shared explicit attributes, ``OPTIONAL``,
    ``SUBTYPE OF (...)``, ``ABSTRACT``, ``SUPERTYPE OF (ONEOF/ANDOR/AND ...)``,
    and the ``DERIVE`` / ``INVERSE`` / ``UNIQUE`` / ``WHERE`` clauses (captured
    structurally; rule expressions are kept as raw text);
  * ``TYPE`` declarations over simple types, ``ENUMERATION``, ``SELECT`` and
    aggregate (``LIST``/``SET``/``ARRAY``/``BAG``) underlying types;
  * parameter types with bounds ``[lo:hi]`` and ``UNIQUE``/``OPTIONAL`` flags;
  * both comment forms ``(* ... *)`` and ``-- ...``.

``FUNCTION``/``PROCEDURE``/``RULE``/``CONSTANT``/``SUBTYPE_CONSTRAINT`` blocks
are recognised and skipped (their algorithmic bodies are out of scope for a
static structural model).  Everything is pure, stdlib-only and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --- type model --------------------------------------------------------------

@dataclass(frozen=True)
class TypeRef:
    """A parsed EXPRESS type reference (attribute type or type underlying).

    ``kind`` selects the interpretation of the remaining fields:

      * ``"simple"``  -> :attr:`name` is ``REAL``/``INTEGER``/``STRING``/...
      * ``"named"``   -> :attr:`name` references an entity or defined type
      * ``"list"``/``"set"``/``"array"``/``"bag"`` -> aggregate over :attr:`base`
      * ``"enum"``    -> :attr:`items` holds the enumeration ids
      * ``"select"``  -> :attr:`types` holds the selectable type names
    """

    kind: str
    name: str = ""
    base: "TypeRef | None" = None
    lower: str | None = None          # bound lower expr (raw text) or None
    upper: str | None = None          # bound upper expr; "?" means indeterminate
    unique: bool = False
    optional: bool = False            # ARRAY ... OF OPTIONAL
    items: tuple = ()                 # enumeration ids
    types: tuple = ()                 # select member type names
    extensible: bool = False
    width: int | None = None          # STRING(n) / BINARY(n)
    fixed: bool = False

    def is_aggregate(self) -> bool:
        return self.kind in ("list", "set", "array", "bag")


@dataclass(frozen=True)
class Attribute:
    name: str
    type_ref: TypeRef
    optional: bool


@dataclass(frozen=True)
class WhereRule:
    label: str | None
    expr: str          # raw expression text


@dataclass(frozen=True)
class UniqueRule:
    label: str | None
    attributes: tuple  # attribute names


@dataclass(frozen=True)
class DerivedAttribute:
    name: str          # possibly a qualified "SELF\\x.y" reference
    type_ref: TypeRef | None
    expr: str          # raw text after ':='


@dataclass(frozen=True)
class InverseAttribute:
    name: str
    raw: str           # the full raw ``... FOR ...`` body


@dataclass
class EntityDef:
    name: str
    attributes: list = field(default_factory=list)      # list[Attribute]
    supertypes: list = field(default_factory=list)       # names from SUBTYPE OF
    is_abstract: bool = False
    supertype_expr: object | None = None                 # SuperTypeExpr or None
    where_rules: list = field(default_factory=list)
    unique_rules: list = field(default_factory=list)
    derived: list = field(default_factory=list)
    inverse: list = field(default_factory=list)

    @property
    def arity(self) -> int:
        return len(self.attributes)


@dataclass(frozen=True)
class SuperTypeExpr:
    """A SUPERTYPE-OF expression: ``op`` in {ref, oneof, andor, and}."""

    op: str
    name: str = ""            # for op == "ref"
    operands: tuple = ()      # for oneof/andor/and


@dataclass
class TypeDef:
    name: str
    underlying: TypeRef
    where_rules: list = field(default_factory=list)


@dataclass
class Schema:
    name: str
    entities: dict = field(default_factory=dict)      # name -> EntityDef
    types: dict = field(default_factory=dict)         # name -> TypeDef
    entity_order: list = field(default_factory=list)
    type_order: list = field(default_factory=list)
    interfaces: list = field(default_factory=list)    # (kind, from_schema, items)

    def entity(self, name: str) -> EntityDef:
        return self.entities[name]


class ExpressParseError(ValueError):
    """Raised when EXPRESS text does not conform to the handled grammar."""


# --- tokenizer ---------------------------------------------------------------

@dataclass(frozen=True)
class Token:
    kind: str          # 'id' | 'num' | 'str' | 'op'
    text: str
    start: int
    end: int


_ID_START = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_"
_ID_BODY = _ID_START + "0123456789"


def tokenize(text: str) -> list:
    """Tokenize EXPRESS source, stripping ``--`` and ``(* *)`` comments."""

    toks: list = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in " \t\r\n":
            i += 1
            continue
        if c == "-" and i + 1 < n and text[i + 1] == "-":
            j = text.find("\n", i)
            i = n if j == -1 else j + 1
            continue
        if c == "(" and i + 1 < n and text[i + 1] == "*":
            depth = 1
            i += 2
            while i < n and depth:
                if text[i] == "(" and i + 1 < n and text[i + 1] == "*":
                    depth += 1
                    i += 2
                elif text[i] == "*" and i + 1 < n and text[i + 1] == ")":
                    depth -= 1
                    i += 2
                else:
                    i += 1
            if depth:
                raise ExpressParseError("unterminated (* comment *)")
            continue
        if c == "'":
            start = i
            i += 1
            buf: list = []
            while i < n:
                if text[i] == "'":
                    if i + 1 < n and text[i + 1] == "'":
                        buf.append("'")
                        i += 2
                        continue
                    i += 1
                    break
                buf.append(text[i])
                i += 1
            else:
                raise ExpressParseError("unterminated string literal")
            toks.append(Token("str", "".join(buf), start, i))
            continue
        if c in _ID_START:
            start = i
            while i < n and text[i] in _ID_BODY:
                i += 1
            toks.append(Token("id", text[start:i], start, i))
            continue
        if c.isdigit():
            start = i
            while i < n and (text[i].isdigit() or text[i] in ".eE+-"):
                # only consume +/- immediately after e/E (exponent sign)
                if text[i] in "+-" and text[i - 1] not in "eE":
                    break
                i += 1
            toks.append(Token("num", text[start:i], start, i))
            continue
        if c == ":" and i + 1 < n and text[i + 1] == "=":
            toks.append(Token("op", ":=", i, i + 2))
            i += 2
            continue
        toks.append(Token("op", c, i, i + 1))
        i += 1
    return toks


# --- keyword-aware cursor ----------------------------------------------------

class _Cursor:
    def __init__(self, toks: list, source: str) -> None:
        self.toks = toks
        self.src = source
        self.i = 0
        self.n = len(toks)

    def at_end(self) -> bool:
        return self.i >= self.n

    def peek(self) -> Token | None:
        return self.toks[self.i] if self.i < self.n else None

    def next(self) -> Token:
        if self.i >= self.n:
            raise ExpressParseError("unexpected end of schema")
        t = self.toks[self.i]
        self.i += 1
        return t

    def is_kw(self, word: str) -> bool:
        t = self.peek()
        return t is not None and t.kind == "id" and t.text.upper() == word

    def is_op(self, ch: str) -> bool:
        t = self.peek()
        return t is not None and t.kind == "op" and t.text == ch

    def eat_kw(self, word: str) -> None:
        if not self.is_kw(word):
            got = self.peek()
            raise ExpressParseError(
                f"expected keyword {word!r}, got {got.text if got else 'EOF'!r}")
        self.i += 1

    def eat_op(self, ch: str) -> None:
        if not self.is_op(ch):
            got = self.peek()
            raise ExpressParseError(
                f"expected {ch!r}, got {got.text if got else 'EOF'!r}")
        self.i += 1

    def id_text(self) -> str:
        t = self.next()
        if t.kind != "id":
            raise ExpressParseError(f"expected identifier, got {t.text!r}")
        return t.text


_KEYWORDS = {
    "OF", "OPTIONAL", "UNIQUE", "LIST", "SET", "ARRAY", "BAG", "SELECT",
    "ENUMERATION", "EXTENSIBLE", "GENERIC_ENTITY", "FIXED",
    "NUMBER", "REAL", "INTEGER", "LOGICAL", "BOOLEAN", "STRING", "BINARY",
}


# --- type expression parsing -------------------------------------------------

_SIMPLE = {"NUMBER", "REAL", "INTEGER", "LOGICAL", "BOOLEAN", "STRING", "BINARY"}


def _parse_width(cur: _Cursor):
    if not cur.is_op("("):
        return None, False
    cur.eat_op("(")
    t = cur.next()
    if t.kind != "num":
        raise ExpressParseError("width must be a number")
    width = int(t.text)
    cur.eat_op(")")
    fixed = False
    if cur.is_kw("FIXED"):
        cur.eat_kw("FIXED")
        fixed = True
    return width, fixed


def _parse_bound(cur: _Cursor):
    """Parse ``[lo:hi]`` returning (lower_text, upper_text) or (None, None)."""

    if not cur.is_op("["):
        return None, None
    cur.eat_op("[")
    lower = _read_until(cur, {":"})
    cur.eat_op(":")
    upper = _read_until(cur, {"]"})
    cur.eat_op("]")
    return lower.strip(), upper.strip()


def _read_until(cur: _Cursor, stop_ops: set) -> str:
    """Consume tokens until a top-level op in ``stop_ops``; return raw text."""

    depth = 0
    start_off = cur.peek().start if cur.peek() else 0
    end_off = start_off
    while not cur.at_end():
        t = cur.peek()
        if t.kind == "op" and t.text in "([":
            depth += 1
        elif t.kind == "op" and t.text in ")]":
            if depth == 0 and t.text in stop_ops:
                break
            depth -= 1
        elif depth == 0 and t.kind == "op" and t.text in stop_ops:
            break
        end_off = t.end
        cur.next()
    return cur.src[start_off:end_off]


def parse_type(cur: _Cursor) -> TypeRef:
    """Parse a parameter/underlying type expression."""

    t = cur.peek()
    if t is None:
        raise ExpressParseError("expected a type")

    if cur.is_kw("LIST") or cur.is_kw("SET") or cur.is_kw("ARRAY") or cur.is_kw("BAG"):
        kw = cur.id_text().lower()
        lower, upper = _parse_bound(cur)
        cur.eat_kw("OF")
        optional = False
        unique = False
        if cur.is_kw("OPTIONAL"):
            cur.eat_kw("OPTIONAL")
            optional = True
        if cur.is_kw("UNIQUE"):
            cur.eat_kw("UNIQUE")
            unique = True
        base = parse_type(cur)
        return TypeRef(kw, base=base, lower=lower, upper=upper,
                       unique=unique, optional=optional)

    extensible = False
    if cur.is_kw("EXTENSIBLE"):
        cur.eat_kw("EXTENSIBLE")
        extensible = True
        if cur.is_kw("GENERIC_ENTITY"):
            cur.eat_kw("GENERIC_ENTITY")

    if cur.is_kw("ENUMERATION"):
        cur.eat_kw("ENUMERATION")
        items: list = []
        # EXTENSIBLE ENUMERATION may omit ``OF (...)`` (an extension point).
        if cur.is_kw("OF"):
            cur.eat_kw("OF")
            cur.eat_op("(")
            items.append(cur.id_text())
            while cur.is_op(","):
                cur.eat_op(",")
                items.append(cur.id_text())
            cur.eat_op(")")
        return TypeRef("enum", items=tuple(items), extensible=extensible)

    if cur.is_kw("SELECT"):
        cur.eat_kw("SELECT")
        # select_extension: SELECT BASED_ON type_ref [ WITH (named_types...) ]
        if cur.is_kw("BASED_ON"):
            cur.eat_kw("BASED_ON")
            base_name = cur.id_text()
            names = [base_name]
            if cur.is_kw("WITH"):
                cur.eat_kw("WITH")
                cur.eat_op("(")
                names.append(cur.id_text())
                while cur.is_op(","):
                    cur.eat_op(",")
                    names.append(cur.id_text())
                cur.eat_op(")")
            return TypeRef("select", types=tuple(names), extensible=extensible)
        # extensible select may also omit the list entirely
        names = []
        if cur.is_op("("):
            cur.eat_op("(")
            names.append(cur.id_text())
            while cur.is_op(","):
                cur.eat_op(",")
                names.append(cur.id_text())
            cur.eat_op(")")
        return TypeRef("select", types=tuple(names), extensible=extensible)

    if t.kind == "id" and t.text.upper() in _SIMPLE:
        name = cur.id_text().upper()
        if name in ("STRING", "BINARY"):
            width, fixed = _parse_width(cur)
            return TypeRef("simple", name=name, width=width, fixed=fixed)
        return TypeRef("simple", name=name)

    # Named entity/type reference.
    if t.kind == "id":
        return TypeRef("named", name=cur.id_text())
    raise ExpressParseError(f"unexpected token in type: {t.text!r}")


# --- supertype expression ----------------------------------------------------

def _parse_supertype_expr(cur: _Cursor) -> SuperTypeExpr:
    return _sup_andor(cur)


def _sup_andor(cur: _Cursor) -> SuperTypeExpr:
    first = _sup_and(cur)
    factors = [first]
    while cur.is_kw("ANDOR"):
        cur.eat_kw("ANDOR")
        factors.append(_sup_and(cur))
    if len(factors) == 1:
        return first
    return SuperTypeExpr("andor", operands=tuple(factors))


def _sup_and(cur: _Cursor) -> SuperTypeExpr:
    first = _sup_term(cur)
    terms = [first]
    while cur.is_kw("AND"):
        cur.eat_kw("AND")
        terms.append(_sup_term(cur))
    if len(terms) == 1:
        return first
    return SuperTypeExpr("and", operands=tuple(terms))


def _sup_term(cur: _Cursor) -> SuperTypeExpr:
    if cur.is_kw("ONEOF"):
        cur.eat_kw("ONEOF")
        cur.eat_op("(")
        exprs = [_parse_supertype_expr(cur)]
        while cur.is_op(","):
            cur.eat_op(",")
            exprs.append(_parse_supertype_expr(cur))
        cur.eat_op(")")
        return SuperTypeExpr("oneof", operands=tuple(exprs))
    if cur.is_op("("):
        cur.eat_op("(")
        inner = _parse_supertype_expr(cur)
        cur.eat_op(")")
        return inner
    return SuperTypeExpr("ref", name=cur.id_text())


def supertype_leaf_names(expr: SuperTypeExpr | None) -> list:
    """Flatten a SUPERTYPE expression to the set of referenced entity names."""

    if expr is None:
        return []
    if expr.op == "ref":
        return [expr.name]
    out: list = []
    for sub in expr.operands:
        for name in supertype_leaf_names(sub):
            if name not in out:
                out.append(name)
    return out


# --- entity declaration ------------------------------------------------------

def _parse_entity(cur: _Cursor) -> EntityDef:
    cur.eat_kw("ENTITY")
    name = cur.id_text()
    ent = EntityDef(name=name)

    # subsuper: [supertype constraint] [subtype declaration]
    if cur.is_kw("ABSTRACT"):
        cur.eat_kw("ABSTRACT")
        ent.is_abstract = True
        if cur.is_kw("SUPERTYPE"):
            cur.eat_kw("SUPERTYPE")
            if cur.is_kw("OF"):
                cur.eat_kw("OF")
                cur.eat_op("(")
                ent.supertype_expr = _parse_supertype_expr(cur)
                cur.eat_op(")")
    elif cur.is_kw("SUPERTYPE"):
        cur.eat_kw("SUPERTYPE")
        cur.eat_kw("OF")
        cur.eat_op("(")
        ent.supertype_expr = _parse_supertype_expr(cur)
        cur.eat_op(")")

    if cur.is_kw("SUBTYPE"):
        cur.eat_kw("SUBTYPE")
        cur.eat_kw("OF")
        cur.eat_op("(")
        ent.supertypes.append(cur.id_text())
        while cur.is_op(","):
            cur.eat_op(",")
            ent.supertypes.append(cur.id_text())
        cur.eat_op(")")

    cur.eat_op(";")

    # explicit attributes
    while cur.peek() is not None and cur.peek().kind == "id" \
            and cur.peek().text.upper() not in (
                "DERIVE", "INVERSE", "UNIQUE", "WHERE", "END_ENTITY"):
        names = [_read_qualified_name(cur)]
        while cur.is_op(","):
            cur.eat_op(",")
            names.append(_read_qualified_name(cur))
        # A redeclared attribute (qualified name) may carry ``RENAMED id``.
        if cur.is_kw("RENAMED"):
            cur.eat_kw("RENAMED")
            cur.id_text()
        cur.eat_op(":")
        optional = False
        if cur.is_kw("OPTIONAL"):
            cur.eat_kw("OPTIONAL")
            optional = True
        ty = parse_type(cur)
        cur.eat_op(";")
        for a in names:
            # A qualified name (contains '\\' or '.') is a redeclaration of an
            # inherited attribute; it restricts a type but does not add a new
            # attribute, so it must not count toward the instance arity.
            if "\\" in a or "." in a:
                continue
            ent.attributes.append(Attribute(a, ty, optional))

    if cur.is_kw("DERIVE"):
        cur.eat_kw("DERIVE")
        _parse_derive(cur, ent)
    if cur.is_kw("INVERSE"):
        cur.eat_kw("INVERSE")
        _parse_inverse(cur, ent)
    if cur.is_kw("UNIQUE"):
        cur.eat_kw("UNIQUE")
        _parse_unique(cur, ent)
    if cur.is_kw("WHERE"):
        cur.eat_kw("WHERE")
        _parse_where(cur, ent.where_rules)

    cur.eat_kw("END_ENTITY")
    cur.eat_op(";")
    return ent


def _read_qualified_name(cur: _Cursor) -> str:
    """Read a possibly-qualified attribute reference like ``SELF\\p.x``."""

    parts = [cur.id_text()]
    while cur.is_op("\\") or cur.is_op("."):
        sep = cur.next().text
        parts.append(sep + cur.id_text())
    return "".join(parts)


def _parse_derive(cur: _Cursor, ent: EntityDef) -> None:
    while cur.peek() is not None and cur.peek().kind == "id" \
            and cur.peek().text.upper() not in (
                "INVERSE", "UNIQUE", "WHERE", "END_ENTITY"):
        name = _read_qualified_name(cur)
        cur.eat_op(":")
        ty = parse_type(cur)
        cur.eat_op(":=")
        expr = _read_until(cur, {";"})
        cur.eat_op(";")
        ent.derived.append(DerivedAttribute(name, ty, expr.strip()))


def _parse_inverse(cur: _Cursor, ent: EntityDef) -> None:
    while cur.peek() is not None and cur.peek().kind == "id" \
            and cur.peek().text.upper() not in (
                "UNIQUE", "WHERE", "END_ENTITY"):
        name = cur.id_text()
        cur.eat_op(":")
        raw = _read_until(cur, {";"})
        cur.eat_op(";")
        ent.inverse.append(InverseAttribute(name, raw.strip()))


def _parse_unique(cur: _Cursor, ent: EntityDef) -> None:
    while cur.peek() is not None and cur.peek().kind == "id" \
            and cur.peek().text.upper() not in ("WHERE", "END_ENTITY"):
        # Rule may be "label : attr, attr;" or just "attr, attr;".
        first = _read_qualified_name(cur)
        label = None
        attrs: list = []
        if cur.is_op(":") and not cur.is_op(":="):
            cur.eat_op(":")
            label = first
        else:
            attrs.append(first)
        # remaining comma-separated attribute references
        if not attrs:
            attrs.append(_read_qualified_name(cur))
        while cur.is_op(","):
            cur.eat_op(",")
            attrs.append(_read_qualified_name(cur))
        cur.eat_op(";")
        ent.unique_rules.append(UniqueRule(label, tuple(attrs)))


def _parse_where(cur: _Cursor, out: list) -> None:
    while cur.peek() is not None and cur.peek().kind == "id" \
            and cur.peek().text.upper() != "END_ENTITY" \
            and cur.peek().text.upper() != "END_TYPE":
        # Optional "label :" prefix, then an expression up to ';'.
        save = cur.i
        label = None
        if cur.peek().kind == "id":
            maybe_label = cur.peek().text
            # peek two tokens ahead for ':' not ':='
            if cur.i + 1 < cur.n and cur.toks[cur.i + 1].kind == "op" \
                    and cur.toks[cur.i + 1].text == ":":
                label = maybe_label
                cur.i += 2
        expr = _read_until(cur, {";"})
        cur.eat_op(";")
        if not expr.strip() and label is None:
            cur.i = save
            break
        out.append(WhereRule(label, expr.strip()))


# --- type declaration --------------------------------------------------------

def _parse_type_decl(cur: _Cursor) -> TypeDef:
    cur.eat_kw("TYPE")
    name = cur.id_text()
    cur.eat_op("=")
    underlying = parse_type(cur)
    cur.eat_op(";")
    td = TypeDef(name=name, underlying=underlying)
    if cur.is_kw("WHERE"):
        cur.eat_kw("WHERE")
        _parse_where(cur, td.where_rules)
    cur.eat_kw("END_TYPE")
    cur.eat_op(";")
    return td


# --- block skipping ----------------------------------------------------------

def _skip_to(cur: _Cursor, end_kw: str) -> None:
    while not cur.at_end():
        if cur.is_kw(end_kw):
            cur.eat_kw(end_kw)
            if cur.is_op(";"):
                cur.eat_op(";")
            return
        cur.next()
    raise ExpressParseError(f"missing {end_kw}")


def _skip_stmt(cur: _Cursor) -> None:
    while not cur.at_end():
        if cur.is_op(";"):
            cur.eat_op(";")
            return
        cur.next()


# --- top-level schema --------------------------------------------------------

def parse_schema(text: str) -> Schema:
    """Parse a single EXPRESS ``SCHEMA ... END_SCHEMA;`` into a :class:`Schema`."""

    cur = _Cursor(tokenize(text), text)
    cur.eat_kw("SCHEMA")
    name = cur.id_text()
    # optional schema_version_id string
    if cur.peek() is not None and cur.peek().kind == "str":
        cur.next()
    cur.eat_op(";")
    schema = Schema(name=name)

    while not cur.at_end():
        if cur.is_kw("END_SCHEMA"):
            cur.eat_kw("END_SCHEMA")
            if cur.is_op(";"):
                cur.eat_op(";")
            break
        if cur.is_kw("ENTITY"):
            ent = _parse_entity(cur)
            schema.entities[ent.name] = ent
            schema.entity_order.append(ent.name)
        elif cur.is_kw("TYPE"):
            td = _parse_type_decl(cur)
            schema.types[td.name] = td
            schema.type_order.append(td.name)
        elif cur.is_kw("USE") or cur.is_kw("REFERENCE"):
            kind = cur.id_text().upper()
            cur.eat_kw("FROM")
            from_schema = cur.id_text()
            schema.interfaces.append((kind, from_schema))
            _skip_stmt(cur)
        elif cur.is_kw("FUNCTION"):
            _skip_to(cur, "END_FUNCTION")
        elif cur.is_kw("PROCEDURE"):
            _skip_to(cur, "END_PROCEDURE")
        elif cur.is_kw("RULE"):
            _skip_to(cur, "END_RULE")
        elif cur.is_kw("CONSTANT"):
            _skip_to(cur, "END_CONSTANT")
        elif cur.is_kw("SUBTYPE_CONSTRAINT"):
            _skip_to(cur, "END_SUBTYPE_CONSTRAINT")
        else:
            # Unknown declaration: skip to next ';' to stay robust.
            _skip_stmt(cur)
    return schema
