"""ISO 10303-21 (STEP part 21) tokenizer / parser and serializer.

STEP-LLM (Shi et al., DATE 2026) generates CAD STEP models directly from
natural language. The LLM itself is external and out of scope here; this module
implements the deterministic substrate the paper relies on: reading, structuring
and re-emitting the STEP part-21 exchange text.

A STEP file is organised into a ``HEADER`` section (metadata: schema, author,
units) and a ``DATA`` section that encodes geometry/topology as a collection of
entity instances ``#N = ENTITY(args);`` linked by cross-references (``#M``). This
module parses both sections into a structured, cross-referenced object model and
serializes it back, supporting an exact round-trip for canonically formatted
text.

The parser is a small recursive-descent scanner over the part-21 grammar:

  * references ``#N``                    -> :class:`Ref`
  * enumerations ``.T.`` / ``.PLANE.``  -> :class:`Enum`
  * strings ``'...'`` (``''`` escapes)  -> ``str``
  * integers ``42``                     -> ``int``
  * reals ``0.`` / ``-1.5E-3``          -> :class:`Real` (keeps literal text)
  * unset ``$`` / derived ``*``         -> :data:`UNSET` / :data:`DERIVED`
  * lists ``(a,b,c)``                   -> ``list``
  * typed values ``NAME(...)``          -> :class:`Typed`
  * complex instances ``(A(..)B(..))``  -> ``list`` of :class:`Typed`

Everything is pure and deterministic; no kernel, no I/O beyond the passed text.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --- value model -------------------------------------------------------------

@dataclass(frozen=True)
class Ref:
    """A cross-reference ``#N`` to another entity instance."""

    id: int

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Ref(#{self.id})"


@dataclass(frozen=True)
class Enum:
    """A part-21 enumeration literal ``.NAME.`` (name stored without dots)."""

    name: str


@dataclass(frozen=True)
class Real:
    """A real literal; the source ``text`` is kept so round-trips are exact."""

    text: str

    @property
    def value(self) -> float:
        return float(self.text)


class _Sentinel:
    __slots__ = ("token",)

    def __init__(self, token: str) -> None:
        self.token = token

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{self.token}>"


UNSET = _Sentinel("$")     # omitted / unset optional attribute
DERIVED = _Sentinel("*")   # attribute value is derived in the schema


@dataclass(frozen=True)
class Typed:
    """A typed parameter or simple-entity body ``KEYWORD(param, ...)``."""

    keyword: str
    params: tuple = ()


@dataclass
class Entity:
    """A DATA-section instance ``#id = keyword(params);``.

    For a *complex* instance ``#id=(A(..)B(..));`` the ``keyword`` is ``None``
    and ``params`` holds the list of :class:`Typed` parts.
    """

    id: int
    keyword: str | None
    params: list


@dataclass
class StepFile:
    """A parsed STEP part-21 file: header statements + DATA entities."""

    header: list = field(default_factory=list)     # list[Typed]
    entities: dict = field(default_factory=dict)    # id -> Entity
    order: list = field(default_factory=list)       # ids in file order

    def get(self, ref) -> Entity | None:
        rid = ref.id if isinstance(ref, Ref) else int(ref)
        return self.entities.get(rid)

    def add(self, entity: Entity) -> None:
        if entity.id in self.entities:
            raise ValueError(f"duplicate entity id #{entity.id}")
        self.entities[entity.id] = entity
        self.order.append(entity.id)


# --- scanning helpers --------------------------------------------------------

class ParseError(ValueError):
    """Raised when the part-21 text is malformed."""


def strip_comments(text: str) -> str:
    """Remove ``/* ... */`` block comments (outside string literals)."""

    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "'":
            j = i + 1
            while j < n:
                if text[j] == "'":
                    if j + 1 < n and text[j + 1] == "'":
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            out.append(text[i:j])
            i = j
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            if j == -1:
                raise ParseError("unterminated /* comment */")
            i = j + 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def split_statements(text: str):
    """Yield top-level ``;``-terminated statements (string-aware)."""

    i, n, start = 0, len(text), 0
    while i < n:
        c = text[i]
        if c == "'":
            i += 1
            while i < n:
                if text[i] == "'":
                    if i + 1 < n and text[i + 1] == "'":
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue
        if c == ";":
            stmt = text[start:i].strip()
            if stmt:
                yield stmt
            i += 1
            start = i
            continue
        i += 1
    tail = text[start:].strip()
    if tail:
        raise ParseError(f"trailing text without terminating ';': {tail!r}")


class _ValueScanner:
    """Recursive-descent scanner over a part-21 parameter expression."""

    def __init__(self, s: str) -> None:
        self.s = s
        self.i = 0
        self.n = len(s)

    def _skip_ws(self) -> None:
        while self.i < self.n and self.s[self.i] in " \t\r\n":
            self.i += 1

    def at_end(self) -> bool:
        self._skip_ws()
        return self.i >= self.n

    def peek(self) -> str:
        self._skip_ws()
        return self.s[self.i] if self.i < self.n else ""

    def parse_value(self):
        self._skip_ws()
        if self.i >= self.n:
            raise ParseError("unexpected end of expression")
        c = self.s[self.i]
        if c == "#":
            return self._read_ref()
        if c == "'":
            return self._read_string()
        if c == "(":
            return self._read_list()
        if c == "$":
            self.i += 1
            return UNSET
        if c == "*":
            self.i += 1
            return DERIVED
        if c == ".":
            return self._read_enum()
        if c == "+" or c == "-" or c.isdigit():
            return self._read_number()
        if c.isalpha() or c == "_":
            return self._read_keyword()
        raise ParseError(f"unexpected character {c!r} at {self.i}")

    def _read_ref(self) -> Ref:
        self.i += 1  # '#'
        start = self.i
        while self.i < self.n and self.s[self.i].isdigit():
            self.i += 1
        if start == self.i:
            raise ParseError("'#' not followed by digits")
        return Ref(int(self.s[start:self.i]))

    def _read_string(self) -> str:
        self.i += 1  # opening quote
        out: list[str] = []
        while self.i < self.n:
            c = self.s[self.i]
            if c == "'":
                if self.i + 1 < self.n and self.s[self.i + 1] == "'":
                    out.append("'")
                    self.i += 2
                    continue
                self.i += 1
                return "".join(out)
            out.append(c)
            self.i += 1
        raise ParseError("unterminated string literal")

    def _read_enum(self) -> Enum:
        self.i += 1  # opening '.'
        start = self.i
        while self.i < self.n and self.s[self.i] != ".":
            self.i += 1
        if self.i >= self.n:
            raise ParseError("unterminated enumeration literal")
        name = self.s[start:self.i]
        self.i += 1  # closing '.'
        return Enum(name)

    def _read_number(self):
        start = self.i
        if self.s[self.i] in "+-":
            self.i += 1
        is_real = False
        while self.i < self.n:
            c = self.s[self.i]
            if c.isdigit():
                self.i += 1
            elif c == ".":
                is_real = True
                self.i += 1
            elif c in "eE":
                is_real = True
                self.i += 1
                if self.i < self.n and self.s[self.i] in "+-":
                    self.i += 1
            else:
                break
        text = self.s[start:self.i]
        if is_real:
            return Real(text)
        return int(text)

    def _read_keyword(self):
        start = self.i
        while self.i < self.n and (self.s[self.i].isalnum() or self.s[self.i] == "_"):
            self.i += 1
        name = self.s[start:self.i]
        self._skip_ws()
        if self.i < self.n and self.s[self.i] == "(":
            params = self._read_list()
            return Typed(name, tuple(params))
        # bare keyword (rare); treat as an enumeration-like token by name.
        return Enum(name)

    def _read_list(self) -> list:
        assert self.s[self.i] == "("
        self.i += 1  # '('
        items: list = []
        self._skip_ws()
        if self.i < self.n and self.s[self.i] == ")":
            self.i += 1
            return items
        while True:
            items.append(self.parse_value())
            self._skip_ws()
            if self.i >= self.n:
                raise ParseError("unterminated list")
            c = self.s[self.i]
            if c == ",":
                self.i += 1
                continue
            if c == ")":
                self.i += 1
                return items
            raise ParseError(f"expected ',' or ')' in list, got {c!r}")

    def parse_complex(self) -> list:
        """Parse ``(A(..)B(..))`` -> list of :class:`Typed` (no commas)."""

        self._skip_ws()
        if self.i >= self.n or self.s[self.i] != "(":
            raise ParseError("complex instance must start with '('")
        self.i += 1
        parts: list = []
        while True:
            self._skip_ws()
            if self.i >= self.n:
                raise ParseError("unterminated complex instance")
            if self.s[self.i] == ")":
                self.i += 1
                return parts
            val = self.parse_value()
            if not isinstance(val, Typed):
                raise ParseError("complex instance parts must be typed values")
            parts.append(val)


# --- top-level parse ---------------------------------------------------------

def parse_expression(text: str):
    """Parse a single part-21 value expression (utility for tests/callers)."""

    sc = _ValueScanner(text)
    val = sc.parse_value()
    if not sc.at_end():
        raise ParseError(f"trailing text after expression: {text[sc.i:]!r}")
    return val


def _parse_entity(stmt: str) -> Entity:
    eq = stmt.index("=")
    lhs = stmt[:eq].strip()
    if not lhs.startswith("#"):
        raise ParseError(f"entity must start with '#': {stmt!r}")
    ent_id = int(lhs[1:])
    rhs = stmt[eq + 1:].strip()
    sc = _ValueScanner(rhs)
    if sc.peek() == "(":
        parts = sc.parse_complex()
        if not sc.at_end():
            raise ParseError(f"trailing text in complex instance: {rhs!r}")
        return Entity(ent_id, None, list(parts))
    val = sc.parse_value()
    if not sc.at_end():
        raise ParseError(f"trailing text after entity body: {rhs!r}")
    if not isinstance(val, Typed):
        raise ParseError(f"entity body must be a typed value: {rhs!r}")
    return Entity(ent_id, val.keyword, list(val.params))


def parse(text: str) -> StepFile:
    """Parse a full STEP part-21 file into a :class:`StepFile`."""

    text = strip_comments(text)
    statements = list(split_statements(text))
    if not statements or statements[0].strip() != "ISO-10303-21":
        raise ParseError("file must start with 'ISO-10303-21;'")
    if statements[-1].strip() != "END-ISO-10303-21":
        raise ParseError("file must end with 'END-ISO-10303-21;'")

    step = StepFile()
    section = None  # None | 'HEADER' | 'DATA'
    for stmt in statements[1:-1]:
        head = stmt.strip()
        upper = head.upper()
        if upper == "HEADER":
            section = "HEADER"
            continue
        if upper == "DATA" or upper.startswith("DATA "):
            section = "DATA"
            continue
        if upper == "ENDSEC":
            section = None
            continue
        if head.startswith("#"):
            if section != "DATA":
                raise ParseError("entity instance outside DATA section")
            step.add(_parse_entity(head))
        elif section == "HEADER":
            val = parse_expression(head)
            if not isinstance(val, Typed):
                raise ParseError(f"malformed header record: {head!r}")
            step.header.append(val)
        else:
            raise ParseError(f"unexpected statement {head!r}")
    return step


# --- serialization -----------------------------------------------------------

def serialize_value(value) -> str:
    if isinstance(value, Ref):
        return f"#{value.id}"
    if isinstance(value, Enum):
        return f".{value.name}."
    if isinstance(value, Real):
        return value.text
    if isinstance(value, bool):  # guard: bool is an int subclass
        return ".T." if value else ".F."
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if value is UNSET:
        return "$"
    if value is DERIVED:
        return "*"
    if isinstance(value, Typed):
        inner = ",".join(serialize_value(p) for p in value.params)
        return f"{value.keyword}({inner})"
    if isinstance(value, (list, tuple)):
        return "(" + ",".join(serialize_value(v) for v in value) + ")"
    raise TypeError(f"cannot serialize value of type {type(value).__name__}")


def serialize_entity(entity: Entity) -> str:
    if entity.keyword is None:
        body = "".join(serialize_value(p) for p in entity.params)
        return f"#{entity.id}=({body});"
    inner = ",".join(serialize_value(p) for p in entity.params)
    return f"#{entity.id}={entity.keyword}({inner});"


def serialize(step: StepFile) -> str:
    """Serialize a :class:`StepFile` back to canonical part-21 text."""

    lines = ["ISO-10303-21;", "HEADER;"]
    for rec in step.header:
        lines.append(serialize_value(rec) + ";")
    lines.append("ENDSEC;")
    lines.append("DATA;")
    for ent_id in step.order:
        lines.append(serialize_entity(step.entities[ent_id]))
    lines.append("ENDSEC;")
    lines.append("END-ISO-10303-21;")
    return "\n".join(lines) + "\n"


# --- reference utilities -----------------------------------------------------

def iter_refs(value):
    """Yield every :class:`Ref` reachable inside a value (recursively)."""

    if isinstance(value, Ref):
        yield value
    elif isinstance(value, Typed):
        for p in value.params:
            yield from iter_refs(p)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from iter_refs(v)


def entity_refs(entity: Entity):
    """All ids referenced by ``entity`` (in first-seen order, deduplicated)."""

    seen: list[int] = []
    for p in entity.params:
        for ref in iter_refs(p):
            if ref.id not in seen:
                seen.append(ref.id)
    return seen
