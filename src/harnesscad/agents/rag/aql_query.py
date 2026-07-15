"""AQL -- a restricted, deterministic query notation for retrieval-as-query.

Paper: *RUBICON -- Agentic AI for Messy Enterprise Data* (Wenz et al., MIT/TU
Darmstadt, VLDB). RUBICON's central argument is that free natural language is an
unreliable interface over heterogeneous sources, and full SQL is too demanding;
the workable middle is **Agentic Query Language (AQL)** -- a "baby SQL" whose
*structural* part (which source, which fields, how results join) is written
explicitly, while only the local predicate may stay natural language (Sec. 2):

    FIND   <columns>
    FROM   <source or relation>
    WHERE  <natural-language predicate>

Queries compose with ``JOIN`` (two source-local FINDs), and the notation carries
mundane housekeeping commands that make exploration explicit rather than
conversational: ``?`` lists sources, ``? <source>`` lists relations,
``? <relation>`` lists attributes, and ``SAVE <name>`` names an intermediate
result. The paper stresses this is about *allocation of responsibility*, not
syntax: the human fixes the structural skeleton, the model handles only the
predicate.

This module is the deterministic core the paper leaves implicit: a **parser and
logical-plan representation** for AQL. It parses AQL text into typed dataclasses
(:class:`FindQuery`, :class:`JoinQuery`, :class:`SchemaCommand`,
:class:`SaveCommand`), classifies each field/source as fully qualified or
under-specified (the paper's "degrees of explicitness"), and reports which parts
are delegated to the LLM (the free predicate). No source is contacted and no
model is called -- this is pure notation processing, so a plan is inspectable,
saveable, and re-executable exactly as the paper requires.

Stdlib-only and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "Column",
    "Source",
    "FindQuery",
    "JoinQuery",
    "SchemaCommand",
    "SaveCommand",
    "ParseError",
    "parse",
    "parse_program",
    "delegation_report",
]


class ParseError(ValueError):
    """Raised when AQL text is malformed."""


@dataclass(frozen=True)
class Column:
    """A projected column, optionally table-qualified (``Faculty.name``)."""

    name: str
    table: str = ""

    @property
    def qualified(self) -> bool:
        return bool(self.table)


@dataclass(frozen=True)
class Source:
    """A FROM target: a source, optionally a ``Source.Relation`` pair."""

    source: str
    relation: str = ""

    @property
    def qualified(self) -> bool:
        return bool(self.relation)


@dataclass
class FindQuery:
    """A single ``FIND ... FROM ... WHERE ...`` block."""

    columns: list          # list[Column]  (empty list means "*"-like)
    source: Source
    predicate: str = ""    # free natural-language text (delegated to the LLM)

    @property
    def star(self) -> bool:
        return len(self.columns) == 0


@dataclass
class JoinQuery:
    """Two source-local FINDs joined into one logical result."""

    left: FindQuery
    right: FindQuery


@dataclass
class SchemaCommand:
    """A ``?`` housekeeping command. ``target`` empty => list sources."""

    target: str = ""


@dataclass
class SaveCommand:
    """A ``SAVE <name>`` naming of the previous intermediate result."""

    name: str


# --- tokenizing helpers -----------------------------------------------------

_KEYWORDS = ("FIND", "FROM", "WHERE", "JOIN")


def _split_keywords(text: str):
    """Yield ``(keyword, payload)`` for a FIND/FROM/WHERE/JOIN clause stream.

    Keyword matching is whole-word and case-insensitive; anything before the
    first keyword is an error. JOIN is emitted with an empty payload as a marker.
    """
    tokens = text.split()
    out = []
    cur_kw = None
    cur = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        up = tok.upper()
        if up in _KEYWORDS:
            if cur_kw is not None:
                out.append((cur_kw, " ".join(cur).strip()))
            elif cur:
                raise ParseError(f"unexpected text before keyword: {' '.join(cur)!r}")
            cur_kw = up
            cur = []
            if up == "JOIN":
                out.append(("JOIN", ""))
                cur_kw = None
        else:
            cur.append(tok)
        i += 1
    if cur_kw is not None:
        out.append((cur_kw, " ".join(cur).strip()))
    elif cur:
        raise ParseError(f"trailing text with no keyword: {' '.join(cur)!r}")
    return out


def _parse_columns(payload: str):
    if payload.strip() in ("", "*"):
        return []
    cols = []
    for raw in payload.split(","):
        item = raw.strip()
        if not item:
            raise ParseError("empty column in FIND list")
        if "." in item:
            table, _, name = item.partition(".")
            cols.append(Column(name=name.strip(), table=table.strip()))
        else:
            cols.append(Column(name=item))
    return cols


def _parse_source(payload: str) -> Source:
    item = payload.strip()
    if not item:
        raise ParseError("FROM requires a source")
    if " " in item:
        raise ParseError(f"FROM source must be a single token: {item!r}")
    if "." in item:
        source, _, relation = item.partition(".")
        return Source(source=source.strip(), relation=relation.strip())
    return Source(source=item)


def _parse_find(clauses) -> FindQuery:
    """Build a FindQuery from an ordered ``(keyword, payload)`` list."""
    cols = None
    src = None
    pred = ""
    for kw, payload in clauses:
        if kw == "FIND":
            cols = _parse_columns(payload)
        elif kw == "FROM":
            src = _parse_source(payload)
        elif kw == "WHERE":
            pred = payload
        else:
            raise ParseError(f"unexpected keyword in FIND block: {kw}")
    if cols is None:
        raise ParseError("query missing FIND")
    if src is None:
        raise ParseError("query missing FROM")
    return FindQuery(columns=cols, source=src, predicate=pred)


# --- public parse -----------------------------------------------------------

def parse(text: str):
    """Parse a single AQL statement into its typed representation.

    Handles ``?``/``? x`` schema commands, ``SAVE name``, a lone ``FIND`` query,
    and a ``FIND ... JOIN FIND ...`` join. Raises :class:`ParseError` otherwise.
    """
    stripped = text.strip()
    if not stripped:
        raise ParseError("empty AQL statement")
    if stripped.startswith("?"):
        return SchemaCommand(target=stripped[1:].strip())
    head = stripped.split(None, 1)
    if head[0].upper() == "SAVE":
        if len(head) < 2 or not head[1].strip():
            raise ParseError("SAVE requires a name")
        return SaveCommand(name=head[1].strip())

    clauses = _split_keywords(stripped)
    # split on the JOIN marker into at most two FIND blocks
    groups = [[]]
    for kw, payload in clauses:
        if kw == "JOIN":
            groups.append([])
        else:
            groups[-1].append((kw, payload))
    if len(groups) == 1:
        return _parse_find(groups[0])
    if len(groups) == 2:
        return JoinQuery(left=_parse_find(groups[0]), right=_parse_find(groups[1]))
    raise ParseError("AQL supports at most one JOIN")


def parse_program(text: str):
    """Parse a newline/`;`-separated sequence of AQL statements into a list."""
    out = []
    for chunk in text.replace(";", "\n").splitlines():
        if chunk.strip():
            out.append(parse(chunk))
    return out


# --- delegation analysis ----------------------------------------------------

def delegation_report(query) -> dict:
    """Summarize how much structure is explicit vs. delegated to the model.

    Returns counts of qualified/unqualified columns and sources plus the free
    predicates handed to the LLM -- the paper's "degrees of explicitness". A plan
    with all fields qualified and no predicate delegates nothing.
    """
    finds = []
    if isinstance(query, FindQuery):
        finds = [query]
    elif isinstance(query, JoinQuery):
        finds = [query.left, query.right]
    else:
        return {"finds": 0, "qualified_columns": 0, "unqualified_columns": 0,
                "qualified_sources": 0, "unqualified_sources": 0,
                "delegated_predicates": []}
    qcol = sum(1 for f in finds for c in f.columns if c.qualified)
    ucol = sum(1 for f in finds for c in f.columns if not c.qualified)
    qsrc = sum(1 for f in finds if f.source.qualified)
    usrc = sum(1 for f in finds if not f.source.qualified)
    preds = [f.predicate for f in finds if f.predicate.strip()]
    return {
        "finds": len(finds),
        "qualified_columns": qcol,
        "unqualified_columns": ucol,
        "qualified_sources": qsrc,
        "unqualified_sources": usrc,
        "delegated_predicates": preds,
    }
