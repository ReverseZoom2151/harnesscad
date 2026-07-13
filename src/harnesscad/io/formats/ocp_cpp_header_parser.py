"""Deterministic C++ header declaration parser (OCCT / OCP flavoured).

OCP (the CadQuery project's pybind11 binding generator for Open CASCADE) works
by *reading the OCCT C++ headers* and mechanically deriving the exported API
surface from them. The binding generation itself (pybind11, CMake, clang) is
out of scope for a stdlib-only harness, but the underlying capability -- turning
a directory of ``.hxx`` headers into a machine-readable inventory of classes,
methods, enums and typedefs -- is a deterministic parsing problem and is exactly
what a text-to-CAD system needs in order to *ground* generated kernel calls
against the real API.

This module is that parser. It is intentionally a pragmatic subset parser, not a
C++ front-end: it recovers declarations, not semantics. It is robust against the
constructs that actually appear in OCCT headers:

  * license/doc comments (``//``, ``//!``, ``/* ... */``)
  * preprocessor lines (``#ifndef``/``#define``/``#include``), incl. backslash
    line continuations
  * include guards and forward declarations (``class gp_Ax1;``)
  * ``Standard_EXPORT`` / ``virtual`` / ``static`` / ``explicit`` qualifiers
  * OCCT macros (``DEFINE_STANDARD_ALLOC``, ``DEFINE_STANDARD_RTTIEXT(...)``)
  * inline member bodies and constructor initialiser lists
  * nested classes, enums (scoped and unscoped), typedefs
  * default arguments, templated parameter types, operators, destructors

Design: comments/preprocessor are stripped first (newline-preserving), then a
brace/semicolon scanner splits the translation unit into ``(header, block)``
statements; class bodies recurse. Everything is pure functions over strings, so
parsing the same text always yields the same structure.

Public entry points:

  * :func:`strip_comments`, :func:`strip_preprocessor` -- normalisation
  * :func:`parse_header` -- text -> :class:`Header`
  * :func:`parse_header_file` -- path -> :class:`Header`
  * :func:`split_params` / :func:`parse_param` -- parameter-list handling
  * :class:`CppClass`.``arity_range`` style helpers on :class:`CppMethod`
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "CppClass",
    "CppEnum",
    "CppMethod",
    "CppParam",
    "CppTypedef",
    "Header",
    "ParseError",
    "iter_header_files",
    "parse_header",
    "parse_header_file",
    "parse_param",
    "split_params",
    "split_top_level",
    "strip_comments",
    "strip_preprocessor",
]


class ParseError(ValueError):
    """Raised when the scanner cannot recover a balanced structure."""


# --------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CppParam:
    """One function parameter."""

    type: str
    name: str = ""
    default: str = ""

    @property
    def has_default(self) -> bool:
        return self.default != ""

    @property
    def is_const_ref(self) -> bool:
        return self.type.startswith("const ") and self.type.rstrip().endswith("&")


@dataclass(frozen=True)
class CppMethod:
    """A member (or free) function declaration."""

    name: str
    return_type: str = ""
    params: Tuple[CppParam, ...] = ()
    access: str = "public"
    is_static: bool = False
    is_virtual: bool = False
    is_const: bool = False
    is_pure: bool = False
    is_explicit: bool = False
    is_exported: bool = False
    is_constructor: bool = False
    is_destructor: bool = False
    is_operator: bool = False

    @property
    def min_args(self) -> int:
        return sum(1 for p in self.params if not p.has_default)

    @property
    def max_args(self) -> int:
        return len(self.params)

    @property
    def arity_range(self) -> Tuple[int, int]:
        return (self.min_args, self.max_args)

    def accepts(self, argc: int) -> bool:
        return self.min_args <= argc <= self.max_args

    def signature(self) -> str:
        args = ", ".join(
            (p.type + (" " + p.name if p.name else "")) for p in self.params
        )
        head = (self.return_type + " ") if self.return_type else ""
        tail = " const" if self.is_const else ""
        return "%s%s(%s)%s" % (head, self.name, args, tail)


@dataclass(frozen=True)
class CppEnum:
    """An enum declaration and its enumerators (values in declaration order)."""

    name: str
    values: Tuple[str, ...] = ()
    is_scoped: bool = False
    access: str = "public"


@dataclass(frozen=True)
class CppTypedef:
    """A ``typedef`` or ``using X = Y;`` alias."""

    name: str
    target: str = ""


@dataclass
class CppClass:
    """A class/struct declaration."""

    name: str
    kind: str = "class"  # class | struct
    bases: Tuple[str, ...] = ()
    methods: List[CppMethod] = field(default_factory=list)
    enums: List[CppEnum] = field(default_factory=list)
    typedefs: List[CppTypedef] = field(default_factory=list)
    nested: List["CppClass"] = field(default_factory=list)
    is_forward: bool = False

    def method_names(self) -> Tuple[str, ...]:
        seen: List[str] = []
        for m in self.methods:
            if m.name not in seen:
                seen.append(m.name)
        return tuple(seen)

    def overloads(self, name: str) -> Tuple[CppMethod, ...]:
        return tuple(m for m in self.methods if m.name == name)

    def constructors(self) -> Tuple[CppMethod, ...]:
        return tuple(m for m in self.methods if m.is_constructor)


@dataclass
class Header:
    """Everything recovered from one header file."""

    path: str = ""
    classes: List[CppClass] = field(default_factory=list)
    enums: List[CppEnum] = field(default_factory=list)
    typedefs: List[CppTypedef] = field(default_factory=list)
    functions: List[CppMethod] = field(default_factory=list)
    forward_decls: List[str] = field(default_factory=list)

    def class_map(self) -> Dict[str, CppClass]:
        return {c.name: c for c in self.classes}

    def find(self, name: str) -> Optional[CppClass]:
        for c in self.classes:
            if c.name == name:
                return c
        return None


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------


def strip_comments(text: str) -> str:
    """Remove C and C++ comments, preserving line structure and string literals."""
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == '"' or c == "'":
            quote = c
            out.append(c)
            i += 1
            while i < n:
                d = text[i]
                out.append(d)
                i += 1
                if d == "\\" and i < n:
                    out.append(text[i])
                    i += 1
                    continue
                if d == quote:
                    break
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            if j < 0:
                j = n
                block = text[i:]
            else:
                block = text[i : j + 2]
            out.append("\n" * block.count("\n"))
            i = j + 2 if j < n else n
            continue
        out.append(c)
        i += 1
    return "".join(out)


def strip_preprocessor(text: str) -> str:
    """Drop preprocessor directives (honouring backslash continuations)."""
    lines = text.split("\n")
    out: List[str] = []
    skipping = False
    for line in lines:
        stripped = line.lstrip()
        if skipping:
            out.append("")
            skipping = line.rstrip().endswith("\\")
            continue
        if stripped.startswith("#"):
            out.append("")
            skipping = line.rstrip().endswith("\\")
            continue
        out.append(line)
    return "\n".join(out)


def _read_block(text: str, start: int) -> Tuple[str, int]:
    """``text[start] == '{'``; return (inner text, index just past matching '}')."""
    if text[start] != "{":
        raise ParseError("expected '{' at %d" % start)
    depth = 0
    i = start
    n = len(text)
    while i < n:
        c = text[i]
        if c == '"' or c == "'":
            quote = c
            i += 1
            while i < n:
                d = text[i]
                i += 1
                if d == "\\":
                    i += 1
                    continue
                if d == quote:
                    break
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
        i += 1
    raise ParseError("unbalanced '{' at %d" % start)


def split_top_level(text: str) -> List[Tuple[str, Optional[str]]]:
    """Split a translation unit / class body into ``(header, block)`` statements.

    ``block`` is ``None`` for plain ``;``-terminated declarations, and the inner
    text of ``{...}`` otherwise (inline bodies, class bodies, enum bodies).
    """
    items: List[Tuple[str, Optional[str]]] = []
    buf: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == '"' or c == "'":
            quote = c
            buf.append(c)
            i += 1
            while i < n:
                d = text[i]
                buf.append(d)
                i += 1
                if d == "\\" and i < n:
                    buf.append(text[i])
                    i += 1
                    continue
                if d == quote:
                    break
            continue
        if c == "{":
            inner, j = _read_block(text, i)
            head = "".join(buf).strip()
            items.append((head, inner))
            buf = []
            i = j
            # consume an optional ';' terminating the block (``class X {...};``)
            while i < n and text[i].isspace():
                i += 1
            if i < n and text[i] == ";":
                i += 1
            continue
        if c == ";":
            head = "".join(buf).strip()
            if head:
                items.append((head, None))
            buf = []
            i += 1
            continue
        buf.append(c)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        items.append((tail, None))
    return items


# --------------------------------------------------------------------------
# declaration parsing
# --------------------------------------------------------------------------

_ACCESS_RE = re.compile(r"\b(public|protected|private)\s*:")
_CLASS_RE = re.compile(
    r"^(?:template\s*<[^>]*>\s*)?(class|struct)\s+"
    r"(?:[A-Za-z_][A-Za-z0-9_]*\s+)?"  # e.g. Standard_EXPORT / alignas-ish token
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?::(?P<bases>[^;{]*))?$"
)
_ENUM_RE = re.compile(
    r"^enum\s+(?:(?P<scoped>class|struct)\s+)?"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)?\s*(?::[^{]*)?$"
)
_MACRO_LINE_RE = re.compile(r"^[A-Z][A-Z0-9_]*(\s*\([^()]*\))?$")
_QUALIFIERS = (
    "virtual",
    "static",
    "inline",
    "explicit",
    "constexpr",
    "friend",
    "Standard_EXPORT",
    "Standard_NODISCARD",
    "Standard_DEPRECATED",
)


def _clean_decl(text: str) -> str:
    """Drop macro-only lines from a raw declaration chunk."""
    kept: List[str] = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        if _MACRO_LINE_RE.match(s):
            continue
        kept.append(s)
    return " ".join(kept).strip()


def _split_commas(text: str) -> List[str]:
    """Split on top-level commas, respecting ``()``, ``[]``, ``<>`` nesting."""
    parts: List[str] = []
    depth = 0
    angle = 0
    buf: List[str] = []
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "<":
            angle += 1
        elif ch == ">" and angle > 0:
            angle -= 1
        if ch == "," and depth == 0 and angle == 0:
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    last = "".join(buf).strip()
    if last:
        parts.append(last)
    return parts


def parse_param(text: str) -> CppParam:
    """Parse a single parameter declaration such as ``const gp_Pnt& theP = gp_Pnt()``."""
    text = text.strip()
    default = ""
    depth = 0
    angle = 0
    for i, ch in enumerate(text):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "<":
            angle += 1
        elif ch == ">" and angle > 0:
            angle -= 1
        elif ch == "=" and depth == 0 and angle == 0:
            default = text[i + 1 :].strip()
            text = text[:i].strip()
            break
    if not text or text == "void":
        return CppParam(type="void" if text == "void" else "", name="", default=default)
    m = re.match(r"^(?P<type>.*?[\w>\*&\]\s])(?P<name>[A-Za-z_][A-Za-z0-9_]*)$", text)
    name = ""
    type_str = text
    if m:
        head = m.group("type").rstrip()
        cand = m.group("name")
        # a bare type ("Standard_Real") must not be mistaken for a name
        if head and not head.endswith(("::", ",")):
            type_str = head
            name = cand
    type_str = re.sub(r"\s+", " ", type_str).strip()
    # array suffix stays with the type
    if name.endswith("]"):
        name = name[: name.index("[")]
    return CppParam(type=type_str, name=name, default=default)


def split_params(text: str) -> Tuple[CppParam, ...]:
    """Parse a whole parameter list body (text between the parentheses)."""
    inner = text.strip()
    if not inner or inner == "void":
        return ()
    return tuple(parse_param(p) for p in _split_commas(inner) if p.strip())


def _find_call_parens(decl: str) -> Optional[Tuple[int, int]]:
    """Index of the '(' opening the parameter list and its matching ')'."""
    depth = 0
    angle = 0
    open_at = -1
    for i, ch in enumerate(decl):
        if ch == "<":
            angle += 1
        elif ch == ">" and angle > 0:
            angle -= 1
        elif ch == "(":
            if depth == 0 and open_at < 0:
                open_at = i
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and open_at >= 0:
                return (open_at, i)
    return None


def _parse_function(decl: str, access: str, class_name: str = "") -> Optional[CppMethod]:
    """Parse a (member) function declaration; return None if it is not one."""
    decl = decl.strip()
    if not decl or decl.startswith(("typedef", "using", "namespace", "template")):
        return None
    span = _find_call_parens(decl)
    if span is None:
        return None
    open_at, close_at = span
    head = decl[:open_at].strip()
    params_text = decl[open_at + 1 : close_at]
    tail = decl[close_at + 1 :].strip()
    if not head:
        return None
    # assignments / returns of a call expression are not declarations
    # (``operator=`` legitimately ends in '=' and must survive this guard)
    if head.endswith("return") or (head.endswith("=") and "operator" not in head):
        return None

    is_exported = "Standard_EXPORT" in head
    tokens = head.replace("*", " * ").replace("&", " & ").split()
    flags = {q: False for q in _QUALIFIERS}
    while tokens and tokens[0] in _QUALIFIERS:
        flags[tokens[0]] = True
        tokens.pop(0)
    if not tokens:
        return None
    head = " ".join(tokens)

    is_operator = False
    idx = head.find("operator")
    if idx >= 0 and (idx == 0 or not head[idx - 1].isalnum()):
        name = head[idx:].strip()
        return_type = head[:idx].strip()
        is_operator = True
    else:
        m = re.search(r"(~?\s*[A-Za-z_][A-Za-z0-9_]*)\s*$", head)
        if not m:
            return None
        name = m.group(1).replace(" ", "")
        return_type = head[: m.start(1)].strip()
    return_type = re.sub(r"\s+", " ", return_type).replace(" *", "*").replace(" &", "&")
    return_type = return_type.strip()
    if "::" in name:
        name = name.rsplit("::", 1)[1]

    is_destructor = name.startswith("~")
    is_constructor = bool(class_name) and not is_destructor and name == class_name and not return_type
    if not is_constructor and not is_destructor and not return_type and not is_operator:
        # e.g. a macro invocation such as ``DEFINE_STANDARD_RTTIEXT(A, B)``
        return None
    is_pure = bool(re.search(r"=\s*0\s*$", tail))
    if re.search(r"=\s*(delete|default)\s*$", tail):
        pass  # still a declaration; keep it
    return CppMethod(
        name=name,
        return_type=return_type,
        params=split_params(params_text),
        access=access,
        is_static=flags["static"],
        is_virtual=flags["virtual"] or is_pure,
        is_const=bool(re.match(r"^const\b", tail)),
        is_pure=is_pure,
        is_explicit=flags["explicit"],
        is_exported=is_exported,
        is_constructor=is_constructor,
        is_destructor=is_destructor,
        is_operator=is_operator,
    )


def _parse_enum_values(body: str) -> Tuple[str, ...]:
    values: List[str] = []
    for part in _split_commas(body):
        part = part.strip()
        if not part:
            continue
        name = part.split("=", 1)[0].strip()
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
            values.append(name)
    return tuple(values)


def _parse_bases(text: str) -> Tuple[str, ...]:
    bases: List[str] = []
    for part in _split_commas(text):
        toks = [t for t in part.replace("virtual", " ").split() if t]
        toks = [t for t in toks if t not in ("public", "protected", "private")]
        if toks:
            bases.append(toks[-1])
    return tuple(bases)


def _parse_class_body(body: str, cls: CppClass) -> None:
    access = "public" if cls.kind == "struct" else "private"
    for head, block in split_top_level(body):
        # access labels may precede the declaration inside the same chunk
        last = None
        for m in _ACCESS_RE.finditer(head):
            last = m
        if last is not None:
            access = last.group(1)
            head = head[last.end() :]
        decl = _clean_decl(head)
        if not decl:
            continue
        if block is not None:
            em = _ENUM_RE.match(decl)
            if em:
                cls.enums.append(
                    CppEnum(
                        name=em.group("name") or "",
                        values=_parse_enum_values(block),
                        is_scoped=bool(em.group("scoped")),
                        access=access,
                    )
                )
                continue
            cm = _CLASS_RE.match(decl)
            if cm:
                sub = CppClass(
                    name=cm.group("name"),
                    kind=cm.group(1),
                    bases=_parse_bases(cm.group("bases") or ""),
                )
                _parse_class_body(block, sub)
                cls.nested.append(sub)
                continue
            # inline member body: the head still carries the declaration,
            # minus any constructor initialiser list
            decl = _strip_init_list(decl)
        td = _parse_typedef(decl)
        if td is not None:
            cls.typedefs.append(td)
            continue
        fn = _parse_function(decl, access, cls.name)
        if fn is not None:
            cls.methods.append(fn)


def _strip_init_list(decl: str) -> str:
    """Remove a constructor initialiser list (``: a(x), b(y)``) from a decl."""
    span = _find_call_parens(decl)
    if span is None:
        return decl
    close_at = span[1]
    tail = decl[close_at + 1 :]
    idx = tail.find(":")
    if idx >= 0 and "::" not in tail[idx : idx + 2]:
        return decl[: close_at + 1] + tail[:idx]
    return decl


def _parse_typedef(decl: str) -> Optional[CppTypedef]:
    if decl.startswith("typedef "):
        rest = decl[len("typedef ") :].strip()
        m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", rest)
        if not m:
            return None
        name = m.group(1)
        target = re.sub(r"\s+", " ", rest[: m.start(1)].strip())
        return CppTypedef(name=name, target=target)
    if decl.startswith("using ") and "=" in decl:
        lhs, rhs = decl[len("using ") :].split("=", 1)
        return CppTypedef(name=lhs.strip(), target=re.sub(r"\s+", " ", rhs.strip()))
    return None


def parse_header(text: str, path: str = "") -> Header:
    """Parse C++ header source text into a :class:`Header` inventory."""
    src = strip_preprocessor(strip_comments(text))
    header = Header(path=path)
    _parse_namespace(src, header)
    return header


def _parse_namespace(src: str, header: Header) -> None:
    for head, block in split_top_level(src):
        decl = _clean_decl(head)
        if not decl:
            continue
        if block is not None:
            if decl.startswith("namespace"):
                _parse_namespace(block, header)
                continue
            if decl.startswith("extern"):
                _parse_namespace(block, header)
                continue
            em = _ENUM_RE.match(decl)
            if em:
                header.enums.append(
                    CppEnum(
                        name=em.group("name") or "",
                        values=_parse_enum_values(block),
                        is_scoped=bool(em.group("scoped")),
                    )
                )
                continue
            cm = _CLASS_RE.match(decl)
            if cm:
                cls = CppClass(
                    name=cm.group("name"),
                    kind=cm.group(1),
                    bases=_parse_bases(cm.group("bases") or ""),
                )
                _parse_class_body(block, cls)
                header.classes.append(cls)
                continue
            decl = _strip_init_list(decl)
        else:
            fm = re.match(r"^(class|struct)\s+([A-Za-z_][A-Za-z0-9_]*)$", decl)
            if fm:
                if fm.group(2) not in header.forward_decls:
                    header.forward_decls.append(fm.group(2))
                continue
        td = _parse_typedef(decl)
        if td is not None:
            header.typedefs.append(td)
            continue
        fn = _parse_function(decl, "public", "")
        if fn is not None:
            header.functions.append(fn)


def parse_header_file(path: str) -> Header:
    """Read and parse a header file from disk (latin-1 tolerant)."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return parse_header(fh.read(), path=path)


def iter_header_files(
    root: str, suffixes: Sequence[str] = (".hxx", ".h", ".hpp")
) -> Iterable[str]:
    """Yield header paths under *root* in deterministic (sorted) order."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if name.endswith(tuple(suffixes)):
                yield os.path.join(dirpath, name)
