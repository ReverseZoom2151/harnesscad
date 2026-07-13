"""OCCT kernel API catalog: a machine-readable inventory of the CAD kernel API.

OCP generates the Python bindings for Open CASCADE by enumerating the OCCT
headers module by module (its ``ocp.toml`` lists the modules to wrap and the
namespaces to exclude). The generated bindings themselves are out of scope for a
stdlib harness, but the *inventory* they encode is not: an LLM that writes
CadQuery / OCC / pythonocc code hallucinates class and method names, and the
cheapest deterministic guard is to check every generated symbol against a
catalog of what the kernel actually exports.

This module builds and queries that catalog:

  * :func:`module_of` -- OCCT symbols are ``Module_Class`` (``gp_Pnt`` -> ``gp``,
    ``BRepPrimAPI_MakeBox`` -> ``BRepPrimAPI``), so the module is recoverable
    from the name alone.
  * :func:`parse_ocp_config` -- reads the module allow-list / namespace
    exclusions out of an ``ocp.toml``-style config (minimal reader, no
    third-party TOML dependency).
  * :class:`OcctApiCatalog` -- classes, methods, enums per module; built either
    from parsed headers (:func:`build_catalog_from_headers`) or from a JSON-able
    dict, and queried for existence / arity / spelling.
  * :meth:`OcctApiCatalog.check_call` -- validate ``Class.Method(argc)`` and
    return a structured verdict with deterministic near-miss suggestions, so a
    code generator can repair a call before it ever reaches the kernel.

Everything is stdlib-only and deterministic: same headers in, same catalog out.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from harnesscad.io.formats.ocp_cpp_header_parser import (
    CppClass,
    Header,
    iter_header_files,
    parse_header_file,
)

__all__ = [
    "CallCheck",
    "MethodEntry",
    "ClassEntry",
    "OcctApiCatalog",
    "OcpConfig",
    "build_catalog_from_headers",
    "module_of",
    "parse_ocp_config",
]


def module_of(symbol: str) -> str:
    """Return the OCCT module of a symbol (``gp_Pnt`` -> ``gp``).

    Symbols without an underscore (``TopoDS``, ``Precision``) are their own
    module. Leading namespace qualifiers are ignored.
    """
    name = symbol.strip()
    if "::" in name:
        name = name.rsplit("::", 1)[1]
    if "_" not in name:
        return name
    return name.split("_", 1)[0]


# --------------------------------------------------------------------------
# ocp.toml (minimal reader)
# --------------------------------------------------------------------------


@dataclass
class OcpConfig:
    """The parts of an ``ocp.toml`` that describe the API surface to export."""

    name: str = ""
    input_folder: str = ""
    output_folder: str = ""
    modules: Tuple[str, ...] = ()
    exclude_namespaces: Tuple[str, ...] = ()
    exclude_classes: Tuple[str, ...] = ()

    def includes_module(self, module: str) -> bool:
        return not self.modules or module in self.modules


_KEY_RE = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$', re.S)


def _strip_toml_comments(text: str) -> str:
    out: List[str] = []
    for line in text.split("\n"):
        in_str = False
        quote = ""
        buf: List[str] = []
        for ch in line:
            if in_str:
                buf.append(ch)
                if ch == quote:
                    in_str = False
                continue
            if ch in "\"'":
                in_str = True
                quote = ch
                buf.append(ch)
                continue
            if ch == "#":
                break
            buf.append(ch)
        out.append("".join(buf))
    return "\n".join(out)


def _toml_strings(text: str) -> Tuple[str, ...]:
    return tuple(a or b for a, b in re.findall(r'"([^"]*)"|\'([^\']*)\'', text))


def parse_ocp_config(text: str) -> OcpConfig:
    """Parse the top-level keys of an ``ocp.toml``-style binding config.

    Only what the catalog needs: scalar strings and flat string arrays at the
    document root. Table sections (``[table]``) are ignored, as are comments.
    """
    src = _strip_toml_comments(text)
    root: Dict[str, str] = {}
    i = 0
    lines = src.split("\n")
    in_table = False
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("["):
            in_table = True
            i += 1
            continue
        m = _KEY_RE.match(line)
        if not m or in_table:
            i += 1
            continue
        key, value = m.group(1), m.group(2).strip()
        if value.startswith("["):
            chunk = [value]
            depth = value.count("[") - value.count("]")
            while depth > 0 and i + 1 < len(lines):
                i += 1
                chunk.append(lines[i])
                depth += lines[i].count("[") - lines[i].count("]")
            value = "\n".join(chunk)
        root[key] = value
        i += 1

    def scalar(key: str) -> str:
        raw = root.get(key, "")
        vals = _toml_strings(raw)
        return vals[0] if vals else ""

    def array(key: str) -> Tuple[str, ...]:
        raw = root.get(key, "")
        if not raw.strip().startswith("["):
            return ()
        return tuple(_toml_strings(raw))

    return OcpConfig(
        name=scalar("name"),
        input_folder=scalar("input_folder"),
        output_folder=scalar("output_folder"),
        modules=array("modules"),
        exclude_namespaces=array("exclude_namespaces"),
        exclude_classes=array("exclude_classes"),
    )


# --------------------------------------------------------------------------
# catalog
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MethodEntry:
    """One callable method name with the arity range across all its overloads."""

    name: str
    min_args: int
    max_args: int
    is_static: bool = False
    overloads: int = 1
    signatures: Tuple[str, ...] = ()

    def accepts(self, argc: int) -> bool:
        return self.min_args <= argc <= self.max_args


@dataclass
class ClassEntry:
    """One OCCT class as exposed to a code generator."""

    name: str
    module: str
    bases: Tuple[str, ...] = ()
    methods: Dict[str, MethodEntry] = field(default_factory=dict)
    enums: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    header: str = ""

    def method_names(self) -> Tuple[str, ...]:
        return tuple(sorted(self.methods))

    def constructor(self) -> Optional[MethodEntry]:
        return self.methods.get(self.name)


@dataclass(frozen=True)
class CallCheck:
    """Verdict for a single ``Class.Method(argc)`` call."""

    ok: bool
    reason: str = ""
    kind: str = ""  # unknown_class | unknown_method | bad_arity | ok
    suggestions: Tuple[str, ...] = ()
    expected: Tuple[int, int] = (0, 0)

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.ok


def _method_entries(cls: CppClass) -> Dict[str, MethodEntry]:
    entries: Dict[str, MethodEntry] = {}
    for m in cls.methods:
        if m.access != "public" or m.is_destructor:
            continue
        prev = entries.get(m.name)
        if prev is None:
            entries[m.name] = MethodEntry(
                name=m.name,
                min_args=m.min_args,
                max_args=m.max_args,
                is_static=m.is_static,
                overloads=1,
                signatures=(m.signature(),),
            )
        else:
            entries[m.name] = MethodEntry(
                name=m.name,
                min_args=min(prev.min_args, m.min_args),
                max_args=max(prev.max_args, m.max_args),
                is_static=prev.is_static or m.is_static,
                overloads=prev.overloads + 1,
                signatures=prev.signatures + (m.signature(),),
            )
    return entries


class OcctApiCatalog:
    """Queryable inventory of kernel classes, methods and enums."""

    def __init__(self, classes: Optional[Dict[str, ClassEntry]] = None) -> None:
        self.classes: Dict[str, ClassEntry] = dict(classes or {})

    # -- construction ------------------------------------------------------

    def add_class(self, entry: ClassEntry) -> None:
        existing = self.classes.get(entry.name)
        if existing is None:
            self.classes[entry.name] = entry
            return
        merged = dict(existing.methods)
        merged.update(entry.methods)
        existing.methods = merged
        existing.enums.update(entry.enums)
        if entry.bases and not existing.bases:
            existing.bases = entry.bases

    def add_header(self, header: Header) -> None:
        """Ingest one parsed header (all of its top-level classes)."""
        for cls in header.classes:
            entry = ClassEntry(
                name=cls.name,
                module=module_of(cls.name),
                bases=cls.bases,
                methods=_method_entries(cls),
                enums={
                    e.name: e.values
                    for e in cls.enums
                    if e.name and e.access == "public"
                },
                header=header.path,
            )
            self.add_class(entry)
        for e in header.enums:
            if not e.name:
                continue
            mod = module_of(e.name)
            holder = self.classes.get(e.name)
            if holder is None:
                self.add_class(
                    ClassEntry(
                        name=e.name,
                        module=mod,
                        enums={e.name: e.values},
                        header=header.path,
                    )
                )
            else:
                holder.enums[e.name] = e.values

    # -- queries -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.classes)

    def __contains__(self, name: str) -> bool:
        return name in self.classes

    def modules(self) -> Tuple[str, ...]:
        return tuple(sorted({c.module for c in self.classes.values()}))

    def classes_in(self, module: str) -> Tuple[str, ...]:
        return tuple(
            sorted(n for n, c in self.classes.items() if c.module == module)
        )

    def get(self, class_name: str) -> Optional[ClassEntry]:
        return self.classes.get(class_name)

    def has_method(self, class_name: str, method: str) -> bool:
        entry = self.classes.get(class_name)
        return bool(entry and method in entry.methods)

    def resolve_method(self, class_name: str, method: str) -> Optional[MethodEntry]:
        """Look the method up on the class, then on its bases (breadth-first)."""
        seen = set()
        queue: List[str] = [class_name]
        while queue:
            name = queue.pop(0)
            if name in seen:
                continue
            seen.add(name)
            entry = self.classes.get(name)
            if entry is None:
                continue
            found = entry.methods.get(method)
            if found is not None:
                return found
            queue.extend(entry.bases)
        return None

    def suggest_class(self, name: str, limit: int = 3) -> Tuple[str, ...]:
        return tuple(
            difflib.get_close_matches(name, sorted(self.classes), n=limit, cutoff=0.6)
        )

    def suggest_method(
        self, class_name: str, method: str, limit: int = 3
    ) -> Tuple[str, ...]:
        entry = self.classes.get(class_name)
        if entry is None:
            return ()
        names = set(entry.methods)
        for base in entry.bases:
            base_entry = self.classes.get(base)
            if base_entry:
                names.update(base_entry.methods)
        return tuple(
            difflib.get_close_matches(method, sorted(names), n=limit, cutoff=0.6)
        )

    def check_call(self, class_name: str, method: str, argc: int) -> CallCheck:
        """Validate one kernel call: class exists, method exists, arity fits."""
        if class_name not in self.classes:
            return CallCheck(
                ok=False,
                kind="unknown_class",
                reason="unknown class %r" % class_name,
                suggestions=self.suggest_class(class_name),
            )
        entry = self.resolve_method(class_name, method)
        if entry is None:
            return CallCheck(
                ok=False,
                kind="unknown_method",
                reason="class %r has no method %r" % (class_name, method),
                suggestions=self.suggest_method(class_name, method),
            )
        if not entry.accepts(argc):
            return CallCheck(
                ok=False,
                kind="bad_arity",
                reason="%s.%s takes %d..%d args, got %d"
                % (class_name, method, entry.min_args, entry.max_args, argc),
                expected=(entry.min_args, entry.max_args),
            )
        return CallCheck(ok=True, kind="ok", expected=(entry.min_args, entry.max_args))

    def check_construction(self, class_name: str, argc: int) -> CallCheck:
        """Validate ``Class(argc)`` -- i.e. a constructor call."""
        return self.check_call(class_name, class_name, argc)

    def filter_modules(self, modules: Sequence[str]) -> "OcctApiCatalog":
        """A new catalog restricted to the given modules (e.g. an OCP allow-list)."""
        allowed = set(modules)
        return OcctApiCatalog(
            {n: c for n, c in self.classes.items() if c.module in allowed}
        )

    def summary(self) -> Dict[str, Dict[str, int]]:
        """Per-module ``{classes, methods, enums}`` counts (deterministic)."""
        out: Dict[str, Dict[str, int]] = {}
        for cls in self.classes.values():
            bucket = out.setdefault(
                cls.module, {"classes": 0, "methods": 0, "enums": 0}
            )
            bucket["classes"] += 1
            bucket["methods"] += len(cls.methods)
            bucket["enums"] += len(cls.enums)
        return {k: out[k] for k in sorted(out)}

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        return {
            "classes": [
                {
                    "name": c.name,
                    "module": c.module,
                    "bases": list(c.bases),
                    "header": c.header,
                    "methods": [
                        {
                            "name": m.name,
                            "min_args": m.min_args,
                            "max_args": m.max_args,
                            "is_static": m.is_static,
                            "overloads": m.overloads,
                            "signatures": list(m.signatures),
                        }
                        for m in sorted(c.methods.values(), key=lambda x: x.name)
                    ],
                    "enums": {k: list(v) for k, v in sorted(c.enums.items())},
                }
                for c in sorted(self.classes.values(), key=lambda x: x.name)
            ]
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "OcctApiCatalog":
        catalog = cls()
        for raw in data.get("classes", []):  # type: ignore[union-attr]
            methods = {
                m["name"]: MethodEntry(
                    name=m["name"],
                    min_args=int(m["min_args"]),
                    max_args=int(m["max_args"]),
                    is_static=bool(m.get("is_static", False)),
                    overloads=int(m.get("overloads", 1)),
                    signatures=tuple(m.get("signatures", ())),
                )
                for m in raw.get("methods", [])
            }
            catalog.add_class(
                ClassEntry(
                    name=raw["name"],
                    module=raw.get("module") or module_of(raw["name"]),
                    bases=tuple(raw.get("bases", ())),
                    methods=methods,
                    enums={k: tuple(v) for k, v in raw.get("enums", {}).items()},
                    header=raw.get("header", ""),
                )
            )
        return catalog

    @classmethod
    def from_json(cls, text: str) -> "OcctApiCatalog":
        return cls.from_dict(json.loads(text))


def build_catalog_from_headers(
    root: str,
    modules: Optional[Sequence[str]] = None,
    paths: Optional[Iterable[str]] = None,
) -> OcctApiCatalog:
    """Parse every header under *root* into a catalog.

    ``modules`` (e.g. ``parse_ocp_config(...).modules``) restricts ingestion to
    headers whose ``Module_`` prefix is on the allow-list, which is both faster
    and matches what OCP actually exports.
    """
    catalog = OcctApiCatalog()
    allowed = set(modules) if modules else None
    for path in paths if paths is not None else iter_header_files(root):
        base = path.replace("\\", "/").rsplit("/", 1)[-1]
        stem = base.rsplit(".", 1)[0]
        if allowed is not None and module_of(stem) not in allowed:
            continue
        catalog.add_header(parse_header_file(path))
    return catalog
