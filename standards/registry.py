"""The versioned standards knowledge-base — typed rules, rule packs, registry.

Concepts (mirroring what ``verifiers/standards.py`` and ``verifiers/compliance.py``
hardcode, but externalised into data):

  * :class:`Rule` — one machine-readable clause: a ``parameter`` constrained by a
    ``comparator`` against a ``limit`` (or a set of ``values``), tagged with the
    ``standard`` + ``version`` it came from, the exact ``clause`` id and a human
    ``citation`` string, and a ``scope`` of material / process / region tags that
    say when it applies. This is the same shape the hardcoded engines encode
    (parameter / comparator / limit) — here it is a record you can swap.
  * :class:`RulePack` — a named, versioned bundle of rules with a ``source``;
    loads / saves as JSON natively and reads a tiny YAML subset (no pyyaml).
  * :class:`StandardsRegistry` — indexes packs by ``(standard, version)``,
    resolves the *active* rules for a material/process/region, lists a standard's
    versions, and diffs two versions (added / removed / changed) so a caller can
    flag when a regulation changed.

Stdlib only; deterministic (all listings/diffs are sorted).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Recognised comparators. ``in`` uses ``values``; the rest use ``limit``.
COMPARATORS = ("<=", ">=", "==", "in", "near")


# --------------------------------------------------------------------------- #
# Rule
# --------------------------------------------------------------------------- #
@dataclass
class Rule:
    """One machine-readable standards clause.

    ``comparator`` is one of :data:`COMPARATORS`. For ``in`` the allowed set is
    ``values``; for ``<=``, ``>=``, ``==`` and ``near`` the bound is ``limit``.
    ``scope`` carries optional ``material`` / ``process`` / ``region`` tags — each
    may be a single string or a list of strings; an absent/empty tag means the
    rule applies regardless of that dimension.
    """

    id: str
    standard: str
    version: str
    parameter: str
    comparator: str
    limit: Optional[float] = None
    values: Optional[List[Any]] = None
    clause: str = ""
    citation: str = ""
    scope: Dict[str, Any] = field(default_factory=dict)
    # Optional unit / tolerance metadata, carried through round-trips.
    unit: Optional[str] = None

    def __post_init__(self) -> None:
        if self.comparator not in COMPARATORS:
            raise ValueError(
                f"unknown comparator {self.comparator!r}; "
                f"expected one of {COMPARATORS}")

    # -- scope resolution ---------------------------------------------------
    def applies_to(self, material: Optional[str] = None,
                   process: Optional[str] = None,
                   region: Optional[str] = None) -> bool:
        """True when this rule is in scope for the given tags.

        A query tag of ``None`` does not filter on that dimension. When a query
        tag *is* given, the rule matches if it carries no constraint for that
        dimension (applies to all) or the query value is among its tags.
        """
        return (
            _scope_match(self.scope.get("material"), material)
            and _scope_match(self.scope.get("process"), process)
            and _scope_match(self.scope.get("region"), region)
        )

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "standard": self.standard,
            "version": self.version,
            "parameter": self.parameter,
            "comparator": self.comparator,
            "clause": self.clause,
            "citation": self.citation,
            "scope": dict(self.scope),
        }
        if self.limit is not None:
            d["limit"] = self.limit
        if self.values is not None:
            d["values"] = list(self.values)
        if self.unit is not None:
            d["unit"] = self.unit
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Rule":
        limit = d.get("limit")
        return cls(
            id=str(d["id"]),
            standard=str(d.get("standard", "")),
            version=str(d.get("version", "")),
            parameter=str(d["parameter"]),
            comparator=str(d["comparator"]),
            limit=None if limit is None else float(limit),
            values=list(d["values"]) if d.get("values") is not None else None,
            clause=str(d.get("clause", "")),
            citation=str(d.get("citation", "")),
            scope=dict(d.get("scope") or {}),
            unit=d.get("unit"),
        )

    # -- identity for diffing ----------------------------------------------
    def signature(self) -> Tuple:
        """A hashable value of the *content* (everything but the id) — two rules
        with the same id but different signatures count as 'changed'."""
        return (
            self.standard, self.parameter, self.comparator,
            self.limit,
            tuple(self.values) if self.values is not None else None,
            self.clause, self.citation,
            _scope_key(self.scope), self.unit,
        )


def _scope_match(tag: Any, query: Optional[str]) -> bool:
    """Whether a rule's single scope ``tag`` admits a ``query`` value."""
    if query is None:
        return True
    if tag is None or tag == "" or tag == []:
        return True  # unconstrained on this dimension -> applies to all
    if isinstance(tag, (list, tuple, set)):
        return query in tag
    return tag == query


def _scope_key(scope: Dict[str, Any]) -> Tuple:
    """Deterministic hashable form of a scope dict for diffing."""
    out = []
    for k in sorted(scope):
        v = scope[k]
        if isinstance(v, (list, tuple, set)):
            v = tuple(sorted(str(x) for x in v))
        out.append((k, v))
    return tuple(out)


# --------------------------------------------------------------------------- #
# RulePack
# --------------------------------------------------------------------------- #
@dataclass
class RulePack:
    """A named, versioned bundle of :class:`Rule` records with a ``source``."""

    name: str
    version: str
    rules: List[Rule] = field(default_factory=list)
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "source": self.source,
            "rules": [r.to_dict() for r in self.rules],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RulePack":
        return cls(
            name=str(d.get("name", "")),
            version=str(d.get("version", "")),
            source=str(d.get("source", "")),
            rules=[Rule.from_dict(r) for r in (d.get("rules") or [])],
        )

    # -- JSON ---------------------------------------------------------------
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "RulePack":
        return cls.from_dict(json.loads(text))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "RulePack":
        """Load a rule pack from JSON (``.json``) or the tiny YAML subset."""
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        low = path.lower()
        if low.endswith((".yaml", ".yml")):
            return cls.from_dict(parse_simple_yaml(text))
        # Default to JSON, but fall back to the YAML subset if JSON fails.
        try:
            return cls.from_dict(json.loads(text))
        except json.JSONDecodeError:
            return cls.from_dict(parse_simple_yaml(text))


# --------------------------------------------------------------------------- #
# Version diff
# --------------------------------------------------------------------------- #
@dataclass
class VersionDiff:
    """Result of :meth:`StandardsRegistry.changed_between`.

    ``added`` / ``removed`` are the rules only in the new / old version; each
    entry of ``changed`` is a ``(before, after)`` pair sharing a rule id but
    differing in content.
    """

    standard: str
    v1: str
    v2: str
    added: List[Rule] = field(default_factory=list)
    removed: List[Rule] = field(default_factory=list)
    changed: List[Tuple[Rule, Rule]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.changed)

    def summary(self) -> str:
        return (f"{self.standard}: {self.v1} -> {self.v2}: "
                f"+{len(self.added)} / -{len(self.removed)} / "
                f"~{len(self.changed)}")


# --------------------------------------------------------------------------- #
# StandardsRegistry
# --------------------------------------------------------------------------- #
class StandardsRegistry:
    """A living codebook: rule packs keyed by ``(standard, version)``.

    Register packs, then resolve the *active* rules for a given
    material/process/region, list a standard's versions, or diff two versions to
    flag when a regulation changed.
    """

    def __init__(self) -> None:
        # (standard, version) -> RulePack
        self._packs: Dict[Tuple[str, str], RulePack] = {}

    # -- registration -------------------------------------------------------
    def register(self, pack: RulePack) -> "StandardsRegistry":
        """Register a pack under every ``(standard, version)`` its rules carry.

        A pack usually holds one standard/version, but a heterogeneous pack is
        indexed under each distinct pair so lookups still resolve. Re-registering
        the same key merges rules (later ids win on duplicate id).
        """
        keys = {(r.standard, r.version) for r in pack.rules}
        if not keys:
            # An empty pack still gets a home under its own name/version.
            keys = {(pack.name, pack.version)}
        for key in keys:
            existing = self._packs.get(key)
            subset = [r for r in pack.rules
                      if (r.standard, r.version) == key] or list(pack.rules)
            if existing is None:
                self._packs[key] = RulePack(
                    name=pack.name, version=key[1],
                    rules=list(subset), source=pack.source)
            else:
                by_id = {r.id: r for r in existing.rules}
                for r in subset:
                    by_id[r.id] = r
                existing.rules = [by_id[i] for i in sorted(by_id)]
        return self

    def get(self, standard: str, version: str) -> Optional[RulePack]:
        return self._packs.get((standard, version))

    def standards(self) -> List[str]:
        return sorted({std for (std, _v) in self._packs})

    def rule_versions(self, standard: str) -> List[str]:
        """All registered versions of ``standard``, sorted."""
        return sorted(v for (std, v) in self._packs if std == standard)

    def latest_version(self, standard: str) -> Optional[str]:
        versions = self.rule_versions(standard)
        return versions[-1] if versions else None

    # -- active rule resolution --------------------------------------------
    def active_rules(self, material: Optional[str] = None,
                     process: Optional[str] = None,
                     region: Optional[str] = None,
                     standard: Optional[str] = None,
                     version: Optional[str] = None) -> List[Rule]:
        """Return the applicable rule set for the given scope.

        By default every registered pack contributes; pass ``standard`` (and
        optionally ``version``) to restrict to one standard. When ``version`` is
        omitted for a standard, only its latest version is used, so you never mix
        two versions of the same standard. Filters each candidate through
        :meth:`Rule.applies_to`. Deterministically ordered.
        """
        # Decide which (standard, version) pairs are in play.
        wanted_standards = [standard] if standard else self.standards()
        keys: List[Tuple[str, str]] = []
        for std in wanted_standards:
            if version is not None:
                if (std, version) in self._packs:
                    keys.append((std, version))
            else:
                latest = self.latest_version(std)
                if latest is not None:
                    keys.append((std, latest))

        out: List[Rule] = []
        for key in keys:
            pack = self._packs[key]
            for r in pack.rules:
                if r.applies_to(material=material, process=process,
                                region=region):
                    out.append(r)
        out.sort(key=lambda r: (r.standard, r.version, r.parameter, r.id))
        return out

    # -- version diffing ----------------------------------------------------
    def changed_between(self, standard: str, v1: str, v2: str) -> VersionDiff:
        """Diff two versions of ``standard`` (added / removed / changed).

        Rules are matched by id: an id only in ``v2`` is *added*, only in ``v1``
        is *removed*, and in both but with a different content signature is
        *changed*. Lets a caller flag exactly what a regulation update altered.
        """
        pack1 = self._packs.get((standard, v1))
        pack2 = self._packs.get((standard, v2))
        if pack1 is None:
            raise KeyError(f"no registered {standard} version {v1!r}")
        if pack2 is None:
            raise KeyError(f"no registered {standard} version {v2!r}")

        old = {r.id: r for r in pack1.rules}
        new = {r.id: r for r in pack2.rules}

        added = [new[i] for i in sorted(new) if i not in old]
        removed = [old[i] for i in sorted(old) if i not in new]
        changed = [
            (old[i], new[i])
            for i in sorted(set(old) & set(new))
            if old[i].signature() != new[i].signature()
        ]
        return VersionDiff(standard=standard, v1=v1, v2=v2,
                           added=added, removed=removed, changed=changed)

    def all_rules(self) -> List[Rule]:
        out: List[Rule] = []
        for key in sorted(self._packs):
            out.extend(self._packs[key].rules)
        return out


# --------------------------------------------------------------------------- #
# Tiny YAML-subset reader (no pyyaml dependency)
# --------------------------------------------------------------------------- #
def parse_simple_yaml(text: str) -> Any:
    """Parse a small, well-behaved YAML subset into Python data.

    Supports exactly what a rule pack needs, deterministically:

      * 2-space-indented block mappings (``key: value``)
      * block sequences (``- item`` and ``- key: value`` mapping items)
      * scalars: int, float, ``true``/``false``, ``null``/``~``, quoted and bare
        strings, and inline flow lists ``[a, b, c]``

    It is intentionally strict/minimal — not a general YAML engine — but round
    trips the JSON-equivalent structure this package emits. Raises ``ValueError``
    on shapes it does not understand rather than guessing.
    """
    lines = _yaml_significant_lines(text)
    value, idx = _yaml_parse_block(lines, 0, 0)
    if idx != len(lines):
        raise ValueError(f"trailing YAML content at line {lines[idx][2]}")
    return value


def _yaml_significant_lines(text: str) -> List[Tuple[int, str, int]]:
    """Return (indent, content, lineno) for non-blank, non-comment lines."""
    out: List[Tuple[int, str, int]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        # Strip trailing comments only when not inside quotes (rule packs keep
        # citations simple, so a naive strip on unquoted content is safe here).
        stripped = raw.rstrip()
        if not stripped.strip():
            continue
        if stripped.lstrip().startswith("#"):
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        content = stripped.strip()
        out.append((indent, content, lineno))
    return out


def _yaml_parse_block(lines, i: int, indent: int):
    """Parse a mapping or sequence block at the given indent; return (value, i)."""
    if i >= len(lines):
        return None, i
    cur_indent, content, _lineno = lines[i]
    if cur_indent < indent:
        return None, i
    if content.startswith("- "):
        return _yaml_parse_sequence(lines, i, cur_indent)
    return _yaml_parse_mapping(lines, i, cur_indent)


def _yaml_parse_mapping(lines, i: int, indent: int):
    result: Dict[str, Any] = {}
    while i < len(lines):
        cur_indent, content, lineno = lines[i]
        if cur_indent < indent:
            break
        if cur_indent > indent:
            raise ValueError(f"unexpected indent at line {lineno}")
        if content.startswith("- "):
            break
        if ":" not in content:
            raise ValueError(f"expected 'key: value' at line {lineno}")
        key, _sep, rest = content.partition(":")
        key = _yaml_scalar(key.strip())
        rest = rest.strip()
        if rest == "":
            child, i = _yaml_parse_block(lines, i + 1, indent + 1)
            result[key] = {} if child is None else child
        else:
            result[key] = _yaml_scalar(rest)
            i += 1
    return result, i


def _yaml_parse_sequence(lines, i: int, indent: int):
    result: List[Any] = []
    while i < len(lines):
        cur_indent, content, lineno = lines[i]
        if cur_indent < indent:
            break
        if cur_indent > indent:
            raise ValueError(f"unexpected indent at line {lineno}")
        if not content.startswith("- "):
            break
        item = content[2:].strip()
        if ":" in item and not _looks_like_scalar(item):
            # Inline first key of a mapping item; re-inject as a mapping line.
            synth = [(indent + 2, item, lineno)]
            j = i + 1
            while j < len(lines) and lines[j][0] > indent:
                synth.append(lines[j])
                j += 1
            value, _ = _yaml_parse_mapping(synth, 0, indent + 2)
            result.append(value)
            i = j
        else:
            result.append(_yaml_scalar(item))
            i += 1
    return result, i


def _looks_like_scalar(item: str) -> bool:
    """A ``- item`` where the ':' is part of a quoted/flow scalar, not a key."""
    return item.startswith(("[", "\"", "'"))


def _yaml_scalar(token: str) -> Any:
    """Convert a YAML scalar token to a Python value."""
    if token == "":
        return ""
    if token[0] in "\"'" and token[-1] == token[0] and len(token) >= 2:
        return token[1:-1]
    if token.startswith("[") and token.endswith("]"):
        inner = token[1:-1].strip()
        if not inner:
            return []
        return [_yaml_scalar(p.strip()) for p in _split_flow(inner)]
    low = token.lower()
    if low in ("null", "~"):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    return token


def _split_flow(inner: str) -> List[str]:
    """Split an inline flow-list body on commas outside quotes."""
    parts: List[str] = []
    buf = ""
    quote = None
    for ch in inner:
        if quote:
            buf += ch
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
            buf += ch
        elif ch == ",":
            parts.append(buf)
            buf = ""
        else:
            buf += ch
    if buf.strip():
        parts.append(buf)
    return parts
