"""The CAD-reference token grammar: hierarchical occurrence and entity selectors.

Ported from ``packages/cadpy/src/cadpy/cad_ref_syntax.py`` in the ``text-to-cad``
(CAD Skills) repository.  Its agent workflow annotates prose and generated code
with ``#``-prefixed tokens that point at a specific piece of a STEP assembly, so
that a later inspection pass can resolve "the face I meant" without ambiguity.

The grammar is small but genuinely load-bearing, and it is *not* the CadQuery-style
predicate selector algebra the harness already carries (``geometry/cq_selector_*``,
``geometry/cascade_entity_selector``).  Those select entities by geometric
predicate (">Z", "|X", tags).  This one is a *positional index into an assembly
occurrence tree*:

* ``o<path>``            -- an occurrence, whose path is a dotted chain of integer
                            instance indices (``o1``, ``o1.2``, ``o3.1.4``);
* ``o<path>.<k><n>``     -- entity ``n`` of kind ``k`` inside that occurrence;
* ``<k><n>``             -- entity ``n`` of kind ``k``, inheriting the occurrence
                            of the previous selector in the same comma list;
* kinds are ``s`` shape, ``f`` face, ``e`` edge, ``v`` vertex.

The subtle rule -- and the reason a shared implementation matters -- is the
*left-to-right occurrence inheritance* inside one token: in ``#o1.2.f3,f4,o5.e1``
the bare ``f4`` belongs to occurrence ``o1.2`` (inherited), while ``e1`` after
``o5`` would belong to ``o5``.  Canonicalisation re-attaches the inherited
occurrence, so every selector round-trips to an absolute form.

Also included is STEP-path normalisation (backslashes folded to ``/``, the
``.step``/``.stp`` suffix dropped, traversal segments rejected).

Deterministic, stdlib-only, no file access.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

CAD_TOKEN_RE = re.compile(r"#(\S*)")
OCCURRENCE_RE = re.compile(r"^o(\d+(?:\.\d+)*)$")
OCCURRENCE_ENTITY_RE = re.compile(r"^o(\d+(?:\.\d+)*)\.([sfev])(\d+)$")
ENTITY_RE = re.compile(r"^([sfev])(\d+)$")

KIND_NAMES = {"s": "shape", "f": "face", "e": "edge", "v": "vertex"}

STEP_SUFFIXES = (".step", ".stp")


@dataclass(frozen=True)
class ParsedSelector:
    """One resolved selector: what kind of thing, in which occurrence, at which index."""

    selector_type: str
    occurrence_id: str
    ordinal: Optional[int]
    canonical: str


@dataclass(frozen=True)
class ParsedToken:
    """One ``#...`` token found in a line of text."""

    line: int
    token: str
    selectors: Tuple[str, ...]


def selector_type_for_kind(kind: str) -> str:
    """Map a one-letter entity kind to its long name."""
    return KIND_NAMES.get(kind, "vertex")


def occurrence_segments(occurrence_id: str) -> Tuple[str, ...]:
    """Split an occurrence id such as ``o1.2.3`` into ``("1", "2", "3")``."""
    text = str(occurrence_id or "").strip()
    match = re.match(r"^o(\d+(?:\.\d+)*)", text, re.IGNORECASE)
    if match is None:
        return ()
    return tuple(part for part in match.group(1).split(".") if part)


def parse_selector(
    raw_selector: str, *, inherited_occurrence_id: str = ""
) -> Optional[ParsedSelector]:
    """Parse one selector, optionally inheriting an occurrence from its left sibling.

    Returns ``None`` for empty input.  Anything that matches no rule is kept
    verbatim as an ``opaque`` selector rather than discarded, so unknown token
    dialects survive a round trip.
    """
    selector = str(raw_selector or "").strip().replace("#", "", 1).strip()
    if not selector:
        return None

    match = OCCURRENCE_ENTITY_RE.match(selector)
    if match is not None:
        occurrence_id = f"o{match.group(1)}"
        kind = match.group(2)
        ordinal = int(match.group(3))
        return ParsedSelector(
            selector_type=selector_type_for_kind(kind),
            occurrence_id=occurrence_id,
            ordinal=ordinal,
            canonical=f"{occurrence_id}.{kind}{ordinal}",
        )

    match = OCCURRENCE_RE.match(selector)
    if match is not None:
        occurrence_id = f"o{match.group(1)}"
        return ParsedSelector(
            selector_type="occurrence",
            occurrence_id=occurrence_id,
            ordinal=None,
            canonical=occurrence_id,
        )

    match = ENTITY_RE.match(selector)
    if match is not None:
        kind = match.group(1)
        ordinal = int(match.group(2))
        if inherited_occurrence_id:
            return ParsedSelector(
                selector_type=selector_type_for_kind(kind),
                occurrence_id=inherited_occurrence_id,
                ordinal=ordinal,
                canonical=f"{inherited_occurrence_id}.{kind}{ordinal}",
            )
        return ParsedSelector(
            selector_type=selector_type_for_kind(kind),
            occurrence_id="",
            ordinal=ordinal,
            canonical=f"{kind}{ordinal}",
        )

    return ParsedSelector(
        selector_type="opaque",
        occurrence_id="",
        ordinal=None,
        canonical=selector,
    )


def parse_selector_list(raw_selector_list: str) -> Tuple[ParsedSelector, ...]:
    """Parse a comma-separated selector list, threading occurrence inheritance."""
    parsed: List[ParsedSelector] = []
    inherited = ""
    for raw_selector in str(raw_selector_list or "").split(","):
        selector = parse_selector(raw_selector, inherited_occurrence_id=inherited)
        if selector is None:
            continue
        parsed.append(selector)
        if selector.occurrence_id:
            inherited = selector.occurrence_id
    return tuple(parsed)


def normalize_selector_list(raw_selector_list: str) -> Tuple[str, ...]:
    """Canonical, occurrence-absolute form of every selector in a list."""
    return tuple(
        selector.canonical for selector in parse_selector_list(raw_selector_list)
    )


def parse_cad_tokens(text: str) -> Tuple[ParsedToken, ...]:
    """Find every ``#...`` CAD token in ``text``, one result per token, line-numbered."""
    tokens: List[ParsedToken] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for match in CAD_TOKEN_RE.finditer(line):
            tokens.append(
                ParsedToken(
                    line=line_number,
                    token=match.group(0),
                    selectors=normalize_selector_list(match.group(1)),
                )
            )
    return tuple(tokens)


def build_cad_token(selectors) -> str:
    """Render selectors back into a single ``#a,b,c`` token."""
    if isinstance(selectors, str):
        parts = normalize_selector_list(selectors)
    else:
        parts = tuple(str(selector).strip() for selector in selectors if str(selector).strip())
    if not parts:
        return "#"
    return "#" + ",".join(parts)


def normalize_cad_path(raw_cad_path: str) -> Optional[str]:
    """Normalise a STEP asset path; ``None`` when it is empty or unsafe.

    Backslashes fold to ``/``, a trailing ``.step``/``.stp`` suffix is dropped,
    and any empty, ``.`` or ``..`` segment rejects the path outright so a token
    can never escape its project root.
    """
    normalized = str(raw_cad_path or "").replace("\\", "/").strip().strip("/")
    if not normalized:
        return None
    lowered = normalized.lower()
    for suffix in STEP_SUFFIXES:
        if lowered.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    parts = normalized.split("/")
    if any(not part or part in (".", "..") for part in parts):
        return None
    return "/".join(parts)


def occurrence_depth(occurrence_id: str) -> int:
    """Nesting depth of an occurrence id (``o1`` is 1, ``o1.2`` is 2)."""
    return len(occurrence_segments(occurrence_id))


def is_descendant_occurrence(candidate: str, ancestor: str) -> bool:
    """True when ``candidate`` sits strictly below ``ancestor`` in the occurrence tree."""
    candidate_segments = occurrence_segments(candidate)
    ancestor_segments = occurrence_segments(ancestor)
    if not candidate_segments or not ancestor_segments:
        return False
    if len(candidate_segments) <= len(ancestor_segments):
        return False
    return candidate_segments[: len(ancestor_segments)] == ancestor_segments


def common_occurrence_prefix(occurrence_ids) -> Tuple[str, ...]:
    """Longest shared prefix of a set of occurrence ids.

    Only ids nested at least two levels deep participate.  If the shared prefix
    would consume the *entire* shallowest path, its last segment is given back:
    a prefix that swallows a whole id leaves nothing to distinguish that id by,
    which would collapse the assembly into a single group.
    """
    paths = [
        segments
        for segments in (occurrence_segments(value) for value in occurrence_ids)
        if len(segments) > 1
    ]
    if not paths:
        return ()
    shortest = min(len(segments) for segments in paths)
    prefix: List[str] = []
    for index in range(shortest):
        value = paths[0][index]
        if not all(segments[index] == value for segments in paths):
            break
        prefix.append(value)
    if len(prefix) >= shortest:
        return tuple(prefix[:-1])
    return tuple(prefix)
