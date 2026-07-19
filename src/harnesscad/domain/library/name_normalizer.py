"""User-provided CAD name hygiene: default-name detection and normalisation.

A corpus of part, body and feature names harvested from real CAD documents is
mostly noise unless it is filtered first.  Every major CAD host auto-names the
things a user does not bother to name -- ``Part 1``, ``Extrude 2``,
``Boss-Extrude12``, ``Pocket001``, ``Front Plane`` -- and those names say
nothing about the geometry, so any study of what humans call their parts has to
throw them away and then deduplicate what is left.  This module is that filter,
expressed as reusable pieces:

* :func:`is_default_name` / :func:`is_user_name` -- recognise host-generated
  names, so a corpus can keep only the names a person actually typed.  The
  grammar is "a known stem, optionally followed by a counter, optionally
  followed by an instance suffix", plus the rule that a name carrying no letter
  at all carries no meaning either.
* :func:`normalize_name` / :func:`tokenize_name` / :func:`name_key` -- fold a
  raw name to a canonical token stream by casefolding and by inserting word
  boundaries at separators, camelCase humps and letter/digit transitions, so
  ``"M6_boltHead <2>"`` and ``"m6 bolt head"`` land on the same string.
* :func:`dedupe_names`, :func:`clean_document`, :func:`clean_corpus` -- apply
  the filter plus an order-preserving dedup to one document or a whole
  ``{document_id: document}`` corpus.
* :func:`document_features`, :func:`name_statistics` -- deterministic corpus
  summaries: per-document stratification flags and per-name-list counts.

Everything here is pure stdlib and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

__all__ = [
    "DEFAULT_STEMS",
    "strip_instance_suffix",
    "is_default_name",
    "is_user_name",
    "normalize_name",
    "tokenize_name",
    "name_key",
    "dedupe_names",
    "CleanDocument",
    "clean_document",
    "clean_corpus",
    "document_features",
    "name_statistics",
]

# ---------------------------------------------------------------------------
# the default-name vocabulary
#
# Grouped by the host that emits it, so a stem can be traced back to why it is
# here and a new host can be added without disturbing the others.  The stems are
# facts about CAD software's naming behaviour, observed per host.
# ---------------------------------------------------------------------------

_GENERIC_STEMS = (
    "part", "part studio", "body", "solid", "surface", "assembly", "document",
    "instance", "component", "component1", "sketch", "plane", "axis", "point",
    "curve", "spline", "line", "circle", "arc", "feature", "op", "operation",
    "shape", "compound", "unnamed", "untitled", "new part", "no name", "none",
)

_FEATURE_OP_STEMS = (
    "extrude", "revolve", "sweep", "loft", "fillet", "chamfer", "shell",
    "draft", "rib", "hole", "thread", "mirror", "pattern", "linear pattern",
    "circular pattern", "split", "boolean", "transform", "move", "copy",
    "import", "derived", "offset", "thicken", "wrap", "helix", "delete",
    "replace face",
)

_SOLIDWORKS_STEMS = (
    "boss-extrude", "cut-extrude", "boss-revolve", "cut-revolve", "boss-sweep",
    "cut-sweep", "boss-loft", "cut-loft", "lpattern", "cirpattern",
    "annotations", "material", "front plane", "top plane", "right plane",
    "origin",
)

_FREECAD_STEMS = (
    "pad", "pocket", "revolution", "groove", "additive", "subtractive", "box",
    "cylinder", "cone", "sphere", "torus", "cut", "fusion", "common",
)

#: Every stem a CAD host is known to auto-name with, deduplicated and sorted so
#: the exported vocabulary is stable regardless of how the groups above grow.
DEFAULT_STEMS: Tuple[str, ...] = tuple(sorted(set(
    _GENERIC_STEMS + _FEATURE_OP_STEMS + _SOLIDWORKS_STEMS + _FREECAD_STEMS
)))

_STEM_SET = frozenset(DEFAULT_STEMS)

# ---------------------------------------------------------------------------
# lexical patterns
# ---------------------------------------------------------------------------

# Instance markers a host appends when the same part appears more than once:
# "<3>", "(2)", ":1", "^assembly". Stripped repeatedly, since hosts stack them.
_INSTANCE_SUFFIX = re.compile(
    r"(?:\s*<\s*\d+\s*>|\s*\(\s*\d+\s*\)|\s*:\s*\d+|\s*\^\S+)\s*$"
)

# A trailing auto-increment counter, with or without a separator in front of it:
# "Extrude 12", "Extrude12", "Extrude_12", "Extrude-12".
_TRAILING_COUNTER = re.compile(r"^(?P<stem>.*?)[\s_\-]*(?P<counter>\d+)$")

# Punctuation that separates words outright.
_SEPARATOR_RUN = re.compile(r"[\s_\-/\\.,;:+&()\[\]{}<>\"'*#]+")

# Zero-width positions that also separate words: a camelCase hump, the start of
# an acronym's final capitalised word, and either direction of a letter/digit
# transition. One alternation rather than several passes -- each branch only
# fires on characters that are still adjacent, so merging them is exact.
_WORD_BOUNDARY = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z])"
    r"|(?<=[A-Z])(?=[A-Z][a-z])"
    r"|(?<=[A-Za-z])(?=\d)"
    r"|(?<=\d)(?=[A-Za-z])"
)

_WHITESPACE_RUN = re.compile(r"\s+")


def strip_instance_suffix(name: str) -> str:
    """Remove a trailing host instance suffix such as ``<3>`` or ``(2)``.

    Applied until the name stops shrinking, because hosts stack these markers
    (``"bracket (2) <1>"``).
    """
    text = name.strip()
    while True:
        shorter = _INSTANCE_SUFFIX.sub("", text).strip()
        if shorter == text:
            return text
        text = shorter


def _canonical_stem(name: str) -> str:
    """The casefolded stem of ``name`` with any trailing counter removed.

    ``"Boss-Extrude12"`` and ``"Boss-Extrude 12"`` both reduce to
    ``"boss-extrude"``.  A name that is *only* a counter has no stem, so the
    whole (casefolded) name is returned instead of an empty string.
    """
    base = _WHITESPACE_RUN.sub(" ", strip_instance_suffix(name).strip())
    match = _TRAILING_COUNTER.match(base)
    if match is not None:
        stem = match.group("stem").strip()
        if stem:
            return stem.casefold()
    return base.casefold()


def is_default_name(name: str) -> bool:
    """True when ``name`` looks auto-generated by a CAD host (not user-typed).

    Three ways to be default: nothing there at all, a known host stem (with or
    without its counter), or a name with no letter in it -- a bare ``"12"`` or
    ``"---"`` names nothing regardless of which host produced it.
    """
    if name is None:
        return True
    base = strip_instance_suffix(str(name)).strip()
    if not base:
        return True
    if _canonical_stem(base) in _STEM_SET:
        return True
    return not any(character.isalpha() for character in base)


def is_user_name(name: str) -> bool:
    """Complement of :func:`is_default_name`."""
    return not is_default_name(name)


# ---------------------------------------------------------------------------
# normalisation / tokenisation
# ---------------------------------------------------------------------------


def normalize_name(name: str, *, lower: bool = True,
                   replace_underscore: bool = True) -> str:
    """Canonical whitespace-separated form of a raw CAD name.

    Strips the instance suffix, turns separator runs into spaces (unless
    ``replace_underscore`` is off), opens up camelCase humps and letter/digit
    transitions, collapses whitespace and casefolds (unless ``lower`` is off).
    Idempotent: normalising an already-normalised name is a no-op.
    """
    text = strip_instance_suffix(str(name))
    if replace_underscore:
        text = _SEPARATOR_RUN.sub(" ", text)
    text = _WORD_BOUNDARY.sub(" ", text)
    text = _WHITESPACE_RUN.sub(" ", text).strip()
    return text.casefold() if lower else text


def tokenize_name(name: str) -> List[str]:
    """Ordered token list of a name (no stopword removal, no stemming)."""
    return normalize_name(name).split()


def name_key(name: str) -> str:
    """Order-insensitive dedup key: sorted unique tokens joined by spaces."""
    return " ".join(sorted(set(tokenize_name(name))))


def dedupe_names(names: Iterable[str]) -> List[str]:
    """Order-preserving dedup of names using :func:`normalize_name` as key.

    The *original* spelling of the first occurrence survives; later spellings
    that normalise to the same string, and names that normalise to nothing, are
    dropped.
    """
    seen = set()
    kept: List[str] = []
    for name in names:
        key = normalize_name(name)
        if key and key not in seen:
            seen.add(key)
            kept.append(name)
    return kept


# ---------------------------------------------------------------------------
# corpus cleaning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CleanDocument:
    """A document after default-name removal and dedup."""

    document_id: str
    document_name: str
    document_description: str = ""
    body_names: Tuple[str, ...] = field(default_factory=tuple)
    feature_names: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def num_parts(self) -> int:
        return len(self.body_names)

    def as_dict(self) -> Dict[str, object]:
        return {
            "body_names": list(self.body_names),
            "feature_names": list(self.feature_names),
            "document_name": self.document_name,
            "document_description": self.document_description,
        }


def _kept_names(raw: object) -> Tuple[str, ...]:
    """User-typed names from one raw name list, deduped and order-preserved."""
    values = [str(item) for item in (raw or [])]
    return tuple(dedupe_names(name for name in values if is_user_name(name)))


def clean_document(document_id: str, doc: Mapping[str, object]) -> CleanDocument:
    """Drop default names and dedupe, keeping the document schema intact.

    A default *document* name is blanked rather than dropped, because the field
    is single-valued and downstream stratification distinguishes "named" from
    "unnamed"; the description is carried through untouched.
    """
    document_name = str(doc.get("document_name", "") or "")
    return CleanDocument(
        document_id=str(document_id),
        document_name="" if is_default_name(document_name) else document_name,
        document_description=str(doc.get("document_description", "") or ""),
        body_names=_kept_names(doc.get("body_names")),
        feature_names=_kept_names(doc.get("feature_names")),
    )


def clean_corpus(
    corpus: Mapping[str, Mapping[str, object]]
) -> Dict[str, CleanDocument]:
    """Clean a whole ``{document_id: document}`` corpus, sorted by id."""
    return {
        document_id: clean_document(document_id, corpus[document_id])
        for document_id in sorted(corpus)
    }


def document_features(doc: CleanDocument) -> Tuple[int, int, int, int, int]:
    """Five 0/1 stratification flags for a cleaned document.

    ``(has_part, two_or_more_parts, has_feature, parts_and_features,
    has_description)``.  A description of one character or less does not count.
    """
    has_part = bool(doc.body_names)
    has_feature = bool(doc.feature_names)
    flags = (
        has_part,
        len(doc.body_names) >= 2,
        has_feature,
        has_part and has_feature,
        len(doc.document_description.strip()) >= 2,
    )
    return tuple(int(flag) for flag in flags)  # type: ignore[return-value]


def name_statistics(names: Sequence[str]) -> Dict[str, object]:
    """Deterministic summary of a name list (for corpus reports)."""
    total = len(names)
    defaults = sum(1 for name in names if is_default_name(name))
    tokens = [token for name in names for token in tokenize_name(name)]
    vocabulary = sorted(set(tokens))
    return {
        "total": total,
        "default": defaults,
        "user": total - defaults,
        "default_ratio": (defaults / total) if total else 0.0,
        "tokens": len(tokens),
        "unique_tokens": len(vocabulary),
        "mean_tokens_per_name": (len(tokens) / total) if total else 0.0,
        "vocabulary": vocabulary,
    }
