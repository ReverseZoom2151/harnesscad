"""CadQuery API reference: signature parsing, intent retrieval, prompt assembly.

CQAsk (the ``CQAsk-main`` repo) is a natural-language -> CadQuery assistant. Its
deterministic core is not the LLM call but the *system prompt* it hands the model:
a hand-curated catalogue of ``cq.Workplane`` methods, each written as a signature
line plus a one-sentence description, e.g. ::

    cq.Workplane.lineTo(x, y[, forConstruction])- Make a line from the current point to the provided point
    cq.Workplane.rect(xLen, yLen[, centered, ...])- Make a rectangle for each item on the stack.

CQAsk always ships the *whole* catalogue on every request. This module makes that
catalogue programmable and cheaper to use:

  * :func:`parse_api_line` / :func:`parse_reference` turn each reference line into a
    structured :class:`ApiCard` (qualified name, method, required params, optional
    params, description) using a small deterministic signature grammar. The
    ``[...]`` brackets mark optional arguments and ``...`` marks a variadic tail.
  * :func:`retrieve` ranks the cards against a natural-language query by
    token-overlap (method-name and description tokens), so only the *relevant*
    API cards need be included in a prompt instead of the entire manual.
  * :func:`build_prompt_context` assembles a bounded, deterministically ordered
    context block from the top-ranked cards.

This complements :mod:`rag.cad_api_knowledge` (which validates/chunks an already
structured API list): here we *parse* free-text signature lines and *retrieve*
against intent. Pure stdlib, deterministic (no wall clock, no network, no model),
and independent of any CadQuery install.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ApiCard:
    """One parsed API reference entry.

    ``qualname`` is the full dotted name (``cq.Workplane.lineTo``); ``method`` is
    the final component (``lineTo``). ``required`` and ``optional`` are the
    positional parameter names; ``variadic`` is True when the signature ended with
    a ``...`` tail (unbounded trailing options). ``description`` is the trailing
    prose, whitespace-normalised.
    """

    qualname: str
    method: str
    required: tuple[str, ...]
    optional: tuple[str, ...]
    variadic: bool
    description: str

    def arity(self) -> tuple[int, float]:
        """Return ``(min_positional, max_positional)`` as ``(int, float)``.

        ``max`` is ``float('inf')`` when the signature is variadic.
        """
        lo = len(self.required)
        hi = float("inf") if self.variadic else float(lo + len(self.optional))
        return (lo, hi)


# A signature line looks like:  <qualname>(<args>)- <description>
# We split on the first ")-" (the repo uses ")-" with no space before the dash).
_LINE_RE = re.compile(r"^(?P<qual>[\w.]+)\((?P<args>.*?)\)\s*-\s*(?P<desc>.*)$")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _split_top_level(args: str) -> list[str]:
    """Split an argument string on commas that are not inside brackets."""
    out: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in args:
        if ch == "[":
            depth += 1
            cur.append(ch)
        elif ch == "]":
            depth = max(0, depth - 1)
            cur.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur))
    return out


def _clean_param(raw: str) -> str:
    """Normalise a single parameter token: strip brackets, defaults, whitespace."""
    p = raw.strip().lstrip("[").rstrip("]").strip()
    # Drop a default value:  angle_units="rad" -> angle_units
    if "=" in p:
        p = p.split("=", 1)[0].strip()
    # Drop a type annotation:  gear: BevelGear -> gear
    if ":" in p:
        p = p.split(":", 1)[0].strip()
    # Leading * for *args / **kwargs is kept meaningless here; strip it.
    return p.lstrip("*").strip()


def parse_signature_args(args: str) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
    """Parse the inside of ``(...)`` into (required, optional, variadic).

    A parameter is *optional* once it (or any preceding sibling within the same
    top-level split) is wrapped in ``[...]``; CadQuery's manual style opens a
    single bracket that covers all trailing optionals, e.g.
    ``rect(xLen, yLen[, centered, ...])``. A literal ``...`` marks the signature
    variadic and contributes no named parameter.
    """
    required: list[str] = []
    optional: list[str] = []
    variadic = False
    opened = False          # True once a '[' has been scanned
    token_optional = False  # optionality captured at the token's first char
    buf: list[str] = []
    started = False         # whether the current token has any content yet

    def flush() -> None:
        nonlocal variadic
        name = _clean_param("".join(buf))
        buf.clear()
        if not name:
            return
        if name == "...":
            variadic = True
            return
        # Optionality is fixed by the bracket state at the token's first character:
        # a name accumulated before its own trailing '[' (e.g. "yLen" in "yLen[")
        # stays required; names that begin after a '[' are optional.
        (optional if token_optional else required).append(name)

    for ch in args:
        if ch == ",":
            flush()
            started = False
            token_optional = opened
            continue
        if ch == "[":
            opened = True
            if not started:
                token_optional = True
            continue
        if ch == "]":
            continue
        if not ch.isspace() and not started:
            started = True
            token_optional = opened
        buf.append(ch)
    flush()
    return (tuple(required), tuple(optional), variadic)


def parse_api_line(line: str) -> ApiCard | None:
    """Parse one reference line into an :class:`ApiCard`, or None if it is not one."""
    m = _LINE_RE.match(line.strip())
    if not m:
        return None
    qual = m.group("qual")
    required, optional, variadic = parse_signature_args(m.group("args"))
    desc = re.sub(r"\s+", " ", m.group("desc")).strip()
    method = qual.rsplit(".", 1)[-1]
    return ApiCard(qual, method, required, optional, variadic, desc)


def parse_reference(text: str) -> tuple[ApiCard, ...]:
    """Parse a multi-line reference block, skipping non-signature lines.

    Deterministic: cards are returned in first-seen order with duplicate
    ``qualname`` collapsed to the first occurrence.
    """
    cards: list[ApiCard] = []
    seen: set[str] = set()
    for line in text.splitlines():
        card = parse_api_line(line)
        if card is None or card.qualname in seen:
            continue
        seen.add(card.qualname)
        cards.append(card)
    return tuple(cards)


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_RE.findall(text.lower()))


def _split_camel(name: str) -> list[str]:
    """Split a camelCase/method name into lowercase word tokens."""
    parts = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z0-9]+|[A-Z]+", name)
    return [p.lower() for p in parts if p]


def card_tokens(card: ApiCard) -> frozenset[str]:
    """The searchable token set for a card: method words + description words."""
    toks = set(_split_camel(card.method))
    toks |= _tokens(card.description)
    for p in card.required + card.optional:
        toks |= set(_split_camel(p))
    return frozenset(toks)


@dataclass(frozen=True)
class ScoredCard:
    """A card paired with its integer overlap score against a query."""

    card: ApiCard
    score: int
    matched: tuple[str, ...] = field(default_factory=tuple)


def retrieve(query: str, cards: tuple[ApiCard, ...], top_k: int = 5) -> tuple[ScoredCard, ...]:
    """Rank ``cards`` by token overlap with ``query``; return the top ``top_k``.

    Deterministic tie-break: higher score first, then a method-name exact-substring
    bonus is folded into the score, then alphabetical ``qualname``. Cards with zero
    overlap are dropped. ``top_k <= 0`` returns everything scored.
    """
    q = _tokens(query)
    scored: list[ScoredCard] = []
    for card in cards:
        ct = card_tokens(card)
        matched = tuple(sorted(q & ct))
        score = len(matched)
        # Bonus: the method name appears verbatim as a query token.
        if card.method.lower() in q:
            score += 2
        if score > 0:
            scored.append(ScoredCard(card, score, matched))
    scored.sort(key=lambda s: (-s.score, s.card.qualname))
    if top_k and top_k > 0:
        return tuple(scored[:top_k])
    return tuple(scored)


def build_prompt_context(
    query: str,
    cards: tuple[ApiCard, ...],
    top_k: int = 5,
    header: str = "Relevant CadQuery API:",
) -> str:
    """Assemble a bounded prompt-context block of the cards most relevant to ``query``.

    Each selected card is rendered as ``qualname(params) - description``. The block
    is deterministic given the same inputs. When nothing matches, only the header is
    returned (callers may then fall back to the full manual).
    """
    hits = retrieve(query, cards, top_k=top_k)
    lines = [header]
    for hit in hits:
        lines.append(render_card(hit.card))
    return "\n".join(lines)


def render_card(card: ApiCard) -> str:
    """Render a card back into a compact single-line signature-plus-description."""
    parts = list(card.required)
    tail = list(card.optional)
    if card.variadic:
        tail.append("...")
    sig = ", ".join(parts)
    if tail:
        opt = ", ".join(tail)
        sig = f"{sig}[, {opt}]" if sig else f"[{opt}]"
    desc = f" - {card.description}" if card.description else ""
    return f"{card.qualname}({sig}){desc}"
