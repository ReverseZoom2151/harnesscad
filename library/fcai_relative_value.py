"""Relative property-edit resolver (deterministic, stdlib-only).

freecad-ai's ``modify_property`` tool lets an LLM nudge a numeric property
without knowing its current value, using a tiny relative-edit mini-language:
``"+10%"`` (grow 10%), ``"-20%"`` (shrink 20%), ``"*1.5"`` (scale), ``"+5"`` /
``"-3"`` (additive delta in mm), or a bare number (absolute set). The live tool
reads the property off the FreeCAD object and applies the edit; the resolution
*rule* -- how a relative token combines with a current value -- is a pure
deterministic function, and that is what this module isolates so the harness can
plan and preview a "make the wall 10% thicker" edit without any FreeCAD host.

Beyond freecad-ai's inline helper this module: classifies each edit into a kind
(:data:`ABSOLUTE`, :data:`PERCENT`, :data:`SCALE`, :data:`DELTA`), reports the
result as a structured :class:`Resolution` (kind, previous, resolved, delta),
and supports optional min/max clamping for safe geometry edits.

Everything here is stdlib-only and deterministic. No FreeCAD, no ``eval``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = [
    "ABSOLUTE",
    "PERCENT",
    "SCALE",
    "DELTA",
    "Resolution",
    "classify",
    "resolve",
]

ABSOLUTE = "absolute"
PERCENT = "percent"
SCALE = "scale"
DELTA = "delta"


@dataclass(frozen=True)
class Resolution:
    """The outcome of applying a relative-edit token to a current value."""
    kind: str
    previous: float
    resolved: float
    token: str

    @property
    def delta(self) -> float:
        return self.resolved - self.previous

    @property
    def changed(self) -> bool:
        return self.resolved != self.previous


def classify(token: str) -> Optional[str]:
    """Return the edit kind for ``token`` or ``None`` if it is not numeric.

    ``"+10%"``/``"-5%"`` -> PERCENT, ``"*1.5"`` -> SCALE,
    ``"+5"``/``"-3"`` -> DELTA, ``"50"``/``"-2"`` (bare) -> ABSOLUTE.
    """
    if not isinstance(token, str):
        return None
    t = token.strip()
    if not t:
        return None

    if t.endswith("%"):
        return PERCENT if _is_float(t[:-1]) else None
    if t.startswith("*"):
        return SCALE if _is_float(t[1:]) else None
    if t.startswith("+"):
        return DELTA if _is_float(t) else None
    # A leading '-' is ambiguous: "-3" as an *edit* means subtract 3 (DELTA),
    # matching freecad-ai's modify_property semantics. A bare positive number
    # or a plain float with no sign is ABSOLUTE.
    if t.startswith("-") and len(t) > 1 and _is_float(t):
        return DELTA
    if _is_float(t):
        return ABSOLUTE
    return None


def resolve(current, token: str,
            minimum: Optional[float] = None,
            maximum: Optional[float] = None) -> Optional[Resolution]:
    """Apply relative-edit ``token`` to ``current``.

    Returns a :class:`Resolution`, or ``None`` when the token is not a
    recognised numeric edit or ``current`` is not numeric (mirroring
    freecad-ai's fall-through, where a non-numeric edit is left to the caller).
    Optional ``minimum`` / ``maximum`` clamp the resolved value.
    """
    kind = classify(token)
    if kind is None:
        return None
    try:
        cur = float(current)
    except (TypeError, ValueError):
        return None

    t = token.strip()
    if kind == PERCENT:
        resolved = cur * (1.0 + float(t[:-1]) / 100.0)
    elif kind == SCALE:
        resolved = cur * float(t[1:])
    elif kind == DELTA:
        resolved = cur + float(t)
    else:  # ABSOLUTE
        resolved = float(t)

    if minimum is not None and resolved < minimum:
        resolved = minimum
    if maximum is not None and resolved > maximum:
        resolved = maximum

    return Resolution(kind=kind, previous=cur, resolved=resolved, token=t)


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False
