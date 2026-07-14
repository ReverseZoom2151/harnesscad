"""quantity — locale-correct numeric entry with a MANDATORY read-back assert.

The bug this module exists to make impossible
---------------------------------------------
Measured on this machine, FreeCAD 1.1.1, live::

    field 'boxLength' shows: '10,00 mm'        <- COMMA decimal separator
    we focus it, select-all, and type: '37.5'  <- an ordinary Python str(float)
    we read it back:           '375,00 mm'     <- THIRTY-SEVEN POINT FIVE BECAME 375

The dot was swallowed by the widget's locale-aware validator. No exception. No
warning. The geometry regenerates happily and the part is **ten times too long**.
A vision-grounded agent cannot catch this — the screenshot says ``375,00 mm`` and
a model that "knows" it asked for 37.5 will read what it expects.

So: every numeric field write goes through :func:`write_quantity`, which

1. **formats in the application's locale** (never ``str(float)``),
2. types it,
3. **reads the field back**,
4. **parses the read-back in that same locale**, and
5. **compares to what we intended, and RAISES on mismatch.**

Step 5 is not optional and there is no path around it. An unverified write is not
a write. When the mismatch is a power of ten, the error says so in as many words,
because that is the signature of a swallowed separator.

Stdlib only. No GUI dependency: the writer takes ``type_text``/``read_text``
callables, so the whole contract is unit-testable against a simulated
comma-locale widget (see the tests) with no FreeCAD running.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple

#: Relative tolerance for the read-back compare. A UI rounds its display (2 dp by
#: default in FreeCAD), so an exact compare is wrong; but it must be tight enough
#: that 37.5 vs 375 — or 37.5 vs 37.6 — is a hard failure.
DEFAULT_REL_TOL = 1e-4
DEFAULT_ABS_TOL = 1e-9


class QuantityError(ValueError):
    """A quantity could not be parsed in the declared locale."""


class QuantityMismatch(RuntimeError):
    """THE READ-BACK DID NOT MATCH WHAT WE INTENDED. The op must not proceed.

    ``factor`` is ``read_back / intended`` when both are non-zero; a factor of
    10/100/0.1 is the fingerprint of a swallowed decimal separator.
    """

    def __init__(self, field: str, intended: float, typed: str, read_text: str,
                 read_value: Optional[float], reason: str = "") -> None:
        self.field = field
        self.intended = intended
        self.typed = typed
        self.read_text = read_text
        self.read_value = read_value
        self.factor = (read_value / intended
                       if (read_value is not None and intended) else None)
        detail = reason
        if not detail and self.factor is not None:
            power = math.log10(abs(self.factor)) if self.factor else None
            if power is not None and abs(power - round(power)) < 1e-9 and round(power) != 0:
                detail = ("read back %g x the intended value — this is a SWALLOWED "
                          "DECIMAL SEPARATOR, the classic silent 10x dimensional "
                          "error" % self.factor)
        super().__init__(
            "field %r: intended %r, typed %r, read back %r (parsed %r)%s"
            % (field, intended, typed, read_text, read_value,
               " -- " + detail if detail else ""))


@dataclass(frozen=True)
class Locale:
    """How the application under control writes numbers. DETECTED, never assumed."""

    decimal: str = "."
    group: str = ","          # thousands separator ('' = none)
    decimals: int = 2         # digits the UI displays

    def __post_init__(self) -> None:
        if self.decimal not in (".", ","):
            raise QuantityError("unsupported decimal separator %r" % self.decimal)
        if self.group == self.decimal:
            raise QuantityError("group separator equals decimal separator (%r)"
                                % self.decimal)


DOT_LOCALE = Locale(decimal=".", group=",")
COMMA_LOCALE = Locale(decimal=",", group=".")

_NUM_RE = re.compile(r"[-+]?[0-9][0-9.,   ]*")
_UNIT_RE = re.compile(r"[A-Za-z°µ]+[A-Za-z0-9^/²³]*\s*$")


def detect_locale(samples: Iterable[str], decimals: int = 2) -> Locale:
    """Infer the app's decimal separator from strings it PRODUCED (e.g. '10,00 mm').

    Only the app's own rendering is evidence. We never assume the process locale:
    FreeCAD's separator comes from its own preferences, not from Python's.
    Ambiguity ('1,234' could be either) resolves to the dot locale only when there
    is no unambiguous witness at all.
    """
    votes = {".": 0, ",": 0}
    for raw in samples:
        text = str(raw)
        m = _NUM_RE.search(text)
        if not m:
            continue
        body = m.group(0).strip()
        for sep in (".", ","):
            i = body.rfind(sep)
            if i < 0:
                continue
            tail = body[i + 1:]
            other = body.count("." if sep == "," else ",")
            # A separator followed by a run of digits that is NOT a 3-group, or
            # one that appears exactly once with the other one also present, is
            # the decimal separator.
            if tail.isdigit() and (len(tail) != 3 or other > 0
                                   or body.count(sep) == 1 and len(tail) == 2):
                if len(tail) != 3 or other > 0:
                    votes[sep] += 1
    if votes[","] > votes["."]:
        return Locale(decimal=",", group=".", decimals=decimals)
    return Locale(decimal=".", group=",", decimals=decimals)


def parse_quantity(text: str, locale: Locale) -> Tuple[float, str]:
    """Parse an app-rendered quantity ('375,00 mm') in ITS locale -> (value, unit).

    Raises :class:`QuantityError` rather than guessing. A string that is empty, or
    whose numeric part is ambiguous under the given locale, is an error — a
    silently-wrong number here would defeat the entire read-back.
    """
    raw = str(text).strip()
    if not raw:
        raise QuantityError("empty quantity")
    unit_m = _UNIT_RE.search(raw)
    unit = unit_m.group(0).strip() if unit_m else ""
    body = raw[: unit_m.start()] if unit_m else raw
    body = body.strip().replace(" ", "").replace(" ", "").replace(" ", "")
    if not body:
        raise QuantityError("no numeric part in %r" % text)
    if locale.group and locale.group in body:
        # A group separator is only legal in 3-digit groups ('1.234,56'). A lone
        # '37.5' in a comma locale is NOT a number this app would ever render, and
        # stripping the dot would silently turn it into 375 -- the very bug this
        # module exists to kill. Refuse instead of guessing.
        head, sep, _tail = body.rpartition(locale.decimal)
        intpart = (head if sep else body).lstrip("+-")
        groups = intpart.split(locale.group)
        if not groups[0] or any(len(g) != 3 for g in groups[1:]):
            raise QuantityError(
                "%r is not a well-formed number in the app's locale "
                "(decimal=%r, group=%r): %r is not a valid thousands group -- "
                "refusing to guess" % (text, locale.decimal, locale.group, body))
        body = body.replace(locale.group, "")
    if locale.decimal != ".":
        if "." in body:
            raise QuantityError(
                "%r contains '.' but the app's decimal separator is %r and its "
                "group separator is %r — refusing to guess"
                % (text, locale.decimal, locale.group))
        body = body.replace(locale.decimal, ".")
    elif "," in body:
        raise QuantityError("%r contains an unexpected ',' for a dot locale" % text)
    try:
        return float(body), unit
    except ValueError as exc:
        raise QuantityError("cannot parse %r in locale %r: %s"
                            % (text, locale.decimal, exc)) from exc


def format_quantity(value: float, locale: Locale, decimals: Optional[int] = None,
                    unit: str = "") -> str:
    """Render a value the way the app EXPECTS TO READ it. Never ``str(float)``.

    No group separator is emitted: a grouped '1.000' is exactly the ambiguity that
    breaks these widgets. The output is a plain sign + digits + the app's decimal
    separator + digits, which every Qt validator accepts.
    """
    nd = locale.decimals if decimals is None else int(decimals)
    if not math.isfinite(value):
        raise QuantityError("cannot type a non-finite quantity %r" % value)
    text = ("%.*f" % (nd, float(value)))
    if locale.decimal != ".":
        text = text.replace(".", locale.decimal)
    return text + ((" " + unit) if unit else "")


def values_match(intended: float, read: float, rel_tol: float = DEFAULT_REL_TOL,
                 abs_tol: float = DEFAULT_ABS_TOL) -> bool:
    return math.isclose(float(intended), float(read),
                        rel_tol=rel_tol, abs_tol=abs_tol)


@dataclass(frozen=True)
class WriteReport:
    """Evidence that a numeric write happened, and happened correctly."""

    field: str
    intended: float
    typed: str
    read_text: str
    read_value: float
    unit: str = ""
    locale_decimal: str = "."
    verified: bool = False

    def to_dict(self) -> dict:
        return {"field": self.field, "intended": self.intended, "typed": self.typed,
                "read_text": self.read_text, "read_value": self.read_value,
                "unit": self.unit, "locale_decimal": self.locale_decimal,
                "verified": self.verified}


def write_quantity(field: str, value: float,
                   type_text: Callable[[str], None],
                   read_text: Callable[[], str],
                   locale: Optional[Locale] = None,
                   decimals: Optional[int] = None,
                   rel_tol: float = DEFAULT_REL_TOL,
                   abs_tol: float = DEFAULT_ABS_TOL,
                   retries: int = 1) -> WriteReport:
    """Write a number into a field and PROVE it landed. The only numeric write path.

    ``type_text`` must already have done focus + select-all (the caller owns the
    widget); ``read_text`` returns the widget's rendering afterwards. If ``locale``
    is None it is detected from the field's CURRENT rendering, which is the only
    honest source: the app's own output.

    Raises :class:`QuantityMismatch` if the read-back disagrees with the intent.
    There is no return path that reports success without having compared.
    """
    before = ""
    try:
        before = str(read_text() or "")
    except Exception:  # noqa: BLE001 - a field we cannot read is a field we cannot write
        before = ""
    loc = locale if locale is not None else detect_locale([before] if before else [])

    nd = loc.decimals if decimals is None else int(decimals)
    # The UI displays a ROUNDED value (2 dp by default), so the read-back can never
    # be exact; the tolerance admits exactly that much and not one bit more. Half a
    # display ulp still makes 37.5 vs 37.6 -- and 37.5 vs 375 -- a hard failure.
    abs_tol = max(abs_tol, 0.5 * (10.0 ** -nd) * 1.001)

    typed = format_quantity(value, loc, decimals=decimals)
    last_text, last_value, last_reason = "", None, ""
    for _ in range(max(1, int(retries) + 1)):
        type_text(typed)
        last_text = str(read_text() or "")
        try:
            parsed, unit = parse_quantity(last_text, loc)
        except QuantityError as exc:
            last_value, last_reason = None, str(exc)
            continue
        last_value = parsed
        if values_match(value, parsed, rel_tol, abs_tol):
            return WriteReport(field=field, intended=float(value), typed=typed,
                               read_text=last_text, read_value=float(parsed),
                               unit=unit, locale_decimal=loc.decimal, verified=True)
        last_reason = ""
    raise QuantityMismatch(field, float(value), typed, last_text, last_value,
                           last_reason)
