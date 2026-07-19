"""annomap_parser — 2D drawing-annotation schema + deterministic callout parser.

The mapping problem is formalised over two pre-extracted structured sets:

  * a **3D AFR feature set** ``F3D = {f_i}`` where ``f_i = (t_i, p_i, c_i, k_i)``
    is a feature type, geometric parameters, an AFR confidence and optional
    metadata (pattern id, symmetry group, centroid / bbox);
  * a **2D drawing entity set** ``E2D = {e_j}`` where ``e_j = (u_j, q_j, tau_j,
    gamma_j)`` is an entity type, parsed semantic values, raw OCR text and local
    context (bbox, neighbours, associated views).

The VLM *semantic-enrichment* step that turns a cropped image + OCR into a typed
descriptor is a learned model handled elsewhere. But the deterministic
*symbolic* parse of the OCR text itself — turning tokens such as ``"Ø10"``, ``"M8x1.25"``, ``"R5"``, ``"4X Ø6.5"``,
``"10 +0.1/-0.0"`` or ``"Ra 3.2"`` into a typed :class:`DrawingEntity` with a
normalized entity type, a nominal value, a symmetric/asymmetric tolerance band and
an inferred target feature category — is a pure, reproducible string operation.
This module implements that schema and parser (stdlib-only, no VLM, no OCR).

Nothing here is executed or networked; input is text, output is data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------- #
# Schema (entity + feature sets)
# --------------------------------------------------------------------------- #

# Normalized 2D entity types (u_j).
ENTITY_DIAMETER = "diameter"
ENTITY_RADIUS = "radius"
ENTITY_LINEAR = "linear"        # a plain length/width/depth dimension
ENTITY_ANGLE = "angle"
ENTITY_THREAD = "thread"
ENTITY_COUNTERBORE = "counterbore"
ENTITY_COUNTERSINK = "countersink"
ENTITY_SURFACE_FINISH = "surface_finish"
ENTITY_GDT = "gdt"
ENTITY_DATUM = "datum"
ENTITY_NOTE = "note"

_KNOWN_ENTITY_TYPES = frozenset({
    ENTITY_DIAMETER, ENTITY_RADIUS, ENTITY_LINEAR, ENTITY_ANGLE, ENTITY_THREAD,
    ENTITY_COUNTERBORE, ENTITY_COUNTERSINK, ENTITY_SURFACE_FINISH, ENTITY_GDT,
    ENTITY_DATUM, ENTITY_NOTE,
})

# Inferred target 3D feature category per entity type (drives the type
# compatibility). ``None`` means "no single preferred target".
_TARGET_FEATURE: Dict[str, Optional[str]] = {
    ENTITY_DIAMETER: "hole",
    ENTITY_RADIUS: "fillet",
    ENTITY_THREAD: "hole",
    ENTITY_COUNTERBORE: "hole",
    ENTITY_COUNTERSINK: "hole",
    ENTITY_LINEAR: None,
    ENTITY_ANGLE: "chamfer",
    ENTITY_SURFACE_FINISH: None,
    ENTITY_GDT: None,
    ENTITY_DATUM: None,
    ENTITY_NOTE: None,
}


@dataclass(frozen=True)
class Tolerance:
    """A tolerance band around a nominal value.

    ``plus`` / ``minus`` are non-negative magnitudes (mm), so the permitted range
    is ``[nominal - minus, nominal + plus]``. A symmetric ``+/-0.1`` has
    ``plus == minus == 0.1``; a limit/asymmetric band is captured directly.
    """

    plus: float = 0.0
    minus: float = 0.0

    @property
    def is_symmetric(self) -> bool:
        return abs(self.plus - self.minus) <= 1e-9

    @property
    def width(self) -> float:
        return self.plus + self.minus

    def bounds(self, nominal: float) -> Tuple[float, float]:
        return (nominal - self.minus, nominal + self.plus)

    def to_dict(self) -> dict:
        return {"plus": self.plus, "minus": self.minus}


@dataclass
class DrawingEntity:
    """A parsed 2D drawing entity (e_j).

    - ``entity_type``    : normalized u_j (one of the ``ENTITY_*`` constants).
    - ``raw_text``       : the original OCR token tau_j.
    - ``value``          : nominal numeric value (q_j), or ``None`` when the
                           entity carries no scalar (a bare datum / note).
    - ``unit``           : measurement unit ("mm" / "deg" / "").
    - ``tolerance``      : parsed :class:`Tolerance` (defaults to exact 0/0).
    - ``target_feature`` : inferred 3D feature category.
    - ``multiplicity``   : ``n`` from an ``nX`` pattern indication (default 1).
    - ``symbol``         : the leading symbol if any ("Ø", "R", "M", ...).
    - ``extra``          : type-specific extras (thread pitch, gdt symbol, etc.).
    - ``context``        : local context gamma_j (bbox / view / neighbours).
    - ``entity_id``      : stable id used by the mapping stage.
    """

    entity_type: str
    raw_text: str = ""
    value: Optional[float] = None
    unit: str = ""
    tolerance: Tolerance = field(default_factory=Tolerance)
    target_feature: Optional[str] = None
    multiplicity: int = 1
    symbol: str = ""
    extra: Dict[str, object] = field(default_factory=dict)
    context: Dict[str, object] = field(default_factory=dict)
    entity_id: str = ""

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "raw_text": self.raw_text,
            "value": self.value,
            "unit": self.unit,
            "tolerance": self.tolerance.to_dict(),
            "target_feature": self.target_feature,
            "multiplicity": self.multiplicity,
            "symbol": self.symbol,
            "extra": dict(self.extra),
            "context": dict(self.context),
        }


@dataclass
class CADFeature:
    """A recognised 3D CAD feature (f_i).

    - ``feature_type`` : t_i (e.g. "hole", "slot", "pocket", "fillet").
    - ``params``       : p_i, a mapping of geometric properties -> value
                         (e.g. {"diameter": 10.0, "depth": 20.0}).
    - ``confidence``   : c_i, the AFR confidence in [0, 1].
    - ``metadata``     : k_i (pattern id, symmetry group, ...).
    - ``centroid``     : optional (x, y, z).
    - ``feature_id``   : stable id used by the mapping stage.
    """

    feature_type: str
    params: Dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    metadata: Dict[str, object] = field(default_factory=dict)
    centroid: Optional[Tuple[float, float, float]] = None
    feature_id: str = ""

    def to_dict(self) -> dict:
        return {
            "feature_id": self.feature_id,
            "feature_type": self.feature_type,
            "params": dict(self.params),
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
            "centroid": list(self.centroid) if self.centroid else None,
        }


# --------------------------------------------------------------------------- #
# Callout parser
# --------------------------------------------------------------------------- #

# Diameter symbol variants OCR / drafting fonts emit for Ø.
_DIAMETER_SYMS = ("Ø", "Ø", "⌀", "%%c", "DIA", "dia")
# Surface-roughness prefixes.
_RA_RE = re.compile(r"\bR([az])\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
# nX / n X / nX-  pattern-multiplicity prefix (e.g. "4X", "4 X", "4x").
_MULT_RE = re.compile(r"^\s*([0-9]+)\s*[xX×]\s+")
# Thread designation, e.g. M8, M8x1.25, M10 X 1.5.
_THREAD_RE = re.compile(
    r"\bM\s*([0-9]+(?:\.[0-9]+)?)\s*(?:[xX×]\s*([0-9]+(?:\.[0-9]+)?))?")
# Counterbore / countersink symbols (drafting: ⌴ / ⌵) plus text forms.
_CBORE_TOKENS = ("⌴", "CBORE", "C'BORE", "SF")
_CSINK_TOKENS = ("⌵", "CSK", "CSINK")
# A signed number, optionally the first scalar in the string.
_NUM_RE = re.compile(r"[-+]?[0-9]*\.?[0-9]+")

# GD&T control symbols (ASME Y14.5 / ISO 1101), keyed by canonical name.
GDT_SYMBOLS: Dict[str, str] = {
    "⌓": "profile_of_a_surface",   # ⌓ (approx)
    "⏤": "straightness",
    "⌒": "arc/profile",
    "◎": "concentricity",
    "⌭": "cylindricity",
    "⌯": "position_alt",
    "⌖": "position",               # ⌖
    "⌷": "total_runout_alt",
    "↗": "runout_arrow",
    "⟂": "perpendicularity",       # ⟂
    "∥": "parallelism",            # ∥
    "∠": "angularity",             # ∠
    "▱": "flatness",               # ▱ (approx placeholder)
    "⬡": "circularity_alt",
}
# Text fallbacks (what OCR usually yields when the glyph is unavailable).
GDT_TEXT: Dict[str, str] = {
    "POSITION": "position",
    "TRUE POSITION": "position",
    "TP": "position",
    "PROFILE": "profile_of_a_surface",
    "PROFILE OF A SURFACE": "profile_of_a_surface",
    "PROFILE OF A LINE": "profile_of_a_line",
    "FLATNESS": "flatness",
    "STRAIGHTNESS": "straightness",
    "CIRCULARITY": "circularity",
    "ROUNDNESS": "circularity",
    "CYLINDRICITY": "cylindricity",
    "PERPENDICULARITY": "perpendicularity",
    "PARALLELISM": "parallelism",
    "ANGULARITY": "angularity",
    "CONCENTRICITY": "concentricity",
    "COAXIALITY": "concentricity",
    "SYMMETRY": "symmetry",
    "RUNOUT": "runout",
    "CIRCULAR RUNOUT": "circular_runout",
    "TOTAL RUNOUT": "total_runout",
}
# Material-condition modifiers (M = MMC, L = LMC, S = RFS).
GDT_MODIFIERS = {"M": "MMC", "L": "LMC", "S": "RFS", "P": "projected"}


def _has_diameter_symbol(text: str) -> bool:
    up = text.upper()
    return any(sym.upper() in up for sym in _DIAMETER_SYMS)


def _strip_multiplicity(text: str) -> Tuple[int, str]:
    m = _MULT_RE.match(text)
    if not m:
        return 1, text
    n = int(m.group(1))
    return max(1, n), text[m.end():]


def _first_number(text: str) -> Optional[float]:
    m = _NUM_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _parse_tolerance(text: str) -> Tolerance:
    """Extract a tolerance band from the trailing part of a dimension token.

    Recognises:
      * symmetric  ``+/-0.1`` / ``±0.1``,
      * bilateral  ``+0.1/-0.05`` / ``+0.1 -0.05``,
      * limit      handled by caller when two full numbers are present.
    Returns an exact ``Tolerance()`` when nothing is found.
    """
    # Symmetric +/- forms.
    sym = re.search(r"(?:±|\+/-|\+/\-)\s*([0-9]*\.?[0-9]+)", text)
    if sym:
        v = float(sym.group(1))
        return Tolerance(plus=v, minus=v)
    plus = re.search(r"\+\s*([0-9]*\.?[0-9]+)", text)
    minus = re.search(r"-\s*([0-9]*\.?[0-9]+)", text)
    if plus or minus:
        p = float(plus.group(1)) if plus else 0.0
        m = float(minus.group(1)) if minus else 0.0
        return Tolerance(plus=p, minus=m)
    return Tolerance()


def parse_gdt_frame(text: str) -> Optional[Dict[str, object]]:
    """Parse a GD&T feature-control frame into ``{symbol, tolerance, datums}``.

    Accepts either the glyph or a text label plus a tolerance value, optional
    diameter zone (``Ø``), material modifier, and up to three datum letters, e.g.
    ``"POSITION Ø0.2 M A B C"`` or ``"⌖ 0.1 A"``. Returns ``None`` when no GD&T
    characteristic can be identified.
    """
    if not text:
        return None
    symbol = None
    for glyph, name in GDT_SYMBOLS.items():
        if glyph in text:
            symbol = name
            break
    if symbol is None:
        up = text.upper()
        # Longest text key first so "TOTAL RUNOUT" wins over "RUNOUT".
        for key in sorted(GDT_TEXT, key=len, reverse=True):
            if key in up:
                symbol = GDT_TEXT[key]
                break
    if symbol is None:
        return None

    diametral = _has_diameter_symbol(text)
    tol = _first_number(text)
    # Material modifier: an isolated M/L/S token after the tolerance.
    modifier = None
    mod = re.search(r"\b([MLS])\b", text.upper())
    if mod and mod.group(1) in GDT_MODIFIERS:
        modifier = GDT_MODIFIERS[mod.group(1)]
    # Datum letters: single capital letters excluding the modifier tokens.
    datums: List[str] = []
    for tok in re.findall(r"\b([A-Z])\b", text.upper()):
        if tok in ("M", "L", "S", "P"):
            continue
        if tok not in datums:
            datums.append(tok)
    return {
        "symbol": symbol,
        "tolerance": tol,
        "diametral_zone": diametral,
        "modifier": modifier,
        "datums": datums[:3],
    }


def parse_callout(text: str, entity_id: str = "",
                  context: Optional[Dict[str, object]] = None) -> DrawingEntity:
    """Parse a single OCR callout string into a typed :class:`DrawingEntity`.

    This is the deterministic symbolic core of the enrichment step: it
    normalises the entity type, extracts the nominal value + unit, the tolerance
    band, the pattern multiplicity (``nX``) and the inferred target feature —
    everything except the learned VLM inference of ambiguous bare numbers.
    """
    ctx = dict(context or {})
    raw = (text or "").strip()
    mult, body = _strip_multiplicity(raw)
    body_s = body.strip()
    up = body_s.upper()

    def _finish(ent: DrawingEntity) -> DrawingEntity:
        ent.multiplicity = mult
        ent.context = ctx
        ent.entity_id = entity_id
        if ent.target_feature is None:
            ent.target_feature = _TARGET_FEATURE.get(ent.entity_type)
        return ent

    # 1) GD&T frame.
    frame = parse_gdt_frame(body_s)
    if frame is not None:
        return _finish(DrawingEntity(
            entity_type=ENTITY_GDT, raw_text=raw,
            value=frame["tolerance"], unit="mm",
            symbol=frame["symbol"], extra=frame))

    # 2) Bare datum feature label, e.g. "A" or "DATUM A" or "[A]".
    dm = re.fullmatch(r"(?:DATUM\s+)?\[?([A-Z])\]?", up)
    if dm:
        return _finish(DrawingEntity(
            entity_type=ENTITY_DATUM, raw_text=raw, symbol=dm.group(1),
            extra={"datum": dm.group(1)}))

    # 3) Surface finish (Ra / Rz).
    ra = _RA_RE.search(body_s)
    if ra:
        return _finish(DrawingEntity(
            entity_type=ENTITY_SURFACE_FINISH, raw_text=raw,
            value=float(ra.group(2)), unit="um", symbol="R" + ra.group(1),
            extra={"parameter": ("R" + ra.group(1)).lower()}))

    # 4) Thread designation.
    th = _THREAD_RE.search(body_s)
    if th and (up.startswith("M") or " M" in up):
        nominal = float(th.group(1))
        pitch = float(th.group(2)) if th.group(2) else None
        return _finish(DrawingEntity(
            entity_type=ENTITY_THREAD, raw_text=raw, value=nominal, unit="mm",
            symbol="M", tolerance=_parse_tolerance(body_s),
            extra={"designation": "M%g" % nominal, "pitch": pitch}))

    # 5) Counterbore / countersink.
    if any(tok in up for tok in _CBORE_TOKENS):
        return _finish(DrawingEntity(
            entity_type=ENTITY_COUNTERBORE, raw_text=raw,
            value=_first_number(body_s), unit="mm",
            tolerance=_parse_tolerance(body_s)))
    if any(tok in up for tok in _CSINK_TOKENS):
        return _finish(DrawingEntity(
            entity_type=ENTITY_COUNTERSINK, raw_text=raw,
            value=_first_number(body_s), unit="mm",
            tolerance=_parse_tolerance(body_s)))

    # 6) Radius, e.g. "R5" or "R 5.0".
    rr = re.match(r"^R\s*([0-9]*\.?[0-9]+)", up)
    if rr:
        return _finish(DrawingEntity(
            entity_type=ENTITY_RADIUS, raw_text=raw, value=float(rr.group(1)),
            unit="mm", symbol="R", tolerance=_parse_tolerance(body_s)))

    # 7) Angle, e.g. "45°" or "30 deg".
    ang = re.search(r"([0-9]*\.?[0-9]+)\s*(?:°|DEG)", up)
    if ang:
        return _finish(DrawingEntity(
            entity_type=ENTITY_ANGLE, raw_text=raw, value=float(ang.group(1)),
            unit="deg", tolerance=_parse_tolerance(body_s)))

    # 8) Diameter, e.g. "Ø10" / "DIA 10" / "10 DIA".
    if _has_diameter_symbol(body_s):
        return _finish(DrawingEntity(
            entity_type=ENTITY_DIAMETER, raw_text=raw,
            value=_first_number(body_s), unit="mm", symbol="Ø",
            tolerance=_parse_tolerance(body_s)))

    # 9) Plain scalar dimension -> linear (an ambiguous bare number; the VLM
    #    would refine the target category, which is left None here).
    val = _first_number(body_s)
    if val is not None:
        return _finish(DrawingEntity(
            entity_type=ENTITY_LINEAR, raw_text=raw, value=val, unit="mm",
            tolerance=_parse_tolerance(body_s)))

    # 10) Anything else is a free note.
    return _finish(DrawingEntity(entity_type=ENTITY_NOTE, raw_text=raw))


def parse_entities(callouts: Sequence[str],
                   contexts: Optional[Sequence[Optional[Dict[str, object]]]] = None,
                   id_prefix: str = "E") -> List[DrawingEntity]:
    """Parse a batch of callouts, assigning stable ids ``E1, E2, ...``."""
    out: List[DrawingEntity] = []
    for idx, text in enumerate(callouts, start=1):
        ctx = None
        if contexts is not None and idx - 1 < len(contexts):
            ctx = contexts[idx - 1]
        out.append(parse_callout(text, entity_id="%s%d" % (id_prefix, idx),
                                 context=ctx))
    return out


def is_known_entity_type(entity_type: str) -> bool:
    return entity_type in _KNOWN_ENTITY_TYPES
