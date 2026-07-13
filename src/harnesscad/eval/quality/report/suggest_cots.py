"""suggest_cots — advisory COTS (commercial-off-the-shelf) standard-part matcher.

The blueprint's "standard-part suggestion" move: scan a model's features (hole
patterns, shaft/bore diameters) and, where a piece of custom geometry lines up
with a catalogued standard part, suggest *replacing* it with the COTS item — a
DIN/ISO fastener for a clearance hole, a metric ball bearing for a shaft bore.

This is purely advisory: it emits :class:`Suggestion` records (what, which
standard, why, how confident) and never mutates the model. Matching is against a
small built-in standards table plus, optionally, the :class:`library.catalog`
(e.g. a shaft bore -> "seat it in a `bearing_seat` part").

Input is flexible: pass a backend (anything exposing ``.features`` / ``.query``,
e.g. StubBackend) and holes are read off its feature log, or pass an explicit
feature list/dict for full control:

    suggest_cots(backend, catalog)
    suggest_cots([{"kind": "hole", "diameter": 5.5}], catalog)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Standard-part reference tables (nominal metric; mm)
# ---------------------------------------------------------------------------
# Metric socket-head cap screws (ISO 4762 / DIN 912) with medium clearance holes
# (ISO 273 "medium"). A custom through-hole near a clearance size is a strong
# signal it exists to pass a standard bolt.
FASTENERS: List[Dict[str, Any]] = [
    {"standard": "ISO 4762 M2.5", "thread": "M2.5", "clearance_hole": 2.9},
    {"standard": "ISO 4762 M3",   "thread": "M3",   "clearance_hole": 3.4},
    {"standard": "ISO 4762 M4",   "thread": "M4",   "clearance_hole": 4.5},
    {"standard": "ISO 4762 M5",   "thread": "M5",   "clearance_hole": 5.5},
    {"standard": "ISO 4762 M6",   "thread": "M6",   "clearance_hole": 6.6},
    {"standard": "ISO 4762 M8",   "thread": "M8",   "clearance_hole": 9.0},
    {"standard": "ISO 4762 M10",  "thread": "M10",  "clearance_hole": 11.0},
    {"standard": "ISO 4762 M12",  "thread": "M12",  "clearance_hole": 13.5},
]

# Metric deep-groove ball bearings (ISO 15 / common designations): bore/OD/width.
BEARINGS: List[Dict[str, Any]] = [
    {"designation": "623",  "bore": 3.0,  "od": 10.0, "width": 4.0},
    {"designation": "624",  "bore": 4.0,  "od": 13.0, "width": 5.0},
    {"designation": "625",  "bore": 5.0,  "od": 16.0, "width": 5.0},
    {"designation": "626",  "bore": 6.0,  "od": 19.0, "width": 6.0},
    {"designation": "608",  "bore": 8.0,  "od": 22.0, "width": 7.0},
    {"designation": "6000", "bore": 10.0, "od": 26.0, "width": 8.0},
    {"designation": "6001", "bore": 12.0, "od": 28.0, "width": 8.0},
    {"designation": "6002", "bore": 15.0, "od": 32.0, "width": 9.0},
    {"designation": "6004", "bore": 20.0, "od": 42.0, "width": 12.0},
]

# Absolute match tolerances (mm). A hole within this of a clearance size, or a
# bore within this of a bearing bore, is considered a match.
FASTENER_TOL = 0.5
BEARING_TOL = 0.4


@dataclass
class Suggestion:
    """One advisory COTS substitution."""

    kind: str                       # "fastener" | "bearing" | "catalog-part"
    standard_part: str              # e.g. "ISO 4762 M5" / bearing "608"
    rationale: str
    confidence: float = 0.0         # 0..1, higher = closer dimensional match
    feature: Dict[str, Any] = field(default_factory=dict)
    source: str = "standard"        # "standard" | "catalog"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "standard_part": self.standard_part,
            "rationale": self.rationale,
            "confidence": round(self.confidence, 4),
            "feature": self.feature,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def _normalize_features(backend_or_features: Any) -> List[Dict[str, Any]]:
    """Coerce the input into a list of ``{"kind", "diameter", ...}`` features.

    Accepts: a normalized list already; a single feature dict; a StubBackend-like
    object (reads holes off ``.features``, with ``.query('summary')`` as a
    fallback signal)."""
    # explicit list of feature dicts
    if isinstance(backend_or_features, (list, tuple)):
        return [_coerce_feature(f) for f in backend_or_features]
    # single feature dict
    if isinstance(backend_or_features, dict):
        return [_coerce_feature(backend_or_features)]
    # a backend: pull holes from its feature log
    feats: List[Dict[str, Any]] = []
    raw = getattr(backend_or_features, "features", None)
    if raw:
        for f in raw:
            if not isinstance(f, dict):
                continue
            if f.get("type") == "hole" and "diameter" in f:
                feats.append({
                    "kind": "hole",
                    "diameter": float(f["diameter"]),
                    "id": f.get("id"),
                    "hole_kind": f.get("kind", "simple"),
                })
    return feats


def _coerce_feature(f: Any) -> Dict[str, Any]:
    if not isinstance(f, dict):
        raise TypeError(f"feature must be a dict, got {type(f).__name__}")
    out = dict(f)
    # map a backend-style {"type": "hole"} onto the normalized {"kind": ...}
    if "kind" not in out and "type" in out:
        out["kind"] = out["type"]
    out.setdefault("kind", "hole")
    if "diameter" not in out and "radius" in out:
        out["diameter"] = 2.0 * float(out["radius"])
    return out


def _nearest(value: float, table: List[Dict[str, Any]], key: str,
             tol: float) -> Optional[Dict[str, Any]]:
    best = None
    best_err = None
    for row in table:
        err = abs(value - row[key])
        if err <= tol and (best_err is None or err < best_err):
            best, best_err = row, err
    return best


def _confidence(err: float, tol: float) -> float:
    """1.0 for an exact match, decaying linearly to 0 at the tolerance edge."""
    return max(0.0, 1.0 - (err / tol)) if tol > 0 else 1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def suggest_cots(backend_or_features: Any,
                 catalog: Any = None) -> List[Suggestion]:
    """Suggest COTS standard-part substitutions for a model's features.

    For each hole, propose the nearest DIN/ISO fastener whose clearance hole
    matches. For each hole/bore/shaft diameter, propose a metric ball bearing
    whose bore matches. When a ``catalog`` is given, also point shaft/bore
    features at a matching library part (e.g. ``bearing_seat``). Advisory only.
    """
    features = _normalize_features(backend_or_features)
    suggestions: List[Suggestion] = []

    for feat in features:
        diameter = feat.get("diameter")
        if diameter is None:
            continue
        kind = feat.get("kind", "hole")

        # 1) fastener for a clearance-style hole
        if kind in ("hole",):
            match = _nearest(diameter, FASTENERS, "clearance_hole", FASTENER_TOL)
            if match is not None:
                err = abs(diameter - match["clearance_hole"])
                suggestions.append(Suggestion(
                    kind="fastener",
                    standard_part=match["standard"],
                    rationale=(
                        f"Custom {diameter:g} mm hole matches the {match['clearance_hole']:g} mm "
                        f"medium clearance hole for a {match['thread']} screw "
                        f"({match['standard']}). Replace the modelled hole with a "
                        f"standard clearance hole + COTS {match['thread']} socket-head cap screw."),
                    confidence=_confidence(err, FASTENER_TOL),
                    feature=feat,
                    source="standard",
                ))

        # 2) bearing for a shaft/bore diameter
        if kind in ("shaft", "bore", "hole"):
            match = _nearest(diameter, BEARINGS, "bore", BEARING_TOL)
            if match is not None:
                err = abs(diameter - match["bore"])
                suggestions.append(Suggestion(
                    kind="bearing",
                    standard_part=f"bearing {match['designation']}",
                    rationale=(
                        f"{kind} of {diameter:g} mm matches the {match['bore']:g} mm bore of a "
                        f"{match['designation']} deep-groove ball bearing "
                        f"(OD {match['od']:g}, width {match['width']:g} mm). "
                        f"Consider a COTS {match['designation']} bearing on this shaft/bore."),
                    confidence=_confidence(err, BEARING_TOL),
                    feature=feat,
                    source="standard",
                ))

        # 3) optional: point a shaft/bore at a matching library part
        if catalog is not None and kind in ("shaft", "bore"):
            found = catalog.find("bearing", k=1)
            if found:
                part = found[0]
                suggestions.append(Suggestion(
                    kind="catalog-part",
                    standard_part=part.name,
                    rationale=(
                        f"A {kind} of {diameter:g} mm could be located by the library "
                        f"'{part.name}' part ({part.description}) rather than modelled ad hoc."),
                    confidence=0.5,
                    feature=feat,
                    source="catalog",
                ))

    suggestions.sort(key=lambda s: -s.confidence)
    return suggestions
