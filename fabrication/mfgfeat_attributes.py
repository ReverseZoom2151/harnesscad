"""Dimensional-attribute extraction for recognised manufacturing features
(Khan et al., "Leveraging Vision-Language Models for Manufacturing Feature
Recognition in CAD Designs", Sec. 5 / ref. [15] "Automatic Feature Recognition
and Dimensional Attributes Extraction From CAD Models").

The paper flags as a key limitation that its VLM "cannot extract geometric
dimensions from recognized features, which are vital for downstream
manufacturing tasks such as process selection, cost estimation, and quality
control". Once a feature is *named* (by the VLM, or by the deterministic
:mod:`reconstruction.mfgfeat_rule_detector`), the dimensional attributes ARE a
deterministic geometric computation. This module provides that step.

For each leaf feature it validates and normalises a raw measurement dict into
the canonical attribute schema declared in
:mod:`fabrication.mfgfeat_taxonomy` (``FEATURE_ATTRIBUTES``), deriving convenient
secondary attributes and classifying subtypes with explicit rules:

  * hole   -- diameter, depth, through/blind, and subtype
              (through | blind | countersink | counterbore | tapered |
               threaded) from entry geometry / thread flag; aspect_ratio =
               depth / diameter (a machinability hint: deep bores need special
               tooling).
  * slot / pocket / step -- width, length, depth, and aspect ratios.
  * chamfer -- width, angle.
  * fillet  -- radius.
  * pipe_tube -- outer/inner diameter, wall thickness, pipe-vs-tube class.
  * draft   -- angle, sufficiency vs a minimum draft.

Pure, deterministic, stdlib-only. No geometry kernel: the raw numbers are
assumed already measured (e.g. from a B-rep query); this layer turns them into
standardised, validated feature attributes.
"""

from __future__ import annotations

from fabrication.mfgfeat_taxonomy import (
    normalize_feature, attributes_of, HOLE_SUBTYPES,
)


def _pos(value, name):
    v = float(value)
    if v <= 0.0:
        raise ValueError("%s must be positive, got %r" % (name, value))
    return v


def _nonneg(value, name):
    v = float(value)
    if v < 0.0:
        raise ValueError("%s must be non-negative, got %r" % (name, value))
    return v


# --------------------------------------------------------------------------- #
# Per-feature extractors
# --------------------------------------------------------------------------- #
def _hole(raw):
    diameter = _pos(raw["diameter"], "diameter")
    out = {"diameter": diameter}
    depth = raw.get("depth")
    through = raw.get("through")

    # A blind hole has finite depth; a through hole passes fully through.
    if through is None and depth is not None:
        through = False
    if depth is not None:
        depth = _pos(depth, "depth")
        out["depth"] = depth
        out["aspect_ratio"] = depth / diameter
    out["through"] = bool(through) if through is not None else False

    # Subtype: explicit wins; else derive.
    subtype = raw.get("subtype")
    if subtype is not None:
        subtype = str(subtype).strip().lower()
        canon = {s.replace("_", "") for s in HOLE_SUBTYPES}
        if subtype.replace(" ", "").replace("_", "") not in canon:
            raise ValueError("unknown hole subtype: %r" % (subtype,))
    else:
        if raw.get("threaded"):
            subtype = "threaded"
        elif raw.get("counterbore_diameter"):
            subtype = "counterbore"
        elif raw.get("countersink_angle"):
            subtype = "countersink"
        elif out["through"]:
            subtype = "through"
        elif depth is not None:
            subtype = "blind"
        else:
            subtype = "simple"
    out["subtype"] = subtype
    return out


def _prismatic(raw, keys):
    out = {}
    for k in keys:
        if k in raw:
            out[k] = _pos(raw[k], k)
    if "depth" in out and "width" in out and out["width"] > 0:
        out["aspect_ratio"] = out["depth"] / out["width"]
    return out


def _chamfer(raw):
    out = {}
    if "width" in raw:
        out["width"] = _pos(raw["width"], "width")
    if "angle" in raw:
        a = float(raw["angle"])
        if not (0.0 < a < 90.0):
            raise ValueError("chamfer angle must be in (0, 90), got %r" % (a,))
        out["angle"] = a
    return out


def _fillet(raw):
    return {"radius": _pos(raw["radius"], "radius")}


def _pipe_tube(raw):
    od = _pos(raw["outer_diameter"], "outer_diameter")
    out = {"outer_diameter": od}
    idm = raw.get("inner_diameter")
    if idm is not None:
        idm = _nonneg(idm, "inner_diameter")
        if idm >= od:
            raise ValueError("inner_diameter must be < outer_diameter")
        out["inner_diameter"] = idm
        out["wall_thickness"] = (od - idm) / 2.0
        # A tube is hollow; a solid extrusion (id == 0) is a pipe/boss stem.
        out["class"] = "tube" if idm > 0.0 else "pipe"
    if "length" in raw:
        out["length"] = _pos(raw["length"], "length")
    return out


def _draft(raw):
    a = float(raw["angle"])
    if a < 0.0:
        raise ValueError("draft angle must be non-negative")
    out = {"angle": a}
    min_draft = raw.get("min_draft")
    if min_draft is not None:
        out["sufficient"] = a >= float(min_draft)
    return out


def _generic(feature, raw):
    """Validate a raw dict against the taxonomy's attribute schema, keeping only
    recognised keys and requiring positive numerics for dimensional ones."""
    allowed = set(attributes_of(feature))
    out = {}
    for k, v in raw.items():
        if k not in allowed:
            continue
        if k in ("count",):
            iv = int(v)
            if iv <= 0:
                raise ValueError("count must be positive")
            out[k] = iv
        elif k in ("subtype",):
            out[k] = v
        else:
            out[k] = _pos(v, k)
    return out


_EXTRACTORS = {
    "hole": _hole,
    "slot": lambda r: _prismatic(r, ("width", "length", "depth")),
    "step": lambda r: _prismatic(r, ("width", "depth")),
    "pocket": lambda r: _prismatic(r, ("width", "length", "depth",
                                       "corner_radius")),
    "chamfer": _chamfer,
    "fillet": _fillet,
    "pipe_tube": _pipe_tube,
    "draft": _draft,
}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def extract_attributes(feature, raw):
    """Extract standardised dimensional attributes for one recognised feature.

    ``feature``: a feature name (free text; normalised through the taxonomy).
    ``raw``: a dict of measured quantities (mm / degrees). Returns a dict of
    canonical attributes for that feature type, with derived quantities
    (aspect_ratio, wall_thickness, subtype, ...) where applicable.

    Raises ``KeyError`` for an unknown feature or missing required key, and
    ``ValueError`` for an out-of-range measurement.
    """
    leaf = normalize_feature(feature)
    extractor = _EXTRACTORS.get(leaf)
    if extractor is not None:
        return extractor(dict(raw))
    return _generic(leaf, dict(raw))
