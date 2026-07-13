"""Manufacturing-feature taxonomy / ontology (Khan et al., "Leveraging Vision-
Language Models for Manufacturing Feature Recognition in CAD Designs", Sec. 3.1
and 3.3.1, Fig. 4).

The paper builds a *hierarchical* manufacturing-feature list used as the fixed
label space for Automatic Feature Recognition (AFR). Features are organised into
five primary categories, each subdivided into subcategories, whose lowest level
holds the individual leaf features that a recogniser must name:

    machining features
        holes              -> hole
        slots              -> slot
        steps              -> step
        pockets            -> pocket
        edges & contours   -> chamfer, fillet
        threads & spirals  -> thread, gear_teeth
        additional         -> neck
    extrusion features     -> pipe_tube, boss
    freeform features      -> depression, protrusion
    molding & casting      -> rib, gusset, draft
    sheet metal features   -> bend

Similar geometries are grouped: e.g. blind / through / countersink / counterbore
/ tapered holes all fall under the single leaf ``hole`` "due to their similar
geometries and tooling requirements" (Sec. 3.3.1). Those finer distinctions are
represented here as *subtypes* (attributes), NOT as separate leaf labels.

This is the paper's DETERMINISTIC contribution: a fixed ontology + a normaliser
that maps free-text feature names (as a VLM or an expert would write them) onto
the canonical leaf label space. It is distinct from ``bench/engdesign_taxonomy``
and ``bench/engdesign_dfm_scoring.MACHINING_FEATURES`` (paper 85's flat
15-machining-feature list): this taxonomy is hierarchical and spans five
manufacturing processes, not machining alone.

stdlib-only, deterministic, no I/O.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# The hierarchy (primary category -> subcategory -> leaf features)
# --------------------------------------------------------------------------- #
# Ordered so iteration is deterministic.
HIERARCHY = (
    ("machining", (
        ("holes", ("hole",)),
        ("slots", ("slot",)),
        ("steps", ("step",)),
        ("pockets", ("pocket",)),
        ("edges_and_contours", ("chamfer", "fillet")),
        ("threads_and_spirals", ("thread", "gear_teeth")),
        ("additional", ("neck",)),
    )),
    ("extrusion", (
        ("pipes_and_tubes", ("pipe_tube",)),
        ("bosses", ("boss",)),
    )),
    ("freeform", (
        ("depressions", ("depression",)),
        ("protrusions", ("protrusion",)),
    )),
    ("molding_casting", (
        ("ribs", ("rib",)),
        ("gussets", ("gusset",)),
        ("drafts", ("draft",)),
    )),
    ("sheet_metal", (
        ("bends", ("bend",)),
    )),
)

PRIMARY_CATEGORIES = tuple(cat for cat, _ in HIERARCHY)

# Flat, ordered tuple of every leaf feature label.
LEAF_FEATURES = tuple(
    leaf
    for _cat, subs in HIERARCHY
    for _sub, leaves in subs
    for leaf in leaves
)

# leaf -> (primary_category, subcategory)
_LEAF_INDEX = {
    leaf: (cat, sub)
    for cat, subs in HIERARCHY
    for sub, leaves in subs
    for leaf in leaves
}

# Hole subtypes grouped under the single ``hole`` leaf (Sec. 3.3.1).
HOLE_SUBTYPES = (
    "simple", "blind", "through", "countersink", "counterbore", "tapered",
    "threaded",
)

# Which leaf features can carry which named dimensional attributes. Used by
# ``fabrication/mfgfeat_attributes`` and to validate attribute dicts.
FEATURE_ATTRIBUTES = {
    "hole": ("diameter", "depth", "through", "subtype"),
    "slot": ("width", "length", "depth"),
    "step": ("width", "depth"),
    "pocket": ("width", "length", "depth", "corner_radius"),
    "chamfer": ("width", "angle"),
    "fillet": ("radius",),
    "thread": ("major_diameter", "pitch", "length"),
    "gear_teeth": ("count", "module"),
    "neck": ("diameter", "width"),
    "pipe_tube": ("outer_diameter", "inner_diameter", "length"),
    "boss": ("diameter", "height"),
    "depression": ("depth",),
    "protrusion": ("height",),
    "rib": ("thickness", "height", "length"),
    "gusset": ("thickness", "height"),
    "draft": ("angle",),
    "bend": ("angle", "radius", "length"),
}

# --------------------------------------------------------------------------- #
# Name normalisation: map free-text feature names onto canonical leaf labels.
# --------------------------------------------------------------------------- #
# Aliases / synonyms a VLM or expert might emit. Keys are normalised (lowercase,
# single-spaced); values are canonical leaf labels. Canonical labels and their
# space/hyphen variants are added automatically below.
_ALIASES = {
    "holes": "hole",
    "hole feature": "hole",
    "drilled hole": "hole",
    "blind hole": "hole",
    "through hole": "hole",
    "thru hole": "hole",
    "countersink": "hole",
    "countersunk hole": "hole",
    "counterbore": "hole",
    "counterbored hole": "hole",
    "tapered hole": "hole",
    "bore": "hole",
    "slots": "slot",
    "rectangular slot": "slot",
    "through slot": "slot",
    "blind slot": "slot",
    "groove": "slot",
    "steps": "step",
    "rectangular step": "step",
    "through step": "step",
    "blind step": "step",
    "pockets": "pocket",
    "rectangular pocket": "pocket",
    "closed pocket": "pocket",
    "cavity": "pocket",
    "chamfers": "chamfer",
    "chamfered edge": "chamfer",
    "bevel": "chamfer",
    "fillets": "fillet",
    "rounded edge": "fillet",
    "round": "fillet",
    "threads": "thread",
    "threaded": "thread",
    "thread and spiral": "thread",
    "spiral": "thread",
    "gear tooth": "gear_teeth",
    "gear teeth": "gear_teeth",
    "teeth": "gear_teeth",
    "necks": "neck",
    "pipe": "pipe_tube",
    "tube": "pipe_tube",
    "pipes and tubes": "pipe_tube",
    "pipe/tube": "pipe_tube",
    "pipe tube": "pipe_tube",
    "bosses": "boss",
    "boss shape": "boss",
    "depressions": "depression",
    "dent": "depression",
    "protrusions": "protrusion",
    "bump": "protrusion",
    "ribs": "rib",
    "gussets": "gusset",
    "drafts": "draft",
    "draft angle": "draft",
    "bends": "bend",
    "sheet metal bend": "bend",
    "flange": "bend",
}


def _norm(name):
    """Lowercase, collapse whitespace, unify separators."""
    s = str(name).strip().lower()
    s = s.replace("-", " ").replace("_", " ").replace("/", " ")
    s = " ".join(s.split())
    return s


# Build the full lookup: canonical labels (and their separator variants) plus
# the alias table.
def _build_lookup():
    table = {}
    for leaf in LEAF_FEATURES:
        table[_norm(leaf)] = leaf
    for alias, leaf in _ALIASES.items():
        table[_norm(alias)] = leaf
    return table


_LOOKUP = _build_lookup()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def is_leaf(name):
    """True iff ``name`` is one of the canonical leaf feature labels."""
    return name in _LEAF_INDEX


def normalize_feature(name):
    """Map a free-text feature name onto a canonical leaf label.

    Returns the canonical leaf label (e.g. "hole"). Raises ``KeyError`` if the
    name cannot be recognised as any known feature or alias.
    """
    key = _norm(name)
    if key in _LOOKUP:
        return _LOOKUP[key]
    raise KeyError("unknown manufacturing feature: %r" % (name,))


def try_normalize(name, default=None):
    """Like :func:`normalize_feature` but returns ``default`` instead of raising."""
    try:
        return normalize_feature(name)
    except KeyError:
        return default


def category_of(feature):
    """Primary category of a leaf feature (canonical label). Raises KeyError."""
    return _LEAF_INDEX[feature][0]


def subcategory_of(feature):
    """Subcategory of a leaf feature (canonical label). Raises KeyError."""
    return _LEAF_INDEX[feature][1]


def leaves_of_category(category):
    """Ordered tuple of leaf features under a primary category."""
    if category not in PRIMARY_CATEGORIES:
        raise KeyError("unknown primary category: %r" % (category,))
    return tuple(
        leaf
        for cat, subs in HIERARCHY if cat == category
        for _sub, leaves in subs
        for leaf in leaves
    )


def attributes_of(feature):
    """Named dimensional attributes a leaf feature can carry (tuple)."""
    return FEATURE_ATTRIBUTES[feature]


def is_hole_subtype(subtype):
    """True iff ``subtype`` is a recognised hole subtype."""
    return _norm(subtype).replace(" ", "") in {
        s.replace("_", "") for s in HOLE_SUBTYPES
    }
