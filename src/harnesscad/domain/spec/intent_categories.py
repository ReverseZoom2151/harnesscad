"""Text-intent -> brick category routing and prompt constraint linting.

Before any model is invoked, this module deterministically maps a free-form
request onto a fixed set of brick-assembly object categories and lints the
prompt against the model's hard constraints (1-unit cuboid bricks, 20x20x20
grid, trained categories only). This is a self-contained keyword / synonym /
semantic-verb router plus a rule-based prompt validator -- reusable for any
pipeline that must snap an open-ended intent onto a closed vocabulary and warn
about out-of-vocabulary requests.

Pure, stdlib only, deterministic. No model, no network.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The 21 trained object categories.
CATEGORIES: tuple[str, ...] = (
    "basket", "bed", "bench", "birdhouse", "bookshelf", "bottle", "bowl",
    "bus", "camera", "car", "chair", "guitar", "jar", "mug", "piano",
    "pot", "sofa", "table", "tower", "train", "vessel",
)

# One-word synonyms that map straight onto a category.
SYNONYMS: dict[str, str] = {
    "cup": "mug",
    "couch": "sofa",
    "shelf": "bookshelf",
    "bookshelves": "bookshelf",
    "truck": "bus",
}

# Verb / theme keywords -> ranked candidate categories.
SEMANTIC_MAP: dict[str, tuple[str, ...]] = {
    "sit": ("chair", "bench", "sofa"),
    "sleep": ("bed",),
    "drink": ("mug", "bottle", "bowl"),
    "eat": ("bowl", "table", "chair"),
    "play": ("guitar", "piano"),
    "store": ("bookshelf", "basket", "jar", "pot"),
    "transport": ("car", "bus", "train"),
    "contain": ("basket", "bowl", "jar", "pot", "vessel"),
    "structure": ("tower", "table"),
    "furniture": ("chair", "sofa", "table", "bed", "bench"),
    "vehicle": ("car", "bus", "train"),
    "musical": ("guitar", "piano"),
}


def map_to_categories(intent: str, top_k: int = 3) -> list[str]:
    """Rank up to ``top_k`` candidate categories for a free-form ``intent``.

    Precedence: direct category words, then synonyms, then semantic verbs, then
    a size/movement-based fallback. Order within a tier is stable.
    """
    text = (intent or "").lower()
    matches: list[str] = []

    def add(cat: str) -> None:
        if cat not in matches:
            matches.append(cat)

    for cat in CATEGORIES:
        if cat in text:
            add(cat)
    for syn, cat in SYNONYMS.items():
        if syn in text:
            add(cat)
    for key, cats in SEMANTIC_MAP.items():
        if key in text:
            for cat in cats:
                add(cat)

    if not matches:
        if any(w in text for w in ("small", "tiny", "mini")):
            matches = ["basket", "bowl", "mug"]
        elif any(w in text for w in ("large", "big", "tall")):
            matches = ["tower", "bookshelf", "table"]
        elif any(w in text for w in ("vehicle", "move", "travel")):
            matches = ["car", "train", "bus"]
        else:
            matches = ["tower", "table", "chair"]

    return matches[:top_k]


def normalize_category(name: str) -> str:
    """Snap a possibly-noisy category name onto the allowed set (default table)."""
    if not isinstance(name, str):
        return "table"
    n = name.strip().lower()
    extra = {
        "arm chair": "chair",
        "glass jar": "jar",
        "vehicle": "car",
        "camera body": "camera",
    }
    n = SYNONYMS.get(n, extra.get(n, n))
    return n if n in CATEGORIES else "table"


# Terms the 1-unit-cuboid / fixed-grid model cannot honour.
_OUT_OF_SCOPE = (
    "round", "circular", "sphere", "cylinder", "cone",
    "organic", "curved", "smooth", "fluid", "liquid",
    "2-story", "multi-story", "double-decker",
    "floating", "flying", "submerged", "underground",
)
_OVERSIZE = ("massive", "enormous", "gigantic", "huge", "colossal")
_NON_BRICK = ("wood", "metal", "glass", "plastic", "fabric", "cloth")


@dataclass
class PromptValidation:
    valid: bool
    warnings: list[str] = field(default_factory=list)
    suggested_category: str | None = None


def validate_prompt(prompt: str) -> PromptValidation:
    """Lint ``prompt`` against the brick-model constraints.

    Flags curved/organic geometry, oversize adjectives, and non-brick materials.
    More than two warnings marks the prompt invalid. Also suggests the best
    category via :func:`map_to_categories`.
    """
    text = (prompt or "").lower()
    warnings: list[str] = []
    for term in _OUT_OF_SCOPE:
        if term in text:
            warnings.append(f"Term '{term}' may not work with 1-unit brick constraints")
    for term in _OVERSIZE:
        if term in text:
            warnings.append("Very large structures may exceed the 20x20x20 grid")
            break
    for term in _NON_BRICK:
        if term in text:
            warnings.append(f"BrickGPT builds with bricks only -- '{term}' becomes brick")
    cats = map_to_categories(prompt)
    return PromptValidation(
        valid=len(warnings) <= 2,
        warnings=warnings,
        suggested_category=cats[0] if cats else None,
    )
