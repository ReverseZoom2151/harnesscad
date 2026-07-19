"""Part-metadata contract: feature tags, params placement, viewer isolation.

A viewer runtime can impose a small, checkable contract on every generated
model, and ship it to the agent as skill text. The deterministic rules (as
opposed to the modeling advice) are:

1. **Feature-tag metadata** (schema ``aicad.part.metadata.v1``): a part may
   expose semantic feature tags -- name, ``kind`` (hole, slot, boss, ...),
   and a *copyable selector expression*. The selector must be written in the
   viewer's copy form: no first part argument (``holes(radius=3, axis=
   Axis.Z)``, never ``holes(part, ...)``). Tags are selector hints, not
   persistent B-rep ids, and generic faces must not be tagged wholesale.
2. **Parameter placement**: root ``params.json`` carries only assembly-level
   values (placement, motion, constraints, anchors, ``__viewer``); each
   part's geometry knobs (teeth, bore, thickness, radii, hole sizes, feature
   counts) live beside that part. Hidden JSON-only parameters are forbidden.
3. **Viewer isolation**: ``__viewer`` values are preview-only. Materials
   must be ``"#rrggbb"`` strings or ``{"preset": ...}`` objects from the
   documented preset list -- bare preset strings are invalid -- and material
   part keys must match placed instance *labels*, not part folder names.
   Nothing in ``__viewer`` may feed dimensions, topology, anchors, joints,
   or constraints.

This contract layer is harness-relevant because it makes a
generated artifact's *metadata* verifiable the same way its geometry is:
a plausible-looking part with a selector that silently names the wrong
topology, or a viewer color that leaks into a dimension, is caught by rule,
not by luck.

Stdlib only, deterministic, absolute imports. ``--selfcheck`` covers all
three rule families.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "METADATA_SCHEMA",
    "VIEWER_PRESETS",
    "ASSEMBLY_LEVEL_KEYS",
    "GEOMETRY_KNOB_PATTERNS",
    "Violation",
    "validate_selector_copy_form",
    "validate_metadata",
    "validate_params_placement",
    "validate_viewer_materials",
    "main",
]

METADATA_SCHEMA = "aicad.part.metadata.v1"

#: The documented preview material presets.
VIEWER_PRESETS: Tuple[str, ...] = (
    "cad_clay", "matte_plastic", "gloss_plastic", "rubber", "painted_metal",
    "anodized_aluminum", "brushed_steel", "dark_steel", "polished_metal",
    "glass_clear",
)

#: Keys that belong in the root (assembly-level) params.json.
ASSEMBLY_LEVEL_KEYS: Tuple[str, ...] = (
    "placement", "motion", "constraints", "anchors", "__viewer",
)

#: Name patterns that identify per-part geometry knobs (for example:
#: teeth, bore, thickness, radii, hole sizes, feature counts, local dims).
GEOMETRY_KNOB_PATTERNS: Tuple[str, ...] = (
    r"teeth", r"bore", r"thickness", r"radius", r"radii", r"diameter",
    r"hole", r"width", r"height", r"depth", r"length", r"count", r"pitch",
    r"wall", r"fillet", r"chamfer",
)

_KNOB_RE = re.compile("|".join(GEOMETRY_KNOB_PATTERNS), re.IGNORECASE)

#: Feature-tag kinds that name functional/user-visible topology.
TAG_KINDS: Tuple[str, ...] = (
    "hole", "slot", "boss", "rim", "rail", "hinge", "cutout", "connector",
    "mount", "gear_bore", "generic",
)

_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

# First positional argument that names a part/result variable, e.g.
# "holes(part, ...)" or "holes(result)". Copy form starts with a keyword
# argument or a literal.
_SELECTOR_CALL_RE = re.compile(r"^\s*(\w+)\s*\(\s*([^,()=]*?)\s*(?:,|\))")


@dataclass(frozen=True)
class Violation:
    """One contract violation, with the prescribed fix."""
    where: str
    kind: str
    message: str
    fix: str = ""

    def to_dict(self) -> dict:
        return {"where": self.where, "kind": self.kind,
                "message": self.message, "fix": self.fix}


# ---------------------------------------------------------------------------
# 1. Feature-tag metadata
# ---------------------------------------------------------------------------

def validate_selector_copy_form(selector: str) -> Optional[str]:
    """Return an error string when a selector is not in copy form.

    Copy form omits the part argument: ``holes(radius=3, axis=Axis.Z)`` is
    valid; ``holes(part, radius=3)`` and ``holes(result)`` are not. A bare
    first *keyword* argument or an empty argument list is fine.
    """
    if not selector.strip():
        return "selector is empty"
    m = _SELECTOR_CALL_RE.match(selector)
    if m is None:
        return None  # not a call expression; nothing to police
    first_arg = m.group(2).strip()
    if first_arg and "=" not in selector[m.start(2):m.end(2) + 1]:
        # a positional first argument that is a bare identifier
        if re.fullmatch(r"[A-Za-z_]\w*", first_arg):
            return (f"selector passes the part positionally ('{first_arg}'); "
                    "write the viewer copy form without it")
    return None


def validate_metadata(metadata: Mapping[str, Any],
                      total_faces: Optional[int] = None) -> List[Violation]:
    """Validate an ``aicad.part.metadata.v1`` metadata mapping.

    ``total_faces``, when given, enables the "do not tag every generic face"
    rule: a single tag claiming every face of the part is flagged.
    """
    violations: List[Violation] = []

    schema = metadata.get("schema")
    if schema != METADATA_SCHEMA:
        violations.append(Violation(
            where="metadata.schema", kind="wrong-schema",
            message=f"expected '{METADATA_SCHEMA}', got {schema!r}",
            fix=f"set schema to '{METADATA_SCHEMA}'"))

    units = metadata.get("units")
    if units != "mm":
        violations.append(Violation(
            where="metadata.units", kind="wrong-units",
            message=f"units must be 'mm', got {units!r}",
            fix="model in millimetres"))

    anchors = metadata.get("anchors", {})
    if not isinstance(anchors, Mapping):
        violations.append(Violation(
            where="metadata.anchors", kind="bad-anchors",
            message="anchors must map names to 3-tuples"))
    else:
        for name, point in anchors.items():
            ok = (isinstance(point, (list, tuple)) and len(point) == 3
                  and all(isinstance(v, (int, float)) for v in point))
            if not ok:
                violations.append(Violation(
                    where=f"metadata.anchors[{name}]", kind="bad-anchor",
                    message=f"anchor '{name}' is not a numeric 3-tuple"))

    tags = metadata.get("features", [])
    seen_names: set = set()
    for index, tag in enumerate(tags):
        where = f"metadata.features[{index}]"
        name = str(tag.get("name", "")).strip()
        if not name:
            violations.append(Violation(where=where, kind="unnamed-tag",
                                         message="feature tag has no name"))
        elif name in seen_names:
            violations.append(Violation(where=where, kind="duplicate-tag",
                                         message=f"duplicate tag name '{name}'"))
        seen_names.add(name)
        kind = str(tag.get("kind", "")).strip()
        if kind not in TAG_KINDS:
            violations.append(Violation(
                where=where, kind="unknown-kind",
                message=f"tag kind '{kind}' is not a known functional kind",
                fix=f"use one of {TAG_KINDS}"))
        selector = str(tag.get("selector", ""))
        err = validate_selector_copy_form(selector)
        if err:
            violations.append(Violation(
                where=where, kind="selector-not-copy-form", message=err,
                fix="omit the part argument so the selector matches "
                    "auto-synthesized selectors"))
        face_count = tag.get("face_count")
        if (total_faces and isinstance(face_count, int)
                and face_count >= total_faces > 1):
            violations.append(Violation(
                where=where, kind="tags-everything",
                message=(f"tag '{name}' claims {face_count} of {total_faces} "
                         "faces; tags are for functional topology, not "
                         "every generic face")))
    return violations


# ---------------------------------------------------------------------------
# 2. Parameter placement
# ---------------------------------------------------------------------------

def validate_params_placement(root_params: Mapping[str, Any],
                              part_params: Mapping[str, Mapping[str, Any]],
                              ) -> List[Violation]:
    """Root params carry assembly-level values only; geometry knobs live
    beside their part. ``part_params`` maps part name -> its params.json."""
    violations: List[Violation] = []
    for key in root_params:
        if key in ASSEMBLY_LEVEL_KEYS:
            continue
        if _KNOB_RE.search(key):
            violations.append(Violation(
                where=f"params.json[{key}]", kind="knob-in-root",
                message=(f"'{key}' looks like a part geometry knob; local "
                         "part geometry does not belong in root params"),
                fix="move it to models/<model>/parts/<part>/params.json"))
    for part, params in part_params.items():
        for key in params:
            if key in ASSEMBLY_LEVEL_KEYS:
                violations.append(Violation(
                    where=f"parts/{part}/params.json[{key}]",
                    kind="assembly-key-in-part",
                    message=(f"assembly-level key '{key}' inside part params"),
                    fix="keep placement/motion/constraints/anchors/__viewer "
                        "in the root params.json"))
    return violations


# ---------------------------------------------------------------------------
# 3. Viewer isolation
# ---------------------------------------------------------------------------

def validate_viewer_materials(viewer: Mapping[str, Any],
                              instance_labels: Sequence[str] = (),
                              ) -> List[Violation]:
    """Validate a ``__viewer`` block: material value shapes and part keys."""
    violations: List[Violation] = []
    materials = viewer.get("materials", {})
    if not isinstance(materials, Mapping):
        return [Violation(where="__viewer.materials", kind="bad-materials",
                          message="materials must be a mapping")]

    def check_value(where: str, value: Any) -> None:
        if isinstance(value, str):
            if _HEX_COLOR_RE.match(value):
                return
            if value in VIEWER_PRESETS:
                violations.append(Violation(
                    where=where, kind="bare-preset-string",
                    message=(f"preset '{value}' given as a bare string; bare "
                             "strings are reserved for explicit colors"),
                    fix=f'use {{"preset": "{value}"}}'))
            else:
                violations.append(Violation(
                    where=where, kind="bad-color",
                    message=f"'{value}' is neither '#rrggbb' nor a preset object"))
            return
        if isinstance(value, Mapping):
            preset = value.get("preset")
            if preset is not None and preset not in VIEWER_PRESETS:
                violations.append(Violation(
                    where=where, kind="unknown-preset",
                    message=f"unknown preset '{preset}'",
                    fix=f"use one of {VIEWER_PRESETS}"))
            color = value.get("color")
            if color is not None and not (isinstance(color, str)
                                          and _HEX_COLOR_RE.match(color)):
                violations.append(Violation(
                    where=where, kind="bad-color",
                    message=f"color {color!r} is not '#rrggbb'"))
            if preset is None and color is None:
                violations.append(Violation(
                    where=where, kind="empty-material",
                    message="material object has neither preset nor color"))
            return
        violations.append(Violation(
            where=where, kind="bad-material-type",
            message=f"material must be a string or object, got "
                    f"{type(value).__name__}"))

    for key, value in materials.items():
        if key == "parts" and isinstance(value, Mapping):
            labels = set(instance_labels)
            for part_key, part_value in value.items():
                if labels and part_key not in labels:
                    violations.append(Violation(
                        where=f"__viewer.materials.parts[{part_key}]",
                        kind="label-mismatch",
                        message=(f"'{part_key}' matches no placed instance "
                                 "label"),
                        fix="keys must match assembly.py instance labels, "
                            "not parts/<name>/ folder names"))
                check_value(f"__viewer.materials.parts[{part_key}]", part_value)
        else:
            check_value(f"__viewer.materials[{key}]", value)
    return violations


# ---------------------------------------------------------------------------
# Selfcheck
# ---------------------------------------------------------------------------

def _selfcheck() -> int:
    failures: List[str] = []

    def check(cond: bool, message: str) -> None:
        if not cond:
            failures.append(message)

    # Selector copy form.
    check(validate_selector_copy_form("holes(radius=3, axis=Axis.Z)") is None,
          "keyword-first selector accepted")
    check(validate_selector_copy_form("holes()") is None, "empty args accepted")
    check(validate_selector_copy_form("holes(part, radius=3)") is not None,
          "positional part rejected")
    check(validate_selector_copy_form("holes(result)") is not None,
          "bare result rejected")
    check(validate_selector_copy_form("faces_of_interest") is None,
          "non-call selectors pass through")

    # Metadata: the skill's own example validates cleanly.
    good_meta = {
        "schema": METADATA_SCHEMA, "units": "mm",
        "anchors": {"origin": (0.0, 0.0, 0.0), "pin": (10.0, 0.0, 4.0)},
        "features": [
            {"name": "mounting_holes", "kind": "hole",
             "selector": "holes(radius=3, axis=Axis.Z)", "face_count": 4},
        ],
    }
    check(validate_metadata(good_meta, total_faces=26) == [],
          "skill example metadata passes")

    bad_meta = {
        "schema": "aicad.part.metadata.v2", "units": "in",
        "anchors": {"origin": (0.0, 0.0)},
        "features": [
            {"name": "", "kind": "swoosh", "selector": "holes(part)"},
            {"name": "everything", "kind": "generic",
             "selector": "faces()", "face_count": 26},
        ],
    }
    kinds = {v.kind for v in validate_metadata(bad_meta, total_faces=26)}
    for expected in ("wrong-schema", "wrong-units", "bad-anchor",
                     "unnamed-tag", "unknown-kind", "selector-not-copy-form",
                     "tags-everything"):
        check(expected in kinds, f"metadata violation '{expected}' detected")

    # Params placement.
    violations = validate_params_placement(
        root_params={"placement": {}, "anchors": {}, "__viewer": {},
                     "gear_teeth": 24},
        part_params={"gear": {"teeth": 24, "placement": {}}})
    pk = {(v.kind, v.where) for v in violations}
    check(("knob-in-root", "params.json[gear_teeth]") in pk,
          "geometry knob in root flagged")
    check(("assembly-key-in-part", "parts/gear/params.json[placement]") in pk,
          "assembly key in part flagged")
    check(validate_params_placement(
        {"placement": {}, "__viewer": {}}, {"gear": {"teeth": 24}}) == [],
        "correct placement passes")

    # Viewer materials.
    good_viewer = {"materials": {
        "default": {"preset": "painted_metal", "color": "#2f80ed"},
        "parts": {"front_arm": {"preset": "rubber"}, "body": "#ff0000"},
    }}
    check(validate_viewer_materials(good_viewer,
                                    ["front_arm", "body"]) == [],
          "good viewer block passes")

    bad_viewer = {"materials": {
        "default": "painted_metal",                 # bare preset string
        "parts": {"arm": {"preset": "chrome"},      # unknown preset, bad label
                  "body": {}},                      # empty material
    }}
    vkinds = {v.kind for v in validate_viewer_materials(bad_viewer, ["body"])}
    for expected in ("bare-preset-string", "unknown-preset", "label-mismatch",
                     "empty-material"):
        check(expected in vkinds, f"viewer violation '{expected}' detected")

    if failures:
        for f in failures:
            print(f"selfcheck FAIL: {f}")
        return 1
    print("part_metadata_contract selfcheck: OK")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="aicad part-metadata and params-placement contract "
                    "(forgent3d)")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args(argv)
    if args.selfcheck:
        return _selfcheck()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
