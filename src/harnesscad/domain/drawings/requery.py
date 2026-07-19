"""requery -- confidence-gated re-query of low-confidence drawing annotations.

When a vision model returns an annotation with confidence below 0.6, the
pipeline crops the bounding-box region from the original image (with 15 percent
padding), re-asks the model with a focused per-type prompt, and applies a decision rule: replace the
original when the re-query is confident enough, otherwise keep the
higher-confidence reading flagged for human review.

The implementation keeps every pure part -- percent-to-pixel crop geometry, the
confidence threshold, the replace-or-flag decision rule, and the response
merge that preserves the original id and bounding box -- and injects the only
impure part (the vision call) as an optional callable. Without a vision
callable, low-confidence annotations are simply flagged needs_review.

Harness gap filled: harnesscad had no self-correction loop over extracted
annotations. Prompts come from harnesscad.domain.drawings.gdt_prompts and
tolerant parsing from harnesscad.domain.drawings.annotation_schema; per-frame
GD&T checks live in harnesscad.domain.drawings.gdt and are not duplicated.

Pure stdlib, deterministic core; no image processing (the caller owns pixels).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace as dc_replace
from typing import Callable, List, Optional, Sequence

from harnesscad.domain.drawings.annotation_schema import (
    Annotation,
    BoundingBox,
    extract_json_blob,
    parse_annotation,
)
from harnesscad.domain.drawings.gdt_prompts import build_focused_requery_prompt

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Confidence threshold below which annotations are re-queried.
REQUERY_CONFIDENCE_THRESHOLD = 0.6

#: Padding fraction applied to each side of the bounding box crop.
CROP_PADDING = 0.15


# --------------------------------------------------------------------------- #
# Crop geometry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CropRegion:
    """A pixel-space crop rectangle (left, top, width, height)."""

    left: int
    top: int
    width: int
    height: int

    def to_dict(self) -> dict:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


def compute_crop_region(
    bbox_percent: BoundingBox,
    image_width: int,
    image_height: int,
    padding: float = CROP_PADDING,
) -> CropRegion:
    """Compute the pixel crop region for a percentage bounding box.

    Mirrors computeCropRegion in requery-service.ts exactly: convert
    percentages to pixels, pad each side by ``padding`` times the bbox
    dimension, clamp to image bounds (floor/ceil like the TS), and guarantee
    at least 1 px in each dimension.
    """
    bbox_left_px = (bbox_percent.x / 100.0) * image_width
    bbox_top_px = (bbox_percent.y / 100.0) * image_height
    bbox_width_px = (bbox_percent.width / 100.0) * image_width
    bbox_height_px = (bbox_percent.height / 100.0) * image_height

    pad_x = bbox_width_px * padding
    pad_y = bbox_height_px * padding

    expanded_left = bbox_left_px - pad_x
    expanded_top = bbox_top_px - pad_y
    expanded_right = bbox_left_px + bbox_width_px + pad_x
    expanded_bottom = bbox_top_px + bbox_height_px + pad_y

    clamped_left = max(0, math.floor(expanded_left))
    clamped_top = max(0, math.floor(expanded_top))
    clamped_right = min(image_width, math.ceil(expanded_right))
    clamped_bottom = min(image_height, math.ceil(expanded_bottom))

    return CropRegion(
        left=clamped_left,
        top=clamped_top,
        width=max(1, clamped_right - clamped_left),
        height=max(1, clamped_bottom - clamped_top),
    )


# --------------------------------------------------------------------------- #
# Decision rule
# --------------------------------------------------------------------------- #


def apply_requery_decision(original: Annotation, requery_result: Annotation) -> Annotation:
    """Choose which annotation to keep after a re-query attempt.

    * If the re-query result has confidence >= REQUERY_CONFIDENCE_THRESHOLD,
      replace the original.
    * Otherwise keep whichever has higher confidence (ties favour the
      re-query result, as in the TS) and set needs_review = True.
    """
    if requery_result.confidence >= REQUERY_CONFIDENCE_THRESHOLD:
        return requery_result
    if requery_result.confidence >= original.confidence:
        return dc_replace(requery_result, needs_review=True)
    return dc_replace(original, needs_review=True)


# --------------------------------------------------------------------------- #
# Re-query loop
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReQueryResult:
    """One annotation after the re-query stage."""

    annotation: Annotation
    was_requeried: bool

    def to_dict(self) -> dict:
        return {
            "annotation": self.annotation.to_dict(),
            "was_requeried": self.was_requeried,
        }


def _merge_requery_response(original: Annotation, content: str) -> Optional[Annotation]:
    """Merge a raw re-query response onto the original annotation.

    Mirrors the TS spread ``{...annotation, ...parsed}``: the original's
    fields (via to_dict, snake_case) act as the base, the parsed response
    (camelCase) overrides, the original id and bounding box are always
    preserved, and a missing/invalid confidence falls back to the original's
    (the tolerant parser clamps it). Returns None when the response carries
    no parseable JSON object or the merged annotation fails validation.
    """
    parsed = extract_json_blob(content)
    if parsed is None:
        return None

    merged = dict(original.to_dict())
    merged.update(parsed)
    merged["id"] = original.id
    merged["type"] = original.type if not isinstance(parsed.get("type"), str) else parsed["type"]
    # Fall back to the original confidence when the response omits a number.
    raw_conf = parsed.get("confidence")
    if isinstance(raw_conf, bool) or not isinstance(raw_conf, (int, float)):
        merged["confidence"] = original.confidence
    # Keep parsing tolerant of a bad bounding box by forcing the original's.
    merged["bounding_box"] = original.bounding_box.to_dict()
    merged.pop("boundingBox", None)

    candidate = parse_annotation(merged, 0)
    if candidate is None:
        return None
    return dc_replace(candidate, id=original.id, bounding_box=original.bounding_box)


def requery_low_confidence(
    annotations: Sequence[Annotation],
    image_width: int,
    image_height: int,
    vision: Optional[Callable[[CropRegion, str], str]] = None,
) -> List[ReQueryResult]:
    """Re-query annotations whose confidence is below the threshold.

    ``vision`` is an optional callable ``(crop_region, prompt) -> raw JSON
    str``; the caller owns image cropping and the model call. For each
    low-confidence annotation:

    * without ``vision``: flag it needs_review = True, was_requeried = False;
    * with ``vision``: compute the padded crop region, build the focused
      per-type prompt, call the model once (max one attempt per annotation),
      merge the response preserving id and bounding box, and apply the
      decision rule. Any failure keeps the original flagged needs_review.
    """
    results: List[ReQueryResult] = []
    for annotation in annotations:
        if annotation.confidence >= REQUERY_CONFIDENCE_THRESHOLD:
            results.append(ReQueryResult(annotation=annotation, was_requeried=False))
            continue

        if vision is None:
            results.append(
                ReQueryResult(
                    annotation=dc_replace(annotation, needs_review=True),
                    was_requeried=False,
                )
            )
            continue

        try:
            crop_region = compute_crop_region(
                annotation.bounding_box, image_width, image_height, CROP_PADDING
            )
            prompt = build_focused_requery_prompt(
                annotation.type, annotation.label, annotation.value
            )
            content = vision(crop_region, prompt)
            requeried = _merge_requery_response(annotation, content)
            if requeried is None:
                results.append(
                    ReQueryResult(
                        annotation=dc_replace(annotation, needs_review=True),
                        was_requeried=True,
                    )
                )
                continue
            chosen = apply_requery_decision(annotation, requeried)
            results.append(ReQueryResult(annotation=chosen, was_requeried=True))
        except Exception:
            results.append(
                ReQueryResult(
                    annotation=dc_replace(annotation, needs_review=True),
                    was_requeried=True,
                )
            )
    return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _synthetic_annotations() -> List[Annotation]:
    from harnesscad.domain.drawings.annotation_schema import (
        DatumAnnotation,
        DimensionAnnotation,
    )

    return [
        DimensionAnnotation(
            id="ann_1",
            label="40.2 +/-0.1",
            value="40.2",
            view="Front View",
            bounding_box=BoundingBox(x=10, y=20, width=15, height=8, color="green"),
            confidence=0.95,
            dimension_type="linear",
            nominal_value=40.2,
            plus_tolerance=0.1,
            minus_tolerance=0.1,
            unit="mm",
        ),
        # low confidence -> re-queried
        DatumAnnotation(
            id="ann_2",
            label="Datum ?",
            value="?",
            view="Front View",
            bounding_box=BoundingBox(x=95, y=95, width=10, height=10, color="red"),
            confidence=0.4,
            datum_letter="Z",
        ),
        # low confidence, re-query also comes back weak
        DatumAnnotation(
            id="ann_3",
            label="Datum ??",
            value="??",
            view="Front View",
            bounding_box=BoundingBox(x=0, y=0, width=5, height=5, color="red"),
            confidence=0.3,
            datum_letter="Y",
        ),
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. ``--selfcheck`` runs the re-query loop on synthetic
    annotations both without a vision callable (flag-only path) and with a
    fake vision double returning one confident and one weak reading, and
    asserts crop clamping, decision-rule, and id/bbox preservation."""
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.domain.drawings.requery",
        description="Confidence-gated re-query of low-confidence annotations "
        ". Vision model injected, never required.",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="exercise the crop geometry, decision rule, and re-query loop "
        "on synthetic annotations with and without a fake vision callable.",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit re-query results as JSON."
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.selfcheck:
        parser.print_help()
        return 0

    annotations = _synthetic_annotations()
    image_w, image_h = 1000, 800

    # Path 1: no vision callable -- low-confidence annotations only flagged.
    flagged = requery_low_confidence(annotations, image_w, image_h)

    # Path 2: fake vision double. ann_2's crop returns a confident corrected
    # reading; ann_3's returns a still-weak one.
    def fake_vision(crop: CropRegion, prompt: str) -> str:
        if "??" in prompt:
            return json.dumps(
                {"type": "datum", "label": "Datum ?", "value": "?",
                 "confidence": 0.35, "datumLetter": "Y"}
            )
        return (
            "```json\n"
            + json.dumps(
                {"type": "datum", "label": "Datum B", "value": "B",
                 "confidence": 0.92, "datumLetter": "B",
                 "id": "should_be_ignored",
                 "boundingBox": {"x": 1, "y": 1, "width": 1, "height": 1}}
            )
            + "\n```"
        )

    requeried = requery_low_confidence(annotations, image_w, image_h, vision=fake_vision)

    crop = compute_crop_region(annotations[1].bounding_box, image_w, image_h)

    if args.json:
        print(
            json.dumps(
                {
                    "crop_ann_2": crop.to_dict(),
                    "no_vision": [r.to_dict() for r in flagged],
                    "with_vision": [r.to_dict() for r in requeried],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print("crop for ann_2 bbox (95,95,10,10 pct on 1000x800): %s" % (crop.to_dict(),))
        for label, results in (("no vision", flagged), ("with vision", requeried)):
            print("%s:" % label)
            for r in results:
                a = r.annotation
                print(
                    "  %-6s conf=%.2f needs_review=%s was_requeried=%s value=%s"
                    % (a.id, a.confidence, a.needs_review, r.was_requeried, a.value)
                )

    ann2 = requeried[1].annotation
    ann3 = requeried[2].annotation
    ok = (
        # crop clamped to image bounds despite bbox spilling past 100 pct
        crop.left == 935 and crop.top == 748
        and crop.left + crop.width <= image_w
        and crop.top + crop.height <= image_h
        # no-vision path: flag only, not marked requeried
        and flagged[0].annotation == annotations[0]
        and flagged[1].annotation.needs_review is True
        and flagged[1].was_requeried is False
        # high-confidence annotation untouched
        and requeried[0].annotation == annotations[0]
        and requeried[0].was_requeried is False
        # confident re-query replaces original, preserving id and bbox
        and ann2.id == "ann_2"
        and ann2.value == "B"
        and ann2.confidence == 0.92
        and ann2.bounding_box == annotations[1].bounding_box
        and requeried[1].was_requeried is True
        # weak re-query keeps higher-confidence reading flagged for review
        and ann3.needs_review is True
        and ann3.confidence == 0.35  # requery 0.35 >= original 0.3, requery wins
        and requeried[2].was_requeried is True
    )
    if not ok:
        print("SELFCHECK FAILED")
        return 1
    print("selfcheck OK")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
