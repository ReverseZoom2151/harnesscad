"""gdt_prompts -- deterministic prompt builders for staged GD&T drawing analysis.

Provides:

  * ``GDT_SYSTEM_PROMPT`` -- the staged GD&T extraction system prompt that
    embeds the enriched-annotation JSON schema (5 annotation types, percentage
    bounding boxes, confidence rules, color-per-type convention);
  * ``BASELINE_SYSTEM_PROMPT`` / ``DETAILED_SYSTEM_PROMPT`` -- simpler and
    thorough generic drawing-analysis prompts;
  * ``build_focused_requery_prompt`` -- the per-type focused re-examination
    prompt used by the confidence-gated re-query stage
    (harnesscad.domain.drawings.requery).

Harness gap filled: harnesscad had no vision-prompt layer for extracting
structured annotations from drawing images. The schema the prompts describe is
parsed by harnesscad.domain.drawings.annotation_schema; per-frame GD&T
validity checks already live in harnesscad.domain.drawings.gdt and are not
duplicated here. Provider-specific bits (model names, token limits, image
payload plumbing) are stripped: this module builds strings only, no LLM calls.
"""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

# --------------------------------------------------------------------------- #
# Staged GD&T extraction system prompt (gdt-prompts.ts GDT_SYSTEM_PROMPT)
# --------------------------------------------------------------------------- #

GDT_SYSTEM_PROMPT = """You are an expert GD&T (Geometric Dimensioning and Tolerancing) analyzer for engineering drawings. Carefully analyze the provided CAD drawing image and extract ALL GD&T annotations, dimensions, feature control frames, datums, surface finish symbols, and notes.

For each annotation, you MUST:
1. Classify it into exactly one type: "dimension", "fcf", "datum", "surface_finish", or "note"
2. Extract type-specific sub-fields as described below
3. Assign a confidence score between 0.0 and 1.0 based on image clarity and your certainty
4. Provide bounding box coordinates as percentages (0-100) of the image dimensions

Return a JSON object with this exact structure:
{
  "annotations": [ ... ],
  "views": ["View 1", "View 2"],
  "description": "Overall description of the drawing"
}

Each annotation MUST follow one of these type schemas:

DIMENSION annotation:
{
  "id": "ann_1",
  "type": "dimension",
  "label": "40.2 +/-0.1",
  "value": "40.2",
  "view": "Front View",
  "boundingBox": { "x": 10, "y": 20, "width": 15, "height": 8, "color": "green" },
  "confidence": 0.95,
  "dimensionType": "linear",
  "nominalValue": 40.2,
  "plusTolerance": 0.1,
  "minusTolerance": -0.1,
  "unit": "mm"
}
dimensionType must be one of: "linear", "angular", "radius", "diameter"

FCF (Feature Control Frame) annotation:
{
  "id": "ann_2",
  "type": "fcf",
  "label": "Position 0.05 MMC A B C",
  "value": "0.05",
  "view": "Front View",
  "boundingBox": { "x": 30, "y": 40, "width": 20, "height": 6, "color": "blue" },
  "confidence": 0.88,
  "geometricCharacteristic": "position",
  "toleranceValue": 0.05,
  "materialCondition": "MMC",
  "datumReferences": ["A", "B", "C"]
}
geometricCharacteristic must be one of: "position", "flatness", "straightness", "circularity", "cylindricity", "perpendicularity", "parallelism", "angularity", "profileOfLine", "profileOfSurface", "circularRunout", "totalRunout", "symmetry", "concentricity"
materialCondition must be one of: "MMC", "LMC", "RFS", or null
datumReferences is an ordered array of up to 3 uppercase letters (A-Z)

DATUM annotation:
{
  "id": "ann_3",
  "type": "datum",
  "label": "Datum A",
  "value": "A",
  "view": "Front View",
  "boundingBox": { "x": 50, "y": 60, "width": 5, "height": 5, "color": "red" },
  "confidence": 0.97,
  "datumLetter": "A"
}
datumLetter must be a single uppercase letter A-Z

SURFACE_FINISH annotation:
{
  "id": "ann_4",
  "type": "surface_finish",
  "label": "Ra 1.6",
  "value": "1.6",
  "view": "Front View",
  "boundingBox": { "x": 70, "y": 30, "width": 8, "height": 8, "color": "orange" },
  "confidence": 0.82,
  "roughnessValue": 1.6,
  "processNote": "Ground"
}

NOTE annotation:
{
  "id": "ann_5",
  "type": "note",
  "label": "UNLESS OTHERWISE SPECIFIED",
  "value": "UNLESS OTHERWISE SPECIFIED, DIMENSIONS ARE IN MM",
  "view": "Title Block",
  "boundingBox": { "x": 5, "y": 90, "width": 30, "height": 5, "color": "purple" },
  "confidence": 0.90
}

Rules:
- boundingBox x, y are the top-left corner as percentages (0-100) of image dimensions
- boundingBox width, height are dimensions as percentages
- Use different colors for different annotation types: green for dimensions, blue for FCFs, red for datums, orange for surface finish, purple for notes
- Extract ALL visible annotations, aim for completeness
- Set confidence lower (< 0.6) when the symbol is partially obscured, blurry, or ambiguous
- Only return valid JSON, no other text"""


# --------------------------------------------------------------------------- #
# Generic drawing-analysis prompts (routes/analyze.ts)
# --------------------------------------------------------------------------- #

BASELINE_SYSTEM_PROMPT = """You are a CAD drawing analyzer. Analyze the provided engineering drawing and extract key annotations and measurements.
Focus only on the most prominent dimensions and labels.
Return a JSON object with this exact structure:
{
  "annotations": [
    {
      "id": "ann_1",
      "label": "R3.2 TYP.",
      "value": "R3.2",
      "view": "View 1",
      "boundingBox": { "x": 10, "y": 20, "width": 15, "height": 8, "color": "green" },
      "description": "optional description"
    }
  ],
  "views": ["View 1", "View 2"],
  "description": "optional overall description"
}
The boundingBox coordinates are percentages of the image dimensions (0-100).
x and y are the top-left corner. width and height are the dimensions.
Use different colors for different sections: green, blue, red, orange, purple.
Only return valid JSON, no other text."""

DETAILED_SYSTEM_PROMPT = """You are an expert CAD drawing analyzer. Carefully analyze the provided engineering drawing and extract ALL annotations, dimensions, measurements, tolerances, notes, and labels.
For each annotation, identify its location in the image as a bounding box (percentage coordinates).
Group annotations by which view they belong to (e.g., "View 1", "View 2", or "Top View", "Front View", etc.).
Return a JSON object with this exact structure:
{
  "annotations": [
    {
      "id": "ann_1",
      "label": "R3.2 TYP.",
      "value": "R3.2",
      "view": "View 1",
      "boundingBox": { "x": 10, "y": 20, "width": 15, "height": 8, "color": "green" },
      "description": "Typical radius of 3.2mm"
    }
  ],
  "views": ["View 1", "View 2"],
  "description": "Overall description of the drawing"
}

Rules:
- The boundingBox coordinates are percentages of the image dimensions (0-100)
- x and y are the top-left corner, width and height are the dimensions of the bounding box
- Use different colors for different sections/views: green for main views, blue for detail views, red for notes/title block, orange for additional views, purple for reference dimensions
- Extract at minimum 6-12 annotations from complex drawings
- label should be a short identifier (e.g. "R3.2 TYP.", "40.2", "M6x1.0")
- value should be the extracted numeric or text value
- Only return valid JSON, no other text."""


# --------------------------------------------------------------------------- #
# Focused re-query prompt builder (gdt-prompts.ts buildFocusedReQueryPrompt)
# --------------------------------------------------------------------------- #

_TYPE_SPECIFIC_INSTRUCTIONS = {
    "dimension": (
        "This appears to be a DIMENSION annotation.\n"
        "Extract: dimensionType (linear|angular|radius|diameter), nominalValue, "
        "plusTolerance, minusTolerance, unit."
    ),
    "fcf": (
        "This appears to be a FEATURE CONTROL FRAME (FCF) annotation.\n"
        "Extract: geometricCharacteristic (position|flatness|straightness|"
        "circularity|cylindricity|perpendicularity|parallelism|angularity|"
        "profileOfLine|profileOfSurface|circularRunout|totalRunout|symmetry|"
        "concentricity), toleranceValue, materialCondition (MMC|LMC|RFS|null), "
        "datumReferences (array of up to 3 uppercase letters)."
    ),
    "datum": (
        "This appears to be a DATUM annotation.\n"
        "Extract: datumLetter (a single uppercase letter A-Z)."
    ),
    "surface_finish": (
        "This appears to be a SURFACE FINISH annotation.\n"
        "Extract: roughnessValue (number), processNote (optional text)."
    ),
    "note": (
        "This appears to be a NOTE annotation.\n"
        "Extract the text content of the note."
    ),
}


def build_focused_requery_prompt(type_hint: str, label: str, value: str) -> str:
    """Build the focused re-examination prompt for a single cropped annotation.

    Mirrors buildFocusedReQueryPrompt in gdt-prompts.ts: embeds the initial
    detection (type, label, value) plus per-type extraction instructions, and
    asks the model to re-read the cropped region and return a single JSON
    object with a clamped confidence.
    """
    type_specific = _TYPE_SPECIFIC_INSTRUCTIONS.get(type_hint, "")

    return (
        "You are an expert GD&T (Geometric Dimensioning and Tolerancing) symbol "
        "reader. You are re-examining a specific cropped region of an "
        "engineering drawing that was initially detected as a GD&T annotation.\n"
        "\n"
        "Initial detection:\n"
        "- Type: %s\n"
        '- Label: "%s"\n'
        '- Value: "%s"\n'
        "\n"
        "%s\n"
        "\n"
        "Carefully examine this cropped image and provide an accurate reading "
        "of the GD&T annotation. Return a JSON object with this exact "
        "structure:\n"
        "{\n"
        '  "type": "%s",\n'
        '  "label": "the annotation label text",\n'
        '  "value": "the annotation value",\n'
        '  "confidence": 0.85,\n'
        "  ... type-specific fields as described above\n"
        "}\n"
        "\n"
        "Rules:\n"
        "- confidence must be between 0.0 and 1.0, reflecting your certainty "
        "in the reading\n"
        "- If you cannot read the annotation clearly, set confidence below 0.5\n"
        "- Only return valid JSON, no other text\n"
        '- Keep the same "type" as the initial detection unless you are very '
        "confident it is a different type"
    ) % (type_hint, label, value, type_specific, type_hint)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. ``--selfcheck`` builds every prompt on synthetic data
    and asserts key structural markers are present."""
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.domain.drawings.gdt_prompts",
        description="Deterministic GD&T / drawing-analysis prompt builders "
        ". No LLM calls.",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="build all prompts on synthetic inputs and verify structural "
        "markers.",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit prompt texts as JSON."
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.selfcheck:
        parser.print_help()
        return 0

    requery_prompts = {
        t: build_focused_requery_prompt(t, "Position 0.05 A B", "0.05")
        for t in ("dimension", "fcf", "datum", "surface_finish", "note")
    }

    if args.json:
        print(
            json.dumps(
                {
                    "gdt_system_prompt": GDT_SYSTEM_PROMPT,
                    "baseline_system_prompt": BASELINE_SYSTEM_PROMPT,
                    "detailed_system_prompt": DETAILED_SYSTEM_PROMPT,
                    "requery_prompts": requery_prompts,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print("GDT_SYSTEM_PROMPT: %d chars" % len(GDT_SYSTEM_PROMPT))
        print("BASELINE_SYSTEM_PROMPT: %d chars" % len(BASELINE_SYSTEM_PROMPT))
        print("DETAILED_SYSTEM_PROMPT: %d chars" % len(DETAILED_SYSTEM_PROMPT))
        for t, p in requery_prompts.items():
            print("requery prompt (%s): %d chars" % (t, len(p)))

    fcf_prompt = requery_prompts["fcf"]
    ok = (
        '"surface_finish"' in GDT_SYSTEM_PROMPT
        and "percentages (0-100)" in GDT_SYSTEM_PROMPT
        and "datumReferences is an ordered array of up to 3 uppercase letters"
        in GDT_SYSTEM_PROMPT
        and "most prominent dimensions" in BASELINE_SYSTEM_PROMPT
        and "6-12 annotations" in DETAILED_SYSTEM_PROMPT
        and "- Type: fcf" in fcf_prompt
        and 'Label: "Position 0.05 A B"' in fcf_prompt
        and "geometricCharacteristic" in fcf_prompt
        and '"type": "fcf"' in fcf_prompt
        and "datumLetter" in requery_prompts["datum"]
        and "roughnessValue" in requery_prompts["surface_finish"]
        and all(p == p for p in requery_prompts.values())
        and build_focused_requery_prompt("fcf", "x", "y")
        == build_focused_requery_prompt("fcf", "x", "y")  # deterministic
    )
    if not ok:
        print("SELFCHECK FAILED")
        return 1
    print("selfcheck OK")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
