"""OpenSCAD Customizer parameter extraction (deterministic, stdlib-only).

Ported from ``shared/parseParameters.ts`` in CADAM (Adam-CAD's open-source
text-to-CAD web app). CADAM's core UX trick: the LLM emits *only* OpenSCAD
source, and the editable parameter panel (sliders, dropdowns, checkboxes) is
derived *deterministically* from the variable declarations at the top of the
file, using the OpenSCAD Customizer annotation vocabulary. Deriving the schema
from the code (instead of asking the model for a parameter list) guarantees the
same source always yields the same UI, regardless of which model produced it.

The harness had no OpenSCAD-source parameter extractor, so this supplies one.

Annotation vocabulary (recognised on the declaration line's trailing comment)::

    // Description of the parameter        <- description on the line ABOVE
    name = 10;  // [1:50]                  <- min:max slider
    name = 10;  // [1:1:50]                <- min:step:max slider
    name = "red";  // [red, green, blue]   <- enum options
    name = "x";  // [a:Apple, b:Banana]    <- enum value:label pairs
    name = 10;  // 5                       <- bare number: step (num) / maxLength (str)
    /* [Group Name] */                     <- starts a new group section

Only declarations ABOVE the first ``module``/``function`` keyword are exposed
(everything below is treated as implementation, not API). A ``name = [a, b, c]``
numeric vector is flattened into scalar entries ``name[0]``, ``name[1]``, ...
with axis-aware labels (X/Y, Width/Depth/Height, or 1..N).

Deterministic: pure regex/string processing, no clock, no randomness. Output is
a list of ``Parameter`` in source order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Declaration: name = value ; optional trailing // comment
_PARAM_RE = re.compile(
    r"^([a-z0-9A-Z_$]+)\s*=\s*([^;]+);[\t\f\x0b ]*(//[^\n]*)?", re.MULTILINE
)
_GROUP_RE = re.compile(r"^/\*\s*\[([^\]]+)\]\s*\*/", re.MULTILINE)
_MODULE_SPLIT_RE = re.compile(r"^(module |function )", re.MULTILINE)


@dataclass
class ParameterOption:
    value: Any
    label: Optional[str] = None


@dataclass
class Parameter:
    name: str
    display_name: str
    value: Any
    type: str
    default_value: Any
    group: str = ""
    description: Optional[str] = None
    range: Dict[str, float] = field(default_factory=dict)
    options: List[ParameterOption] = field(default_factory=list)


def _convert_type(raw: str):
    """Infer (value, type) from an OpenSCAD literal, or None if not a constant."""
    raw = raw.strip()
    if re.fullmatch(r"-?\d+(\.\d+)?", raw):
        return float(raw), "number"
    if raw in ("true", "false"):
        return raw == "true", "boolean"
    if re.fullmatch(r'".*"', raw, re.DOTALL):
        return raw[1:-1], "string"
    if raw.startswith("[") and raw.endswith("]"):
        items = [it.strip() for it in raw[1:-1].split(",")]
        if items and all(re.fullmatch(r"-?\d+(\.\d+)?", it) for it in items):
            return [float(it) for it in items], "number[]"
        if items and all(re.fullmatch(r'".*"', it) for it in items):
            return [it[1:-1] for it in items], "string[]"
        if items and all(it in ("true", "false") for it in items):
            return [it == "true" for it in items], "boolean[]"
        return None
    return None


def _number_array_labels(name: str, length: int) -> List[str]:
    if length == 2:
        return ["X", "Y"]
    if length != 3:
        return [str(i + 1) for i in range(length)]
    lower = name.lower()
    if any(k in lower for k in ("size", "dimension", "body", "torso", "head", "foot", "base")):
        return ["Width", "Depth", "Height"]
    return ["X", "Y", "Z"]


def _display_name_for_array_item(display_name: str, label: str) -> str:
    if label in ("Width", "Depth", "Height"):
        return re.sub(r"\s+Size$", "", display_name, flags=re.IGNORECASE) + f" {label}"
    return f"{display_name} {label}"


def _title_case(name: str) -> str:
    if name == "$fn":
        return "Resolution"
    words = [w for w in name.replace("_", " ").split(" ") if w]
    return " ".join(w[0].upper() + w[1:] for w in words)


def _parse_comment(comment: str, ptype: str):
    """Parse a trailing ``// ...`` annotation into (range, options)."""
    range_: Dict[str, float] = {}
    options: List[ParameterOption] = []
    raw = re.sub(r"^//\s*", "", comment).strip()
    cleaned = re.sub(r"^\[+|\]+$", "", raw)

    # Bare number -> step (numeric) or maxLength (string).
    if _is_number(raw):
        if ptype == "string":
            range_ = {"max": float(cleaned)}
        else:
            range_ = {"step": float(cleaned)}
    elif raw.startswith("[") and "," in cleaned:
        # Enum options: `[a, b:Label, c]`.
        for opt in cleaned.split(","):
            parts = opt.strip().split(":")
            val: Any = parts[0]
            label = parts[1] if len(parts) > 1 else None
            if ptype == "number":
                try:
                    val = float(val)
                except ValueError:
                    pass
            options.append(ParameterOption(value=val, label=label))
    elif re.search(r"([0-9]+:?)+", cleaned):
        # Slider bounds: `[min:max]` or `[min:step:max]`.
        parts = cleaned.strip().split(":")
        mn = parts[0] if len(parts) > 0 else ""
        mid = parts[1] if len(parts) > 1 else ""
        mx = parts[2] if len(parts) > 2 else ""
        if mn and (mid or mx):
            range_ = {"min": float(mn)}
        if mx or mid or mn:
            range_["max"] = float(mx or mid or mn)
        if mx and mid:
            range_["step"] = float(mid)
    return range_, options


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_parameters(script: str) -> List[Parameter]:
    """Extract Customizer parameters from OpenSCAD source (top-of-file only)."""
    # Discard everything from the first module/function declaration onward.
    script = _MODULE_SPLIT_RE.split(script)[0]

    # Build group sections keyed by source offset.
    sections = [{"start": 0, "group": "", "code": ""}]
    for m in _GROUP_RE.finditer(script):
        sections.append({"start": m.start(), "group": m.group(1).strip(), "code": ""})
    for i, sec in enumerate(sections):
        end = sections[i + 1]["start"] if i + 1 < len(sections) else len(script)
        sec["code"] = script[sec["start"]:end]

    parameters: "Dict[str, Parameter]" = {}
    for sec in sections:
        for match in _PARAM_RE.finditer(sec["code"]):
            name = match.group(1)
            value = match.group(2).strip()
            conv = _convert_type(value)
            if conv is None:
                continue
            pvalue, ptype = conv

            # Skip values that reference another variable or span lines.
            if value not in ("true", "false") and (
                re.match(r"^[a-zA-Z_]", value) or len(value.split("\n")) > 1
            ):
                continue

            range_: Dict[str, float] = {}
            options: List[ParameterOption] = []
            if match.group(3):
                range_, options = _parse_comment(match.group(3), ptype)

            # Description: comment on the line immediately above the declaration.
            description = None
            above = re.split(
                r"^" + re.escape(match.group(0)), script, maxsplit=1, flags=re.MULTILINE
            )[0]
            if above.endswith("\n"):
                above = above[:-1]
            last_line = above.split("\n")[-1] if above else ""
            if last_line.strip().startswith("//"):
                description = re.sub(r"^//+\s*", "", last_line.strip())
                if not description:
                    description = None

            display_name = _title_case(name)

            # Flatten numeric vectors into scalar sliders.
            if ptype == "number[]" and isinstance(pvalue, list):
                labels = _number_array_labels(name, len(pvalue))
                for idx, item in enumerate(pvalue):
                    key = f"{name}[{idx}]"
                    parameters[key] = Parameter(
                        name=key,
                        display_name=_display_name_for_array_item(display_name, labels[idx]),
                        value=item,
                        type="number",
                        default_value=item,
                        group=sec["group"],
                        description=description,
                        range=dict(range_),
                        options=list(options),
                    )
                continue

            parameters[name] = Parameter(
                name=name,
                display_name=display_name,
                value=pvalue,
                type=ptype,
                default_value=pvalue,
                group=sec["group"],
                description=description,
                range=range_,
                options=options,
            )

    return list(parameters.values())
