"""Extract editable parameters from OpenSCAD source (Customizer annotation vocab).

This is a deterministic, stdlib-only parameter extractor. It answers
"what does this OpenSCAD model expose as an
adjustable input?" by reading the top-of-file variable declarations and their
Customizer-style trailing-comment annotations -- the same convention the OpenSCAD
Customizer GUI uses.

The recognised annotation vocabulary::

    // Description of the parameter
    name = 10;   // [1:50]                 -> min:max slider
    name = 10;   // [1:1:50]               -> min:step:max slider
    name = "red";// [red, green, blue]     -> enum options
    name = "red";// [r:Red, g:Green]       -> enum options with labels
    name = "hi"; // 20                      -> maxLength (string) / step (number)
    /* [Group Name] */                     -> starts a new group section

Rules enforced:

*   Only declarations *before* the first ``module`` or ``function`` keyword are
    exposed; everything after is implementation, not API.
*   Values that reference another variable, or span multiple lines, are treated
    as computed expressions and skipped (and end the current group scan).
*   ``name = [a, b, c]`` numeric vectors are flattened into scalar parameters
    ``name[0]``, ``name[1]``, ... with X/Y/Z (or Width/Depth/Height) labels.
*   ``$fn`` is given the display name "Resolution".

Deterministic: same source -> same ordered parameter list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

__all__ = ["Parameter", "ParameterOption", "parse_parameters"]

Scalar = Union[float, bool, str]

_PARAM_RE = re.compile(r"^([A-Za-z0-9_$]+)\s*=\s*([^;]+);[\t\f\v ]*(//[^\n]*)?", re.M)
_GROUP_RE = re.compile(r"^/\*\s*\[([^\]]+)\]\s*\*/", re.M)
_NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?$")


@dataclass(frozen=True)
class ParameterOption:
    value: Union[float, str]
    label: Optional[str] = None


@dataclass
class Parameter:
    name: str
    display_name: str
    type: str  # "number" | "boolean" | "string"
    value: Scalar
    default_value: Scalar
    group: str = ""
    description: Optional[str] = None
    options: List[ParameterOption] = field(default_factory=list)
    range: dict = field(default_factory=dict)  # keys: min / max / step


def _convert_type(raw: str) -> Optional[Tuple[Scalar, str]]:
    """(value, type) for a scalar/vector literal, or None for arrays that flatten.

    Raises ValueError for expressions that are not constants (handled by caller).
    Returns a special ("__array__", values) sentinel via a list for numeric vectors.
    """
    raw = raw.strip()
    if _NUMBER_RE.match(raw):
        return (float(raw), "number")
    if raw == "true" or raw == "false":
        return (raw == "true", "boolean")
    if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
        return (raw[1:-1], "string")
    if raw.startswith("[") and raw.endswith("]"):
        items = [it.strip() for it in raw[1:-1].split(",")]
        if items and all(_NUMBER_RE.match(it) for it in items):
            # signalled to caller via a list payload (numeric vector -> flatten)
            return ([float(it) for it in items], "number[]")  # type: ignore[return-value]
        raise ValueError(f"unsupported array literal: {raw}")
    raise ValueError(f"not a constant: {raw}")


def _number_array_labels(name: str, length: int) -> List[str]:
    if length == 2:
        return ["X", "Y"]
    if length != 3:
        return [str(i + 1) for i in range(length)]
    lower = name.lower()
    if any(k in lower for k in ("size", "dimension", "body", "torso", "head", "foot", "base")):
        return ["Width", "Depth", "Height"]
    return ["X", "Y", "Z"]


def _display_name(name: str) -> str:
    if name == "$fn":
        return "Resolution"
    words = [w for w in name.replace("_", " ").split(" ") if w]
    return " ".join(w[0].upper() + w[1:] for w in words)


def _parse_comment(raw_comment: str, ptype: str) -> Tuple[dict, List[ParameterOption]]:
    """Parse a Customizer trailing comment into (range, options)."""
    rng: dict = {}
    options: List[ParameterOption] = []
    cleaned = re.sub(r"^\[+|\]+$", "", raw_comment)
    # Bare number -> step (numeric) / maxLength (string).
    try:
        float(raw_comment)
        is_bare_number = True
    except ValueError:
        is_bare_number = False
    if is_bare_number:
        if ptype == "string":
            rng = {"max": float(cleaned)}
        else:
            rng = {"step": float(cleaned)}
        return rng, options
    if raw_comment.startswith("[") and "," in cleaned:
        for opt in cleaned.split(","):
            parts = opt.strip().split(":")
            val: Union[float, str] = parts[0]
            label = parts[1] if len(parts) > 1 else None
            if ptype == "number":
                try:
                    val = float(val)
                except ValueError:
                    pass
            options.append(ParameterOption(value=val, label=label))
        return rng, options
    if re.search(r"([0-9]+:?)+", cleaned):
        bits = cleaned.strip().split(":")
        min_s = bits[0] if len(bits) > 0 else ""
        mid_s = bits[1] if len(bits) > 1 else ""
        max_s = bits[2] if len(bits) > 2 else ""
        if min_s and (mid_s or max_s):
            rng["min"] = float(min_s)
        if max_s or mid_s or min_s:
            rng["max"] = float(max_s or mid_s or min_s)
        if max_s and mid_s:
            rng["step"] = float(mid_s)
    return rng, options


def parse_parameters(script: str) -> List[Parameter]:
    """Extract the exposed parameters from a piece of OpenSCAD ``script``."""
    # Everything from the first module/function onward is implementation.
    script = re.split(r"^(module |function )", script, maxsplit=1, flags=re.M)[0]

    # Partition the source into group sections keyed off /* [Group] */ markers.
    sections: List[Tuple[int, str, str]] = [(0, "", script)]
    for m in _GROUP_RE.finditer(script):
        sections.append((m.start(), m.group(1).strip(), ""))
    resolved: List[Tuple[str, str]] = []
    for idx, (start, group, _code) in enumerate(sections):
        end = sections[idx + 1][0] if idx + 1 < len(sections) else len(script)
        resolved.append((group, script[start:end]))

    params: "dict[str, Parameter]" = {}
    for group, code in resolved:
        for match in _PARAM_RE.finditer(code):
            name, value, comment = match.group(1), match.group(2), match.group(3)
            value = value.strip()
            # Skip expressions referencing variables / spanning lines.
            if value not in ("true", "false") and (
                re.match(r"^[A-Za-z_]", value) or len(value.splitlines()) > 1
            ):
                continue
            try:
                converted = _convert_type(value)
            except ValueError:
                continue
            if converted is None:
                continue
            cvalue, ctype = converted

            rng: dict = {}
            options: List[ParameterOption] = []
            if comment:
                raw_comment = re.sub(r"^//\s*", "", comment).strip()
                base_type = "string" if ctype == "string" else (
                    "number" if ctype in ("number", "number[]") else ctype
                )
                rng, options = _parse_comment(raw_comment, base_type)

            # Description: a // comment on the line immediately above the decl.
            decl_pos = script.find(match.group(0))
            description: Optional[str] = None
            if decl_pos > 0:
                preceding = script[:decl_pos].rstrip("\n")
                last_line = preceding.split("\n")[-1] if preceding else ""
                if last_line.strip().startswith("//"):
                    description = re.sub(r"^//+\s*", "", last_line.strip()) or None

            display = _display_name(name)

            if ctype == "number[]":
                labels = _number_array_labels(name, len(cvalue))  # type: ignore[arg-type]
                for k, item in enumerate(cvalue):  # type: ignore[arg-type]
                    item_name = f"{name}[{k}]"
                    lbl = labels[k]
                    if lbl in ("Width", "Depth", "Height"):
                        item_disp = re.sub(r"\s+Size$", "", display, flags=re.I) + f" {lbl}"
                    else:
                        item_disp = f"{display} {lbl}"
                    params[item_name] = Parameter(
                        name=item_name, display_name=item_disp, type="number",
                        value=item, default_value=item, group=group,
                        description=description, options=list(options), range=dict(rng),
                    )
                continue

            params[name] = Parameter(
                name=name, display_name=display, type=ctype, value=cvalue,
                default_value=cvalue, group=group, description=description,
                options=options, range=rng,
            )
    return list(params.values())
