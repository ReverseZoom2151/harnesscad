"""Unified code-CAD parameter schema across languages (deterministic, stdlib-only).

One neutral parameter model, with a pure converter per language (OpenSCAD
Customizer manifest, JSCAD ``getParameterDefinitions()`` output, CadQuery
``--getparams`` JSON) into that model. One panel, N languages -- the conversion is
pure and testable.

The harness had an OpenSCAD *source* parameter extractor
(``programs/cadam_scad_customizer``), but nothing that (a) ingests the parameter
manifests the other toolchains emit, (b) reduces all of them to one schema, or
(c) validates/coerces a set of user (or model-proposed) values against that
schema before they are fed back into a render. This module supplies all three,
plus the OpenSCAD ``params.json`` parameter-set writer used by the CLI runner
(``-p params.json -P default``, ``fileFormatVersion: "1"``).

Neutral model
-------------
type:  number | string | boolean
input: default-number | default-string | default-boolean | choice-number | choice-string

Deterministic: pure dict/list transforms, source order preserved, no clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

TYPE_NUMBER = "number"
TYPE_STRING = "string"
TYPE_BOOLEAN = "boolean"

INPUT_NUMBER = "default-number"
INPUT_STRING = "default-string"
INPUT_BOOLEAN = "default-boolean"
INPUT_CHOICE_NUMBER = "choice-number"
INPUT_CHOICE_STRING = "choice-string"


class UnknownParamLanguage(KeyError):
    """Raised when asked to convert a language with no converter."""


@dataclass(frozen=True)
class Option:
    name: str  # label
    value: Any  # number or string


@dataclass(frozen=True)
class Param:
    """One neutral parameter."""

    name: str
    type: str
    input: str
    initial: Any
    caption: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    decimal: Optional[int] = None
    max_length: Optional[int] = None
    placeholder: str = ""
    options: Tuple[Option, ...] = field(default_factory=tuple)

    def option_values(self) -> List[Any]:
        return [o.value for o in self.options]

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "name": self.name,
            "type": self.type,
            "input": self.input,
            "initial": self.initial,
            "caption": self.caption,
        }
        if self.options:
            out["options"] = [{"name": o.name, "value": o.value} for o in self.options]
        for key in ("min", "max", "step", "decimal", "max_length"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        if self.placeholder:
            out["placeholder"] = self.placeholder
        return out


def _get(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


# ---------------------------------------------------------------------------
# OpenSCAD customizer manifest  (openscad -o customizer.param)
# ---------------------------------------------------------------------------


def from_openscad(defs: Iterable[Mapping[str, Any]]) -> List[Param]:
    out: List[Param] = []
    for raw in defs:
        ptype = _get(raw, "type", default="")
        name = _get(raw, "name", default="")
        caption = _get(raw, "caption", default="") or ""
        initial = _get(raw, "initial", "value", "default")
        options = _get(raw, "options")
        if ptype == TYPE_BOOLEAN:
            out.append(
                Param(
                    name=name,
                    type=TYPE_BOOLEAN,
                    input=INPUT_BOOLEAN,
                    initial=bool(initial),
                    caption=caption,
                )
            )
        elif ptype == TYPE_STRING:
            if isinstance(options, list) and options:
                out.append(
                    Param(
                        name=name,
                        type=TYPE_STRING,
                        input=INPUT_CHOICE_STRING,
                        initial=str(initial),
                        caption=caption,
                        options=_options(options, str),
                    )
                )
            else:
                out.append(
                    Param(
                        name=name,
                        type=TYPE_STRING,
                        input=INPUT_STRING,
                        initial=str(initial) if initial is not None else "",
                        caption=caption,
                        max_length=_get(raw, "maxLength", "max_length"),
                    )
                )
        elif ptype == TYPE_NUMBER:
            if isinstance(options, list) and options:
                out.append(
                    Param(
                        name=name,
                        type=TYPE_NUMBER,
                        input=INPUT_CHOICE_NUMBER,
                        initial=_num(initial),
                        caption=caption,
                        options=_options(options, _num),
                    )
                )
            elif isinstance(initial, list):
                # vector parameter: CadHub drops these (TODO in source); we skip too
                continue
            else:
                out.append(
                    Param(
                        name=name,
                        type=TYPE_NUMBER,
                        input=INPUT_NUMBER,
                        initial=_num(initial),
                        caption=caption,
                        min=_maybe_num(_get(raw, "min")),
                        max=_maybe_num(_get(raw, "max")),
                        step=_maybe_num(_get(raw, "step")),
                    )
                )
    return out


# ---------------------------------------------------------------------------
# JSCAD getParameterDefinitions()
# ---------------------------------------------------------------------------

_JSCAD_NUMERIC = {"int", "number", "float", "slider"}
_JSCAD_TEXTUAL = {"text", "url", "email", "password", "color", "date"}
_JSCAD_PLACEHOLDER = {"text", "date", "url"}
_JSCAD_MAXLEN = {"text", "url"}


def from_jscad(defs: Iterable[Mapping[str, Any]]) -> List[Param]:
    out: List[Param] = []
    for raw in defs:
        ptype = _get(raw, "type", default="")
        name = _get(raw, "name", default="")
        caption = _get(raw, "caption", default="") or ""
        initial = _get(raw, "initial", "default")
        if ptype == "group":
            continue
        if ptype in _JSCAD_NUMERIC:
            step = _maybe_num(_get(raw, "step"))
            value = _num(initial)
            decimal = 0 if (step is not None and step % 1 == 0 and value % 1 == 0) else 2
            out.append(
                Param(
                    name=name,
                    type=TYPE_NUMBER,
                    input=INPUT_NUMBER,
                    initial=value,
                    caption=caption,
                    min=_maybe_num(_get(raw, "min")),
                    max=_maybe_num(_get(raw, "max")),
                    step=step,
                    decimal=decimal,
                )
            )
        elif ptype in _JSCAD_TEXTUAL:
            out.append(
                Param(
                    name=name,
                    type=TYPE_STRING,
                    input=INPUT_STRING,
                    initial="" if initial is None else str(initial),
                    caption=caption,
                    placeholder=str(_get(raw, "placeholder", default=""))
                    if ptype in _JSCAD_PLACEHOLDER
                    else "",
                    max_length=_get(raw, "maxLength", "max_length")
                    if ptype in _JSCAD_MAXLEN
                    else None,
                )
            )
        elif ptype == "checkbox":
            out.append(
                Param(
                    name=name,
                    type=TYPE_BOOLEAN,
                    input=INPUT_BOOLEAN,
                    initial=bool(_get(raw, "initial", "checked", default=False)),
                    caption=caption,
                )
            )
        elif ptype in ("choice", "radio"):
            values = list(_get(raw, "values", default=[]))
            captions = list(_get(raw, "captions", default=[])) or values
            numeric = bool(values) and isinstance(values[0], (int, float)) and not isinstance(
                values[0], bool
            )
            if numeric:
                options = tuple(
                    Option(name=str(captions[i]), value=_num(v))
                    for i, v in enumerate(values)
                )
                out.append(
                    Param(
                        name=name,
                        type=TYPE_NUMBER,
                        input=INPUT_CHOICE_NUMBER,
                        initial=_num(initial),
                        caption=caption,
                        options=options,
                    )
                )
            else:
                options = tuple(
                    Option(name=str(captions[i]), value=str(v))
                    for i, v in enumerate(values)
                )
                out.append(
                    Param(
                        name=name,
                        type=TYPE_STRING,
                        input=INPUT_CHOICE_STRING,
                        initial="" if initial is None else str(initial),
                        caption=caption,
                        options=options,
                    )
                )
    return out


# ---------------------------------------------------------------------------
# CadQuery (cq-cli --getparams)
# ---------------------------------------------------------------------------


def from_cadquery(defs: Iterable[Mapping[str, Any]]) -> List[Param]:
    out: List[Param] = []
    for raw in defs:
        name = _get(raw, "name", default="")
        initial = _get(raw, "initial", "default")
        ptype = _get(raw, "type")
        if ptype is None:
            ptype = _infer_type(initial)
        if ptype == TYPE_NUMBER:
            out.append(
                Param(
                    name=name,
                    type=TYPE_NUMBER,
                    input=INPUT_NUMBER,
                    initial=_num(initial) if initial is not None else 0,
                )
            )
        elif ptype == TYPE_STRING:
            out.append(
                Param(
                    name=name,
                    type=TYPE_STRING,
                    input=INPUT_STRING,
                    initial=str(initial) if initial else "",
                )
            )
        elif ptype == TYPE_BOOLEAN:
            out.append(
                Param(
                    name=name,
                    type=TYPE_BOOLEAN,
                    input=INPUT_BOOLEAN,
                    initial=bool(initial) if initial is not None else False,
                )
            )
    return out


_CONVERTERS = {
    "openscad": from_openscad,
    "jscad": from_jscad,
    "cadquery": from_cadquery,
}


def normalize(language: str, defs: Iterable[Mapping[str, Any]]) -> List[Param]:
    """Convert a language's raw parameter manifest into the neutral schema."""
    key = language.strip().lower()
    if key not in _CONVERTERS:
        raise UnknownParamLanguage(language)
    return _CONVERTERS[key](defs)


# ---------------------------------------------------------------------------
# Value handling
# ---------------------------------------------------------------------------


def defaults(params: Sequence[Param]) -> Dict[str, Any]:
    """The initial value of every parameter, in declaration order."""
    return {p.name: p.initial for p in params}


@dataclass(frozen=True)
class ValueIssue:
    name: str
    kind: str  # unknown | type | range | option | length
    detail: str


def validate_values(
    params: Sequence[Param], values: Mapping[str, Any]
) -> Tuple[Dict[str, Any], List[ValueIssue]]:
    """Coerce ``values`` onto the schema; return (clean values, issues).

    Missing keys fall back to the parameter's initial value. Numbers out of
    ``[min, max]`` are clamped, over-long strings truncated, non-member choices
    reset to the initial -- every repair is reported as a ``ValueIssue`` so a
    caller can surface or reject it. Keys with no matching parameter are dropped.
    """
    by_name = {p.name: p for p in params}
    issues: List[ValueIssue] = []
    clean: Dict[str, Any] = {}

    for key in values:
        if key not in by_name:
            issues.append(ValueIssue(key, "unknown", "no such parameter"))

    for param in params:
        if param.name not in values:
            clean[param.name] = param.initial
            continue
        raw = values[param.name]
        if param.type == TYPE_BOOLEAN:
            if not isinstance(raw, bool):
                issues.append(ValueIssue(param.name, "type", "expected boolean"))
                clean[param.name] = bool(raw)
            else:
                clean[param.name] = raw
        elif param.type == TYPE_NUMBER:
            try:
                value = _num(raw)
            except (TypeError, ValueError):
                issues.append(ValueIssue(param.name, "type", "expected number"))
                clean[param.name] = param.initial
                continue
            if param.options:
                if value not in param.option_values():
                    issues.append(ValueIssue(param.name, "option", "not an allowed option"))
                    value = param.initial
            else:
                if param.min is not None and value < param.min:
                    issues.append(ValueIssue(param.name, "range", "below min"))
                    value = _num(param.min)
                if param.max is not None and value > param.max:
                    issues.append(ValueIssue(param.name, "range", "above max"))
                    value = _num(param.max)
            clean[param.name] = value
        else:  # string
            value = raw if isinstance(raw, str) else str(raw)
            if not isinstance(raw, str):
                issues.append(ValueIssue(param.name, "type", "expected string"))
            if param.options:
                if value not in param.option_values():
                    issues.append(ValueIssue(param.name, "option", "not an allowed option"))
                    value = param.initial
            elif param.max_length is not None and len(value) > param.max_length:
                issues.append(ValueIssue(param.name, "length", "over maxLength"))
                value = value[: param.max_length]
            clean[param.name] = value
    return clean, issues


def openscad_parameter_set(
    values: Mapping[str, Any], set_name: str = "default"
) -> Dict[str, Any]:
    """The ``params.json`` payload OpenSCAD's ``-p/-P`` flags consume.

    OpenSCAD reads every value as a string in this file format.
    """
    encoded = {name: _scad_value(values[name]) for name in sorted(values)}
    return {
        "parameterSets": {set_name: encoded},
        "fileFormatVersion": "1",
    }


def _scad_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def schema_digest(params: Sequence[Param]) -> str:
    """Stable digest of the parameter schema (order-sensitive)."""
    import hashlib

    parts = []
    for p in params:
        parts.append(
            "|".join(
                [
                    p.name,
                    p.type,
                    p.input,
                    repr(p.initial),
                    repr(p.min),
                    repr(p.max),
                    repr(p.step),
                    ",".join("%s=%r" % (o.name, o.value) for o in p.options),
                ]
            )
        )
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _options(raw_options: Sequence[Mapping[str, Any]], cast) -> Tuple[Option, ...]:
    out: List[Option] = []
    for opt in raw_options:
        value = cast(_get(opt, "value"))
        label = _get(opt, "name", "caption", default=None)
        out.append(Option(name=str(value) if label is None else str(label), value=value))
    return tuple(out)


def _num(value: Any) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    number = float(value)
    return int(number) if number.is_integer() else number


def _maybe_num(value: Any) -> Optional[float]:
    return None if value is None else _num(value)


def _infer_type(value: Any) -> str:
    if isinstance(value, bool):
        return TYPE_BOOLEAN
    if isinstance(value, (int, float)):
        return TYPE_NUMBER
    return TYPE_STRING
