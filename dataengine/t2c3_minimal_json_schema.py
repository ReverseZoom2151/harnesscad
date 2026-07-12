"""The Text2CAD *minimal JSON* schema: the exact LLM-facing CAD serialisation.

Reference implementation: the ``_json()`` methods of ``CADSequence``,
``CoordinateSystem``, ``SketchSequence``, ``FaceSequence``, ``LoopSequence``,
``Line``, ``Arc``, ``Circle`` and the corresponding ``from_minimal_json()``
constructors in the released Text2CAD code (Khan et al., NeurIPS 2024), i.e. the
output of ``CadSeqProc/minimal_cad_json.py``.

Relation to ``dataengine.text2cad_minimal_metadata``
---------------------------------------------------
That module implements the *Minimal Metadata Generator* as a transformation: strip
DeepCAD's random uuid keys and redundant bookkeeping, renaming entities positionally.
It does not fix the resulting **document schema** -- and the schema is the actual
contract the LLM annotator (and every downstream parser) sees. This module is that
schema, with a builder and an exactly inverse parser:

    {
      "final_name": "", "final_shape": "",
      "parts": {
        "part_1": {
          "coordinate_system": {"Euler Angles": [deg, deg, deg],
                                "Translation Vector": [x, y, z]},
          "sketch": {"face_1": {"loop_1": {"line_1": {"Start Point": [x, y],
                                                      "End Point":   [x, y]},
                                           "arc_1":  {"Start Point": [x, y],
                                                      "Mid Point":   [x, y],
                                                      "End Point":   [x, y]},
                                           "circle_1": {"Centre": [x, y],
                                                        "Radius": r}}}},
          "extrusion": {"extrude_depth_towards_normal": e1,
                        "extrude_depth_opposite_normal": e2,
                        "sketch_scale": s,
                        "operation": "NewBodyFeatureOperation"},
          "description": {"name": "", "shape": "",
                          "length": l, "width": w, "height": h}
        }
      }
    }

Load-bearing details reproduced here:

* keys are **1-based positional names** (``part_1``, ``face_1``, ``loop_1``) and
  curves are numbered **per type within a loop** (``line_1``, ``line_2``, ``arc_1``);
* the coordinate system is stored as **Euler angles in degrees** (ZYX intrinsic) plus
  a translation vector -- radians are used everywhere else in the pipeline, degrees
  only here, and ``from_minimal_json`` converts back;
* every float is rounded to **4 decimals** (``float_round``), and ``-0.0`` is
  normalised to ``0.0`` so two identical models serialise byte-identically;
* the extrusion keys are the verbose, LLM-readable ones
  (``extrude_depth_towards_normal`` / ``extrude_depth_opposite_normal`` /
  ``sketch_scale`` / ``operation``), and ``operation`` is the *name* of the boolean,
  never its index;
* ``description.height`` is ``extent_one + extent_two`` and ``length`` / ``width`` are
  the sketch bbox dimensions -- fields the LLM is explicitly told to describe.

Deterministic, stdlib-only. Angle conversion is plain ``math.degrees`` /
``math.radians``; no rotation matrices are needed at this layer.
"""

from __future__ import annotations

import math
from collections import OrderedDict

ROUNDING = 4

EXTRUDE_OPERATIONS: tuple[str, ...] = (
    "NewBodyFeatureOperation", "JoinFeatureOperation",
    "CutFeatureOperation", "IntersectFeatureOperation",
)

CURVE_TYPES: tuple[str, ...] = ("line", "arc", "circle")


class MinimalJsonError(ValueError):
    """Raised for malformed models or malformed minimal-JSON documents."""


# --- numeric formatting -----------------------------------------------------
def float_round(value: float, rounding: int = ROUNDING) -> float:
    """Round to 4 decimals and normalise ``-0.0`` to ``0.0`` (reference ``float_round``)."""
    out = round(float(value), rounding)
    return 0.0 if out == 0 else out


def _point(pt) -> list[float]:
    if len(pt) != 2:
        raise MinimalJsonError(f"expected a 2D point, got {pt!r}")
    return [float_round(pt[0]), float_round(pt[1])]


# --- serialisation ----------------------------------------------------------
def coordinate_system_json(origin, euler_radians) -> dict:
    """``{"Euler Angles": [deg x3], "Translation Vector": [x, y, z]}``."""
    if len(origin) != 3 or len(euler_radians) != 3:
        raise MinimalJsonError("origin and euler angles must each hold 3 values")
    return {
        "Euler Angles": [float_round(math.degrees(a)) for a in euler_radians],
        "Translation Vector": [float_round(v) for v in origin],
    }


def curve_json(curve: dict) -> dict:
    kind = curve["type"]
    if kind == "line":
        return {"Start Point": _point(curve["start"]), "End Point": _point(curve["end"])}
    if kind == "arc":
        return {
            "Start Point": _point(curve["start"]),
            "Mid Point": _point(curve["mid"]),
            "End Point": _point(curve["end"]),
        }
    if kind == "circle":
        return {"Centre": _point(curve["center"]), "Radius": float_round(curve["radius"])}
    raise MinimalJsonError(f"unknown curve type {kind!r}")


def loop_json(loop: list[dict]) -> dict:
    """Curves keyed ``<type>_<n>``, numbered per type within the loop."""
    counters = {t: 1 for t in CURVE_TYPES}
    out = OrderedDict()
    for curve in loop:
        kind = curve["type"]
        if kind not in counters:
            raise MinimalJsonError(f"unknown curve type {kind!r}")
        out[f"{kind}_{counters[kind]}"] = curve_json(curve)
        counters[kind] += 1
    if not out:
        raise MinimalJsonError("loop has no curves")
    return out


def sketch_json(sketch: list[list[list[dict]]]) -> dict:
    out = OrderedDict()
    for i, face in enumerate(sketch):
        face_out = OrderedDict()
        for j, loop in enumerate(face):
            face_out[f"loop_{j + 1}"] = loop_json(loop)
        if not face_out:
            raise MinimalJsonError("face has no loops")
        out[f"face_{i + 1}"] = face_out
    if not out:
        raise MinimalJsonError("sketch has no faces")
    return out


def extrusion_json(extrusion: dict) -> dict:
    boolean = extrusion["boolean"]
    if not 0 <= boolean < len(EXTRUDE_OPERATIONS):
        raise MinimalJsonError(f"boolean index {boolean} out of range")
    return {
        "extrude_depth_towards_normal": float_round(extrusion["extent_one"]),
        "extrude_depth_opposite_normal": float_round(extrusion["extent_two"]),
        "sketch_scale": float_round(extrusion["sketch_size"]),
        "operation": EXTRUDE_OPERATIONS[boolean],
    }


def sketch_dimension(sketch: list[list[list[dict]]]) -> tuple[float, float]:
    """``(length, width)`` = the sketch bounding-box extents, as ``description`` uses."""
    xs: list[float] = []
    ys: list[float] = []
    for face in sketch:
        for loop in face:
            for curve in loop:
                if curve["type"] == "circle":
                    cx, cy = curve["center"]
                    r = curve["radius"]
                    xs += [cx - r, cx + r]
                    ys += [cy - r, cy + r]
                else:
                    pts = [curve["start"], curve["end"]]
                    if curve["type"] == "arc":
                        pts.append(curve["mid"])
                    xs += [p[0] for p in pts]
                    ys += [p[1] for p in pts]
    if not xs:
        raise MinimalJsonError("sketch has no curves")
    return (float_round(max(xs) - min(xs)), float_round(max(ys) - min(ys)))


def build_minimal_json(model: list[dict], *, final_name: str = "",
                       final_shape: str = "") -> dict:
    """Serialise ``[{sketch, extrusion, [coordinate_system], [description]}, ...]``.

    Each part must carry ``coordinate_system`` as ``{"origin": (x, y, z),
    "euler": (theta, phi, gamma)}`` with the angles in **radians**; they are emitted
    in degrees. ``description`` fields ``name``/``shape`` default to the empty string
    the annotator later fills in.
    """
    if not model:
        raise MinimalJsonError("model has no parts")
    doc: dict = OrderedDict()
    doc["final_name"] = final_name
    doc["final_shape"] = final_shape
    doc["parts"] = OrderedDict()
    for i, part in enumerate(model):
        coord = part["coordinate_system"]
        sketch = part["sketch"]
        extrusion = part["extrusion"]
        length, width = sketch_dimension(sketch)
        height = float_round(extrusion["extent_one"] + extrusion["extent_two"])
        desc = part.get("description", {})
        doc["parts"][f"part_{i + 1}"] = OrderedDict([
            ("coordinate_system", coordinate_system_json(coord["origin"], coord["euler"])),
            ("sketch", sketch_json(sketch)),
            ("extrusion", extrusion_json(extrusion)),
            ("description", OrderedDict([
                ("name", desc.get("name", "")),
                ("shape", desc.get("shape", "")),
                ("length", length),
                ("width", width),
                ("height", height),
            ])),
        ])
    return doc


# --- parsing (inverse) ------------------------------------------------------
def _sorted_keys(mapping: dict, prefix: str) -> list[str]:
    keys = [k for k in mapping if k.startswith(prefix + "_")]
    try:
        return sorted(keys, key=lambda k: int(k.split("_")[-1]))
    except ValueError as exc:
        raise MinimalJsonError(f"malformed {prefix} key in {list(mapping)}") from exc


def parse_curve(key: str, value: dict) -> dict:
    kind = key.rsplit("_", 1)[0]
    if kind == "line":
        return {"type": "line", "start": tuple(value["Start Point"]),
                "end": tuple(value["End Point"])}
    if kind == "arc":
        return {"type": "arc", "start": tuple(value["Start Point"]),
                "mid": tuple(value["Mid Point"]), "end": tuple(value["End Point"])}
    if kind == "circle":
        return {"type": "circle", "center": tuple(value["Centre"]),
                "radius": value["Radius"]}
    raise MinimalJsonError(f"unknown curve key {key!r}")


def parse_minimal_json(doc: dict) -> list[dict]:
    """Inverse of :func:`build_minimal_json`; Euler angles come back in **radians**."""
    if "parts" not in doc:
        raise MinimalJsonError("document has no 'parts'")
    model = []
    for part_key in _sorted_keys(doc["parts"], "part"):
        part = doc["parts"][part_key]
        coord = part["coordinate_system"]
        sketch = []
        for face_key in _sorted_keys(part["sketch"], "face"):
            face = []
            for loop_key in _sorted_keys(part["sketch"][face_key], "loop"):
                loop_doc = part["sketch"][face_key][loop_key]
                face.append([parse_curve(k, v) for k, v in loop_doc.items()])
            sketch.append(face)
        ext = part["extrusion"]
        if ext["operation"] not in EXTRUDE_OPERATIONS:
            raise MinimalJsonError(f"unknown operation {ext['operation']!r}")
        model.append({
            "coordinate_system": {
                "origin": tuple(coord["Translation Vector"]),
                "euler": tuple(math.radians(a) for a in coord["Euler Angles"]),
            },
            "sketch": sketch,
            "extrusion": {
                "extent_one": ext["extrude_depth_towards_normal"],
                "extent_two": ext["extrude_depth_opposite_normal"],
                "sketch_size": ext["sketch_scale"],
                "boolean": EXTRUDE_OPERATIONS.index(ext["operation"]),
            },
            "description": part.get("description", {}),
        })
    if not model:
        raise MinimalJsonError("document has no parts")
    return model
