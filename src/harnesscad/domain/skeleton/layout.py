"""Skeleton — a top-down master layout emitted as CISP ops.

``build_skeleton(brief_or_spec)`` reads a natural-language brief (or a structured
spec dict) and lays out, heuristically and deterministically:

  * a bounding **envelope** (width x height x depth),
  * a named **datum** reference frame (origin, centre, X/Y/Z axes, XY/XZ/YZ
    planes),
  * a **master sketch** of the envelope boundary plus a **reference point per
    named feature** (e.g. hole centres on a bolt circle),
  * an editable **parameter table** of the driving dimensions.

``Skeleton.to_ops()`` realises the master sketch as a ``list[cisp.ops.Op]``
(NewSketch / AddPoint / AddLine / Constrain) that applies cleanly through a
:class:`loop.HarnessSession`: the sketch is intentionally left *under*-constrained
(a warning, never an error) so the layout always verifies ``ok`` while leaving
degrees of freedom for the driving dimensions to edit.

Everything is stdlib-only and deterministic. An LLM may be injected to parse the
brief; the default is a pure heuristic parser so no network is required.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from harnesscad.core.cisp.ops import AddLine, AddPoint, Constrain, NewSketch, Op

# Words that name a round through-feature we lay a reference point down for.
_HOLE_WORDS = ("hole", "bolt", "mount", "fastener", "screw")
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12,
}


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Envelope:
    """The bounding box the whole part lives inside (mm)."""

    width: float = 100.0
    height: float = 100.0
    depth: float = 10.0

    def to_dict(self) -> dict:
        return {"width": self.width, "height": self.height, "depth": self.depth}


@dataclass(frozen=True)
class Datum:
    """A named reference-frame element: a ``point``, ``axis`` or ``plane``.

    ``x/y/z`` is the anchor; ``dx/dy/dz`` is the direction (for an axis) or the
    normal (for a plane), and is ignored for a point.
    """

    name: str
    kind: str  # "point" | "axis" | "plane"
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name, "kind": self.kind,
            "x": self.x, "y": self.y, "z": self.z,
            "dx": self.dx, "dy": self.dy, "dz": self.dz,
        }


@dataclass
class Skeleton:
    """A top-down master layout: datums + master sketch + parameter table."""

    name: str
    envelope: Envelope
    plane: str = "XY"
    datums: List[Datum] = field(default_factory=list)
    reference_points: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    parameters: Dict[str, float] = field(default_factory=dict)

    # --- lookup -----------------------------------------------------------
    def datum(self, name: str) -> Datum:
        for d in self.datums:
            if d.name == name:
                return d
        raise KeyError(f"no datum named '{name}'")

    def datum_names(self) -> List[str]:
        return [d.name for d in self.datums]

    # --- op emission ------------------------------------------------------
    def to_ops(self) -> List[Op]:
        """Emit the CISP ops that realise the master sketch.

        Entity ids are predicted to match ``backends.stub.StubBackend``'s
        allocation order (each AddPoint/AddLine consumes the next ``eN``), so the
        emitted Constrain ops reference exactly the lines they intend to. The DOF
        budget is kept strictly positive (points +2, lines +4, minus a handful of
        horizontal/vertical/distance constraints) so the sketch stays
        under-constrained: a warning, never an over-constrained error.
        """
        w, h = self.envelope.width, self.envelope.height
        ops: List[Op] = [NewSketch(plane=self.plane)]  # -> sk1
        sid = "sk1"
        counter = {"e": 0}

        def add_point(x: float, y: float) -> str:
            ops.append(AddPoint(sketch=sid, x=float(x), y=float(y)))
            counter["e"] += 1
            return f"e{counter['e']}"

        def add_line(x1: float, y1: float, x2: float, y2: float) -> str:
            ops.append(AddLine(sketch=sid, x1=float(x1), y1=float(y1),
                               x2=float(x2), y2=float(y2)))
            counter["e"] += 1
            return f"e{counter['e']}"

        # Envelope corner datum points (bl, br, tr, tl).
        add_point(0.0, 0.0)
        add_point(w, 0.0)
        add_point(w, h)
        add_point(0.0, h)

        # Envelope boundary lines (bottom, right, top, left).
        l_bottom = add_line(0.0, 0.0, w, 0.0)
        l_right = add_line(w, 0.0, w, h)
        l_top = add_line(w, h, 0.0, h)
        l_left = add_line(0.0, h, 0.0, 0.0)

        # Orient + dimension the envelope (leaves the sketch under-constrained).
        ops.append(Constrain(kind="horizontal", a=l_bottom))
        ops.append(Constrain(kind="horizontal", a=l_top))
        ops.append(Constrain(kind="vertical", a=l_right))
        ops.append(Constrain(kind="vertical", a=l_left))
        ops.append(Constrain(kind="distance", a=l_bottom, value=w))
        ops.append(Constrain(kind="distance", a=l_left, value=h))

        # Centre datum + one reference point per named feature.
        add_point(w / 2.0, h / 2.0)
        for _name, (px, py) in self.reference_points.items():
            add_point(px, py)

        return ops

    # --- serialisation ----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "envelope": self.envelope.to_dict(),
            "plane": self.plane,
            "datums": [d.to_dict() for d in self.datums],
            "reference_points": {k: list(v) for k, v in self.reference_points.items()},
            "parameters": dict(self.parameters),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Brief -> spec parsing (heuristic default; optional injected LLM)
# ---------------------------------------------------------------------------
def _parse_spec(brief_or_spec: Any, llm: Optional[Any]) -> dict:
    """Return a normalised spec ``{width, height, depth, hole_count, name}``.

    Accepts a structured dict spec directly, or a free-text brief parsed by the
    heuristic (or by an injected LLM, with heuristic fallback on any failure).
    """
    if isinstance(brief_or_spec, Mapping):
        spec = dict(brief_or_spec)
        text = str(spec.get("brief", spec.get("name", "")))
    else:
        text = str(brief_or_spec or "")
        spec = {}

    if llm is not None:
        try:
            spec = {**_llm_spec(text, llm), **spec}
        except Exception:
            pass  # deterministic heuristic fallback; never fail the build

    heur = _heuristic_spec(text)
    # An explicit spec value wins over the heuristic; heuristic fills the gaps.
    merged = {**heur, **{k: v for k, v in spec.items() if v is not None}}
    return merged


def _heuristic_spec(text: str) -> dict:
    """Pure-Python brief parser: dimensions + feature counts. Deterministic."""
    low = text.lower()
    w = h = d = None

    # "100 x 50 x 10" / "100x50" (optionally with mm units).
    triple = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*[x×*]\s*(\d+(?:\.\d+)?)\s*(?:mm)?"
        r"(?:\s*[x×*]\s*(\d+(?:\.\d+)?))?",
        low)
    if triple:
        w = float(triple.group(1))
        h = float(triple.group(2))
        if triple.group(3):
            d = float(triple.group(3))

    # Named dimensions override the shorthand where present.
    for key, names in (("w", ("width", "length")), ("h", ("height", "depth ")),
                       ("d", ("thickness", "depth"))):
        for nm in names:
            m = re.search(nm + r"\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)\s*(?:mm)?", low)
            if m:
                val = float(m.group(1))
                if key == "w":
                    w = val
                elif key == "h":
                    h = val
                else:
                    d = val
                break

    # Hole / bolt count: a digit or spelled-out number *adjacent* to a hole word
    # (allowing one optional adjective). A leading (?<!...) keeps us from picking
    # up a dimension number like the 40 in "80x40".
    n_holes = 0
    lb = r"(?<![\dx×*.])"
    for word in _HOLE_WORDS:
        if word not in low:
            continue
        count = None
        for pat in (lb + r"(\d+)\s+" + word,
                    lb + r"(\d+)\s+\w+\s+" + word):
            m = re.search(pat, low)
            if m:
                count = int(m.group(1))
                break
        if count is None:
            for nw, nv in _NUMBER_WORDS.items():
                if re.search(r"\b" + nw + r"\s+(?:\w+\s+)?" + word, low):
                    count = nv
                    break
        if count is None:
            count = 1
        n_holes = max(n_holes, count)

    return {
        "width": w if w is not None else 100.0,
        "height": h if h is not None else 100.0,
        "depth": d if d is not None else 10.0,
        "hole_count": n_holes,
        "name": (text.strip()[:60] or "part"),
    }


def _llm_spec(text: str, llm: Any) -> dict:
    """Optional: ask an injected LLM to extract the spec as JSON.

    Uses only the vendor-neutral ``llm.base`` vocabulary. Any failure (no JSON,
    network error, bad shape) raises and the caller falls back to the heuristic.
    """
    from harnesscad.agents.llm.base import system, user

    prompt = (
        "Extract a CAD layout spec from the brief. Reply with ONLY a JSON object "
        '{"width":<mm>,"height":<mm>,"depth":<mm>,"hole_count":<int>,"name":<str>}.'
    )
    res = llm.complete([system(prompt), user(text)])
    raw = getattr(res, "text", "") or ""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in LLM reply")
    data = json.loads(match.group(0))
    out: dict = {}
    for k in ("width", "height", "depth"):
        if data.get(k) is not None:
            out[k] = float(data[k])
    if data.get("hole_count") is not None:
        out["hole_count"] = int(data["hole_count"])
    if data.get("name"):
        out["name"] = str(data["name"])
    return out


def _hole_centers(n: int, w: float, h: float) -> Dict[str, Tuple[float, float]]:
    """Deterministic feature reference points.

    n == 1  -> the centre. n >= 2 -> evenly spaced on a bolt circle whose radius
    is 35% of the smaller envelope side, starting at angle 0 (no randomness).
    """
    if n <= 0:
        return {}
    cx, cy = w / 2.0, h / 2.0
    if n == 1:
        return {"hole_1_center": (cx, cy)}
    r = 0.35 * min(w, h)
    pts: Dict[str, Tuple[float, float]] = {}
    for k in range(n):
        ang = 2.0 * math.pi * k / n
        px = round(cx + r * math.cos(ang), 6)
        py = round(cy + r * math.sin(ang), 6)
        pts[f"hole_{k + 1}_center"] = (px, py)
    return pts


# ---------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------
def build_skeleton(brief_or_spec: Any, llm: Optional[Any] = None,
                   sizing: Optional[List[Mapping[str, Any]]] = None) -> Skeleton:
    """Build a top-down master layout from a brief or structured spec.

    Parameters
    ----------
    brief_or_spec:
        A natural-language brief (str) or a structured spec dict with any of
        ``width/height/depth/hole_count/name``.
    llm:
        Optional injected LLM (``llm.base.LLM``) used to parse the brief. The
        default is ``None`` -> a deterministic heuristic parser (no network).
    sizing:
        Optional list of :meth:`sizing.calc.SizingCalc.size` result dicts; each
        ``{dimension, value}`` is merged into the parameter table, so engineering
        sizing drives the layout dimensions.
    """
    spec = _parse_spec(brief_or_spec, llm)
    w = float(spec["width"])
    h = float(spec["height"])
    d = float(spec["depth"])
    n_holes = int(spec.get("hole_count", 0))
    env = Envelope(width=w, height=h, depth=d)

    # Named datum reference frame (origin, centre, primary axes, principal planes).
    datums: List[Datum] = [
        Datum("origin", "point", 0.0, 0.0, 0.0),
        Datum("center", "point", w / 2.0, h / 2.0, 0.0),
        Datum("x_axis", "axis", 0.0, 0.0, 0.0, dx=1.0),
        Datum("y_axis", "axis", 0.0, 0.0, 0.0, dy=1.0),
        Datum("z_axis", "axis", 0.0, 0.0, 0.0, dz=1.0),
        Datum("XY", "plane", 0.0, 0.0, 0.0, dz=1.0),
        Datum("XZ", "plane", 0.0, 0.0, 0.0, dy=1.0),
        Datum("YZ", "plane", 0.0, 0.0, 0.0, dx=1.0),
    ]

    reference_points = _hole_centers(n_holes, w, h)

    # Editable driving-dimension table.
    parameters: Dict[str, float] = {
        "envelope_width": w,
        "envelope_height": h,
        "envelope_depth": d,
    }
    if n_holes > 0:
        parameters["hole_count"] = float(n_holes)
        parameters["hole_diameter"] = round(0.12 * min(w, h), 6)
        if n_holes >= 2:
            parameters["hole_circle_radius"] = round(0.35 * min(w, h), 6)
        for name, (px, py) in reference_points.items():
            parameters[f"{name}_x"] = px
            parameters[f"{name}_y"] = py

    # Fold in any engineering-sizing results as driving dimensions.
    for res in (sizing or []):
        dim = res.get("dimension")
        val = res.get("value")
        if dim is None or val is None:
            continue
        key = str(res.get("formula", dim))
        parameters[f"sized_{key}"] = float(val)

    return Skeleton(
        name=str(spec.get("name", "part")),
        envelope=env,
        plane="XY",
        datums=datums,
        reference_points=reference_points,
        parameters=parameters,
    )
