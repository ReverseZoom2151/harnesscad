"""action_stream — ShowUI's interleaved action format and grounding-pair format.

ShowUI settles two representation questions every GUI agent has to answer, and it
answers both with *normalised coordinates and a flat dict*, which is why they port
cleanly to a CAD viewport whose only honest coordinates are fractional.

1.  **The action format** (``data/template/shared_navigation.py``). Every action
    is one dict::

        {'action': 'CLICK', 'value': None, 'position': [x, y]}

    ``position`` is scaled to ``0..1`` (a *fraction* of the screen, resolution-
    free), ``value`` carries text where applicable, and a two-point gesture
    (select/drag) stores ``[[x1, y1], [x2, y2]]``. Navigation is *interleaved
    streaming*: the model emits ONE action, waits for the next observation, emits
    the next — never a whole plan blind. That single-step contract is what lets a
    step be graded before the next is chosen (compare :mod:`harnesscad.agents.cua.step_eval`).

2.  **The grounding-pair format** (``data/dset_shared_grounding.py``). A grounding
    example is ``(instruction, point)`` with the point normalised to ``0..1`` — the
    same currency :mod:`harnesscad.eval.grounding.cadspot` scores in, and the same
    one this repo's viewport corpus already emits, so this module is the *interop
    shim*: it converts between the normalised ShowUI form and the pixel-space form
    the rest of the CUA surface uses, in one place, tested, instead of every call
    site dividing by width by hand and getting the y-axis wrong.

Why normalised-first matters here specifically: a CAD viewport is resized every
time a task panel docks (see :mod:`harnesscad.eval.grounding.cadspot`), so a pixel
coordinate is meaningless without the rect it was taken in. A *fraction* survives
the resize. Storing actions and grounding points as fractions is not a stylistic
choice; it is the only form that stays correct across the resizes this GUI does on
its own.

Pure stdlib, import-safe. No image, no model, no app.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple, Union

#: ShowUI's navigation action vocabulary (web + a couple CAD-useful additions).
#: An action names WHAT to do; ``position`` says WHERE (a fraction), ``value``
#: carries any payload text/direction. Kept as data so a validator can check an
#: emitted action against it without a model.
ACTION_SPACE = {
    "CLICK": {"position": True, "value": False},
    "INPUT": {"position": True, "value": True},
    "SELECT": {"position": True, "value": False},
    "HOVER": {"position": True, "value": False},
    "SCROLL": {"position": False, "value": True},   # value = direction
    "SELECT_TEXT": {"position": "pair", "value": False},
    "ENTER": {"position": False, "value": False},
    "ANSWER": {"position": False, "value": True},
}

Point = Tuple[float, float]


class ActionFormatError(ValueError):
    """An action dict did not match the ShowUI schema."""


def _is_fraction(v: float) -> bool:
    return -1e-9 <= float(v) <= 1.0 + 1e-9


@dataclass(frozen=True)
class Action:
    """One interleaved-streaming action, ShowUI form.

    ``position`` is either ``None``, a single fraction point ``(x, y)`` or, for a
    two-point gesture, ``((x1, y1), (x2, y2))``. Fractions, always: a raw pixel
    here is a bug, and :meth:`validate` says so.
    """

    action: str
    value: Optional[str] = None
    position: Optional[Union[Point, Tuple[Point, Point]]] = None

    def validate(self) -> "Action":
        spec = ACTION_SPACE.get(self.action)
        if spec is None:
            raise ActionFormatError("unknown action %r (have: %s)"
                                    % (self.action, ", ".join(sorted(ACTION_SPACE))))
        need_pos = spec["position"]
        if need_pos == "pair":
            if not (isinstance(self.position, tuple) and len(self.position) == 2
                    and all(isinstance(p, tuple) and len(p) == 2 for p in self.position)):
                raise ActionFormatError("%s needs a [[x1,y1],[x2,y2]] position" % self.action)
            for p in self.position:
                if not (_is_fraction(p[0]) and _is_fraction(p[1])):
                    raise ActionFormatError("%s position must be 0..1 fractions, got %r"
                                            % (self.action, self.position))
        elif need_pos is True:
            if not (isinstance(self.position, tuple) and len(self.position) == 2
                    and all(isinstance(c, (int, float)) for c in self.position)):
                raise ActionFormatError("%s needs a single [x,y] position" % self.action)
            if not (_is_fraction(self.position[0]) and _is_fraction(self.position[1])):
                raise ActionFormatError("%s position must be 0..1 fractions, got %r"
                                        % (self.action, self.position))
        elif self.position is not None:
            raise ActionFormatError("%s takes no position" % self.action)
        if spec["value"] and self.value is None:
            raise ActionFormatError("%s requires a value" % self.action)
        if not spec["value"] and self.value is not None:
            raise ActionFormatError("%s takes no value" % self.action)
        return self

    def to_dict(self) -> dict:
        pos: Any = None
        if self.position is not None:
            if isinstance(self.position[0], tuple):
                pos = [list(p) for p in self.position]      # type: ignore[union-attr]
            else:
                pos = list(self.position)
        return {"action": self.action, "value": self.value, "position": pos}

    def to_stream(self) -> str:
        """The single-line streaming form ShowUI emits/parses (compact JSON)."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: dict) -> "Action":
        pos = d.get("position")
        position: Any = None
        if pos is not None:
            if pos and isinstance(pos[0], (list, tuple)):
                position = tuple(tuple(float(c) for c in p) for p in pos)
            else:
                position = tuple(float(c) for c in pos)
        return cls(action=str(d["action"]), value=d.get("value"), position=position)

    @classmethod
    def parse_stream(cls, line: str) -> "Action":
        """Parse one streamed action line back to an :class:`Action`."""
        try:
            d = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ActionFormatError("not a JSON action: %s" % exc)
        return cls.from_dict(d)

    # -- pixel <-> fraction interop ---------------------------------------
    def to_pixels(self, width: int, height: int) -> Optional[Union[Tuple[int, int],
                                                                    Tuple[Tuple[int, int],
                                                                          Tuple[int, int]]]]:
        """This action's position in image pixels for a ``width x height`` frame.

        ``None`` if the action has no position. y is measured DOWN from the top
        (image convention), which is what a screenshot uses — the same convention
        :meth:`harnesscad.io.cua.viewport.OrthoCamera.to_image_xy` targets, so a
        ShowUI point and a projected pick agree without a hidden flip.
        """
        if self.position is None:
            return None
        if isinstance(self.position[0], tuple):
            return tuple((int(round(p[0] * width)), int(round(p[1] * height)))  # type: ignore[return-value]
                         for p in self.position)
        return (int(round(self.position[0] * width)),
                int(round(self.position[1] * height)))


def click(x: float, y: float) -> Action:
    return Action("CLICK", position=(x, y)).validate()


def input_text(x: float, y: float, text: str) -> Action:
    return Action("INPUT", value=text, position=(x, y)).validate()


def from_pixel_click(px: int, py: int, width: int, height: int) -> Action:
    """A CLICK from a pixel prediction — the shim from the app's world to ShowUI's.

    Divides by the frame size to get a fraction, so a pick this repo computed in
    pixels can be stored, replayed, and compared in the resolution-free form.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width/height must be positive")
    return click(px / float(width), py / float(height))


@dataclass
class GroundingSample:
    """One ``(instruction, point)`` grounding example in ShowUI's format.

    ``point`` is a ``0..1`` fraction. ``bbox`` (also fractional,
    ``(left, top, right, bottom)``) is optional and, when present, lets this
    interop with :mod:`harnesscad.eval.grounding.cadspot`'s ``point_in_bbox``
    metric without conversion.
    """

    instruction: str
    point: Point
    bbox: Optional[Tuple[float, float, float, float]] = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"instruction": self.instruction, "point": list(self.point)}
        if self.bbox is not None:
            d["bbox"] = list(self.bbox)
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "GroundingSample":
        return cls(instruction=d["instruction"],
                   point=tuple(float(c) for c in d["point"]),
                   bbox=(tuple(float(c) for c in d["bbox"]) if d.get("bbox") else None),
                   meta=d.get("meta", {}))

    def point_pixels(self, width: int, height: int) -> Tuple[int, int]:
        return (int(round(self.point[0] * width)), int(round(self.point[1] * height)))

    def hit(self, x: float, y: float) -> bool:
        """Does fractional prediction ``(x, y)`` fall in this sample's bbox?"""
        if self.bbox is None:
            raise ValueError("sample has no bbox to test against")
        l, t, r, b = self.bbox
        return l <= x <= r and t <= y <= b


def save_grounding(path: str, samples: Sequence[GroundingSample]) -> str:
    """Write grounding samples as JSONL (one sample per line)."""
    with open(path, "w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s.to_dict(), sort_keys=True) + "\n")
    return path


def load_grounding(path: str) -> List[GroundingSample]:
    out: List[GroundingSample] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(GroundingSample.from_dict(json.loads(line)))
    return out
