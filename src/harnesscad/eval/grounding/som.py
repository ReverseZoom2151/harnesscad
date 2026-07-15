"""som — the Set-of-Marks element list: a screenshot as a NUMBERED click table.

Ported from cua-main's Omniparser loop (and the same idea in ghost-os's
``annotate``): instead of asking a model for pixel coordinates, you overlay a
numbered box on every clickable element and let the model pick an ID. The loop
carries an ``id2xy`` map from element id -> pixel, so the model never emits a
coordinate — it emits ``4``, and the harness looks up where ``4`` is. The tool
description is verbatim intent: "each UI element has been assigned a unique ID
number ... use the element's ID number instead of pixel coordinates."

Two grounding sources, one structure
------------------------------------
cua-main gets the elements from OmniParser (a vision model). HarnessCAD does not
need one: :mod:`harnesscad.io.cua.uia` already returns the accessibility tree as
:class:`~harnesscad.io.cua.uia.Element` rects, and
:mod:`harnesscad.io.cua.viewport` returns projected B-rep entities. This module
is the model-free adaptor: it turns EITHER source (a11y elements, or any list of
labelled boxes) into the numbered element list, and — crucially — into the
``id2xy`` lookup, so a policy can address the FreeCAD chrome by mark id with no
VLM in the loop. That is the same "no a11y tree needed on the model side, no
pixels on the model side" ergonomics, grounded on data we already OWN.

Distinct from the existing grounding surface: :mod:`corpus` and :mod:`cadspot`
build (screenshot, description, point) PAIRS for a benchmark; this builds the
live per-frame numbered mark list an agent reads and answers against. Everything
here is pure: boxes in, marks + id2xy out, deterministic id ordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# The tool description cua-main hands the model, kept verbatim so a HarnessCAD
# policy prompt reads identically to the ported one.
SOM_TOOL_DESCRIPTION = (
    "This tool shows a screenshot with numbered elements overlaid on it. Each UI "
    "element has been assigned a unique ID number that you can see in the image. "
    "Use the element's ID number to interact with any element instead of pixel "
    "coordinates."
)


@dataclass(frozen=True)
class BBox:
    """An axis-aligned box in IMAGE pixels (left, top, right, bottom)."""

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("degenerate bbox %r" % (self,))

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @classmethod
    def from_rect(cls, rect: Sequence[int]) -> "BBox":
        """From a (left, top, right, bottom) tuple — the shape
        :class:`harnesscad.io.cua.uia.Element.rect` already carries."""
        left, top, right, bottom = (int(rect[0]), int(rect[1]),
                                    int(rect[2]), int(rect[3]))
        return cls(left, top, right, bottom)

    def to_dict(self) -> dict:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


@dataclass(frozen=True)
class Mark:
    """One numbered clickable element: its id, box, centre, and a label.

    ``label`` is the human/semantic name (a toolbar's ``Name``, a field's
    ``objectName``); ``source`` records where the box came from (``a11y`` /
    ``viewport`` / ``vision``) so a mixed frame stays auditable.
    """

    id: int
    bbox: BBox
    label: str = ""
    kind: str = ""
    source: str = ""

    @property
    def center(self) -> Tuple[int, int]:
        return self.bbox.center

    def to_dict(self) -> dict:
        return {"id": self.id, "bbox": self.bbox.to_dict(), "label": self.label,
                "kind": self.kind, "source": self.source,
                "center": list(self.center)}


class SetOfMarks:
    """A frame's numbered element list plus the id->pixel lookup.

    Built from any sequence of ``(bbox, label, ...)`` boxes. Ids are assigned
    deterministically in a stable reading order (top-to-bottom, then
    left-to-right) so the SAME screen always yields the SAME numbering — a
    property cua-main's vision path does not have and the reason this one is
    testable and replayable.
    """

    def __init__(self, marks: Sequence[Mark]) -> None:
        self.marks: List[Mark] = list(marks)
        self._by_id: Dict[int, Mark] = {m.id: m for m in self.marks}

    # -- construction ------------------------------------------------------
    @classmethod
    def from_boxes(cls, boxes: Sequence[Dict[str, Any]], *,
                   start_id: int = 1, source: str = "") -> "SetOfMarks":
        """Number a list of ``{"bbox"|"rect", "label", "kind"}`` dicts.

        ``bbox`` may be a :class:`BBox`, a ``{x1,y1,x2,y2}`` dict, or a
        ``(l,t,r,b)`` rect. Ordering is stable (reading order) BEFORE ids are
        assigned, so numbering is independent of the caller's input order.
        """
        prepared: List[Tuple[BBox, str, str, str]] = []
        for b in boxes:
            raw = b.get("bbox", b.get("rect"))
            if isinstance(raw, BBox):
                box = raw
            elif isinstance(raw, dict):
                box = BBox(int(raw["x1"]), int(raw["y1"]),
                           int(raw["x2"]), int(raw["y2"]))
            else:
                box = BBox.from_rect(raw)
            prepared.append((box, str(b.get("label", "")),
                             str(b.get("kind", "")),
                             str(b.get("source", source))))
        # Stable reading order: top, then left. This is what makes the id of an
        # element a function of the SCREEN, not of enumeration order.
        prepared.sort(key=lambda t: (t[0].y1, t[0].x1))
        marks = [Mark(id=start_id + i, bbox=box, label=label, kind=kind,
                      source=src)
                 for i, (box, label, kind, src) in enumerate(prepared)]
        return cls(marks)

    @classmethod
    def from_elements(cls, elements: Sequence[Any], *,
                      clickable_only: bool = True,
                      start_id: int = 1) -> "SetOfMarks":
        """From :class:`harnesscad.io.cua.uia.Element` objects — the model-free
        path. Skips zero-area/off-screen nodes; with ``clickable_only`` keeps only
        enabled, interactable control types (buttons, menu items, fields, tabs)."""
        interactable = {"ButtonControl", "MenuItemControl", "TabItemControl",
                        "CheckBoxControl", "RadioButtonControl", "EditControl",
                        "SpinnerControl", "ComboBoxControl", "ListItemControl",
                        "HyperlinkControl", "SplitButtonControl"}
        boxes: List[Dict[str, Any]] = []
        for e in elements:
            rect = getattr(e, "rect", None)
            if not rect:
                continue
            left, top, right, bottom = rect
            if right - left <= 0 or bottom - top <= 0:
                continue
            if clickable_only:
                if not getattr(e, "enabled", True):
                    continue
                if getattr(e, "control_type", "") not in interactable:
                    continue
            boxes.append({"rect": rect, "label": getattr(e, "name", ""),
                          "kind": getattr(e, "control_type", ""),
                          "source": "a11y"})
        return cls.from_boxes(boxes, start_id=start_id)

    # -- the lookup (the whole point) --------------------------------------
    def id2xy(self) -> Dict[int, Tuple[int, int]]:
        """The map a driving loop carries: element id -> click pixel. This is what
        lets the model answer ``4`` and the harness click the right place."""
        return {m.id: m.center for m in self.marks}

    def center_of(self, element_id: int) -> Optional[Tuple[int, int]]:
        m = self._by_id.get(int(element_id))
        return None if m is None else m.center

    def get(self, element_id: int) -> Optional[Mark]:
        return self._by_id.get(int(element_id))

    def find(self, label: str, *, exact: bool = False) -> Optional[Mark]:
        """The first mark whose label matches — the reverse lookup a planner uses
        to turn "the Pad button" into its mark id."""
        low = label.lower()
        for m in self.marks:
            ml = m.label.lower()
            if (ml == low) if exact else (low in ml):
                return m
        return None

    # -- serialisation -----------------------------------------------------
    def element_list(self) -> List[Dict[str, Any]]:
        """The numbered list shown to the model, compact: ``[{id, label, kind}]``.
        Pixels are deliberately omitted here — the model addresses by id only."""
        return [{"id": m.id, "label": m.label, "kind": m.kind}
                for m in self.marks]

    def to_dict(self) -> dict:
        return {"description": SOM_TOOL_DESCRIPTION,
                "elements": [m.to_dict() for m in self.marks],
                "id2xy": {str(k): list(v) for k, v in self.id2xy().items()}}

    def __len__(self) -> int:
        return len(self.marks)
