"""Safe deterministic SVG primitive-ID label overlays."""

from __future__ import annotations

from dataclasses import dataclass
from xml.sax.saxutils import escape


@dataclass(frozen=True)
class LabelPlacement:
    id: str
    anchor: tuple[float, float]
    bbox: tuple[float, float, float, float]


def place_labels(anchors, *, width=24.0, height=14.0, gap=2.0):
    placements = []
    for entity_id, anchor in sorted(anchors.items()):
        x, y = map(float, anchor)
        while any(not (x + width + gap <= p.bbox[0] or x >= p.bbox[2] + gap
                           or y + height + gap <= p.bbox[1] or y >= p.bbox[3] + gap)
                  for p in placements):
            y += height + gap
        placements.append(LabelPlacement(str(entity_id), (x, y),
                                         (x, y, x + width, y + height)))
    return tuple(placements)


def overlay_svg(anchors):
    placements = place_labels(anchors)
    labels = "".join(
        f'<g data-entity-id="{escape(item.id)}"><rect x="{item.bbox[0]}" '
        f'y="{item.bbox[1]}" width="{item.bbox[2]-item.bbox[0]}" '
        f'height="{item.bbox[3]-item.bbox[1]}"/><text x="{item.bbox[0]+2}" '
        f'y="{item.bbox[1]+11}">{escape(item.id)}</text></g>'
        for item in placements)
    return f'<svg xmlns="http://www.w3.org/2000/svg">{labels}</svg>', placements
