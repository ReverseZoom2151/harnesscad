"""Four-view engineering-drawing SVG metrics (deterministic).

Each candidate CAD solid is rendered to a four-view engineering drawing
(Isometric / Top / Front / Right) as an SVG; a judge feeds a handful of
*deterministic* structural measurements about that drawing into a
rubric-deduction engine (path count and estimated component count are read by
several deduction rules).

This module extracts those measurements from the SVG text with the stdlib XML
parser only -- no rasteriser, no rsvg, no VTK:

  * ``view_labels``               -- which of the four standard views are present;
  * ``total_path_count``          -- number of ``<path>`` elements;
  * ``text_count``                -- number of ``<text>`` elements;
  * ``estimated_component_count`` -- connected-components estimate obtained by
    grouping ``<path>`` bounding boxes that overlap (single-linkage union over
    axis-aligned boxes); a proxy for how many distinct solids the drawing shows;
  * ``width_mm`` / ``height_mm``  -- parsed sheet dimensions.

The component estimator is the transferable geometry: a stdlib single-linkage
grouping of axis-aligned bounding boxes with a tolerance band, distinct from the
solid-count / assembly-graph reasoning already in the harness.

No wall clock, no randomness.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

# The four canonical view labels.
VIEW_LABELS = ("Isometric", "Top", "Front", "Right")
_SVG_NS = "http://www.w3.org/2000/svg"
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def parse_dimension(value):
    """Parse an SVG length like '120mm' / '96px' / '120' to a float (0 on fail)."""
    if not value:
        return 0.0
    cleaned = str(value).replace("mm", "").replace("px", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def path_bbox(path_d):
    """Axis-aligned bbox (xmin, ymin, xmax, ymax) from a path 'd' string.

    Treats the numeric tokens in ``d`` as an alternating x,y coordinate stream
    (as the drawing generator emits) and takes their extent. Returns None
    if fewer than two coordinate pairs are present.
    """
    numbers = [float(t) for t in _NUMBER.findall(path_d or "")]
    if len(numbers) < 4:
        return None
    xs = numbers[0::2]
    ys = numbers[1::2]
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def boxes_overlap(a, b, tol=0.5):
    """True iff two axis-aligned boxes overlap within tolerance ``tol``."""
    return not (
        a[2] < b[0] - tol
        or b[2] < a[0] - tol
        or a[3] < b[1] - tol
        or b[3] < a[1] - tol
    )


def estimate_components(path_boxes, tol=0.5):
    """Single-linkage grouping of overlapping boxes; returns group count.

    Greedy single pass matching the repo: each box joins the first existing
    group containing an overlapping box, else starts a new group. Deterministic
    in input order.
    """
    groups = []
    for box in path_boxes:
        matched = False
        for group in groups:
            if any(boxes_overlap(box, existing, tol) for existing in group):
                group.append(box)
                matched = True
                break
        if not matched:
            groups.append([box])
    return len(groups)


def analyze_svg_text(svg_text, tol=0.5):
    """Structural metrics for an engineering-drawing SVG given as a string.

    Returns a dict:
      view_labels               : tuple of present canonical view labels
      total_path_count          : int
      text_count                : int
      estimated_component_count : int
      width_mm, height_mm       : float
    """
    root = ET.fromstring(svg_text)

    def _local(tag):
        return tag.split("}", 1)[1] if "}" in tag else tag

    texts, paths = [], []
    for el in root.iter():
        name = _local(el.tag)
        if name == "text":
            texts.append(el)
        elif name == "path":
            paths.append(el)

    present = []
    label_set = set()
    for t in texts:
        content = "".join(t.itertext()).strip()
        if content in VIEW_LABELS:
            label_set.add(content)
    present = tuple(v for v in VIEW_LABELS if v in label_set)

    boxes = [b for b in (path_bbox(p.get("d", "")) for p in paths) if b]
    component_count = estimate_components(boxes, tol)

    return {
        "view_labels": present,
        "total_path_count": len(paths),
        "text_count": len(texts),
        "estimated_component_count": component_count,
        "width_mm": parse_dimension(root.get("width")),
        "height_mm": parse_dimension(root.get("height")),
    }


def all_views_present(metrics):
    """True iff the four canonical views are all labelled in the drawing."""
    return set(metrics.get("view_labels", ())) == set(VIEW_LABELS)
