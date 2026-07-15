"""SPCC component segmentation and extrusion-direction labelling (CAD-Llama, Li
et al., CVPR 2025, "Leveraging Large Language Models for Computer-Aided Design
Parametric 3D Model Generation").

CAD-Llama's hierarchical annotation pipeline needs two deterministic
pre-processing rules over a DeepCAD-style sketch-extrude command stream, both
drawn from the paper's supplementary Section D:

* **Component segmentation (D.1).** A single sketch-extrude pair is normally one
  component. But when identical pairs occur *consecutively* (e.g. ten cylinders
  in a circular array) and their count exceeds a threshold (3 in the paper),
  they are collapsed into one component. Two pairs are *equivalent* iff all
  commands and parameters match except the sketch-plane origin (px, py, pz).

* **Extrusion-direction labelling (D.3).** The annotation prompt names the
  extrusion direction only when it lies along a canonical axis
  (up/down/left/right/front/back); the paper reports >95% of DeepCAD extrusions
  fall in these six categories. This module classifies an extrusion vector into
  the dominant signed axis and its direction word, or ``None`` when the vector
  is off-axis (skew) beyond a tolerance.

Both routines are deterministic and stdlib-only. Inputs are plain descriptors so
the rules stay independent of any particular sequence encoding.
"""

from __future__ import annotations

DEFAULT_COMPONENT_THRESHOLD = 3

# Canonical world-axis -> direction word (paper D.3).
_AXIS_WORD = {
    (1, 0, 0): "right", (-1, 0, 0): "left",
    (0, 1, 0): "back", (0, -1, 0): "front",
    (0, 0, 1): "up", (0, 0, -1): "down",
}


def classify_extrusion_direction(vector, ratio_tol: float = 0.1):
    """Classify an extrusion vector into a signed canonical axis direction.

    ``vector`` is a 3-tuple (dx, dy, dz). The dominant component defines the
    axis; the vector is considered on-axis only if every *other* component is at
    most ``ratio_tol`` times the dominant magnitude (a skew/off-axis extrusion
    returns ``None``, matching the paper's "only when extruded in a specific
    direction" rule).

    Returns a dict ``{"axis": (sx, sy, sz), "word": str}`` or ``None``.
    A zero vector returns ``None``.
    """
    dx, dy, dz = (float(v) for v in vector)
    mags = (abs(dx), abs(dy), abs(dz))
    dom = max(mags)
    if dom == 0.0:
        return None
    dom_i = mags.index(dom)
    for i, m in enumerate(mags):
        if i != dom_i and m > ratio_tol * dom:
            return None  # off-axis / skew
    axis = [0, 0, 0]
    comp = (dx, dy, dz)[dom_i]
    axis[dom_i] = 1 if comp > 0 else -1
    key = tuple(axis)
    return {"axis": key, "word": _AXIS_WORD[key]}


def _signature(pair, ignore=("px", "py", "pz", "origin")):
    """Canonical hashable signature of a pair ignoring the origin fields."""
    if isinstance(pair, dict):
        items = tuple(sorted((k, v) for k, v in pair.items() if k not in ignore))
        return items
    # Fall back to the object itself (already a signature-like value).
    return pair


def collapse_equivalent_components(pairs, threshold: int = DEFAULT_COMPONENT_THRESHOLD):
    """Collapse consecutive equivalent sketch-extrude pairs into components.

    ``pairs`` is an ordered iterable of sketch-extrude pair descriptors (dicts;
    the origin keys ``px/py/pz`` -- or ``origin`` -- are ignored when testing
    equivalence). A maximal run of equivalent consecutive pairs of length
    ``> threshold`` becomes a single *collapsed* component; otherwise each pair
    in the run stays an individual component.

    Returns a list of component dicts, each with:
      signature : the shared equivalence signature.
      count     : number of pairs in the component.
      indices   : tuple of source indices.
      collapsed : True iff the run was collapsed (count > threshold).
    """
    if threshold < 1:
        raise ValueError("threshold must be >= 1")
    seq = list(pairs)
    components = []
    i = 0
    n = len(seq)
    while i < n:
        sig = _signature(seq[i])
        j = i + 1
        while j < n and _signature(seq[j]) == sig:
            j += 1
        run = list(range(i, j))
        if len(run) > threshold:
            components.append({"signature": sig, "count": len(run),
                               "indices": tuple(run), "collapsed": True})
        else:
            for k in run:
                components.append({"signature": _signature(seq[k]), "count": 1,
                                   "indices": (k,), "collapsed": False})
        i = j
    return components
