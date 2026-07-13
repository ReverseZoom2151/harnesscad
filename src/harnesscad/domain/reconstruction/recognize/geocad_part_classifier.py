"""Simple / complex local-part categorisation for GeoCAD (Zhang et al. 2025).

GeoCAD's complementary captioning (paper Sec. 3.1) routes each local loop to one of
two captioners based on its *internal side types and numbers*:

    "we categorize local parts into simple and complex groups based on their internal
     side types and numbers. Simple parts correspond to common geometric shapes
     (e.g., triangles with three lines, quadrilaterals with four lines, sectors with
     two lines and an arc), making up roughly 50% of the entire set of local parts,
     while complex parts typically exhibit more intricate visual patterns."

Simple parts go to the deterministic vertex-based captioner
(:mod:`geometry.geocad_vertex_caption`); complex parts go to the VLLM (external).
This module implements the deterministic *routing decision* -- it inspects a FlexCAD
:class:`~reconstruction.flexcad_text.Loop` (curve composition) and decides the branch
and shape family. The VLLM captioner itself is out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass

from harnesscad.domain.reconstruction.translate.flexcad_text import LINE, ARC, CIRCLE, Loop

# Captioning branches (paper Sec. 3.1 / Fig. 2).
BRANCH_VERTEX = "vertex"   # simple -> deterministic vertex-based captioning
BRANCH_VLLM = "vllm"       # complex -> external VLLM captioning

# Simple shape families keyed by their curve composition.
FAMILY_TRIANGLE = "triangle"
FAMILY_QUADRILATERAL = "quadrilateral"
FAMILY_CIRCLE = "circle"
FAMILY_SECTOR = "sector"          # two lines + an arc
FAMILY_SEMICIRCLE = "semicircle"  # one line (diameter) + an arc
FAMILY_COMPLEX = "complex"


@dataclass(frozen=True)
class PartClass:
    """Outcome of classifying one local loop."""

    is_simple: bool
    family: str
    branch: str
    n_lines: int
    n_arcs: int
    n_circles: int


def _counts(loop: Loop) -> tuple[int, int, int]:
    n_line = sum(1 for c in loop.curves if c.type == LINE)
    n_arc = sum(1 for c in loop.curves if c.type == ARC)
    n_circle = sum(1 for c in loop.curves if c.type == CIRCLE)
    return n_line, n_arc, n_circle


def classify_loop(loop: Loop) -> PartClass:
    """Categorise a local loop as simple (vertex branch) or complex (VLLM branch).

    Simple templates (paper Sec. 3.1 + appendix C):

    * 3 lines           -> triangle
    * 4 lines           -> quadrilateral
    * exactly 1 circle  -> circle
    * 2 lines + 1 arc   -> sector (quarter / three-quarter / arc loop)
    * 1 line  + 1 arc   -> semicircle-like loop

    Anything else is complex.
    """
    n_line, n_arc, n_circle = _counts(loop)
    total = len(loop.curves)

    family = FAMILY_COMPLEX
    if n_circle == 1 and total == 1:
        family = FAMILY_CIRCLE
    elif n_arc == 0 and n_circle == 0 and n_line == 3:
        family = FAMILY_TRIANGLE
    elif n_arc == 0 and n_circle == 0 and n_line == 4:
        family = FAMILY_QUADRILATERAL
    elif n_circle == 0 and n_arc == 1 and n_line == 2:
        family = FAMILY_SECTOR
    elif n_circle == 0 and n_arc == 1 and n_line == 1:
        family = FAMILY_SEMICIRCLE

    is_simple = family != FAMILY_COMPLEX
    branch = BRANCH_VERTEX if is_simple else BRANCH_VLLM
    return PartClass(is_simple, family, branch, n_line, n_arc, n_circle)


def partition(loops: list[Loop]) -> tuple[list[int], list[int]]:
    """Split loop indices into (simple_indices, complex_indices)."""
    simple: list[int] = []
    complex_: list[int] = []
    for i, lp in enumerate(loops):
        (simple if classify_loop(lp).is_simple else complex_).append(i)
    return simple, complex_
