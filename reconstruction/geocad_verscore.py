"""Ver-score: vertex-based text-to-CAD consistency for GeoCAD (Zhang et al. 2025).

GeoCAD proposes a *vertex-based score* (Ver-score) to evaluate whether generated
**simple** local parts follow the user's geometric instruction (paper Sec. 4.1 #3,
appendix D):

    "to compute Ver-score, we extract vertex coordinates from the generated local
     parts and analyze their geometric attributes to determine whether they align
     with the given geometric instructions."

This is the deterministic local-edit *controllability* metric. It re-captions a
generated local part via the same vertex-based captioner used to build the training
instructions (:mod:`geometry.geocad_vertex_caption`) and checks whether the recovered
caption matches the target instruction. Since the captioner is similarity-invariant,
Ver-score is invariant to where/how the part was placed -- it measures shape identity.
The complementary VLLM-score (for complex parts) and human Realism are external.

The target instruction is normalised (lower-cased, leading article and any trailing
"with <dimensions>" clause stripped) so that, e.g., "An Isosceles Right Triangle" and
"an isosceles right triangle with side 10" both match a recovered
"an isosceles right triangle".
"""

from __future__ import annotations

from dataclasses import dataclass

from geometry.geocad_vertex_caption import caption_polygon, caption_arc_loop


def normalise_instruction(text: str) -> str:
    """Canonicalise a geometric instruction for comparison.

    Lower-cases, strips a trailing dimensional clause ("... with radius 5"), and
    removes a leading indefinite article so shape identity is compared.
    """
    s = text.strip().lower()
    if " with " in s:
        s = s.split(" with ", 1)[0].strip()
    for article in ("an ", "a "):
        if s.startswith(article):
            s = s[len(article):]
            break
    return s


def _shape_key(caption: str) -> str:
    return normalise_instruction(caption)


@dataclass(frozen=True)
class VerResult:
    """Per-sample Ver-score outcome."""

    matched: bool
    recovered: str
    target: str


def score_polygon(vertices: list[tuple[float, float]], instruction: str) -> VerResult:
    """Ver-score a straight-sided generated part against its instruction."""
    recovered = caption_polygon(vertices)
    return VerResult(
        _shape_key(recovered) == _shape_key(instruction), recovered, instruction
    )


def score_arc_loop(sweep_deg: float, instruction: str) -> VerResult:
    """Ver-score an arc-bounded generated part against its instruction."""
    recovered = caption_arc_loop(sweep_deg)
    return VerResult(
        _shape_key(recovered) == _shape_key(instruction), recovered, instruction
    )


def ver_score(results: list[VerResult]) -> float:
    """Aggregate Ver-score = fraction of samples whose shape matches its instruction."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.matched) / len(results)
