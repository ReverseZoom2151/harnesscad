"""SldprtNet five-modality aligned sample record.

Each SldprtNet sample (Li et al., ICRA 2026, Sec. III.C) is a fully aligned set
of **five modalities**:

  1. ``sldprt`` - the native SolidWorks part file (feature-tree history);
  2. ``step``   - a neutral STEP export for cross-platform validation;
  3. ``image``  - a single *composite* PNG merging seven rendered views
     (six orthographic - front/back/left/right/top/bottom - and one isometric);
  4. ``encoder_txt`` - the parametric modeling script from the encoder tool
     (see :mod:`reconstruction.sldprtnet_feature_tree`);
  5. ``description`` - a natural-language caption of appearance + function.

This differs from :mod:`dataengine.omnicad_record` (which models text / multiview
image / point-cloud with camera frames) by pinning the *exact seven SldprtNet
standard views* and treating the two geometry file formats (sldprt + step) and
the parametric script as first-class distinct modalities. Only the deterministic
schema and completeness accounting live here (no rendering, no COM, no LLM).

Stdlib-only and deterministic: modality presence is derived from which fields are
populated, and :func:`multimodal_completeness` is a pure function of the record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Tuple

#: The seven standard SldprtNet render views merged into one composite image.
STANDARD_VIEWS: Tuple[str, ...] = (
    "front", "back", "left", "right", "top", "bottom", "isometric",
)
STANDARD_VIEW_SET = frozenset(STANDARD_VIEWS)

#: The five aligned modalities of a SldprtNet sample.
MODALITIES: Tuple[str, ...] = (
    "sldprt", "step", "image", "encoder_txt", "description",
)


@dataclass(frozen=True)
class CompositeImage:
    """A composite render: one digest per merged view plus the merged digest."""

    view_digests: Mapping[str, str] = field(default_factory=dict)
    composite_digest: str = ""

    def __post_init__(self) -> None:
        for name in self.view_digests:
            if name not in STANDARD_VIEW_SET:
                raise ValueError(f"unknown view: {name!r}")

    @property
    def covered_views(self) -> Tuple[str, ...]:
        return tuple(v for v in STANDARD_VIEWS if v in self.view_digests)

    @property
    def is_complete(self) -> bool:
        """True iff all seven standard views are present and merged."""
        return (
            frozenset(self.view_digests) == STANDARD_VIEW_SET
            and bool(self.composite_digest)
        )

    @property
    def view_coverage(self) -> float:
        return len(self.covered_views) / len(STANDARD_VIEWS)


@dataclass(frozen=True)
class SldprtNetRecord:
    """One aligned five-modality SldprtNet sample.

    A field is considered *present* when it is non-empty. ``sldprt_digest`` and
    ``step_digest`` stand in for the two geometry file formats; ``image`` is a
    :class:`CompositeImage`; ``encoder_txt`` is the parametric script text; and
    ``description`` is the natural-language caption.
    """

    id: str
    sldprt_digest: str = ""
    step_digest: str = ""
    image: CompositeImage = field(default_factory=CompositeImage)
    encoder_txt: str = ""
    description: str = ""
    feature_count: int = 0
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise ValueError("record id is required")
        if self.feature_count < 0:
            raise ValueError("feature_count must be non-negative")

    def modality_present(self) -> Dict[str, bool]:
        """Presence flag for each of the five modalities (sorted keys)."""
        return {
            "description": bool(self.description),
            "encoder_txt": bool(self.encoder_txt),
            "image": self.image.is_complete,
            "sldprt": bool(self.sldprt_digest),
            "step": bool(self.step_digest),
        }

    @property
    def present_modalities(self) -> frozenset:
        return frozenset(m for m, ok in self.modality_present().items() if ok)

    @property
    def is_fully_aligned(self) -> bool:
        """True iff all five modalities are present (paper's 'fully aligned')."""
        return len(self.present_modalities) == len(MODALITIES)

    @property
    def completeness(self) -> float:
        """Fraction of the five modalities present (0..1)."""
        return len(self.present_modalities) / len(MODALITIES)


def multimodal_completeness(record: SldprtNetRecord) -> float:
    """Convenience: per-record fraction of the five modalities present."""
    return record.completeness


def modality_coverage(records) -> Dict[str, float]:
    """Per-modality coverage across a collection (fraction of records present).

    Returns a dict keyed by modality name (sorted) mapping to the fraction of
    records in which that modality is present. Empty input yields all zeros.
    """
    records = list(records)
    n = len(records)
    coverage: Dict[str, float] = {m: 0.0 for m in sorted(MODALITIES)}
    if n == 0:
        return coverage
    for rec in records:
        for m, ok in rec.modality_present().items():
            if ok:
                coverage[m] += 1
    return {m: coverage[m] / n for m in sorted(MODALITIES)}


def fully_aligned_rate(records) -> float:
    """Fraction of records with all five modalities present."""
    records = list(records)
    if not records:
        return 0.0
    aligned = sum(1 for r in records if r.is_fully_aligned)
    return aligned / len(records)
