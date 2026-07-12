"""Segmentation-to-answer grounding for CAD question answering
(Kienle et al., "QueryCAD: Grounded Question Answering for CAD Models",
Sec. III-A "SegCAD" and III-A.4 "View-Specific Retrieval").

QueryCAD grounds a question by first *segmenting* the CAD model: SegCAD renders
the model, runs open-vocabulary detection (GroundingDINO + SAM) to select the
parts whose appearance matches the free-text part description, then filters those
parts by (a) a coverage cap -- masks covering more than 45 % of the model are
discarded because they are the whole model, not a part (Sec. III-A.2) -- and
(b) view visibility -- for "view-related questions" only parts actually visible
from the requested sides are kept (Sec. III-A.4).

The learned appearance model (DINO/SAM) is external and skipped. This module
implements the DETERMINISTIC grounding logic that surrounds it, operating on an
already-parsed structured CAD model whose parts carry a canonical feature label,
measured attributes, a set of sides they are visible from, and a coverage
fraction. It reuses the open-vocabulary label space of
:mod:`fabrication.mfgfeat_taxonomy`: a free-text description is normalised onto a
canonical leaf label (so "thru hole", "bore", "drilled hole" all select
``hole`` parts), which is exactly the open-set matching SegCAD provides.

Given a structured model and a part description (+ optional required views),
:func:`ground` returns the deterministically-selected parts -- the bridge from
"segmentation" to the answer engine (:mod:`reconstruction.querycad_answer_engine`).

Pure, deterministic, stdlib-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional, Tuple

from fabrication.mfgfeat_taxonomy import try_normalize

# Default coverage cap: SegCAD discards masks covering > 45 % of the model
# (Sec. III-A.2), assuming they are the whole model rather than a part.
DEFAULT_MAX_COVERAGE = 0.45

_MODEL_WORDS = frozenset({"", "model", "object", "part", "the object",
                          "the model", "whole model", "everything", "all"})


@dataclass(frozen=True)
class Part:
    """A segmentable part of a structured CAD model.

    ``id``            unique identifier within the model.
    ``feature``       canonical leaf label (from mfgfeat_taxonomy) or free label.
    ``attrs``         measured attributes, e.g. {"diameter": 5.0, ...}.
    ``visible_views`` sides the part is visible from (subset of the 6 views).
    ``coverage``      fraction (0..1] of the rendered model the part occupies.
    ``aliases``       extra free-text names that should also select this part.
    """
    id: str
    feature: str
    attrs: Dict[str, float] = field(default_factory=dict)
    visible_views: FrozenSet[str] = frozenset()
    coverage: float = 0.0
    aliases: Tuple[str, ...] = ()

    def __post_init__(self):
        if not (0.0 <= self.coverage <= 1.0):
            raise ValueError("coverage must be in [0, 1], got %r"
                             % (self.coverage,))
        object.__setattr__(self, "visible_views",
                           frozenset(str(v).strip().lower()
                                     for v in self.visible_views))
        object.__setattr__(self, "aliases",
                           tuple(str(a).strip().lower() for a in self.aliases))


def _norm(text):
    return " ".join(str(text).strip().lower().replace("-", " ")
                    .replace("_", " ").replace("/", " ").split())


def matches_description(part, description):
    """True iff ``part`` matches the free-text ``description``.

    Matching is open-vocabulary via the mfgfeat taxonomy: the description is
    normalised onto a canonical leaf label and compared to the part's feature.
    Direct id / feature / alias equality also matches. An empty / "model"
    description matches nothing here (the whole-model case is handled upstream).
    """
    desc = _norm(description)
    if desc in _MODEL_WORDS:
        return False

    # Direct identity / label / alias hit.
    if desc == _norm(part.id) or desc == _norm(part.feature):
        return True
    if desc in part.aliases:
        return True

    # Open-vocabulary feature match via the taxonomy normaliser.
    desc_leaf = try_normalize(desc)
    part_leaf = try_normalize(part.feature)
    if desc_leaf is not None and part_leaf is not None:
        return desc_leaf == part_leaf
    return False


def visible_from(part, views):
    """True iff the part is visible from at least one of ``views``.

    Empty ``views`` means no view constraint => always visible.
    """
    wanted = frozenset(str(v).strip().lower() for v in views)
    if not wanted:
        return True
    return bool(part.visible_views & wanted)


def ground(parts, description, *, views=(), max_coverage=DEFAULT_MAX_COVERAGE):
    """Select the parts a question grounds on.

    Applies, in order:
      1. coverage cap  -- discard parts covering > ``max_coverage`` of the model
                          (SegCAD's whole-model filter, Sec. III-A.2);
      2. description match -- open-vocabulary feature/label match;
      3. view filter   -- keep only parts visible from a requested side
                          (Sec. III-A.4).

    Returns a tuple of matching :class:`Part` in input order (deterministic).
    """
    out = []
    for p in parts:
        if not isinstance(p, Part):
            raise TypeError("expected Part, got %r" % (type(p).__name__,))
        if p.coverage > max_coverage:
            continue
        if not matches_description(p, description):
            continue
        if not visible_from(p, views):
            continue
        out.append(p)
    return tuple(out)


def ground_ids(parts, description, **kw):
    """Convenience: :func:`ground` returning only the matched part ids."""
    return tuple(p.id for p in ground(parts, description, **kw))
