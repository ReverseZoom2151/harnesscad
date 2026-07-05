"""annomap_spec — unified manufacturing specification + 2D-link evaluation metrics.

The final deterministic artefacts of Khan et al. (Sec. IV-G, Fig. 2, and the
evaluation protocol of Sec. V-B) are:

  * a **unified manufacturing specification** that binds each accepted 2D drawing
    constraint to its 3D CAD feature, explicitly lists unmapped entities, and
    records provenance — the mapping *method* (deterministic / VLM-assisted /
    LLM-escalated / human), a confidence, a rationale, and any human edits; and
  * an **evaluation protocol** at the 2D-link level: for each 3D feature the
    predicted set of 2D associations is compared to ground truth via set
    intersection, giving precision, recall and F1 (Eq. 8), plus the exact-match
    and partial-match rates, macro-averaged across parts.

Both are pure, reproducible data operations, so they are implemented here over the
:class:`annomap_scoring.Assignment` output. The escalation *decisions* themselves
(what the VLM / GPT-4o pick) are external; this module only records their
provenance and scores whatever mapping set is produced.

Stdlib-only, deterministic; no wall clock (provenance timestamps are caller-
supplied, defaulting to a stable sentinel).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# Provenance mapping methods (Sec. IV-F resolution pathways).
METHOD_DETERMINISTIC = "deterministic"
METHOD_VLM = "vlm_assisted"
METHOD_LLM = "llm_escalated"
METHOD_HUMAN = "human"

_KNOWN_METHODS = frozenset({METHOD_DETERMINISTIC, METHOD_VLM, METHOD_LLM,
                            METHOD_HUMAN})


@dataclass
class MappingRecord:
    """One accepted feature<->entity binding with provenance (Fig. 2)."""

    feature_id: str
    entity_id: str
    confidence: float
    method: str = METHOD_DETERMINISTIC
    rationale: str = ""
    human_edited: bool = False

    def to_dict(self) -> dict:
        return {
            "feature_id": self.feature_id,
            "entity_id": self.entity_id,
            "confidence": self.confidence,
            "method": self.method,
            "rationale": self.rationale,
            "human_edited": self.human_edited,
        }


@dataclass
class UnifiedSpec:
    """The unified manufacturing specification document."""

    mappings: List[MappingRecord] = field(default_factory=list)
    unmapped_entities: List[str] = field(default_factory=list)
    unmapped_features: List[str] = field(default_factory=list)
    flagged_for_review: List[str] = field(default_factory=list)

    def pairs(self) -> Set[Tuple[str, str]]:
        return {(m.feature_id, m.entity_id) for m in self.mappings}

    def to_dict(self) -> dict:
        return {
            "mappings": [m.to_dict() for m in self.mappings],
            "unmapped_entities": list(self.unmapped_entities),
            "unmapped_features": list(self.unmapped_features),
            "flagged_for_review": list(self.flagged_for_review),
        }


def build_spec(assignments: Sequence["object"],
               all_entity_ids: Sequence[str],
               review_threshold: float = 0.5,
               method: str = METHOD_DETERMINISTIC) -> UnifiedSpec:
    """Assemble a :class:`UnifiedSpec` from scoring assignments.

    ``assignments`` is a sequence of :class:`annomap_scoring.Assignment` (duck-
    typed: needs ``feature_id``, ``entity_ids``, ``best_score`` and
    ``breakdowns``). Every retained (feature, entity) pair becomes a
    :class:`MappingRecord`; features with no candidate are recorded as unmapped;
    entities never mapped are listed as ``unmapped_entities``; any mapping whose
    per-pair composite is below ``review_threshold`` is flagged for HITL review.
    """
    if method not in _KNOWN_METHODS:
        raise ValueError("unknown provenance method: %r" % method)

    spec = UnifiedSpec()
    mapped_entities: Set[str] = set()
    for a in assignments:
        entity_ids = list(getattr(a, "entity_ids", []))
        breakdowns = list(getattr(a, "breakdowns", []))
        score_by_entity = {b.entity_id: b for b in breakdowns}
        if not entity_ids:
            spec.unmapped_features.append(a.feature_id)
            continue
        for eid in entity_ids:
            bd = score_by_entity.get(eid)
            conf = bd.composite if bd is not None else a.best_score
            rationale = "; ".join(bd.rationale) if bd is not None else ""
            spec.mappings.append(MappingRecord(
                feature_id=a.feature_id, entity_id=eid, confidence=conf,
                method=method, rationale=rationale))
            mapped_entities.add(eid)
            if conf < review_threshold:
                spec.flagged_for_review.append("%s<->%s" % (a.feature_id, eid))

    for eid in all_entity_ids:
        if eid not in mapped_entities:
            spec.unmapped_entities.append(eid)
    # Unmapped entities are also review candidates.
    spec.flagged_for_review.extend(
        "unmapped:%s" % eid for eid in spec.unmapped_entities)
    return spec


def apply_human_edit(spec: UnifiedSpec,
                     add: Optional[Iterable[Tuple[str, str]]] = None,
                     remove: Optional[Iterable[Tuple[str, str]]] = None,
                     all_entity_ids: Optional[Sequence[str]] = None) -> UnifiedSpec:
    """Post-fusion HITL edit: add / remove bindings, recording provenance.

    Returns a NEW :class:`UnifiedSpec` (does not mutate the input). Added bindings
    carry ``method=human`` and ``human_edited=True`` with confidence 1.0.
    """
    add = list(add or [])
    remove = set(remove or [])
    kept = [m for m in spec.mappings if (m.feature_id, m.entity_id) not in remove]
    existing = {(m.feature_id, m.entity_id) for m in kept}
    for fid, eid in add:
        if (fid, eid) in existing:
            continue
        kept.append(MappingRecord(feature_id=fid, entity_id=eid, confidence=1.0,
                                  method=METHOD_HUMAN, rationale="human edit",
                                  human_edited=True))
        existing.add((fid, eid))
    mapped = {eid for _, eid in existing}
    ent_ids = list(all_entity_ids) if all_entity_ids is not None else \
        sorted({eid for _, eid in existing} | set(spec.unmapped_entities))
    unmapped = [eid for eid in ent_ids if eid not in mapped]
    return UnifiedSpec(
        mappings=kept,
        unmapped_entities=unmapped,
        unmapped_features=list(spec.unmapped_features),
        flagged_for_review=["unmapped:%s" % e for e in unmapped])


# --------------------------------------------------------------------------- #
# Evaluation (Sec. V-B, Eq. 8)
# --------------------------------------------------------------------------- #

@dataclass
class MappingMetrics:
    precision: float
    recall: float
    f1: float
    exact_match_rate: float
    partial_match_rate: float
    n_features: int

    def to_dict(self) -> dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "exact_match_rate": self.exact_match_rate,
            "partial_match_rate": self.partial_match_rate,
            "n_features": self.n_features,
        }


def _f1(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0.0 else 2.0 * p * r / (p + r)


def evaluate_mapping(predicted: Iterable[Tuple[str, str]],
                     ground_truth: Iterable[Tuple[str, str]],
                     feature_ids: Optional[Sequence[str]] = None) -> MappingMetrics:
    """Precision / recall / F1 + exact & partial match rates at the 2D-link level.

    ``predicted`` and ``ground_truth`` are iterables of ``(feature_id,
    entity_id)`` pairs (Eq. 8: M and M*). Global precision/recall are computed on
    the pair sets; exact/partial match rates are per-feature (a feature's
    predicted 2D-entity set exactly equals / non-trivially intersects its ground-
    truth set), averaged over ``feature_ids`` (defaulting to all features that
    appear in either set).
    """
    pred = set(predicted)
    gt = set(ground_truth)
    inter = pred & gt
    precision = 1.0 if not pred else len(inter) / len(pred)
    recall = 1.0 if not gt else len(inter) / len(gt)
    f1 = _f1(precision, recall)

    # Per-feature entity sets.
    def _by_feature(pairs: Set[Tuple[str, str]]) -> Dict[str, Set[str]]:
        d: Dict[str, Set[str]] = {}
        for fid, eid in pairs:
            d.setdefault(fid, set()).add(eid)
        return d

    pred_f = _by_feature(pred)
    gt_f = _by_feature(gt)
    if feature_ids is None:
        feats = sorted(set(pred_f) | set(gt_f))
    else:
        feats = list(feature_ids)

    if not feats:
        return MappingMetrics(precision, recall, f1, 1.0, 1.0, 0)

    exact = 0
    partial = 0
    for fid in feats:
        p = pred_f.get(fid, set())
        g = gt_f.get(fid, set())
        if p == g:
            exact += 1
        if p & g:
            partial += 1
    n = len(feats)
    return MappingMetrics(precision, recall, f1,
                          exact / n, partial / n, n)


def macro_average(per_part: Sequence[MappingMetrics]) -> MappingMetrics:
    """Macro-average per-part metrics across parts (Sec. V-B)."""
    if not per_part:
        return MappingMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0)
    n = len(per_part)
    return MappingMetrics(
        precision=sum(m.precision for m in per_part) / n,
        recall=sum(m.recall for m in per_part) / n,
        f1=sum(m.f1 for m in per_part) / n,
        exact_match_rate=sum(m.exact_match_rate for m in per_part) / n,
        partial_match_rate=sum(m.partial_match_rate for m in per_part) / n,
        n_features=sum(m.n_features for m in per_part),
    )
