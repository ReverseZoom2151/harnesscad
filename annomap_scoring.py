"""annomap_scoring — deterministic 2D-entity -> 3D-feature correspondence scoring.

This implements the *deterministic-first* scoring and assignment core of Khan et
al. (Sec. IV-C, IV-D, IV-E) verbatim, over the schema in :mod:`annomap_parser`.
The learned VLM enrichment and the GPT-4o escalation stages are explicitly NOT
here (they are research-heavy/external); what is here is the interpretable,
reproducible metric that produces the candidate ranking the paper says "always
produces the initial candidate ranking".

Composite score for a feature/entity pair (Eq. 3):

    S_ij = 0                                     if S_type == 0   (hard type gate)
    S_ij = w_t*S_type + w_d*S_dim + w_c*S_ctx + h_ij   otherwise

with the paper's fixed weights ``w_t = 0.4, w_d = 0.4, w_c = 0.2`` and heuristic
adjustments ``h_ij`` (Sec. IV-E). Components:

  * **Type compatibility** ``S_type`` (Eq. 4): 1.0 for an exact type match,
    0.9 for a semantically-equivalent group ({hole,bore,drill},
    {slot,pocket,groove}, {fillet,round,radius}), else 0 (hard reject).
  * **Dimensional agreement** ``S_dim`` (Eq. 5): stepped tolerance match — 1.0
    within ε, 0.7 within 2ε, else 0, routed to the geometrically appropriate 3D
    property. A radius entity also checks against ``diameter/2``. If a numeric 2D
    dimension exists but S_dim == 0, the composite is multiplied by 0.3.
  * **Context consistency** ``S_ctx`` (Eq. 6): the VLM confidence when spatial
    cues exist, else the neutral 0.5 (conservative; never overrides type/dim).
  * **Heuristics** ``h_ij`` (Sec. IV-E): Ø-symbol preference, thread->cylinder
    restriction, GD&T attachment priors, plus the "diameter without Ø" penalty.

Assignment (Sec. IV-D) is a greedy near-tie rule (Eq. 7): keep every entity whose
score is within ratio ``ρ = 0.9`` of the best score for that feature, supporting
the one-to-many correspondences that arise across multiple views.

Deterministic and stdlib-only; ``S_ctx`` reads a caller-supplied confidence, never
a live model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from annomap_parser import (
    CADFeature,
    DrawingEntity,
    ENTITY_ANGLE,
    ENTITY_COUNTERBORE,
    ENTITY_COUNTERSINK,
    ENTITY_DATUM,
    ENTITY_DIAMETER,
    ENTITY_GDT,
    ENTITY_LINEAR,
    ENTITY_RADIUS,
    ENTITY_SURFACE_FINISH,
    ENTITY_THREAD,
)

# --------------------------------------------------------------------------- #
# Fixed constants from the paper (held out on 5 parts, fixed thereafter).
# --------------------------------------------------------------------------- #
W_TYPE = 0.4
W_DIM = 0.4
W_CTX = 0.2
EPSILON = 0.1        # mm, base dimensional tolerance
THETA_CAND = 0.3     # candidate retention threshold
RHO = 0.9            # near-tie ratio
NEUTRAL_CTX = 0.5    # S_ctx when spatial cues unavailable

# Semantic-equivalence groups (Omega_semantic, Eq. 4). Each entity's inferred
# target-feature category is compared against a 3D feature type through these.
_SEMANTIC_GROUPS: Tuple[frozenset, ...] = (
    frozenset({"hole", "bore", "drill", "counterbore", "countersink",
               "cylinder", "cylindrical", "boss"}),
    frozenset({"slot", "pocket", "groove"}),
    frozenset({"fillet", "round", "radius"}),
    frozenset({"chamfer", "bevel"}),
    frozenset({"plane", "planar", "face", "surface"}),
)

# Which 3D property a given entity type constrains (dimensional routing).
_PROPERTY_FOR_ENTITY: Dict[str, str] = {
    ENTITY_DIAMETER: "diameter",
    ENTITY_THREAD: "diameter",
    ENTITY_COUNTERBORE: "diameter",
    ENTITY_COUNTERSINK: "diameter",
    ENTITY_RADIUS: "radius",
    ENTITY_ANGLE: "angle",
    ENTITY_LINEAR: "length",
}

_CYLINDRICAL = frozenset({"hole", "bore", "drill", "cylinder", "cylindrical",
                          "boss", "counterbore", "countersink", "shaft"})
_PLANAR = frozenset({"plane", "planar", "face", "surface"})


@dataclass
class ScoreBreakdown:
    """Full, auditable breakdown of a single (feature, entity) score."""

    feature_id: str
    entity_id: str
    s_type: float
    s_dim: float
    s_ctx: float
    heuristic: float
    composite: float
    rationale: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "feature_id": self.feature_id,
            "entity_id": self.entity_id,
            "s_type": self.s_type,
            "s_dim": self.s_dim,
            "s_ctx": self.s_ctx,
            "heuristic": self.heuristic,
            "composite": self.composite,
            "rationale": list(self.rationale),
        }


def _same_semantic_group(a: str, b: str) -> bool:
    a = (a or "").lower()
    b = (b or "").lower()
    for grp in _SEMANTIC_GROUPS:
        if a in grp and b in grp:
            return True
    return False


def type_compatibility(feature: CADFeature,
                       entity: DrawingEntity) -> Tuple[float, str]:
    """S_type (Eq. 4): 1.0 exact, 0.9 semantic group, else 0 (hard gate)."""
    ft = (feature.feature_type or "").lower()
    target = (entity.target_feature or "").lower()

    # Notes / bare datums / surface-finish / GD&T have no scalar target type; they
    # attach via heuristics, so they clear the hard gate with a neutral semantic
    # score rather than being rejected outright.
    if entity.entity_type in (ENTITY_GDT, ENTITY_DATUM, ENTITY_SURFACE_FINISH):
        return 0.9, "non-dimensional entity clears type gate (attaches via priors)"

    if not target:
        # An ambiguous bare linear dimension can constrain any feature's length.
        if entity.entity_type == ENTITY_LINEAR:
            return 0.9, "linear dimension can constrain any feature length"
        return 0.0, "no inferred target feature -> type gate rejects"

    if ft == target:
        return 1.0, "exact type match %s<->%s" % (ft, target)
    if _same_semantic_group(ft, target):
        return 0.9, "semantic group match %s<->%s" % (ft, target)
    return 0.0, "type incompatible %s vs %s (hard reject)" % (ft, target)


def dimensional_agreement(feature: CADFeature,
                          entity: DrawingEntity,
                          epsilon: float = EPSILON) -> Tuple[float, bool, str]:
    """S_dim (Eq. 5). Returns ``(score, has_numeric, rationale)``.

    ``has_numeric`` reports whether the 2D entity carried a scalar at all — the
    caller needs it for the "numeric dim but S_dim==0 -> x0.3" suppression.
    """
    x2d = entity.value
    if x2d is None:
        return 0.0, False, "no numeric 2D value -> dimensional agreement N/A"

    prop = _PROPERTY_FOR_ENTITY.get(entity.entity_type)
    candidates: List[Tuple[str, float]] = []
    if prop and prop in feature.params:
        candidates.append((prop, feature.params[prop]))
    # Radius also matches diameter/2 (Sec. IV-C).
    if entity.entity_type == ENTITY_RADIUS and "diameter" in feature.params:
        candidates.append(("diameter/2", feature.params["diameter"] / 2.0))
    # Diameter-type entity may match a stored radius*2 symmetrically.
    if prop == "diameter" and "diameter" not in feature.params \
            and "radius" in feature.params:
        candidates.append(("radius*2", feature.params["radius"] * 2.0))
    # A bare linear dimension may match any stored scalar; pick the best.
    if prop == "length" or prop is None:
        for k, v in feature.params.items():
            candidates.append((k, v))

    if not candidates:
        return 0.0, True, "no comparable 3D property for %s" % entity.entity_type

    best_score = 0.0
    best_prop = ""
    best_delta = None
    for name, x3d in candidates:
        delta = abs(x2d - x3d)
        if delta <= epsilon:
            score = 1.0
        elif delta <= 2.0 * epsilon:
            score = 0.7
        else:
            score = 0.0
        if score > best_score or (score == best_score and
                                  (best_delta is None or delta < best_delta)):
            best_score = score
            best_prop = name
            best_delta = delta

    return best_score, True, "dim %s vs %s |Δ|=%.4g -> %.1f" % (
        entity.entity_type, best_prop, best_delta if best_delta is not None else -1,
        best_score)


def context_consistency(entity: DrawingEntity,
                        vlm_confidence: Optional[float] = None) -> Tuple[float, str]:
    """S_ctx (Eq. 6): the VLM confidence when spatial cues exist, else 0.5.

    Spatial cues are considered available when the caller supplies a confidence
    or the entity context carries a bbox / view association.
    """
    if vlm_confidence is not None:
        v = max(0.0, min(1.0, float(vlm_confidence)))
        return v, "spatial cue present -> S_ctx=%.3f" % v
    ctx = entity.context or {}
    if "confidence" in ctx:
        v = max(0.0, min(1.0, float(ctx["confidence"])))
        return v, "context confidence -> S_ctx=%.3f" % v
    if ("bbox" in ctx or "view" in ctx) and ctx.get("spatial", True):
        # A cue exists but no confidence value: fall back to neutral.
        return NEUTRAL_CTX, "spatial cue but no confidence -> neutral 0.5"
    return NEUTRAL_CTX, "spatial cues unavailable -> neutral 0.5"


def engineering_heuristics(feature: CADFeature,
                           entity: DrawingEntity) -> Tuple[float, float, List[str]]:
    """h_ij plus post-multipliers (Sec. IV-E).

    Returns ``(additive_h, multiplier, rationale)``. The multiplier captures the
    "diameter labelled but no Ø symbol -> x0.7" and thread/GD&T restrictions.
    """
    h = 0.0
    mult = 1.0
    notes: List[str] = []
    ft = (feature.feature_type or "").lower()

    # Diameter-symbol preference for hole features.
    if entity.entity_type == ENTITY_DIAMETER and ft in _CYLINDRICAL:
        if entity.symbol == "Ø" or "Ø" in (entity.raw_text or ""):
            h += 0.1
            notes.append("Ø symbol on hole -> +0.1")
        else:
            mult *= 0.7
            notes.append("diameter without Ø symbol -> x0.7")

    # Thread callouts restricted to cylindrical features.
    if entity.entity_type == ENTITY_THREAD:
        if ft in _CYLINDRICAL:
            notes.append("thread on cylindrical feature (allowed)")
        else:
            mult *= 0.0
            notes.append("thread on non-cylindrical feature -> reject")

    # GD&T attachment priors.
    if entity.entity_type == ENTITY_GDT:
        sym = str(entity.symbol or "")
        if sym in ("position", "profile_of_a_surface", "profile_of_a_line"):
            if ft in _CYLINDRICAL or ft in ("pocket", "slot", "groove"):
                h += 0.1
                notes.append("position/profile prior -> hole/pocket +0.1")
        if "runout" in sym:
            if ft in _CYLINDRICAL:
                h += 0.1
                notes.append("runout prior -> cylindrical +0.1")
            else:
                mult *= 0.3
                notes.append("runout on non-cylindrical -> x0.3")

    # Datum references prefer planar or cylindrical primitives.
    if entity.entity_type == ENTITY_DATUM:
        if ft in _PLANAR or ft in _CYLINDRICAL:
            h += 0.1
            notes.append("datum prior -> planar/cylindrical +0.1")
        else:
            mult *= 0.5
            notes.append("datum on non-planar/cylindrical -> x0.5")

    return h, mult, notes


def score_pair(feature: CADFeature,
               entity: DrawingEntity,
               vlm_confidence: Optional[float] = None,
               weights: Tuple[float, float, float] = (W_TYPE, W_DIM, W_CTX),
               epsilon: float = EPSILON) -> ScoreBreakdown:
    """Compute the full composite score S_ij for one (feature, entity) pair."""
    w_t, w_d, w_c = weights
    rationale: List[str] = []

    s_type, r = type_compatibility(feature, entity)
    rationale.append(r)
    if s_type == 0.0:
        # Hard type gate (Eq. 3): reject immediately.
        return ScoreBreakdown(feature.feature_id, entity.entity_id,
                              0.0, 0.0, 0.0, 0.0, 0.0, rationale)

    s_dim, has_numeric, r = dimensional_agreement(feature, entity, epsilon)
    rationale.append(r)
    s_ctx, r = context_consistency(entity, vlm_confidence)
    rationale.append(r)
    h, mult, hnotes = engineering_heuristics(feature, entity)
    rationale.extend(hnotes)

    composite = w_t * s_type + w_d * s_dim + w_c * s_ctx + h

    # Apply heuristic multiplier (Ø-less penalty, thread/runout restrictions).
    composite *= mult

    # Numeric dimension present but no agreement -> strong suppression (x0.3).
    if has_numeric and s_dim == 0.0:
        composite *= 0.3
        rationale.append("numeric 2D dim but S_dim=0 -> composite x0.3")

    if composite < 0.0:
        composite = 0.0
    return ScoreBreakdown(feature.feature_id, entity.entity_id,
                          s_type, s_dim, s_ctx, h, composite, rationale)


@dataclass
class Assignment:
    """The retained near-tie candidate set for one 3D feature (Eq. 7)."""

    feature_id: str
    entity_ids: List[str] = field(default_factory=list)
    best_score: float = 0.0
    breakdowns: List[ScoreBreakdown] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "feature_id": self.feature_id,
            "entity_ids": list(self.entity_ids),
            "best_score": self.best_score,
            "breakdowns": [b.to_dict() for b in self.breakdowns],
        }


def assign_features(features: Sequence[CADFeature],
                    entities: Sequence[DrawingEntity],
                    vlm_confidence: Optional[Dict[str, float]] = None,
                    theta_cand: float = THETA_CAND,
                    rho: float = RHO,
                    weights: Tuple[float, float, float] = (W_TYPE, W_DIM, W_CTX),
                    epsilon: float = EPSILON) -> List[Assignment]:
    """Deterministic per-feature assignment with greedy near-tie filtering.

    For each feature: score every entity, drop those below ``theta_cand``, then
    keep every entity whose score is within ratio ``rho`` of the best (Eq. 7).
    Ties are broken deterministically by (score desc, entity_id asc). A feature
    with no surviving candidate yields an empty entity list.
    """
    conf = vlm_confidence or {}
    out: List[Assignment] = []
    for feat in features:
        scored: List[ScoreBreakdown] = []
        for ent in entities:
            b = score_pair(feat, ent, vlm_confidence=conf.get(ent.entity_id),
                           weights=weights, epsilon=epsilon)
            if b.composite >= theta_cand:
                scored.append(b)
        if not scored:
            out.append(Assignment(feat.feature_id, [], 0.0, []))
            continue
        best = max(b.composite for b in scored)
        kept = [b for b in scored if b.composite >= rho * best]
        kept.sort(key=lambda b: (-b.composite, b.entity_id))
        out.append(Assignment(feat.feature_id,
                              [b.entity_id for b in kept], best, kept))
    return out
