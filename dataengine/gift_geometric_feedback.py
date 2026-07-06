"""GIFT geometric-feedback mechanism for image-to-CAD program synthesis.

Implements the deterministic core of "GIFT: Bootstrapping Image-to-CAD Program
Synthesis via Geometric Feedback" (Giannone et al.). The learned synthesizer is
external; what is deterministic and locally buildable is the render-compare-
correct feedback at the *program* level:

  * execute a synthesized CadQuery-like program, compare its geometry to the
    target via a geometric-agreement score f(z) = IoU(z, z_gt) supplied by an
    injected kernel adapter;
  * turn that agreement into a corrective category and a discrepancy signal;
  * partition K sampled candidates by IoU into disjoint bands and build the two
    GIFT augmentation sets:
      - Soft-Rejection Sampling (SRS): output-space augmentation. Retain diverse
        valid alternative programs with tau_valid <= f(z) < tau_match and pair
        them with the ORIGINAL image (broadens the target distribution).
      - Failure-Driven Augmentation (FDA): input-space augmentation. Take
        near-miss hard negatives tau_low <= f(z) < tau_valid, render them back
        into a synthetic input phi(d(z)) and pair that noisy input with the
        ground-truth code (a geometric denoising objective).

Distinct from vlmcadcode_verify_loop (VLM question/answer feedback) and
cad2program (prismatic-box lifting): here the signal is a deterministic geometric
IoU band that selects new supervised training pairs.

Deterministic, stdlib-only. IoU-best computation itself lives in bench.solid_iou.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Default filtering thresholds from the paper (Section 3.1).
TAU_LOW = 0.5      # below this: degenerate / non-executable -> discard.
TAU_VALID = 0.9    # separates recoverable near-miss (FDA) from high fidelity.
TAU_MATCH = 0.99   # near-exact match of the ground truth (excluded from SRS).


def _check_thresholds(tau_low, tau_valid, tau_match):
    if not (0.0 <= tau_low <= tau_valid <= tau_match <= 1.0):
        raise ValueError("require 0 <= tau_low <= tau_valid <= tau_match <= 1")


@dataclass(frozen=True)
class Candidate:
    """One sampled program and its geometric agreement with the target.

    ``iou`` is f(z) in [0, 1]; a non-executable program should be passed with
    iou=0.0 (it falls in the reject band). ``program`` is the code string used
    for de-duplication / diversity.
    """

    program: str
    iou: float

    def __post_init__(self):
        if not (0.0 <= self.iou <= 1.0):
            raise ValueError("iou must lie in [0, 1]")


def geometric_agreement(generated, target, iou_fn):
    """Render-compare step: f(z) = IoU(Execute(d(z)), Execute(d(z_gt))).

    ``iou_fn`` is the injected geometric kernel comparison (e.g. a closure over
    bench.solid_iou.best_solid_iou). Returns a float in [0, 1].
    """
    score = float(iou_fn(generated, target))
    if not (0.0 <= score <= 1.0):
        raise ValueError("iou_fn must return a value in [0, 1]")
    return score


def srs_indicator(iou, tau_valid=TAU_VALID, tau_match=TAU_MATCH):
    """w_srs(z) = 1[tau_valid <= f(z) < tau_match]: diverse valid alternatives."""
    return 1 if tau_valid <= iou < tau_match else 0


def fda_indicator(iou, tau_low=TAU_LOW, tau_valid=TAU_VALID):
    """w_fda(z) = 1[tau_low <= f(z) < tau_valid]: recoverable near-miss failures."""
    return 1 if tau_low <= iou < tau_valid else 0


def feedback_category(iou, tau_low=TAU_LOW, tau_valid=TAU_VALID, tau_match=TAU_MATCH):
    """Map an agreement score to one of the four disjoint GIFT bands."""
    _check_thresholds(tau_low, tau_valid, tau_match)
    if iou >= tau_match:
        return "match"
    if iou >= tau_valid:
        return "valid"
    if iou >= tau_low:
        return "near_miss"
    return "reject"


def geometric_feedback(iou, tau_low=TAU_LOW, tau_valid=TAU_VALID, tau_match=TAU_MATCH):
    """The corrective signal derived from one geometric comparison.

    Returns agreement (f(z)), discrepancy (1 - f(z)), the band, and which
    augmentation the sample feeds (if any).
    """
    category = feedback_category(iou, tau_low, tau_valid, tau_match)
    return {
        "agreement": float(iou),
        "discrepancy": float(1.0 - iou),
        "category": category,
        "feeds_srs": bool(srs_indicator(iou, tau_valid, tau_match)),
        "feeds_fda": bool(fda_indicator(iou, tau_low, tau_valid)),
    }


def partition_candidates(candidates, tau_low=TAU_LOW, tau_valid=TAU_VALID,
                         tau_match=TAU_MATCH):
    """Partition K candidates into the four disjoint IoU bands.

    ``candidates`` is an iterable of Candidate. Returns a dict with keys
    ``match``, ``valid``, ``near_miss``, ``reject`` mapping to lists of
    Candidate, each list ordered by descending IoU then program string.
    """
    _check_thresholds(tau_low, tau_valid, tau_match)
    buckets = {"match": [], "valid": [], "near_miss": [], "reject": []}
    for cand in candidates:
        buckets[feedback_category(cand.iou, tau_low, tau_valid, tau_match)].append(cand)
    for key in buckets:
        buckets[key].sort(key=lambda c: (-c.iou, c.program))
    return buckets


def _dedup_by_program(candidates, exclude=frozenset()):
    """Keep the highest-IoU representative of each distinct program string.

    Enforces the SRS diversity requirement: multiple identical strings collapse
    to one, and anything in ``exclude`` (e.g. the ground-truth string) is
    dropped. Result ordered by descending IoU then program string.
    """
    best = {}
    for cand in candidates:
        if cand.program in exclude:
            continue
        prev = best.get(cand.program)
        if prev is None or cand.iou > prev.iou:
            best[cand.program] = cand
    return sorted(best.values(), key=lambda c: (-c.iou, c.program))


def build_srs_dataset(image_id, gt_code, candidates, tau_valid=TAU_VALID,
                      tau_match=TAU_MATCH):
    """SRS output augmentation: (original image, diverse valid alt program).

    Selects programs in [tau_valid, tau_match), de-duplicates them and drops any
    that equal the ground-truth string (SRS targets alternatives *distinct* from
    the exact ground truth). Returns a list of (image_id, program) pairs.
    """
    selected = [c for c in candidates if srs_indicator(c.iou, tau_valid, tau_match)]
    diverse = _dedup_by_program(selected, exclude=frozenset({gt_code}))
    return [(image_id, c.program) for c in diverse]


def build_fda_dataset(image_id, gt_code, candidates, render_fn,
                      tau_low=TAU_LOW, tau_valid=TAU_VALID):
    """FDA input augmentation: (synthetic rendered input phi(d(z)), gt code).

    ``render_fn(program) -> synthetic_input`` is the inverse mapping phi that
    projects a failed program's geometry back into the image domain. Near-miss
    programs (deduplicated) become noisy inputs paired with the *ground-truth*
    code, so the model learns to recover z_gt from an imperfect render. Returns
    a list of (synthetic_input, gt_code) pairs. ``image_id`` is accepted for API
    symmetry (the synthetic input replaces it) and is otherwise unused.
    """
    selected = [c for c in candidates if fda_indicator(c.iou, tau_low, tau_valid)]
    diverse = _dedup_by_program(selected)
    return [(render_fn(c.program), gt_code) for c in diverse]


@dataclass
class AugmentationRecord:
    """Per-image outcome of one geometric-feedback pass."""

    image_id: object
    srs_pairs: list = field(default_factory=list)
    fda_pairs: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)


def augment_example(image_id, gt_code, candidates, render_fn=None,
                    tau_low=TAU_LOW, tau_valid=TAU_VALID, tau_match=TAU_MATCH):
    """Run the full render-compare-correct pass for one training image.

    Builds SRS pairs always; FDA pairs only when ``render_fn`` is supplied.
    Returns an AugmentationRecord with band counts.
    """
    _check_thresholds(tau_low, tau_valid, tau_match)
    cands = list(candidates)
    buckets = partition_candidates(cands, tau_low, tau_valid, tau_match)
    srs = build_srs_dataset(image_id, gt_code, cands, tau_valid, tau_match)
    fda = ([] if render_fn is None
           else build_fda_dataset(image_id, gt_code, cands, render_fn,
                                   tau_low, tau_valid))
    counts = {k: len(v) for k, v in buckets.items()}
    counts["srs"] = len(srs)
    counts["fda"] = len(fda)
    return AugmentationRecord(image_id=image_id, srs_pairs=srs, fda_pairs=fda,
                              counts=counts)


def build_augmented_dataset(base_dataset, sampled, render_fn=None,
                            tau_low=TAU_LOW, tau_valid=TAU_VALID,
                            tau_match=TAU_MATCH):
    """Assemble the GIFT training set: base SFT pairs + SRS + FDA.

    ``base_dataset`` is a list of (image_id, gt_code). ``sampled`` maps
    image_id -> iterable of Candidate for that image. Returns a dict with the
    combined ``pairs`` list and per-source ``counts`` (base / srs / fda), plus
    the per-image ``records``.
    """
    base = list(base_dataset)
    srs_all, fda_all, records = [], [], []
    for image_id, gt_code in base:
        cands = list(sampled.get(image_id, ()))
        rec = augment_example(image_id, gt_code, cands, render_fn,
                              tau_low, tau_valid, tau_match)
        srs_all.extend(rec.srs_pairs)
        fda_all.extend(rec.fda_pairs)
        records.append(rec)
    pairs = list(base) + srs_all + fda_all
    counts = {"base": len(base), "srs": len(srs_all), "fda": len(fda_all),
              "total": len(pairs)}
    return {"pairs": pairs, "counts": counts, "records": records}
