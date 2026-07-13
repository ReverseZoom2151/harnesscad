"""GIFT bootstrapping self-training loop and inference-time-scaling analysis.

Amortizes inference-time geometric search into supervised training data. Given a
base image-to-CAD dataset and a sampler that draws K candidate programs per image
from the current policy, each round:

  1. samples candidates (inference-time scaling as a *data-generation* tool);
  2. scores them by geometric agreement (IoU) with the ground truth;
  3. keeps high-agreement diverse programs (SRS) and near-miss failures (FDA)
     via dataengine.gift_geometric_feedback;
  4. accumulates them into a growing augmented pool (keeping the best-IoU
     representative of each distinct (image, program) pair across rounds).

Also provides the two deterministic ITS-analysis quantities used in the paper:
the amortization gap (pass@k oracle minus pass@1) and the inverse-temperature
budget schedule (small budgets sample with a wider/higher temperature to explore;
large budgets clamp to low temperature for precision).

Distinct from vlmcadcode_verify_loop (VLM Q&A). Deterministic, stdlib-only;
sampling is delegated to an injected callback so no wall clock / RNG is used
here.
"""

from __future__ import annotations

from harnesscad.data.dataengine.selftrain.gift_geometric_feedback import (
    TAU_LOW, TAU_MATCH, TAU_VALID, Candidate, augment_example,
)


def inverse_temperature_schedule(budgets, temp_high=1.2, temp_low=0.2):
    """Assign a sampling temperature per compute budget (inverse strategy).

    Smaller budgets (e.g. N=8) get the widest/highest temperature to maximise
    exploration; larger budgets (e.g. N=128) get the lowest temperature to
    prioritise precision. Temperatures interpolate linearly in log2(N) between
    ``temp_high`` (min budget) and ``temp_low`` (max budget). Returns a dict
    mapping each budget to its temperature. A singleton budget maps to
    ``temp_low``.
    """
    if not budgets:
        raise ValueError("budgets must be non-empty")
    if temp_low > temp_high:
        raise ValueError("require temp_low <= temp_high")
    uniq = sorted({int(b) for b in budgets})
    if any(b <= 0 for b in uniq):
        raise ValueError("budgets must be positive")
    from math import log2
    lo, hi = log2(uniq[0]), log2(uniq[-1])
    span = hi - lo
    out = {}
    for b in uniq:
        if span == 0:
            out[b] = temp_low
        else:
            frac = (log2(b) - lo) / span
            out[b] = temp_high + frac * (temp_low - temp_high)
    return out


def pass_at_1(per_image_scores):
    """Mean single-shot IoU: average of the first candidate's score per image."""
    vals = []
    for scores in per_image_scores:
        scores = list(scores)
        if not scores:
            raise ValueError("each image needs at least one candidate")
        vals.append(float(scores[0]))
    if not vals:
        raise ValueError("need at least one image")
    return sum(vals) / len(vals)


def pass_at_k(per_image_scores):
    """Oracle best-of-k IoU: average of the max candidate score per image."""
    vals = []
    for scores in per_image_scores:
        scores = list(scores)
        if not scores:
            raise ValueError("each image needs at least one candidate")
        vals.append(max(float(s) for s in scores))
    if not vals:
        raise ValueError("need at least one image")
    return sum(vals) / len(vals)


def amortization_gap(per_image_scores):
    """Gap between oracle pass@k and single-shot pass@1 (Table 3).

    A smaller gap means the model's quality is better amortized into a single
    generation. Returns absolute gap, its two components, and the relative gap.
    """
    p1 = pass_at_1(per_image_scores)
    pk = pass_at_k(per_image_scores)
    gap = pk - p1
    return {
        "pass_at_1": p1,
        "pass_at_k": pk,
        "gap": gap,
        "relative_gap": (gap / pk) if pk > 0 else 0.0,
    }


def _merge_pairs(pool, new_pairs, scores):
    """Insert (key -> program) pairs into ``pool`` keeping the higher IoU.

    ``pool`` maps (key, program) -> iou. ``scores`` gives the IoU used for a
    dedup tie-break; SRS/FDA pairs from augment_example carry no score so we use
    the presence, storing the max seen. Returns number of newly added keys.
    """
    added = 0
    for (key, program), iou in zip(new_pairs, scores):
        k = (key, program)
        if k not in pool:
            pool[k] = iou
            added += 1
        elif iou > pool[k]:
            pool[k] = iou
    return added


def bootstrap_round(base_dataset, sampler, render_fn=None,
                    tau_low=TAU_LOW, tau_valid=TAU_VALID, tau_match=TAU_MATCH):
    """One offline bootstrapping pass over the base dataset.

    ``sampler(image_id, gt_code) -> iterable[Candidate]`` draws candidates for
    an image. Returns a dict with the SRS and FDA pairs collected this round and
    band counts. This is the render-compare-correct step amortized over the
    whole dataset for a single round.
    """
    srs_all, fda_all, records = [], [], []
    for image_id, gt_code in base_dataset:
        cands = [c if isinstance(c, Candidate) else Candidate(*c)
                 for c in sampler(image_id, gt_code)]
        rec = augment_example(image_id, gt_code, cands, render_fn,
                              tau_low, tau_valid, tau_match)
        srs_all.extend(rec.srs_pairs)
        fda_all.extend(rec.fda_pairs)
        records.append(rec)
    return {"srs_pairs": srs_all, "fda_pairs": fda_all, "records": records,
            "srs": len(srs_all), "fda": len(fda_all)}


def bootstrap_selftrain(base_dataset, sampler, rounds=3, render_fn=None,
                        tau_low=TAU_LOW, tau_valid=TAU_VALID, tau_match=TAU_MATCH,
                        stop_when_no_growth=True):
    """Iterative self-training data-selection loop.

    Runs up to ``rounds`` bootstrapping passes, accumulating SRS and FDA pairs
    into growing pools (de-duplicated, keeping the highest-IoU representative).
    Stops early when a round adds no new pairs (``stop_when_no_growth``). Returns
    the final SRS/FDA pair lists, the combined augmented dataset (base first),
    and per-round growth history.

    Note: this loop only *selects* data. Re-fitting the policy on the augmented
    set (the learned synthesizer) is external and out of scope.
    """
    base = list(base_dataset)
    srs_pool, fda_pool = {}, {}
    history = []
    for r in range(rounds):
        out = bootstrap_round(base, sampler, render_fn,
                              tau_low, tau_valid, tau_match)
        # SRS pairs are (image_id, program); score them by absence -> use 1.0
        # tie value is irrelevant here (dedup only), so store a constant.
        srs_added = _merge_pairs(srs_pool, out["srs_pairs"],
                                 [1.0] * len(out["srs_pairs"]))
        fda_added = _merge_pairs(fda_pool, out["fda_pairs"],
                                 [1.0] * len(out["fda_pairs"]))
        history.append({"round": r, "srs_added": srs_added,
                        "fda_added": fda_added,
                        "srs_total": len(srs_pool), "fda_total": len(fda_pool)})
        if stop_when_no_growth and srs_added == 0 and fda_added == 0:
            break
    srs_pairs = [k for k in srs_pool]
    fda_pairs = [k for k in fda_pool]
    augmented = list(base) + srs_pairs + fda_pairs
    return {
        "srs_pairs": srs_pairs,
        "fda_pairs": fda_pairs,
        "augmented": augmented,
        "counts": {"base": len(base), "srs": len(srs_pairs),
                   "fda": len(fda_pairs), "total": len(augmented)},
        "history": history,
    }
