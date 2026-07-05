"""cadrille verifiable reward shaping and hard-example mining (ICLR 2026).

The RL fine-tuning stage of cadrille scores each sampled Python program with a
programmatically computed reward::

    R(tau) = r_IoU(tau) + r_invalid(tau)

where ``r_IoU`` is the IoU between the CAD model produced by ``tau`` and the
ground-truth mesh, multiplied by ``IOU_SCALE`` (=10) to enforce precise
reconstruction, and ``r_invalid`` penalises predictions that fail to produce a
valid CAD model with ``INVALID_PENALTY`` (=-10), and 0 otherwise.

To speed up convergence cadrille performs *hard-example mining*: only inputs
whose reward, averaged over ``MINING_SAMPLES`` (=3) SFT samples, falls below the
threshold ``HARD_MINING_THRESHOLD`` (=7.5) are retained for RL fine-tuning.

Pure-stdlib, deterministic. The IoU / validity signals are supplied by the
caller (they come from executing the program and comparing solids); this module
only defines the reward composition and the mining selection.
"""

from __future__ import annotations

IOU_SCALE = 10.0
INVALID_PENALTY = -10.0
HARD_MINING_THRESHOLD = 7.5
MINING_SAMPLES = 3


def r_iou(iou: float) -> float:
    """Scaled IoU term of the reward. ``iou`` must lie in [0, 1]."""
    value = float(iou)
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"iou must be in [0, 1], got {value!r}")
    return IOU_SCALE * value


def r_invalid(valid: bool) -> float:
    """Validity term: 0 for a valid program, INVALID_PENALTY otherwise."""
    return 0.0 if valid else INVALID_PENALTY


def cadrille_reward(iou: float, valid: bool = True) -> float:
    """Total reward R(tau) = r_IoU + r_invalid.

    An invalid program short-circuits to the penalty (its IoU is undefined), so
    ``iou`` is ignored when ``valid`` is False.
    """
    if not valid:
        return r_invalid(False)
    return r_iou(iou) + r_invalid(True)


def reward_components(iou: float, valid: bool = True) -> dict:
    """Return the individual reward terms and their total for inspection."""
    if not valid:
        return {"r_iou": 0.0, "r_invalid": INVALID_PENALTY, "total": INVALID_PENALTY}
    ri = r_iou(iou)
    return {"r_iou": ri, "r_invalid": 0.0, "total": ri}


def mean_reward(rewards) -> float:
    """Mean of a non-empty sequence of rewards."""
    values = [float(r) for r in rewards]
    if not values:
        raise ValueError("rewards must be non-empty")
    return sum(values) / len(values)


def is_hard_example(rewards, threshold: float = HARD_MINING_THRESHOLD) -> bool:
    """An example is 'hard' when its mean sampled reward is below ``threshold``."""
    return mean_reward(rewards) < float(threshold)


def mine_hard_examples(samples, threshold: float = HARD_MINING_THRESHOLD):
    """Filter ``(key, rewards)`` pairs down to the hard examples.

    ``samples`` is an iterable of ``(key, rewards)`` where ``rewards`` is the
    list of per-sample rewards produced by the frozen SFT model for that input.
    Returns the ordered list of ``key`` values retained for RL fine-tuning.
    """
    kept = []
    for key, rewards in samples:
        if is_hard_example(rewards, threshold):
            kept.append(key)
    return kept


def mining_report(samples, threshold: float = HARD_MINING_THRESHOLD) -> dict:
    """Summarise a hard-mining pass: total, kept, retained fraction, threshold."""
    items = list(samples)
    kept = mine_hard_examples(items, threshold)
    total = len(items)
    return {
        "total": total,
        "kept": len(kept),
        "retained_fraction": (len(kept) / total) if total else 0.0,
        "threshold": float(threshold),
        "kept_keys": kept,
    }
