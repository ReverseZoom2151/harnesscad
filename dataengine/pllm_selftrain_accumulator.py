"""PLLM iterative self-training accumulator with drift detection.

PLLM (Section 3) runs a self-reinforcing loop: each iteration samples programs,
selects the best-of-k pseudo-label per shape (see
``dataengine.pllm_pseudo_label_selection``), and accumulates the accepted
(shape, program) pairs into a growing synthetic training set that is reused
across iterations. This module models that accumulator.

Two PLLM-specific concerns are implemented deterministically:

  * De-duplication across rounds. Because the same shape can be re-labelled in
    later, improved iterations, the pool keeps at most ONE label per shape and
    replaces it only when a later round produces a strictly lower Chamfer
    Distance. This keeps the pool anchored to executions of the target-domain
    shapes (PLLM's partial-Criterion-3 anchoring).

  * Drift detection. PLLM's ablation (Section 5.4.1, Baseline 1) observes that
    training on generated-program executions "increasingly drifts from the
    target shape distribution", degrading performance. We detect this by
    tracking the mean Chamfer of the accepted pool per round: a sustained
    increase (worse fidelity) across rounds signals drift and can trigger an
    early stop.

Also provides the label-efficiency metric: how much supervision is obtained per
unit of unlabeled data (coverage of the shape pool that yielded accepted
labels, and Chamfer improvement per round).

Distinct from ``gift_bootstrap_loop`` (IoU SRS/FDA band accumulation keyed by
(image, program) with best-IoU tie-break): here dedup is keyed by *shape* only,
keeps a single improving label per shape, uses Chamfer, and adds drift
detection. Deterministic, stdlib-only.
"""

from __future__ import annotations


class PseudoLabelPool:
    """Accumulates one best (lowest-Chamfer) pseudo-label per shape across rounds."""

    def __init__(self):
        # shape_id -> {"program", "chamfer", "round", "length"}
        self._labels = {}

    def __len__(self):
        return len(self._labels)

    def get(self, shape_id):
        return self._labels.get(shape_id)

    def items(self):
        return dict(self._labels)

    def add_round(self, round_index, accepted):
        """Merge a round's accepted labels; keep strictly lower-Chamfer entries.

        ``accepted`` is an iterable of dicts with keys ``shape_id``,
        ``program``, ``chamfer`` and optionally ``length``. Returns a dict with
        counts of newly added shapes, improved (replaced) shapes, and unchanged
        shapes for this round.
        """
        added = improved = unchanged = 0
        for rec in accepted:
            sid = rec["shape_id"]
            cd = float(rec["chamfer"])
            entry = {"program": rec["program"], "chamfer": cd,
                     "round": round_index, "length": rec.get("length")}
            cur = self._labels.get(sid)
            if cur is None:
                self._labels[sid] = entry
                added += 1
            elif cd < cur["chamfer"]:
                self._labels[sid] = entry
                improved += 1
            else:
                unchanged += 1
        return {"round": round_index, "added": added, "improved": improved,
                "unchanged": unchanged, "pool_size": len(self._labels)}

    def mean_chamfer(self):
        """Mean Chamfer Distance over the current pool (None when empty)."""
        if not self._labels:
            return None
        return sum(e["chamfer"] for e in self._labels.values()) / len(self._labels)


def detect_drift(mean_chamfer_history, min_rounds=3, tol=0.0):
    """Flag distribution drift from a per-round mean-Chamfer history.

    PLLM's Baseline-1 failure mode is a pool whose fidelity gets *worse* over
    successive rounds. Given the sequence of pool mean-Chamfer values (one per
    round), drift is flagged when the last ``min_rounds`` values are
    monotonically non-decreasing by more than ``tol`` each step (i.e. fidelity
    consistently degrading). Returns a dict with ``drift`` (bool), the trailing
    window inspected, and the total increase across that window.
    """
    if min_rounds < 2:
        raise ValueError("min_rounds must be >= 2")
    if tol < 0:
        raise ValueError("tol must be >= 0")
    hist = [h for h in mean_chamfer_history if h is not None]
    if len(hist) < min_rounds:
        return {"drift": False, "window": hist, "increase": 0.0}
    window = hist[-min_rounds:]
    rising = all(window[i + 1] - window[i] > tol for i in range(len(window) - 1))
    return {"drift": rising, "window": window,
            "increase": window[-1] - window[0]}


def label_efficiency(pool_size, total_unlabeled, mean_chamfer_history=None):
    """Supervision obtained per unit of unlabeled data.

    ``coverage`` is the fraction of the unlabeled shape pool that produced an
    accepted pseudo-label. When a per-round mean-Chamfer history is given,
    ``fidelity_gain`` reports the first-to-last reduction in mean Chamfer
    (positive = improved) and ``gain_per_round`` its average per-round rate.
    """
    if total_unlabeled < 0:
        raise ValueError("total_unlabeled must be >= 0")
    coverage = (pool_size / total_unlabeled) if total_unlabeled else 0.0
    out = {"coverage": coverage, "labeled": pool_size,
           "unlabeled": total_unlabeled}
    if mean_chamfer_history:
        hist = [h for h in mean_chamfer_history if h is not None]
        if len(hist) >= 2:
            gain = hist[0] - hist[-1]
            out["fidelity_gain"] = gain
            out["gain_per_round"] = gain / (len(hist) - 1)
        else:
            out["fidelity_gain"] = 0.0
            out["gain_per_round"] = 0.0
    return out


def run_selftraining(round_accepts, total_unlabeled, drift_min_rounds=3,
                     drift_tol=0.0, stop_on_drift=True):
    """Accumulate a sequence of rounds' accepted labels with drift-aware stop.

    ``round_accepts`` is a list where element r is the accepted-label list for
    round r. Rounds are merged into a :class:`PseudoLabelPool` in order; after
    each merge the pool mean Chamfer is recorded and drift is tested. When
    ``stop_on_drift`` and drift is detected, iteration halts early. Returns the
    final pool, per-round history, drift verdict, and the label-efficiency
    summary.
    """
    pool = PseudoLabelPool()
    history, mean_hist = [], []
    stopped_round = None
    for r, accepted in enumerate(round_accepts):
        merge = pool.add_round(r, accepted)
        mc = pool.mean_chamfer()
        merge["mean_chamfer"] = mc
        mean_hist.append(mc)
        history.append(merge)
        drift = detect_drift(mean_hist, drift_min_rounds, drift_tol)
        if stop_on_drift and drift["drift"]:
            stopped_round = r
            break
    final_drift = detect_drift(mean_hist, drift_min_rounds, drift_tol)
    return {
        "pool": pool,
        "pool_size": len(pool),
        "history": history,
        "mean_chamfer_history": mean_hist,
        "drift": final_drift,
        "stopped_round": stopped_round,
        "efficiency": label_efficiency(len(pool), total_unlabeled, mean_hist),
    }
