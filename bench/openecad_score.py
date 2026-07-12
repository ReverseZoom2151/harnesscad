"""OpenECAD generation scoring metric (Yuan et al., 2024, Sec. 6.1, Table 4, Eq. 1).

Because a 3D shape can be built many ways and the input views lack absolute
dimensions, the paper does not compare generated code token-for-token. Instead it
defines a 100-point scoring algorithm that rewards executability plus the
correctness of curves and loops (Table 4):

    Is the code executable?      10 * exec_flag
    Accuracy of All Curves       45 * acc_c
    Accuracy of Loops count       5 * acc_l
    Loops integrated score       40 * loops_score

    per-loop score (out of 100): 100 if the loop is absolutely correct
                                 (same curve types, same order), else 90 * acc_i

Equation 1::

    score = 10 e + 45 acc_c + 5 acc_l
            + (40 / (100 L)) * sum_{i=1..L} loop_score_i

with ``e, acc_c, acc_l, acc_i in [0, 1]`` and ``L`` the number of loops. Each
``loop_score_i`` is the Table 4 per-loop score (100 when absolutely correct, else
``90 * acc_i``); the paper's printed ``100 s_i + 90 acc_i`` is that same either/or
quantity, since ``s_i`` and the ``90 acc_i`` branch are mutually exclusive. The
four weights sum to 100. This module computes the metric deterministically
over loops of curve calls from :mod:`programs.openecad_script` (or over plain
curve-type sequences), independent of any renderer or model.
"""

from __future__ import annotations

from programs import openecad_script as oe

# Table 4 weights.
W_EXEC = 10.0
W_CURVES = 45.0
W_LOOPS_COUNT = 5.0
W_LOOPS_SCORE = 40.0


def _types(loop) -> tuple[str, ...]:
    """Curve-type sequence of a loop given as Calls or already as type strings."""
    out = []
    for c in loop:
        if isinstance(c, oe.Call):
            out.append(c.func)
        else:
            out.append(str(c))
    return tuple(out)


def curve_accuracy(pred_loop, target_loop) -> float:
    """Fraction of positionally matching curve types (both empty -> 1.0).

    Uses ``max(len)`` as denominator so both missing and extra curves are
    penalised.
    """
    p, t = _types(pred_loop), _types(target_loop)
    n = max(len(p), len(t))
    if n == 0:
        return 1.0
    matches = sum(1 for a, b in zip(p, t) if a == b)
    return matches / n


def loop_absolutely_correct(pred_loop, target_loop) -> bool:
    """True when curve types and their order match exactly (Table 4)."""
    return _types(pred_loop) == _types(target_loop)


def loop_score(absolutely_correct: bool, curve_acc: float) -> float:
    """Per-loop score out of 100 (Table 4)."""
    return 100.0 if absolutely_correct else 90.0 * curve_acc


def loops_count_accuracy(pred_count: int, target_count: int) -> float:
    """Ratio accuracy of the number of loops (both zero -> 1.0)."""
    hi = max(pred_count, target_count)
    if hi == 0:
        return 1.0
    return min(pred_count, target_count) / hi


def score(executable: float, curves_acc: float, loops_count_acc: float,
          per_loop: list[tuple[bool, float]]) -> float:
    """Equation 1 given already-computed components.

    *per_loop* is a list of ``(absolutely_correct, curve_acc_i)`` for the ``L``
    reference loops.
    """
    for name, v in (("executable", executable), ("curves_acc", curves_acc),
                    ("loops_count_acc", loops_count_acc)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {v}")
    L = len(per_loop)
    if L == 0:
        loops_term = 0.0
    else:
        # Table 4 per-loop score: 100 if absolutely correct, else 90 * acc.
        total = sum(loop_score(s, a) for s, a in per_loop)
        loops_term = (W_LOOPS_SCORE / (100.0 * L)) * total
    return (W_EXEC * executable + W_CURVES * curves_acc
            + W_LOOPS_COUNT * loops_count_acc + loops_term)


def evaluate(pred_loops, target_loops, executable: float = 1.0) -> dict:
    """Full scoring of predicted vs target loops (lists of curve loops).

    Loops are aligned by index against the ``target_loops`` reference; a missing
    predicted loop counts as empty. Returns the component accuracies and the
    overall score (0-100).
    """
    if not 0.0 <= executable <= 1.0:
        raise ValueError("executable must be in [0, 1]")

    # Global curve accuracy: flatten curve types loop-by-loop.
    flat_pred, flat_target = [], []
    for lp in pred_loops:
        flat_pred.extend(_types(lp))
    for lt in target_loops:
        flat_target.extend(_types(lt))
    curves_acc = curve_accuracy(flat_pred, flat_target)

    loops_count_acc = loops_count_accuracy(len(pred_loops), len(target_loops))

    per_loop: list[tuple[bool, float]] = []
    for i, target_loop in enumerate(target_loops):
        pred_loop = pred_loops[i] if i < len(pred_loops) else []
        si = loop_absolutely_correct(pred_loop, target_loop)
        acci = curve_accuracy(pred_loop, target_loop)
        per_loop.append((si, acci))

    overall = score(executable, curves_acc, loops_count_acc, per_loop)
    return {
        "executable": executable,
        "curves_accuracy": curves_acc,
        "loops_count_accuracy": loops_count_acc,
        "per_loop": per_loop,
        "loops_score": (
            0.0 if not per_loop
            else sum(loop_score(s, a) for s, a in per_loop) / len(per_loop)),
        "overall": overall,
    }
