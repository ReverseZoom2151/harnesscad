"""Command / parameter reconstruction accuracy for ContrastCAD (Eq. 8).

Jung, Kim & Kim, *ContrastCAD* (2024), Section 5.3.2 (following DeepCAD).

To score how faithfully an autoencoder reconstructs a construction sequence, the
paper reports two position-aligned accuracies over the fixed-length sequence:

    ACC_cmd   = (1/N) * sum_k  I[ t_k == t_hat_k ]                       (Eq. 8)

    ACC_param = (1/T) * sum_k sum_l  I[ |p_{k,l} - p_hat_{k,l}| < eta ]
                                     * I[ t_k == t_hat_k ]

where ``T = sum_k I[t_k == t_hat_k] * |p_k|`` is the count of parameters belonging
to correctly-typed commands, ``|p_k|`` the number of parameter slots (16), ``eta``
a quantised-value tolerance, and only parameters of a *correctly predicted command
type* are eligible (a parameter on a mistyped command is never counted).

This differs from the repository's ``bench/command_metrics.py`` (a tolerant
*set-matching F1* per curve family): here the comparison is strictly
**position-aligned** and split into a command-type accuracy and a tolerant,
type-gated parameter accuracy, exactly as ContrastCAD / DeepCAD define them.

Commands use the quantised-integer dict representation of
``datagen/contrastcad_rre.py``. Fully deterministic, stdlib only.
"""

from __future__ import annotations

from typing import Sequence

# The 16 parameter slots of DeepCAD/ContrastCAD Table 1, in a fixed order so a
# command's parameters compare position-by-position. A command that omits a slot
# is treated as the sentinel (unused) for both sequences.
PARAM_SLOTS = (
    "x", "y", "theta", "c", "r",
    "alpha", "beta", "gamma",
    "ox", "oy", "oz",
    "s", "delta1", "delta2", "b", "w",
)
N_PARAM_SLOTS = len(PARAM_SLOTS)  # == 16
UNUSED = -1


def _param_vector(cmd) -> list:
    return [int(cmd.get(name, UNUSED)) for name in PARAM_SLOTS]


def command_accuracy(actual: Sequence[dict], expected: Sequence[dict]) -> float:
    """ACC_cmd: fraction of positions whose command type matches (Eq. 8).

    Both sequences must be the same (padded) length ``N``.
    """
    if len(actual) != len(expected):
        raise ValueError("sequences must have equal length")
    if not actual:
        raise ValueError("empty sequence")
    correct = sum(1 for a, e in zip(actual, expected)
                  if a.get("type") == e.get("type"))
    return correct / len(actual)


def parameter_accuracy(actual: Sequence[dict], expected: Sequence[dict],
                       eta: int = 3) -> float:
    """ACC_param: type-gated, tolerant parameter accuracy (Eq. 8).

    A parameter counts as correct only when its command type is right *and* its
    quantised value is within ``eta`` of the target. The denominator ``T`` is the
    number of parameters on correctly-typed commands; if no command type matches,
    ``T == 0`` and the accuracy is defined as 0.0.
    """
    if len(actual) != len(expected):
        raise ValueError("sequences must have equal length")
    if eta < 0:
        raise ValueError("eta must be non-negative")
    total = 0
    correct = 0
    for a, e in zip(actual, expected):
        if a.get("type") != e.get("type"):
            continue
        pa = _param_vector(a)
        pe = _param_vector(e)
        total += N_PARAM_SLOTS
        for va, ve in zip(pa, pe):
            if abs(va - ve) < eta:
                correct += 1
    if total == 0:
        return 0.0
    return correct / total


def reconstruction_accuracy(actual: Sequence[dict], expected: Sequence[dict],
                            eta: int = 3) -> dict:
    """Return both ACC_cmd and ACC_param for one reconstructed sequence."""
    return {
        "acc_cmd": command_accuracy(actual, expected),
        "acc_param": parameter_accuracy(actual, expected, eta),
    }
