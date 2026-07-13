"""DeepCAD's official autoencoder accuracy metric (``evaluation/evaluate_ae_acc.py``).

Source: the DeepCAD reference code (Wu, Xiao & Zheng, ICCV 2021). This is the exact
scoring script behind the ``ACC_cmd`` / ``ACC_param`` numbers in the paper's Table 2,
reproduced here on the quantised 17-column rows of
``reconstruction.deepcad2_vector_layout``.

Difference from ``bench/contrastcad_recon_accuracy``
----------------------------------------------------
That module implements ContrastCAD's *paraphrase* of the metric: a single tolerance
over all 16 slots of every correctly-typed command. The released DeepCAD script has
four extra behaviours that materially change the number, all reproduced here:

1. ``SOL`` and ``EOS`` positions are **excluded** from ACC_param entirely (they have
   no arguments), while still counting toward ACC_cmd;
2. only the slots in ``CMD_ARGS_MASK[cmd]`` are scored -- padded ``-1`` slots never
   inflate the score;
3. two slot families are compared **strictly equal**, not within tolerance, because
   they are categorical: the ``Ext`` row's last two args (``b`` = boolean op,
   ``u`` = extent type) and the ``Arc`` row's ``f`` (counter-clockwise flag, arg
   index 3);
4. the tolerance is ``|out - gt| < 3`` on the *quantised* levels (strict ``<``), and
   the per-model scores are **macro-averaged** over models (mean of per-model means),
   not pooled over all commands.

The routines also return the reference's diagnostic breakdowns: accuracy per command
type and per (command, argument) slot.

Ground-truth command types drive the bookkeeping (``gt_cmd`` selects the row of the
mask), exactly as in the script. Deterministic, stdlib only.
"""

from __future__ import annotations

from typing import Sequence

from harnesscad.domain.reconstruction.tokens.deepcad_vector_layout import (
    ALL_COMMANDS, ARC_IDX, ARG_NAMES, CMD_ARGS_MASK, EOS_IDX, EXT_IDX,
    N_ARGS, SOL_IDX,
)

TOLERANCE = 3

Row = Sequence[int]


def _check(out: Sequence[Row], gt: Sequence[Row]) -> None:
    if len(out) != len(gt):
        raise ValueError("output and ground-truth sequences must have equal length")
    if not gt:
        raise ValueError("empty sequence")
    for row in list(out) + list(gt):
        if len(row) != 1 + N_ARGS:
            raise ValueError(f"row must have {1 + N_ARGS} entries, got {len(row)}")


def slot_hits(out_row: Row, gt_row: Row, tolerance: int = TOLERANCE) -> list[int]:
    """Per-argument 0/1 hits for one row pair, before masking.

    Tolerant (``|d| < tolerance``) everywhere except the categorical slots, which must
    match exactly: ``Ext``'s last two args (``b``, ``u``) and ``Arc``'s ``f``.
    """
    cmd = gt_row[0]
    hits = [int(abs(out_row[1 + i] - gt_row[1 + i]) < tolerance) for i in range(N_ARGS)]
    if cmd == EXT_IDX:
        for i in (N_ARGS - 2, N_ARGS - 1):
            hits[i] = int(out_row[1 + i] == gt_row[1 + i])
    elif cmd == ARC_IDX:
        hits[3] = int(out_row[4] == gt_row[4])  # 'f', arg index 3
    return hits


def command_accuracy(out: Sequence[Row], gt: Sequence[Row]) -> float:
    """ACC_cmd for one model: fraction of positions with the right command type."""
    _check(out, gt)
    return sum(1 for o, g in zip(out, gt) if o[0] == g[0]) / len(gt)


def parameter_accuracy(out: Sequence[Row], gt: Sequence[Row],
                       tolerance: int = TOLERANCE) -> float:
    """ACC_param for one model: mean over the masked args of correctly-typed rows.

    ``SOL``/``EOS`` rows are skipped. Returns ``0.0`` when no argument-bearing command
    was typed correctly (the reference would produce NaN there).
    """
    _check(out, gt)
    scored: list[int] = []
    for out_row, gt_row in zip(out, gt):
        cmd = gt_row[0]
        if cmd in (SOL_IDX, EOS_IDX) or out_row[0] != cmd:
            continue
        hits = slot_hits(out_row, gt_row, tolerance)
        mask = CMD_ARGS_MASK[cmd]
        scored.extend(h for h, m in zip(hits, mask) if m)
    if not scored:
        return 0.0
    return sum(scored) / len(scored)


def evaluate_model(out: Sequence[Row], gt: Sequence[Row],
                   tolerance: int = TOLERANCE) -> dict:
    """Both accuracies for a single model."""
    return {"acc_cmd": command_accuracy(out, gt),
            "acc_param": parameter_accuracy(out, gt, tolerance)}


def evaluate_dataset(pairs: Sequence[tuple[Sequence[Row], Sequence[Row]]],
                     tolerance: int = TOLERANCE) -> dict:
    """The full script: macro-averaged accuracies plus the two breakdowns.

    ``pairs`` is a sequence of ``(out_vec, gt_vec)`` models. Returns::

        {"acc_cmd", "acc_param",                # macro-averages over models
         "each_cmd_acc": {command_name: acc},   # per ground-truth command type
         "each_cmd_count": {command_name: n},
         "each_param_acc": {command_name: {arg_name: acc}}}

    A command type / slot never seen in the ground truth is reported as ``0.0``
    (the reference divides by ``count + 1e-6``).
    """
    if not pairs:
        raise ValueError("no models to evaluate")

    per_model_cmd: list[float] = []
    per_model_param: list[float] = []
    cmd_count = [0] * len(ALL_COMMANDS)
    cmd_hits = [0] * len(ALL_COMMANDS)
    param_count = [[0] * N_ARGS for _ in ALL_COMMANDS]
    param_hits = [[0] * N_ARGS for _ in ALL_COMMANDS]

    for out, gt in pairs:
        _check(out, gt)
        for out_row, gt_row in zip(out, gt):
            cmd = gt_row[0]
            cmd_count[cmd] += 1
            cmd_hits[cmd] += int(out_row[0] == cmd)
            if cmd in (SOL_IDX, EOS_IDX) or out_row[0] != cmd:
                continue
            hits = slot_hits(out_row, gt_row, tolerance)
            for i in range(N_ARGS):
                param_count[cmd][i] += 1
                param_hits[cmd][i] += hits[i]
        per_model_cmd.append(command_accuracy(out, gt))
        per_model_param.append(parameter_accuracy(out, gt, tolerance))

    each_cmd_acc = {}
    each_cmd_count = {}
    each_param_acc = {}
    for c, name in enumerate(ALL_COMMANDS):
        each_cmd_count[name] = cmd_count[c]
        each_cmd_acc[name] = (cmd_hits[c] / cmd_count[c]) if cmd_count[c] else 0.0
        slots = {}
        for i, arg in enumerate(ARG_NAMES):
            if not CMD_ARGS_MASK[c][i]:
                continue
            n = param_count[c][i]
            slots[arg] = (param_hits[c][i] / n) if n else 0.0
        if slots:
            each_param_acc[name] = slots

    return {
        "acc_cmd": sum(per_model_cmd) / len(per_model_cmd),
        "acc_param": sum(per_model_param) / len(per_model_param),
        "each_cmd_acc": each_cmd_acc,
        "each_cmd_count": each_cmd_count,
        "each_param_acc": each_param_acc,
    }
