"""Exact GenCAD loss masking + objective formulas (``utils/loss.py``, ``model_utils.py``).

Both GenCAD objectives are deterministic once the logits are given, and both hinge on
*which tokens are counted* -- a detail no paper-level module carries. Reproduced here
verbatim so a decoder's reported command/argument accuracy is comparable with the
reference implementation.

**CAD reconstruction loss** (``CADLoss``):

* Autoregressive shift: targets drop token 0, logits drop the last token.
* ``visibility_mask``: a sequence is *visible* only if it contains fewer than
  ``S - 1`` EOS tokens (i.e. it is not the degenerate all-EOS sequence).
* ``padding_mask``: ``(cmd == EOS).cumsum() == 0`` -- everything strictly before the
  first EOS -- then *extended* by OR-ing in the mask shifted by 3, which pulls the
  terminating EOS itself into the loss.
* Argument tokens are selected by ``CMD_ARGS_MASK[command]``: each command type
  contributes only the argument slots it actually uses (Line: x, y; Arc: x, y, alpha,
  f; Circle: x, y, r; Ext: the 11 extrude slots; SOL/EOS: none).
* Argument *targets* are shifted by ``+1`` because ``PAD_VAL = -1`` occupies class 0,
  so the argument vocabulary is ``args_dim + 1 = 257`` wide.

**CCIP contrastive loss** (``CCIPLoss``, open_clip-style): symmetric cross-entropy of
``logit_scale * A @ B.T`` against ``labels = arange(batch)``, averaged over the two
directions. Note it is *not* the NT-Xent of ``bench.contrastcad_contrastive`` (which
contrasts dropout views of one modality): here the two modalities are the batch's
image and CAD embeddings, and the positives are the diagonal.

Pure standard library, deterministic.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

# --- command vocabulary (GenCAD macro.py) ------------------------------------
ALL_COMMANDS = ("Line", "Arc", "Circle", "EOS", "SOL", "Ext")
LINE_IDX = 0
ARC_IDX = 1
CIRCLE_IDX = 2
EOS_IDX = 3
SOL_IDX = 4
EXT_IDX = 5

N_ARGS_SKETCH = 5     # x, y, alpha, f, r
N_ARGS_PLANE = 3      # theta, phi, gamma
N_ARGS_TRANS = 4      # px, py, pz, s
N_ARGS_EXT_PARAM = 4  # e1, e2, b, u
N_ARGS_EXT = N_ARGS_PLANE + N_ARGS_TRANS + N_ARGS_EXT_PARAM  # 11
N_ARGS = N_ARGS_SKETCH + N_ARGS_EXT                          # 16

PAD_VAL = -1
ARGS_DIM = 256

_SK = N_ARGS_SKETCH
_EX = N_ARGS_EXT

#: Row per command type, column per argument slot: 1 = the slot is used.
CMD_ARGS_MASK: Tuple[Tuple[int, ...], ...] = (
    (1, 1, 0, 0, 0) + (0,) * _EX,   # Line   -> x, y
    (1, 1, 1, 1, 0) + (0,) * _EX,   # Arc    -> x, y, alpha, f
    (1, 1, 0, 0, 1) + (0,) * _EX,   # Circle -> x, y, r
    (0,) * _SK + (0,) * _EX,        # EOS    -> none
    (0,) * _SK + (0,) * _EX,        # SOL    -> none
    (0,) * _SK + (1,) * _EX,        # Ext    -> the 11 extrude slots
)


def args_mask(command: int) -> Tuple[int, ...]:
    """The 16-slot argument mask of a command type."""
    if not 0 <= command < len(CMD_ARGS_MASK):
        raise ValueError("unknown command index: {}".format(command))
    return CMD_ARGS_MASK[command]


def used_arg_slots(command: int) -> Tuple[int, ...]:
    """Indices of the argument slots a command actually uses."""
    return tuple(i for i, bit in enumerate(args_mask(command)) if bit)


def args_vocab_size(args_dim: int = ARGS_DIM) -> int:
    """``args_dim + 1``: the extra class encodes the ``-1`` padding value."""
    return args_dim + 1


def shift_arg_target(value: int) -> int:
    """Argument class index: ``value + 1`` (so ``PAD_VAL = -1`` maps to class 0)."""
    return int(value) + 1


def unshift_arg_target(index: int) -> int:
    """Inverse of :func:`shift_arg_target`."""
    return int(index) - 1


# --- masks --------------------------------------------------------------------
def padding_mask(commands: Sequence[int], extended: bool = False) -> List[int]:
    """``(cmd == EOS).cumsum() == 0``, optionally extended to cover the final EOS.

    The extension is the reference's shift-by-3 OR (``narrow(3, S-3).add_(narrow(0,
    S-3)).clamp_(max=1)``), evaluated against the *unextended* mask.
    """
    seen_eos = False
    base: List[int] = []
    for c in commands:
        base.append(0 if seen_eos else 1)
        if c == EOS_IDX:
            seen_eos = True
    if not extended:
        return base
    out = list(base)
    for i in range(3, len(base)):
        out[i] = min(1, base[i] + base[i - 3])
    return out


def visibility_mask(commands: Sequence[int]) -> int:
    """1 when the sequence carries real content: ``count(EOS) < len(seq) - 1``."""
    s = len(commands)
    return 1 if sum(1 for c in commands if c == EOS_IDX) < s - 1 else 0


def command_loss_mask(commands: Sequence[int]) -> List[int]:
    """Per-token mask used for the command cross-entropy (padding AND visibility)."""
    vis = visibility_mask(commands)
    return [m * vis for m in padding_mask(commands, extended=True)]


def arg_loss_mask(commands: Sequence[int]) -> List[List[int]]:
    """Per-token, per-slot mask used for the argument cross-entropy.

    The reference indexes ``CMD_ARGS_MASK`` by the *target* command, so a slot is
    counted whenever the ground-truth command uses it -- the padding mask is not
    applied here (an EOS/SOL row is all zeros anyway).
    """
    return [list(args_mask(c)) for c in commands]


def autoregressive_pairs(commands: Sequence[int]) -> Tuple[List[int], List[int]]:
    """The reference's shift: ``targets = cmd[1:]``, and logits index ``t`` predicts it.

    Returns ``(target_commands, logit_positions)``, both of length ``len(cmd) - 1``.
    """
    if len(commands) < 2:
        raise ValueError("sequence must have at least 2 tokens to shift")
    return list(commands[1:]), list(range(len(commands) - 1))


def selected_command_tokens(commands: Sequence[int]) -> List[int]:
    """Positions in the *shifted* target sequence that enter the command loss."""
    targets, _ = autoregressive_pairs(commands)
    mask = command_loss_mask(targets)
    return [i for i, m in enumerate(mask) if m]


def selected_arg_tokens(commands: Sequence[int]) -> List[Tuple[int, int]]:
    """``(token, slot)`` pairs of the *shifted* targets that enter the argument loss."""
    targets, _ = autoregressive_pairs(commands)
    out: List[Tuple[int, int]] = []
    for t, c in enumerate(targets):
        for slot in used_arg_slots(c):
            out.append((t, slot))
    return out


# --- cross-entropy -------------------------------------------------------------
def log_softmax(logits: Sequence[float]) -> List[float]:
    """Numerically stable log-softmax."""
    if not logits:
        raise ValueError("empty logits")
    m = max(logits)
    denom = math.log(sum(math.exp(v - m) for v in logits))
    return [v - m - denom for v in logits]


def cross_entropy(logits: Sequence[float], target: int) -> float:
    """``-log softmax(logits)[target]``."""
    ls = log_softmax(logits)
    if not 0 <= target < len(ls):
        raise ValueError("target out of range: {}".format(target))
    return -ls[target]


def mean_cross_entropy(logit_rows: Sequence[Sequence[float]],
                       targets: Sequence[int]) -> float:
    """Mean CE over the selected rows (the reference reduces with ``mean``)."""
    if len(logit_rows) != len(targets):
        raise ValueError("row/target count mismatch")
    if not logit_rows:
        return 0.0
    return sum(cross_entropy(r, t) for r, t in zip(logit_rows, targets)) / len(logit_rows)


def cad_loss(commands: Sequence[int], args: Sequence[Sequence[int]],
             command_logits: Sequence[Sequence[float]],
             args_logits: Sequence[Sequence[Sequence[float]]],
             loss_cmd_weight: float = 1.0,
             loss_args_weight: float = 2.0) -> Dict[str, float]:
    """The full ``CADLoss``: masked, shifted command and argument cross-entropies.

    ``command_logits`` is ``[S][n_commands]`` and ``args_logits`` is
    ``[S][16][args_dim + 1]``, both *unshifted* (position ``t`` predicts token
    ``t + 1``). Returns ``{"loss_cmd", "loss_args"}``, already weighted.
    """
    targets, _ = autoregressive_pairs(commands)
    if len(command_logits) != len(commands) or len(args_logits) != len(commands):
        raise ValueError("logits must be given for every input token")

    cmd_rows, cmd_tgts = [], []
    for i in selected_command_tokens(commands):
        cmd_rows.append(command_logits[i])   # logit position i predicts target i
        cmd_tgts.append(targets[i])

    arg_rows, arg_tgts = [], []
    for t, slot in selected_arg_tokens(commands):
        arg_rows.append(args_logits[t][slot])
        arg_tgts.append(shift_arg_target(args[t + 1][slot]))

    return {"loss_cmd": loss_cmd_weight * mean_cross_entropy(cmd_rows, cmd_tgts),
            "loss_args": loss_args_weight * mean_cross_entropy(arg_rows, arg_tgts)}


# --- CCIP symmetric contrastive loss ------------------------------------------
def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("dimension mismatch")
    return sum(x * y for x, y in zip(a, b))


def logit_matrix(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]],
                 logit_scale: float) -> List[List[float]]:
    """``logit_scale * A @ B.T`` (the reference does *not* re-normalise here)."""
    if len(a) != len(b):
        raise ValueError("both modalities must have the same batch size")
    return [[logit_scale * _dot(u, v) for v in b] for u in a]


def ccip_loss(image_features: Sequence[Sequence[float]],
              cad_features: Sequence[Sequence[float]],
              logit_scale: float) -> float:
    """Symmetric cross-modal InfoNCE, ground truth = ``arange(batch)`` (open_clip)."""
    n = len(image_features)
    if n == 0:
        raise ValueError("empty batch")
    per_image = logit_matrix(image_features, cad_features, logit_scale)
    per_cad = logit_matrix(cad_features, image_features, logit_scale)
    labels = list(range(n))
    return (mean_cross_entropy(per_image, labels)
            + mean_cross_entropy(per_cad, labels)) / 2


def ccip_ground_truth(batch_size: int) -> List[int]:
    """``arange(batch_size)`` -- the positives sit on the diagonal."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return list(range(batch_size))


def clamped_logit_scale(log_value: float, max_value: float = 100.0) -> float:
    """``exp(log_scale)`` clamped at 100, as CLIP-family implementations do."""
    return min(math.exp(log_value), max_value)
