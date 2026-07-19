"""Text-to-CAD training-time token accuracy.

This mirrors an ``AccuracyCalculator``-style metric used for a text-to-CAD
sequence model -- the metric the Transformer is monitored on while it
learns to emit the 2-column ``cad_vec`` stream of
:mod:`reconstruction.t2c3_cad_vec_codec`.

The rules are specific and none of them are shared with ``bench.deepcad2_ae_accuracy``
(which scores the 17-column CAD command rows through ``CMD_ARGS_MASK``) or
``bench.contrastcad_recon_accuracy``:

1. Prediction and target are ``(N, 2)`` token streams. If they differ in length both
   are **truncated to the shorter one** -- a prediction that stops early is not
   penalised for the tokens it never produced, only for the ones it got wrong.
2. A position is scored only if the *target* is a real value token: the mask is
   ``(target > DISCARD_TOKEN).any(axis=-1)`` with ``DISCARD_TOKEN = 6``. All ids
   ``0..6`` are structural (PADDING, START, END_SKETCH, END_FACE, END_LOOP, END_CURVE,
   END_EXTRUSION), so a position whose target is *purely* structural is dropped --
   including the boolean tokens' second slot (which is 0), but **not** the boolean
   token itself (ids 7..10 pass the ``> 6`` test in the first slot, and because the
   mask is taken with ``.any()`` and then broadcast over both slots, both of its slots
   are scored).
3. Both slots of a kept position are scored **independently**, and a slot counts as
   correct when ``|pred - target| < tolerance`` on the *token ids* (default 3, i.e.
   within 2 quantisation levels). Equality is not required -- this is a regression-like
   tolerance over a classification head.
4. Accuracy is ``correct_slots / kept_slots`` pooled over the batch (not a mean of
   per-sample accuracies).

Pure stdlib: streams are lists of ``(px, py)`` int pairs; no tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

DISCARD_TOKEN = 6       # ids 0..6 are structural
DEFAULT_TOLERANCE = 3

Token = tuple[int, int]


class TokenAccuracyError(ValueError):
    """Raised for malformed token streams."""


def _check(stream) -> list[Token]:
    out = []
    for tok in stream:
        if len(tok) != 2:
            raise TokenAccuracyError(f"token {tok!r} is not a 2-tuple")
        out.append((int(tok[0]), int(tok[1])))
    return out


def align(pred: list[Token], target: list[Token]) -> tuple[list[Token], list[Token]]:
    """Truncate both streams to the shorter length (reference behaviour)."""
    n = min(len(pred), len(target))
    return pred[:n], target[:n]


def value_mask(target: list[Token], discard_token: int = DISCARD_TOKEN) -> list[bool]:
    """``True`` where the target position carries a value in *either* slot."""
    return [(t[0] > discard_token or t[1] > discard_token) for t in target]


@dataclass(frozen=True)
class TokenAccuracy:
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def token_accuracy(pred, target, *, tolerance: int = DEFAULT_TOLERANCE,
                   discard_token: int = DISCARD_TOKEN) -> TokenAccuracy:
    """Masked, tolerance-based slot accuracy for one ``(N, 2)`` stream pair."""
    if tolerance <= 0:
        raise TokenAccuracyError("tolerance must be positive")
    p, t = align(_check(pred), _check(target))
    mask = value_mask(t, discard_token)
    correct = 0
    total = 0
    for keep, pt, tt in zip(mask, p, t):
        if not keep:
            continue
        for slot in (0, 1):
            total += 1
            if abs(pt[slot] - tt[slot]) < tolerance:
                correct += 1
    return TokenAccuracy(correct=correct, total=total)


def batch_token_accuracy(preds, targets, *, tolerance: int = DEFAULT_TOLERANCE,
                         discard_token: int = DISCARD_TOKEN) -> TokenAccuracy:
    """Pool correct/total slot counts over a batch, then divide (as the reference does)."""
    preds = list(preds)
    targets = list(targets)
    if len(preds) != len(targets):
        raise TokenAccuracyError("batch size mismatch")
    correct = 0
    total = 0
    for p, t in zip(preds, targets):
        score = token_accuracy(p, t, tolerance=tolerance, discard_token=discard_token)
        correct += score.correct
        total += score.total
    return TokenAccuracy(correct=correct, total=total)


def accuracy_from_logits(logit_stream, target, *, tolerance: int = DEFAULT_TOLERANCE,
                         discard_token: int = DISCARD_TOKEN) -> TokenAccuracy:
    """``calculateAccMulti2DFromProbability``: argmax over the class axis, then score.

    ``logit_stream`` is ``[[slot_x_scores, slot_y_scores], ...]`` -- one score vector
    per slot per position. Ties break to the lowest class id, matching ``argmax``.
    """
    pred: list[Token] = []
    for position in logit_stream:
        if len(position) != 2:
            raise TokenAccuracyError("each position needs scores for both slots")
        pred.append(tuple(max(range(len(s)), key=lambda i, s=s: s[i]) for s in position))
    return token_accuracy(pred, target, tolerance=tolerance, discard_token=discard_token)
