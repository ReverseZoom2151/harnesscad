"""CADMorph planning stage: relative-contribution masking of a CAD sequence.

From CADMorph (Ma et al., NeurIPS 2025), Section 3.4, "Planning". Given the
current parametric sequence ``C_{r-1}`` and a target shape ``S'``, the planner
decides *which segments to edit*. A naive planner masks segments at random,
wasting effort on parts that already match the target; CADMorph instead
concentrates edits on the segments most responsible for the discrepancy.

The paper's raw per-segment signal ``M(C(i), S)`` is a cross-attention score
read out of the learned P2S diffusion model — that read-out is the learned
part and stays outside this module (supply it as a callable). What *is*
deterministic and buildable is the algorithm layered on top:

  * **Scale-invariant relative score** (paper Eq. 2)::

        J(i) = | M(C(i), S') - M(C(i), S_{r-1}) |

    i.e. how much segment ``i``'s influence *changes* between the current
    rendered shape ``S_{r-1}`` and the target ``S'``. Taking the absolute
    difference cancels each segment's baseline magnitude (the paper notes
    ``[SOL]`` tokens always score low), so segments are compared fairly.

  * **Mean-threshold selection** — rank segments by ``J(i)`` and mask those
    whose score exceeds the mean ``J̄`` (paper: "the K largest ones ... those
    above the mean"). This focuses computation on the mismatched segments.

The masked segments are collapsed into ``<mask>`` tokens (the same token
:mod:`editing.locate_infill` uses, so a downstream infiller sees a familiar
shape), yielding ``C_mask_r``.

To make the planner usable end-to-end without the P2S model, we also provide
:func:`leave_one_out_contribution`, a deterministic geometry-based stand-in for
``M``: a segment's contribution to a shape ``S`` is how much *worse* the
sequence fits ``S`` when that segment is removed. It needs only a renderer and
a distance and never touches a learned model.

Stdlib-only and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

# Reuse the mask token the locate-then-infill layer already uses so a candidate
# generator (MPP stand-in) sees a consistent representation. Importing the
# constant does not modify that module.
from harnesscad.domain.editing.locate_infill import MASK


Segment = object
Contribution = Callable[[Sequence, object], List[float]]


@dataclass(frozen=True)
class PlanResult:
    """Outcome of the planning stage.

    ``masked_sequence`` is ``C_mask_r`` with selected segments collapsed to
    ``<mask>`` tokens; ``masked_indices`` are the original segment positions
    chosen; ``scores`` are the per-segment ``J(i)`` values; ``threshold`` is the
    mean ``J̄`` used to select.
    """

    masked_sequence: Tuple
    masked_indices: Tuple[int, ...]
    scores: Tuple[float, ...]
    threshold: float


def relative_scores(contrib_current: Sequence[float],
                    contrib_target: Sequence[float]) -> Tuple[float, ...]:
    """Compute ``J(i) = |M(C(i), S') - M(C(i), S_{r-1})|`` (paper Eq. 2).

    ``contrib_current[i]`` is ``M(C(i), S_{r-1})`` and ``contrib_target[i]`` is
    ``M(C(i), S')``. The two must be the same length (one score per segment).
    """
    if len(contrib_current) != len(contrib_target):
        raise ValueError(
            f"contribution length mismatch: {len(contrib_current)} vs "
            f"{len(contrib_target)}")
    return tuple(abs(float(t) - float(c))
                 for c, t in zip(contrib_current, contrib_target))


def select_mask_indices(scores: Sequence[float],
                        *, max_k: Optional[int] = None,
                        ensure_progress: bool = True) -> Tuple[int, ...]:
    """Pick the segments to edit: those with ``J(i)`` above the mean ``J̄``.

    Ranks by score (desc, stable by index on ties). ``max_k`` optionally caps
    how many are returned. When ``ensure_progress`` is set and the mean rule
    selects nothing (e.g. all scores equal) yet some segment has a positive
    score, the single highest-scoring segment is masked so the outer loop can
    still make progress.
    """
    n = len(scores)
    if n == 0:
        return ()
    mean = sum(scores) / n
    above = [i for i in range(n) if scores[i] > mean]
    # Rank the selected indices by descending score, stable on index.
    above.sort(key=lambda i: (-scores[i], i))
    if not above and ensure_progress:
        best = max(range(n), key=lambda i: (scores[i], -i))
        if scores[best] > 0.0:
            above = [best]
    if max_k is not None and max_k >= 0:
        above = above[:max_k]
    return tuple(sorted(above))


def apply_mask(sequence: Sequence, indices: Sequence[int],
               *, mask_token: object = MASK) -> Tuple:
    """Replace ``indices`` in ``sequence`` with ``<mask>``, collapsing runs.

    Consecutive masked positions collapse into a single mask token (mirroring
    :func:`editing.locate_infill.locate_mask`), so an infiller fills one span
    rather than N adjacent holes.
    """
    chosen = set(indices)
    out: List = []
    prev_masked = False
    for i, seg in enumerate(sequence):
        if i in chosen:
            if not prev_masked:
                out.append(mask_token)
            prev_masked = True
        else:
            out.append(seg)
            prev_masked = False
    return tuple(out)


def plan_mask(sequence: Sequence,
              contrib_current: Sequence[float],
              contrib_target: Sequence[float],
              *, max_k: Optional[int] = None,
              mask_token: object = MASK) -> PlanResult:
    """Full planning stage: scores -> select -> mask.

    Combines :func:`relative_scores`, :func:`select_mask_indices` and
    :func:`apply_mask` into the ``C_{r-1} -> C_mask_r`` step. ``sequence`` must
    have one entry per contribution score.
    """
    if len(sequence) != len(contrib_current):
        raise ValueError(
            f"sequence length {len(sequence)} != contribution length "
            f"{len(contrib_current)}")
    scores = relative_scores(contrib_current, contrib_target)
    idx = select_mask_indices(scores, max_k=max_k)
    masked = apply_mask(sequence, idx, mask_token=mask_token)
    mean = sum(scores) / len(scores) if scores else 0.0
    return PlanResult(masked, idx, scores, mean)


# --------------------------------------------------------------------------- #
# Deterministic contribution stand-in for the learned P2S cross-attention
# --------------------------------------------------------------------------- #
def leave_one_out_contribution(
        render: Callable[[Sequence], object],
        distance: Callable[[object, object], float],
) -> Contribution:
    """Build a deterministic ``M(C(i), S)`` from a renderer and a distance.

    A segment's *contribution* to a shape ``S`` is defined here as how much the
    sequence's fit to ``S`` degrades when that segment is dropped::

        M(C(i), S) = distance(render(C without segment i), S)
                     - distance(render(C), S)

    A large value means removing segment ``i`` moves the rendered shape *away*
    from ``S`` (the segment matters for reproducing ``S``); ~0 means the segment
    is irrelevant to ``S``. This is a purely geometric, learned-model-free
    stand-in for the paper's cross-attention read-out, suitable for driving the
    planner via :func:`plan_mask`.

    The returned callable has signature ``contribution(sequence, shape) ->
    list[float]``.
    """

    def contribution(sequence: Sequence, shape: object) -> List[float]:
        seq = tuple(sequence)
        base = distance(render(seq), shape)
        out: List[float] = []
        for i in range(len(seq)):
            reduced = seq[:i] + seq[i + 1:]
            out.append(distance(render(reduced), shape) - base)
        return out

    return contribution
