"""Dataset statistics and last-step data augmentation for CADParser workflows.

Two deterministic dataset tools from the paper, both independent of the learned
model:

1. Statistical analysis of a corpus of command workflows (paper Tables 1 & 2):
   the operation-type ratio and the sequence-length distribution, where -- crucially
   -- "when calculating the sequence length, we treat the sketch feature as a single
   unit, although it comprises multiple line and segment operations."

2. The "back-to-front" data augmentation (Sec. 3.1): because a CAD model is
   strictly constrained and cannot be randomly generated, new samples are produced
   by iteratively cutting the last modeling step from the workflow (a sketch and its
   consuming extrusion/revolution count as one step), yielding valid prefixes.

Workflows here are lists of :class:`reconstruction.cadparser_schema.Command`.
"""

from __future__ import annotations

from reconstruction.cadparser_schema import Command, SOS, EOS, PAD


# Commands that consume a sketch profile and complete a modeling *step*.
_STEP_CLOSERS = frozenset({"E", "Ec", "R", "Rc"})
# Sketch-curve commands that build the profile of a step.
_CURVES = frozenset({"L", "A", "C"})
_TERMINATORS = frozenset({SOS, EOS, PAD})


def strip_terminators(workflow: list[Command]) -> list[Command]:
    """Drop <SOS>/<EOS>/<PAD> so only content commands remain."""
    return [c for c in workflow if c.type not in _TERMINATORS]


def split_steps(workflow: list[Command]) -> list[list[Command]]:
    """Partition a workflow into modeling *steps*.

    A step is a run of sketch curves plus the extrusion/revolution (and optional
    ``Ax``) that consumes them, so a sketch and its extrusion form one unit (paper
    Sec. 3.1 / Table 2). A standalone edge feature (``F``/``Cf``) is its own step.
    """
    content = strip_terminators(workflow)
    steps: list[list[Command]] = []
    current: list[Command] = []
    for cmd in content:
        current.append(cmd)
        if cmd.type in _STEP_CLOSERS:
            steps.append(current)
            current = []
        elif cmd.type in ("F", "Cf") and all(c.type in ("F", "Cf") for c in current):
            steps.append(current)
            current = []
    if current:
        steps.append(current)
    return steps


def sequence_length(workflow: list[Command]) -> int:
    """Length counting each sketch feature as a single unit (paper Table 2).

    Each modeling step contributes 1; within a step the sketch curves collapse to a
    single unit and the consuming operation is folded in, matching the paper's
    convention that ``sketch + extrusion`` is one step.
    """
    return len(split_steps(workflow))


def operation_ratio(corpus: list[list[Command]]) -> dict:
    """Fraction of each command type across a corpus (paper Table 1).

    Terminators are ignored. Returns a dict ``{command_type: fraction}`` summing to
    1.0 over the observed content commands (empty corpus -> empty dict).
    """
    counts: dict[str, int] = {}
    total = 0
    for workflow in corpus:
        for cmd in strip_terminators(workflow):
            counts[cmd.type] = counts.get(cmd.type, 0) + 1
            total += 1
    if total == 0:
        return {}
    return {k: counts[k] / total for k in sorted(counts)}


# Default histogram buckets mirroring paper Table 2 (0-5, 5-10, 10-15, 15-20, 20+).
DEFAULT_LENGTH_BUCKETS: tuple[tuple[int, int], ...] = (
    (0, 5), (5, 10), (10, 15), (15, 20), (20, 10 ** 9),
)


def _bucket_label(low: int, high: int) -> str:
    return f"{low}-{high}" if high < 10 ** 9 else f"{low}+"


def length_distribution(corpus: list[list[Command]],
                        buckets: tuple[tuple[int, int], ...] = DEFAULT_LENGTH_BUCKETS) -> dict:
    """Fraction of workflows whose step-length falls in each bucket (Table 2).

    A length ``n`` falls in bucket ``[low, high)``. Buckets should cover the range;
    a length matching no bucket is ignored. Empty corpus -> zeros.
    """
    labels = [_bucket_label(lo, hi) for lo, hi in buckets]
    counts = {label: 0 for label in labels}
    total = 0
    for workflow in corpus:
        n = sequence_length(workflow)
        for (lo, hi), label in zip(buckets, labels):
            if lo <= n < hi:
                counts[label] += 1
                total += 1
                break
    if total == 0:
        return {label: 0.0 for label in labels}
    return {label: counts[label] / total for label in labels}


def truncate_last_step(workflow: list[Command]) -> list[Command] | None:
    """Remove the final modeling step (paper's back-to-front augmentation).

    Returns the content-command prefix with the last step cut, or ``None`` when the
    workflow has one step or fewer (nothing left to augment). Terminators are
    dropped; re-wrap with :func:`reconstruction.cadparser_schema.pad_sequence` to
    obtain a padded sequence.
    """
    steps = split_steps(workflow)
    if len(steps) <= 1:
        return None
    prefix = steps[:-1]
    return [cmd for step in prefix for cmd in step]


def augment(workflow: list[Command], max_variants: int | None = None) -> list[list[Command]]:
    """Generate augmented prefixes by iteratively cutting the last step.

    Deterministic: variants are produced longest-first (cut one step, then two,
    ...). ``max_variants`` caps the count. A workflow with a single step yields no
    variants.
    """
    variants: list[list[Command]] = []
    current = strip_terminators(workflow)
    while True:
        cut = truncate_last_step(current)
        if cut is None:
            break
        variants.append(cut)
        current = cut
        if max_variants is not None and len(variants) >= max_variants:
            break
    return variants
