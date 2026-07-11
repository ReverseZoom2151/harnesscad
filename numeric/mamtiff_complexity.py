"""Complexity measures for *complex parametric CAD command sequences*.

MamTiff-CAD targets what it calls **complex parametric sequences** -- long
(60-256 command) CAD programs whose difficulty is the whole reason for the
multi-scale design. The paper's ABC-256 dataset is built precisely by *filtering
on sequence complexity* (Sec. 4 / Supp. 1): keep only models "with complete
design operations", "excluding simpler cases with only sketching and extrusion",
ensuring sequence lengths of 60-256. That filter is a deterministic notion of
complexity, and this module makes it concrete -- distinct from any learned model.

A CAD command sequence is modelled as in DeepCAD / MamTiff-CAD: a tuple of
``(command_type, params)`` pairs where ``command_type`` is a small integer (the
paper defines six types incl. ``SOL`` start-of-loop and ``EOS``) and ``params``
is a fixed-width vector with unused entries set to ``-1`` (Sec. 3.2, Supp. 2).

Provided measures (all deterministic, stdlib-only):

* :func:`effective_length` -- non-padding command count (``EOS``/pad excluded);
* :func:`command_type_entropy` -- Shannon entropy (bits) of the command-type
  histogram, high when many distinct operations are interleaved;
* :func:`loop_structure` -- number of sketch loops and their nesting via the
  ``SOL`` markers, plus max/avg loop length (topological richness);
* :func:`parameter_richness` -- fraction of parameter slots actually used (not
  ``-1``), i.e. how much geometric detail the sequence carries;
* :func:`transition_diversity` -- distinct command-type bigrams over the total,
  measuring how varied the operation ordering is;
* :func:`sequence_complexity` -- a single normalised score combining the above
  with explicit weights (deterministic, reproducible);
* :func:`is_complex` -- the ABC-256-style boolean filter (length band + a
  minimum operation variety), replicating "exclude sketch+extrude-only" models;
* :func:`multiscale_complexity_profile` -- complexity evaluated over a pyramid of
  coarsened sub-sequences, giving a per-scale complexity signature.
"""

from __future__ import annotations

import math

Command = tuple[int, tuple[float, ...]]
Program = tuple[Command, ...]


def effective_length(program: Program, eos_type: int = -1,
                     pad_type: int = -2) -> int:
    """Number of real commands, excluding the ``EOS`` token and padding.

    A command whose type equals ``eos_type`` terminates the sequence (it and
    everything after it is ignored); commands of type ``pad_type`` are skipped.
    """
    count = 0
    for ctype, _ in program:
        if ctype == eos_type:
            break
        if ctype == pad_type:
            continue
        count += 1
    return count


def _active(program: Program, eos_type: int, pad_type: int) -> Program:
    active: list[Command] = []
    for cmd in program:
        if cmd[0] == eos_type:
            break
        if cmd[0] == pad_type:
            continue
        active.append(cmd)
    return tuple(active)


def command_type_entropy(program: Program, eos_type: int = -1,
                         pad_type: int = -2) -> float:
    """Shannon entropy (in bits) of the command-type distribution over the
    active commands. ``0`` for an empty or single-type sequence; grows as more
    distinct command types appear in balanced proportion.
    """
    active = _active(program, eos_type, pad_type)
    if not active:
        return 0.0
    hist: dict[int, int] = {}
    for ctype, _ in active:
        hist[ctype] = hist.get(ctype, 0) + 1
    n = len(active)
    ent = 0.0
    for c in hist.values():
        p = c / n
        ent -= p * math.log2(p)
    return ent


def loop_structure(program: Program, sol_type: int, eos_type: int = -1,
                   pad_type: int = -2) -> dict[str, float]:
    """Sketch-loop statistics keyed on the ``SOL`` (start-of-loop) marker.

    Returns a dict with ``num_loops`` (count of ``SOL`` markers), ``max_loop``
    and ``avg_loop`` (commands between consecutive ``SOL`` markers, exclusive of
    the marker). More loops and longer loops indicate greater topological
    complexity of the 2D profiles.
    """
    active = _active(program, eos_type, pad_type)
    loop_lengths: list[int] = []
    cur = None
    for ctype, _ in active:
        if ctype == sol_type:
            if cur is not None:
                loop_lengths.append(cur)
            cur = 0
        elif cur is not None:
            cur += 1
    if cur is not None:
        loop_lengths.append(cur)
    num = len(loop_lengths)
    if num == 0:
        return {"num_loops": 0.0, "max_loop": 0.0, "avg_loop": 0.0}
    return {
        "num_loops": float(num),
        "max_loop": float(max(loop_lengths)),
        "avg_loop": sum(loop_lengths) / num,
    }


def parameter_richness(program: Program, unused: float = -1.0,
                       eos_type: int = -1, pad_type: int = -2) -> float:
    """Fraction of parameter slots that are *used* (not equal to ``unused``)
    across all active commands, in ``[0, 1]``. High values mean each command
    carries rich geometric information rather than mostly-empty parameters.
    """
    active = _active(program, eos_type, pad_type)
    total = 0
    used = 0
    for _, params in active:
        for p in params:
            total += 1
            if p != unused:
                used += 1
    if total == 0:
        return 0.0
    return used / total


def transition_diversity(program: Program, eos_type: int = -1,
                         pad_type: int = -2) -> float:
    """Ratio of *distinct* command-type bigrams to the number of transitions,
    in ``[0, 1]``. ``0`` for fewer than two commands; near ``1`` when almost
    every adjacent pair of operations is different.
    """
    active = _active(program, eos_type, pad_type)
    if len(active) < 2:
        return 0.0
    bigrams = set()
    trans = 0
    for i in range(len(active) - 1):
        bigrams.add((active[i][0], active[i + 1][0]))
        trans += 1
    return len(bigrams) / trans


def sequence_complexity(program: Program, sol_type: int, eos_type: int = -1,
                        pad_type: int = -2, length_norm: int = 256) -> float:
    """Single normalised complexity score in ``[0, 1]`` combining length,
    command-type entropy, loop count, parameter richness, and transition
    diversity with fixed deterministic weights.

    The weighting mirrors the paper's emphasis: length and operation variety
    dominate (that is what makes a sequence "complex"), while parameter richness
    and loop structure refine the score. Reproducible for identical input.
    """
    active = _active(program, eos_type, pad_type)
    if not active:
        return 0.0
    length_term = min(1.0, len(active) / length_norm)
    # entropy normalised by log2 of the distinct type count ceiling (6 types)
    entropy_term = min(1.0, command_type_entropy(program, eos_type, pad_type)
                       / math.log2(6))
    loops = loop_structure(program, sol_type, eos_type, pad_type)
    loop_term = min(1.0, loops["num_loops"] / 8.0)
    param_term = parameter_richness(program, -1.0, eos_type, pad_type)
    trans_term = transition_diversity(program, eos_type, pad_type)
    score = (0.35 * length_term + 0.25 * entropy_term + 0.15 * loop_term
             + 0.10 * param_term + 0.15 * trans_term)
    return score


def is_complex(program: Program, sol_type: int, min_len: int = 60,
               max_len: int = 256, min_types: int = 3, eos_type: int = -1,
               pad_type: int = -2) -> bool:
    """ABC-256-style complexity filter (Sec. 4).

    Returns ``True`` when the effective length is within ``[min_len, max_len]``
    **and** the sequence uses at least ``min_types`` distinct command types
    (rejecting the "only sketching and extrusion" simple cases the paper
    explicitly excludes).
    """
    active = _active(program, eos_type, pad_type)
    n = len(active)
    if not (min_len <= n <= max_len):
        return False
    distinct = {ctype for ctype, _ in active}
    return len(distinct) >= min_types


def multiscale_complexity_profile(program: Program, sol_type: int,
                                  levels: int = 3, factor: int = 2,
                                  eos_type: int = -1,
                                  pad_type: int = -2) -> tuple[float, ...]:
    """Per-scale complexity signature.

    Level 0 scores the full active sequence; each finer level scores the
    sequence *stride-subsampled* by ``factor`` (keeping every ``factor``-th
    command), giving a coarse-to-fine view of how complexity concentrates. This
    is the deterministic, complexity-domain analogue of the feature pyramid in
    ``numeric.mamtiff_pyramid``.
    """
    if levels <= 0:
        raise ValueError("levels must be positive")
    if factor < 2:
        raise ValueError("factor must be >= 2")
    active = _active(program, eos_type, pad_type)
    profile: list[float] = []
    stride = 1
    for _ in range(levels):
        sub = active[::stride]
        profile.append(sequence_complexity(sub, sol_type, eos_type, pad_type))
        stride *= factor
    return tuple(profile)
