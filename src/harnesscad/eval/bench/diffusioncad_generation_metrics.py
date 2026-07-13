"""Sequence-level generation metrics from Diffusion-CAD (Table III).

Diffusion-CAD reports six generation-quality metrics. Three are point-cloud
based (COV / MMD / JSD) and depend on geometry sampling (out of scope here — the
repo's ``bench`` geometry metrics cover Chamfer/coverage). The remaining three
are defined purely over the generated *CAD command sequences* and are fully
deterministic:

* **Unique %** — percentage of generated sequences that occur exactly once in
  the generated set (the paper: "occur only once in the generated dataset").
* **Novel %** — percentage of generated sequences that do NOT appear in the
  training set (the paper: "do not appear in the training set").
* **Invalidity %** — percentage of generated sequences that are ill-formed. The
  paper's notion is topological (point clouds not extractable); the deterministic
  proxy implemented here is *sequence well-formedness* against the DeepCAD /
  Diffusion-CAD command grammar (SOL / L / A / C / E / EOS): every loop opens
  with ``<SOL>``, geometry commands only occur inside an open loop, an extrusion
  ``E`` closes the current sketch, and the sequence terminates with ``<EOS>``.

Sequences are hashable token tuples. Stdlib-only, deterministic.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

# Canonical Diffusion-CAD command-type tokens (Table I).
SOL = "<SOL>"
LINE = "L"
ARC = "A"
CIRCLE = "C"
EXTRUDE = "E"
EOS = "<EOS>"

_GEOMETRY = frozenset({LINE, ARC, CIRCLE})


def _key(seq: Sequence) -> tuple:
    return tuple(seq)


def unique_percent(generated: Iterable[Sequence]) -> float:
    """Percentage of generated sequences that occur exactly once."""
    keys = [_key(s) for s in generated]
    if not keys:
        return 0.0
    counts = Counter(keys)
    singletons = sum(1 for k in keys if counts[k] == 1)
    return 100.0 * singletons / len(keys)


def novel_percent(
    generated: Iterable[Sequence], training: Iterable[Sequence]
) -> float:
    """Percentage of generated sequences absent from the training set."""
    train = {_key(s) for s in training}
    keys = [_key(s) for s in generated]
    if not keys:
        return 0.0
    novel = sum(1 for k in keys if k not in train)
    return 100.0 * novel / len(keys)


def is_wellformed(commands: Sequence) -> bool:
    """Grammar well-formedness of a single command-type sequence.

    ``commands`` is a sequence of command-type tokens (SOL / L / A / C / E /
    EOS). Rules:

    * a loop opens with ``<SOL>``; geometry (L/A/C) may only appear while a loop
      is open;
    * a circle ``C`` is itself a closed loop, so it is only valid immediately
      inside an open loop;
    * an extrusion ``E`` requires at least one completed loop since the last
      extrusion and closes the current sketch (loops must be closed before E);
    * the sequence must contain at least one extrusion and terminate at ``EOS``;
      nothing follows ``EOS``.
    """
    toks = list(commands)
    if not toks or toks[-1] != EOS:
        return False
    loop_open = False
    loop_has_geom = False
    completed_loops = 0  # since last extrusion
    extrusions = 0
    seen_eos = False
    for tok in toks:
        if seen_eos:
            return False  # nothing after EOS
        if tok == EOS:
            if loop_open:
                return False  # dangling open loop
            seen_eos = True
        elif tok == SOL:
            if loop_open:
                return False  # nested/unterminated loop
            loop_open = True
            loop_has_geom = False
        elif tok in _GEOMETRY:
            if not loop_open:
                return False  # geometry outside a loop
            loop_has_geom = True
            if tok == CIRCLE:
                # A circle is a self-contained loop.
                loop_open = False
                completed_loops += 1
        elif tok == EXTRUDE:
            if loop_open:
                # implicit close only if the open loop had geometry
                if not loop_has_geom:
                    return False
                loop_open = False
                completed_loops += 1
            if completed_loops == 0:
                return False  # nothing to extrude
            extrusions += 1
            completed_loops = 0
        else:
            return False  # unknown token
    return extrusions >= 1


def invalidity_percent(generated: Iterable[Sequence]) -> float:
    """Percentage of generated command-type sequences that are ill-formed."""
    seqs = list(generated)
    if not seqs:
        return 0.0
    bad = sum(1 for s in seqs if not is_wellformed(s))
    return 100.0 * bad / len(seqs)


def generation_report(
    generated: Iterable[Sequence],
    training: Iterable[Sequence],
    command_view: Iterable[Sequence] | None = None,
) -> dict:
    """Bundle unique / novel / invalidity for a generated set.

    ``generated`` and ``training`` are full token sequences used for the
    uniqueness/novelty set comparison. ``command_view`` (optional) supplies the
    command-type-only views used for the well-formedness check; when omitted,
    ``generated`` is assumed to already be command-type sequences.
    """
    gen = list(generated)
    cmd = list(command_view) if command_view is not None else gen
    return {
        "count": len(gen),
        "unique_pct": unique_percent(gen),
        "novel_pct": novel_percent(gen, training),
        "invalidity_pct": invalidity_percent(cmd),
    }
