"""PLLM program-level data augmentation: length diversification.

PLLM (Section 3.2, Figure 1b) enriches the pseudo-label set by synthetically
*expanding* and *shortening* the selected programs so the synthetic dataset
spans a broader range of program lengths and structural complexities:

  * Program Expansion -- extend a program by appending additional operations to
    existing workspaces or by spawning new workspaces with procedurally
    generated sketch-feature sequences, capping the total number of workspaces
    at W_max = 5 to keep programs compact and executable.

  * Program Shortening -- generate shorter variants by removing top-level
    boolean operations (union / cut / intersect) while preserving syntactic
    validity, producing more concise programs without drastically altering the
    resulting geometry.

Both transformations create additional valid program-shape pairs (the edited
program is Z, its execution is X), giving perfectly consistent supervision.

This is a *structural* program editor and is distinct from datagen/augment,
datagen/evolution and datagen/contrastcad_permute (parameter / token
perturbation). Here a program is modelled abstractly as an ordered list of
workspaces, each workspace an ordered list of operations, plus a list of
top-level boolean ops -- no CadQuery dependency. Deterministic: any procedural
generation is driven by an injected ``random.Random(seed)``. Stdlib-only.
"""

from __future__ import annotations

import random
from collections import namedtuple

W_MAX = 5  # paper cap on total workspaces

# A structural CAD program.
#   workspaces:   list of workspaces, each a list of operation tokens (str).
#   booleans:     list of top-level boolean op tokens ("union"/"cut"/"intersect").
Program = namedtuple("Program", ["workspaces", "booleans"])

BOOLEAN_OPS = ("union", "cut", "intersect")


def _as_program(p):
    if isinstance(p, Program):
        return Program(list(p.workspaces), list(p.booleans))
    ws, bo = p
    return Program([list(w) for w in ws], list(bo))


def program_length(program):
    """Total token length: all operations across workspaces plus booleans."""
    p = _as_program(program)
    return sum(len(w) for w in p.workspaces) + len(p.booleans)


def expand_append(program, ops, workspace_index=-1):
    """Append ``ops`` to an existing workspace (in-place-safe copy).

    ``workspace_index`` selects which workspace to grow (default: the last).
    Returns a new :class:`Program`. Raises if the program has no workspaces.
    """
    p = _as_program(program)
    if not p.workspaces:
        raise ValueError("program has no workspaces to append to")
    idx = workspace_index if workspace_index >= 0 else len(p.workspaces) + workspace_index
    if not (0 <= idx < len(p.workspaces)):
        raise IndexError("workspace_index out of range")
    p.workspaces[idx] = list(p.workspaces[idx]) + list(ops)
    return p


def expand_spawn(program, new_workspace, boolean="union", w_max=W_MAX):
    """Spawn a new workspace joined by a boolean op, respecting ``w_max``.

    Appends ``new_workspace`` (a list of op tokens) and its connecting boolean
    op. Returns a new :class:`Program`, or the unchanged program (copy) when the
    workspace cap ``w_max`` is already reached. ``boolean`` must be a valid
    top-level boolean op.
    """
    if boolean not in BOOLEAN_OPS:
        raise ValueError("boolean must be one of %r" % (BOOLEAN_OPS,))
    p = _as_program(program)
    if len(p.workspaces) >= w_max:
        return p
    p.workspaces.append(list(new_workspace))
    p.booleans.append(boolean)
    return p


def procedural_workspace(rng, min_ops=1, max_ops=3, prefix="op"):
    """Procedurally generate a small sketch-feature workspace deterministically.

    ``rng`` is a ``random.Random`` instance; draws ``min_ops``..``max_ops``
    numbered operation tokens. Same seed -> same workspace.
    """
    if min_ops < 1 or max_ops < min_ops:
        raise ValueError("require 1 <= min_ops <= max_ops")
    n = rng.randint(min_ops, max_ops)
    return ["%s_%d" % (prefix, rng.randint(0, 999)) for _ in range(n)]


def shorten_remove_boolean(program, boolean_index=-1):
    """Remove one top-level boolean op and its most-recent spawned workspace.

    Removing a boolean drops the workspace it introduced (the one at the
    matching position), yielding a shorter but still syntactically valid
    program. Default removes the last boolean. Returns a new :class:`Program`.
    Raises if there are no top-level booleans to remove.
    """
    p = _as_program(program)
    if not p.booleans:
        raise ValueError("program has no top-level boolean ops to remove")
    idx = boolean_index if boolean_index >= 0 else len(p.booleans) + boolean_index
    if not (0 <= idx < len(p.booleans)):
        raise IndexError("boolean_index out of range")
    p.booleans.pop(idx)
    # boolean i connects workspace (i + 1); drop that workspace when present,
    # never the base workspace 0, keeping at least one workspace.
    ws_idx = idx + 1
    if len(p.workspaces) > 1 and ws_idx < len(p.workspaces):
        p.workspaces.pop(ws_idx)
    elif len(p.workspaces) > 1:
        p.workspaces.pop()
    return p


def diversify(program, seed=0, n_expand=1, n_shorten=1, w_max=W_MAX):
    """Produce length-diversified variants of a program (expansion + shortening).

    Generates up to ``n_expand`` expanded variants (each spawns a procedurally
    generated workspace via a seeded RNG, capped at ``w_max``) and up to
    ``n_shorten`` shortened variants (each removes a top-level boolean op).
    Variants that cannot change the program (cap reached / no booleans) are
    skipped. Returns a dict with the original, ``expanded`` and ``shortened``
    variant lists; deterministic for a fixed ``seed``.
    """
    rng = random.Random(seed)
    p = _as_program(program)
    expanded = []
    cur = p
    for _ in range(n_expand):
        if len(cur.workspaces) >= w_max:
            break
        ws = procedural_workspace(rng)
        boolean = BOOLEAN_OPS[rng.randrange(len(BOOLEAN_OPS))]
        cur = expand_spawn(cur, ws, boolean, w_max)
        expanded.append(cur)
    shortened = []
    cur = p
    for _ in range(n_shorten):
        if not cur.booleans:
            break
        cur = shorten_remove_boolean(cur)
        shortened.append(cur)
    return {"original": p, "expanded": expanded, "shortened": shortened,
            "lengths": {"original": program_length(p),
                        "expanded": [program_length(e) for e in expanded],
                        "shortened": [program_length(s) for s in shortened]}}
