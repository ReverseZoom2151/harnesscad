"""Grammar/template-based CAD-program synthesis from a design procedure.

Chen, Shu, Hong, Taber, Li & Klenk, *Learning From Design Procedure To Generate
CAD Programs for Data Augmentation* (NeurIPS 2025 Workshop).

The paper generates NEW CAD programs by conditioning an LLM on a reference
surface program + a design procedure (Sec. 3). The generation model is external.
This module implements the deterministic *procedural* half: given a validated
``DesignProcedure`` (see ``datagen.designproc_procedure``) it synthesises a NEW
CAD **command sequence** by expanding each design step through a fixed grammar
of command templates -- producing programs that, unlike the baselines, contain
B-Spline faces and curves induced by the reference surface.

This is deliberately DISTINCT from the two neighbouring augmenters:

  * ``datagen.contrastcad_rre`` RESAMPLES / recombines existing programs' sketch
    loops (Random Replacement/Extraction) -- it never introduces new B-Spline
    geometry.
  * ``datagen.gencad3d_synthbal`` BALANCES an existing corpus by perturbing and
    replacing commands of real programs across sequence-length classes.

Here nothing is resampled: programs are grown from templates of the *design
procedure* grammar, so every reference-surface procedure yields a program whose
B-Spline face/curve counts are positive by construction (the paper's key
contribution -- Table 1: 77 % w/ B-Spline faces vs 0 % for the baselines).

Each emitted command is a dict with an ``op`` and geometric-accounting fields
(``faces``, ``curves``, ``bspline_faces``, ``bspline_curves``, ``lines``) that
the metrics module (``datagen.designproc_bspline_metrics``) consumes. Nothing
is executed; this is symbolic program construction.

Determinism: randomness flows through ``random.Random``/seed; stdlib only.
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence

from datagen.designproc_procedure import (
    ADD_PRIMITIVE,
    BOOLEAN_OP,
    CONFORM_TO_SURFACE,
    DesignProcedure,
    EXPORT,
    FILLET,
    REMOVE_REFERENCE_SURFACE,
    SELECT_REFERENCE_SURFACE,
    build_procedure,
    is_valid_procedure,
)

Command = Dict[str, object]
Program = List[Command]


def _rng(seed) -> random.Random:
    if isinstance(seed, random.Random):
        return seed
    return random.Random(seed)


def _cmd(op: str, *, lines: int, faces: int = 0, curves: int = 0,
         bspline_faces: int = 0, bspline_curves: int = 0, **extra) -> Command:
    c: Command = {
        "op": op,
        "lines": lines,
        "faces": faces,
        "curves": curves,
        "bspline_faces": bspline_faces,
        "bspline_curves": bspline_curves,
    }
    c.update(extra)
    return c


# ---------------------------------------------------------------------------
# Per-step command templates (the design-procedure grammar)
# ---------------------------------------------------------------------------

# Number of source lines a step's template contributes (a STEP-file "#lines"
# proxy; the paper measures program complexity via STEP line count, Sec. 4.1).
_PRIMITIVE_SHAPES = ("hole", "slot", "pocket", "boss")


def _expand_step(step_kind: str, detail: str, rng: random.Random) -> List[Command]:
    """Expand one design step into one or more CAD commands."""
    if step_kind == SELECT_REFERENCE_SURFACE:
        # A B-Spline reference surface: 1 spline face bounded by 4 spline edges.
        return [_cmd("make_spline_surface", lines=12, faces=1, curves=4,
                     bspline_faces=1, bspline_curves=4, detail=detail)]
    if step_kind == CONFORM_TO_SURFACE:
        # Conforming the object top to the surface: adds spline faces/edges that
        # "ripple" the curvature into the solid (Sec. 3.1).
        n = rng.randint(1, 3)
        return [_cmd("conform_face", lines=6, faces=1, curves=2,
                     bspline_faces=1, bspline_curves=2, index=i)
                for i in range(n)]
    if step_kind == ADD_PRIMITIVE:
        shape = _PRIMITIVE_SHAPES[rng.randrange(len(_PRIMITIVE_SHAPES))]
        # A standard sketch+extrude feature: flat faces + straight/circular edges.
        # Circular features contribute one (non-B-Spline) circle curve.
        curves = 1 if shape in ("hole", "boss") else 4
        return [
            _cmd("sketch", lines=4, faces=1, curves=curves, shape=shape),
            _cmd("extrude", lines=3, faces=2, curves=0, shape=shape,
                 boolean="cut" if shape in ("hole", "slot", "pocket") else "add"),
        ]
    if step_kind == BOOLEAN_OP:
        return [_cmd("boolean", lines=2, faces=0, curves=0, detail=detail)]
    if step_kind == FILLET:
        # Fillets round edges into B-Spline (blend) surfaces.
        return [_cmd("fillet", lines=3, faces=1, curves=1,
                     bspline_faces=1, bspline_curves=1, detail=detail)]
    if step_kind == REMOVE_REFERENCE_SURFACE:
        # Removing the surface deletes its own face/edges but the object keeps the
        # conformed curvature already baked into it (net accounting = negative
        # only for the surface's own primitive).
        return [_cmd("delete_surface", lines=2, faces=-1, curves=-4,
                     bspline_faces=-1, bspline_curves=-4, detail=detail)]
    if step_kind == EXPORT:
        return [_cmd("export", lines=2, detail=detail)]
    raise ValueError("no template for step kind %r" % step_kind)  # pragma: no cover


# ---------------------------------------------------------------------------
# Program synthesis
# ---------------------------------------------------------------------------

def synthesize_program(proc: DesignProcedure, seed) -> Program:
    """Synthesise a CAD command sequence from a validated design procedure.

    Each step is expanded through :func:`_expand_step`. Raises ``ValueError`` if
    the procedure is structurally invalid (see ``validate_procedure``), so no
    ill-formed program can be produced. Deterministic given ``seed``.
    """
    if not is_valid_procedure(proc):
        raise ValueError("cannot synthesise from an invalid procedure")
    rng = _rng(seed)
    program: Program = []
    for step in proc.steps:
        program.extend(_expand_step(step.kind, step.detail, rng))
    return program


def program_totals(program: Sequence[Command]) -> Dict[str, int]:
    """Sum the geometric-accounting fields over a program (clamped at 0).

    Returns ``{"lines", "faces", "curves", "bspline_faces", "bspline_curves"}``.
    Face/curve totals are clamped to be non-negative (the reference-surface
    removal subtracts the surface's own primitives).
    """
    keys = ("lines", "faces", "curves", "bspline_faces", "bspline_curves")
    out = {k: 0 for k in keys}
    for c in program:
        for k in keys:
            out[k] += int(c.get(k, 0))
    for k in ("faces", "curves", "bspline_faces", "bspline_curves"):
        out[k] = max(0, out[k])
    out["n_commands"] = len(program)
    return out


def has_bspline_geometry(program: Sequence[Command]) -> bool:
    """True if the finished program retains any B-Spline face or curve."""
    t = program_totals(program)
    return t["bspline_faces"] > 0 or t["bspline_curves"] > 0


# ---------------------------------------------------------------------------
# Family synthesis (diverse augmentation set)
# ---------------------------------------------------------------------------

def synthesize_family(category: str, surface_kinds: Sequence[str],
                      descriptions: Sequence[str], seed,
                      mode: str = "full") -> List[dict]:
    """Grow a diverse augmentation set of NEW programs.

    The Cartesian product of ``surface_kinds`` x ``descriptions`` seeds distinct
    procedures (feature count varies with the description index); each is
    synthesised into a program. ``mode="full"`` uses reference surfaces (Ours);
    ``mode="none"`` builds the surface-free baseline (ablation ours(-RT)).

    Returns a list of dicts ``{"category", "surface_kind", "description",
    "program", "totals"}``. Deterministic in all inputs.
    """
    if mode not in ("full", "none"):
        raise ValueError("mode must be 'full' or 'none'")
    rng = _rng(seed)
    out: List[dict] = []
    for si, surface_kind in enumerate(surface_kinds):
        for di, desc in enumerate(descriptions):
            n_primitives = 1 + ((si + di) % 3)
            proc = build_procedure(
                category, surface_kind, n_primitives=n_primitives,
                with_reference_surface=(mode == "full"))
            prog = synthesize_program(proc, rng.randint(0, 2 ** 31 - 1))
            out.append({
                "category": category,
                "surface_kind": surface_kind if mode == "full" else "",
                "description": desc,
                "program": prog,
                "totals": program_totals(prog),
            })
    return out
