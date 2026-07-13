"""Infeasibility taxonomy for *generated* CAD command sequences.

Tsuji, Flores Medina, Gupta & Alam, *GenCAD-Self-Repairing: Feasibility
Enhancement for 3D CAD Generation* (MIT, 2024/25).

GenCAD generates a CAD model as an autoregressive *command sequence* (in the
DeepCAD vocabulary: ``SOL / Line / Arc / Circle / Ext / EOS``). The paper's core
observation is that ~10% of those generated sequences are **infeasible** — the
OpenCASCADE geometry kernel cannot decode them into a valid B-rep (Fig. 2). The
paper repairs these in *latent* space with a learned SSL regressor; that learned
half is out of scope. What is deterministic and locally buildable is the thing
the learned model is a proxy for: a **program-level feasibility check** that,
given a command sequence, names *why* it cannot form a solid.

This module is that structural diagnosis. It is distinct from
:mod:`reliability.repair` (which HEALS a finished OCCT B-rep solid) and from the
:mod:`verifiers` package (which validates a *built* model): here nothing is
built — we reason purely over the token/command sequence *before* it ever
reaches the kernel, exactly the stage at which GenCAD's infeasibilities arise.

Infeasibility taxonomy (deterministic, sequence-level)::

    CURVE_BEFORE_LOOP        a Line/Arc/Circle appears before any SOL opens a loop
    COMMANDS_AFTER_EOS       tokens follow the terminating EOS
    MISSING_EOS              the sequence never terminates with EOS
    EMPTY_LOOP               a SOL is not followed by any curve
    DEGENERATE_PROFILE       a loop cannot bound an area (single Line/Arc, no Circle)
    EXTRUDE_WITHOUT_PROFILE  an Ext with no completed loop to extrude
    TRAILING_PROFILE         loops opened but never consumed by an Ext
    NONPOSITIVE_RADIUS       a Circle/Arc radius is <= 0
    PARAM_OUT_OF_RANGE       a continuous parameter falls outside its normalised band
    PARAM_NOT_DISCRETE       a flag/enum parameter is not one of its allowed values
    DEGENERATE_EXTRUDE       an Ext with zero thickness (e1==e2==0) or scale s<=0

Pure stdlib, fully deterministic. Operates on the canonical DeepCAD
:class:`reconstruction.deepcad_command_spec.Command` objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
from typing import Dict, List, Sequence, Set, Tuple

from harnesscad.domain.reconstruction.tokens.deepcad_command_spec import (
    ARC,
    CIRCLE,
    COMMAND_INDEX,
    EOS,
    EXT,
    LINE,
    SOL,
    Command,
    param_names,
)

# --- feasibility codes -----------------------------------------------------
CURVE_BEFORE_LOOP = "CURVE_BEFORE_LOOP"
COMMANDS_AFTER_EOS = "COMMANDS_AFTER_EOS"
MISSING_EOS = "MISSING_EOS"
EMPTY_LOOP = "EMPTY_LOOP"
DEGENERATE_PROFILE = "DEGENERATE_PROFILE"
EXTRUDE_WITHOUT_PROFILE = "EXTRUDE_WITHOUT_PROFILE"
TRAILING_PROFILE = "TRAILING_PROFILE"
NONPOSITIVE_RADIUS = "NONPOSITIVE_RADIUS"
PARAM_OUT_OF_RANGE = "PARAM_OUT_OF_RANGE"
PARAM_NOT_DISCRETE = "PARAM_NOT_DISCRETE"
DEGENERATE_EXTRUDE = "DEGENERATE_EXTRUDE"

# Every code this taxonomy can emit (stable, for tests / dashboards).
FEASIBILITY_CODES: Tuple[str, ...] = (
    CURVE_BEFORE_LOOP,
    COMMANDS_AFTER_EOS,
    MISSING_EOS,
    EMPTY_LOOP,
    DEGENERATE_PROFILE,
    EXTRUDE_WITHOUT_PROFILE,
    TRAILING_PROFILE,
    NONPOSITIVE_RADIUS,
    PARAM_OUT_OF_RANGE,
    PARAM_NOT_DISCRETE,
    DEGENERATE_EXTRUDE,
)

_CURVES = frozenset({LINE, ARC, CIRCLE})

# --- parameter validity domains (DeepCAD 2x2x2 normalised cube) ------------
# Continuous slots: inclusive [low, high] bands.
PARAM_RANGES: Dict[str, Tuple[float, float]] = {
    "x": (-1.0, 1.0),
    "y": (-1.0, 1.0),
    "alpha": (0.0, 2.0 * pi),
    "r": (0.0, 2.0),          # radius: (0, 2]; the <=0 case is a distinct code
    "theta": (-2.0 * pi, 2.0 * pi),
    "phi": (-2.0 * pi, 2.0 * pi),
    "gamma": (-2.0 * pi, 2.0 * pi),
    "px": (-1.0, 1.0),
    "py": (-1.0, 1.0),
    "pz": (-1.0, 1.0),
    "s": (0.0, 2.0),          # profile scale: (0, 2]; s<=0 is DEGENERATE_EXTRUDE
    "e1": (-2.0, 2.0),
    "e2": (-2.0, 2.0),
}

# Discrete slots: the exact allowed values (DeepCAD conventions).
DISCRETE_PARAMS: Dict[str, Set[float]] = {
    "f": {0.0, 1.0},                # arc counter-clockwise flag
    "b": {0.0, 1.0, 2.0, 3.0},      # boolean: new / join / cut / intersect
    "u": {0.0, 1.0, 2.0},           # extrude type: one-sided / symmetric / two-sided
}


@dataclass(frozen=True)
class Finding:
    """One infeasibility located in a command sequence.

    ``code`` is one of :data:`FEASIBILITY_CODES`; ``index`` is the offending
    command's position (``-1`` for whole-sequence findings such as a missing
    EOS); ``param`` names the slot for parameter findings; ``message`` is a
    human-readable explanation.
    """

    code: str
    index: int
    message: str
    param: str = ""

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "index": self.index,
            "message": self.message,
            "param": self.param,
        }


@dataclass
class Diagnosis:
    """Structured result of diagnosing one command sequence."""

    findings: List[Finding] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return not self.findings

    def codes(self) -> List[str]:
        return [f.code for f in self.findings]

    def to_dict(self) -> dict:
        return {
            "feasible": self.feasible,
            "findings": [f.to_dict() for f in self.findings],
        }


def _validate_params(cmd: Command, index: int) -> List[Finding]:
    """Range/discrete/positivity checks over a command's populated slots."""
    out: List[Finding] = []
    for name in param_names(cmd.type):
        value = cmd.get(name)
        if name == "r" and value <= 0.0:
            out.append(Finding(
                NONPOSITIVE_RADIUS, index,
                f"{cmd.type} radius r={value} must be positive", name))
            continue
        if name in DISCRETE_PARAMS:
            if value not in DISCRETE_PARAMS[name]:
                allowed = sorted(DISCRETE_PARAMS[name])
                out.append(Finding(
                    PARAM_NOT_DISCRETE, index,
                    f"{cmd.type} flag {name}={value} not in {allowed}", name))
            continue
        if name in PARAM_RANGES:
            low, high = PARAM_RANGES[name]
            if not (low <= value <= high):
                out.append(Finding(
                    PARAM_OUT_OF_RANGE, index,
                    f"{cmd.type} {name}={value} outside [{low}, {high}]", name))
    return out


def diagnose(commands: Sequence[Command]) -> Diagnosis:
    """Diagnose a generated CAD command sequence for structural infeasibility.

    Deterministic single pass: returns a :class:`Diagnosis` whose ``findings``
    are in sequence order (parameter findings for a command follow its
    structural finding). An empty ``findings`` list means the sequence is
    structurally feasible (necessary conditions for the kernel to build it).
    """
    for cmd in commands:
        if cmd.type not in COMMAND_INDEX:
            raise ValueError(f"unknown command type: {cmd.type!r}")

    findings: List[Finding] = []

    # 1. EOS terminality: content after the first EOS, or no EOS at all.
    eos_at = next((i for i, c in enumerate(commands) if c.type == EOS), None)
    if eos_at is None:
        if commands:
            findings.append(Finding(
                MISSING_EOS, len(commands),
                "sequence does not terminate with EOS"))
        body = list(commands)
    else:
        if eos_at != len(commands) - 1:
            findings.append(Finding(
                COMMANDS_AFTER_EOS, eos_at + 1,
                f"{len(commands) - eos_at - 1} command(s) follow the EOS at "
                f"index {eos_at}"))
        body = list(commands[:eos_at])

    # 2. Loop / profile / extrude structure over the body (pre-EOS commands).
    open_loop_start = None          # index where the current SOL opened
    loop_curves: List[str] = []     # curve types collected in the current loop
    loop_has_circle = False
    pending_valid_loops = 0         # completed valid loops awaiting an Ext

    def _close_loop(at_index: int) -> None:
        nonlocal open_loop_start, loop_curves, loop_has_circle
        nonlocal pending_valid_loops
        if open_loop_start is None:
            return
        if not loop_curves:
            findings.append(Finding(
                EMPTY_LOOP, open_loop_start,
                "SOL opened a loop with no curves"))
        elif not loop_has_circle and len(loop_curves) < 2:
            findings.append(Finding(
                DEGENERATE_PROFILE, open_loop_start,
                f"loop of a single {loop_curves[0]} cannot bound an area"))
        else:
            pending_valid_loops += 1
        open_loop_start = None
        loop_curves = []
        loop_has_circle = False

    for i, cmd in enumerate(body):
        if cmd.type == SOL:
            _close_loop(i)
            open_loop_start = i
        elif cmd.type in _CURVES:
            if open_loop_start is None:
                findings.append(Finding(
                    CURVE_BEFORE_LOOP, i,
                    f"{cmd.type} at index {i} appears before any SOL"))
                # Treat as an implicit loop so we can still assess the profile.
                open_loop_start = i
            loop_curves.append(cmd.type)
            if cmd.type == CIRCLE:
                loop_has_circle = True
        elif cmd.type == EXT:
            _close_loop(i)
            if pending_valid_loops == 0:
                findings.append(Finding(
                    EXTRUDE_WITHOUT_PROFILE, i,
                    "Ext has no completed profile loop to extrude"))
            pending_valid_loops = 0
            e1 = cmd.get("e1")
            e2 = cmd.get("e2")
            s = cmd.get("s")
            if e1 == 0.0 and e2 == 0.0:
                findings.append(Finding(
                    DEGENERATE_EXTRUDE, i,
                    "Ext has zero thickness (e1 == e2 == 0)"))
            if s <= 0.0:
                findings.append(Finding(
                    DEGENERATE_EXTRUDE, i,
                    f"Ext profile scale s={s} must be positive", "s"))
        # parameter checks apply to every command carrying params
        findings.extend(_validate_params(cmd, i))

    # A loop left open at end-of-body, or valid loops never extruded.
    _close_loop(len(body))
    if pending_valid_loops > 0:
        findings.append(Finding(
            TRAILING_PROFILE, len(body),
            f"{pending_valid_loops} completed loop(s) never consumed by an Ext"))

    # Keep findings in stable sequence order (index, then code).
    findings.sort(key=lambda f: (f.index, FEASIBILITY_CODES.index(f.code)))
    return Diagnosis(findings)


def is_feasible(commands: Sequence[Command]) -> bool:
    """True iff :func:`diagnose` finds no structural infeasibility."""
    return diagnose(commands).feasible
