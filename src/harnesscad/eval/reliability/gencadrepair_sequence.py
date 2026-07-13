"""Sequence-level self-repair for infeasible generated CAD command programs.

Tsuji, Flores Medina, Gupta & Alam, *GenCAD-Self-Repairing* (MIT).

The paper's self-repair *pipeline* (Sec. 3.3.2) takes a command sequence the
geometry kernel rejected and produces a corrected sequence with "a better chance
of corresponding to a valid geometry". Their correction is a learned latent-space
regressor (out of scope). This module is the deterministic, program-level analogue:
given a command sequence and the structural infeasibilities named by
:mod:`reliability.gencadrepair_taxonomy`, it rewrites the sequence into a
structurally feasible one — closing the same feasibility gap the learned regressor
targets, but by explicit, auditable edits rather than a black-box map.

Repair operators (each addresses one taxonomy code)::

    COMMANDS_AFTER_EOS       truncate the sequence at its first EOS
    MISSING_EOS              append a terminal EOS
    CURVE_BEFORE_LOOP        insert an implicit SOL to open the dangling loop
    EMPTY_LOOP               drop a SOL that opened no curves
    DEGENERATE_PROFILE       drop a loop that cannot bound an area
    EXTRUDE_WITHOUT_PROFILE  drop an Ext that has nothing to extrude
    TRAILING_PROFILE         append a default Ext to consume orphan loops
    NONPOSITIVE_RADIUS       raise a non-positive radius to a small positive default
    PARAM_OUT_OF_RANGE       clamp a continuous parameter into its band
    PARAM_NOT_DISCRETE       snap a flag/enum parameter to its nearest allowed value
    DEGENERATE_EXTRUDE       give a zero-thickness/zero-scale Ext a valid depth/scale

The result is guaranteed feasible under :func:`gencadrepair_taxonomy.is_feasible`
and the repair is **idempotent**: repairing an already-feasible sequence returns
it unchanged with no fixes. Pure stdlib, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from harnesscad.domain.reconstruction.deepcad_command_spec import (
    ARC,
    CIRCLE,
    COMMAND_INDEX,
    EOS,
    EXT,
    LINE,
    SOL,
    Command,
    command,
    param_names,
)
from harnesscad.eval.reliability.gencadrepair_taxonomy import (
    DISCRETE_PARAMS,
    PARAM_RANGES,
    Diagnosis,
    diagnose,
)

_CURVES = frozenset({LINE, ARC, CIRCLE})

# Defaults injected when a value is unrepairable in place.
DEFAULT_RADIUS = 0.1
DEFAULT_EXTRUDE_DEPTH = 1.0
DEFAULT_SCALE = 1.0


@dataclass
class RepairOutcome:
    """Result of repairing one command sequence."""

    repaired: List[Command]
    fixes: List[str] = field(default_factory=list)
    diagnosis_before: Diagnosis = field(default_factory=Diagnosis)
    diagnosis_after: Diagnosis = field(default_factory=Diagnosis)

    @property
    def changed(self) -> bool:
        return bool(self.fixes)

    @property
    def feasible(self) -> bool:
        return self.diagnosis_after.feasible

    def to_dict(self) -> dict:
        return {
            "fixes": list(self.fixes),
            "changed": self.changed,
            "feasible": self.feasible,
            "repaired_types": [c.type for c in self.repaired],
            "before": self.diagnosis_before.to_dict(),
            "after": self.diagnosis_after.to_dict(),
        }


def _snap_discrete(value: float, allowed) -> float:
    """Nearest allowed discrete value (ties resolve to the smaller value)."""
    return min(sorted(allowed), key=lambda a: (abs(a - value), a))


def _clean_params(cmd: Command, fixes: List[str], index: int) -> Command:
    """Return a copy of ``cmd`` with every parameter forced into its domain."""
    if cmd.type not in ("Line", "Arc", "Circle", "Ext"):
        return cmd
    values: Dict[str, float] = {}
    for name in param_names(cmd.type):
        value = cmd.get(name)
        if name == "r" and value <= 0.0:
            values[name] = DEFAULT_RADIUS
            fixes.append(
                f"[{index}] {cmd.type}.r={value} -> {DEFAULT_RADIUS} "
                "(non-positive radius)")
        elif name in DISCRETE_PARAMS and value not in DISCRETE_PARAMS[name]:
            snapped = _snap_discrete(value, DISCRETE_PARAMS[name])
            values[name] = snapped
            fixes.append(
                f"[{index}] {cmd.type}.{name}={value} -> {snapped} "
                "(snap to allowed flag)")
        elif name in PARAM_RANGES:
            low, high = PARAM_RANGES[name]
            if value < low or value > high:
                clamped = min(high, max(low, value))
                values[name] = clamped
                fixes.append(
                    f"[{index}] {cmd.type}.{name}={value} -> {clamped} "
                    f"(clamp into [{low}, {high}])")
            else:
                values[name] = value
        else:
            values[name] = value
    return command(cmd.type, **values)


def _loop_is_valid(loop: List[Command]) -> bool:
    """A loop bounds an area iff it contains a Circle or >= 2 curves."""
    curves = [c for c in loop if c.type in _CURVES]
    return any(c.type == CIRCLE for c in curves) or len(curves) >= 2


def _fix_extrude(cmd: Command, fixes: List[str], index: int) -> Command:
    """Give a degenerate Ext a valid non-zero depth and positive scale."""
    values = {name: cmd.get(name) for name in param_names(EXT)}
    if values.get("e1", 0.0) == 0.0 and values.get("e2", 0.0) == 0.0:
        values["e1"] = DEFAULT_EXTRUDE_DEPTH
        fixes.append(
            f"[{index}] Ext e1=e2=0 -> e1={DEFAULT_EXTRUDE_DEPTH} "
            "(zero thickness)")
    if values.get("s", 0.0) <= 0.0:
        fixes.append(
            f"[{index}] Ext scale s={values.get('s')} -> {DEFAULT_SCALE} "
            "(non-positive scale)")
        values["s"] = DEFAULT_SCALE
    return command(EXT, **values)


def _default_extrude() -> Command:
    return command(EXT, theta=0.0, phi=0.0, gamma=0.0, px=0.0, py=0.0, pz=0.0,
                   s=DEFAULT_SCALE, e1=DEFAULT_EXTRUDE_DEPTH, e2=0.0, b=0.0, u=0.0)


def repair_sequence(commands: Sequence[Command]) -> RepairOutcome:
    """Rewrite a command sequence into a structurally feasible one.

    Deterministic and idempotent. The returned :class:`RepairOutcome` carries the
    repaired sequence, an ordered list of human-readable fixes, and the
    before/after :class:`Diagnosis`. ``diagnosis_after`` is always feasible.
    """
    for cmd in commands:
        if cmd.type not in COMMAND_INDEX:
            raise ValueError(f"unknown command type: {cmd.type!r}")

    before = diagnose(commands)
    fixes: List[str] = []

    # 1. Truncate at the first EOS (drops any COMMANDS_AFTER_EOS).
    eos_at = next((i for i, c in enumerate(commands) if c.type == EOS), None)
    if eos_at is not None and eos_at != len(commands) - 1:
        fixes.append(
            f"[{eos_at + 1}] dropped {len(commands) - eos_at - 1} "
            "command(s) after EOS")
    body = list(commands[:eos_at]) if eos_at is not None else list(commands)

    # 2. Parameter cleaning (range / discrete / positive radius).
    cleaned = [_clean_params(c, fixes, i) for i, c in enumerate(body)]

    # 3. Structural rebuild: group loops, drop degenerate ones, attach Ext.
    out: List[Command] = []
    pending: List[List[Command]] = []   # completed valid loops awaiting an Ext
    current: List[Command] = []         # commands of the loop being built

    def _finish_loop(pos: int) -> None:
        nonlocal current
        if not current:
            return
        curves = [c for c in current if c.type in _CURVES]
        if not curves:
            fixes.append(f"[{pos}] dropped empty loop (SOL with no curves)")
        elif not _loop_is_valid(current):
            fixes.append(
                f"[{pos}] dropped degenerate profile "
                f"(single {curves[0].type})")
        else:
            pending.append(list(current))
        current = []

    for i, cmd in enumerate(cleaned):
        if cmd.type == SOL:
            _finish_loop(i)
            current = [cmd]
        elif cmd.type in _CURVES:
            if not current:
                current = [Command(SOL)]
                fixes.append(f"[{i}] inserted SOL before {cmd.type} "
                             "(curve before loop)")
            current.append(cmd)
        elif cmd.type == EXT:
            _finish_loop(i)
            fixed_ext = _fix_extrude(cmd, fixes, i)
            if not pending:
                fixes.append(f"[{i}] dropped Ext with no profile to extrude")
            else:
                for loop in pending:
                    out.extend(loop)
                out.append(fixed_ext)
                pending = []
        # EOS inside body is impossible (truncated); ignore any others.

    _finish_loop(len(cleaned))
    if pending:
        for loop in pending:
            out.extend(loop)
        out.append(_default_extrude())
        fixes.append(
            f"[{len(cleaned)}] appended default Ext for "
            f"{len(pending)} orphan loop(s)")

    # 4. Terminate with a single EOS.
    if eos_at is None and commands:
        fixes.append(f"[{len(commands)}] appended missing EOS")
    out.append(Command(EOS))

    after = diagnose(out)
    return RepairOutcome(
        repaired=out, fixes=fixes,
        diagnosis_before=before, diagnosis_after=after)
