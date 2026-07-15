"""Construction-sequence -> natural-language prompt templating (GuideCAD).

Mined from *GuideCAD: A Lightweight Multimodal Framework for 3D CAD Model
Generation via Prefix Embedding*. GuideCAD's model (a prefix-embedding mapping
network feeding a frozen LLM) is trained, but its **dataset construction** step is
a deterministic algorithm: each DeepCAD construction command is emitted as a
"Command Line" and then rendered into a text prompt via a fixed per-command
template (paper Sec. 3.2, Table 1 & 2). This module ports that template pipeline.

The command vocabulary is the paper's ``Nc = 6`` commands
``{<SOL>, Line, Arc, Circle, Extrusion, <EOS>}`` over the ``Np = 16`` parameters.
Given a structured construction sequence, :func:`render_prompt` produces a
deterministic textual description; :func:`command_lines` exposes the intermediate
one-line-per-command form the paper feeds through the template.

Stdlib-only, no model calls, fully deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

__all__ = [
    "Command",
    "SOL",
    "Line",
    "Arc",
    "Circle",
    "Extrusion",
    "EOS",
    "command_line",
    "command_lines",
    "render_prompt",
    "COMMANDS",
    "PARAMETERS",
]

#: The Nc = 6 command tokens (paper Table 1).
COMMANDS: Tuple[str, ...] = ("<SOL>", "Line", "Arc", "Circle", "Extrusion", "<EOS>")

#: The Np = 16 parameter symbols (paper Table 1).
PARAMETERS: Tuple[str, ...] = (
    "x", "y", "alpha", "c", "r",
    "px", "py", "pz", "ox", "oy", "oz",
    "s", "e1", "e2", "b", "w",
)


@dataclass(frozen=True)
class Command:
    """One construction command with its parameter mapping."""

    kind: str
    params: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in COMMANDS:
            raise ValueError(f"unknown command {self.kind!r}; expected one of {COMMANDS}")
        for key in self.params:
            if key not in PARAMETERS:
                raise ValueError(f"unknown parameter {key!r}")


def SOL() -> Command:
    """Start-of-loop marker."""
    return Command("<SOL>")


def Line(x: float, y: float) -> Command:
    """A line to end point ``(x, y)``."""
    return Command("Line", {"x": x, "y": y})


def Arc(x: float, y: float, alpha: float, c: float) -> Command:
    """An arc to ``(x, y)`` with sweep angle ``alpha`` and ccw flag ``c``."""
    return Command("Arc", {"x": x, "y": y, "alpha": alpha, "c": c})


def Circle(x: float, y: float, r: float) -> Command:
    """A circle centred at ``(x, y)`` with radius ``r``."""
    return Command("Circle", {"x": x, "y": y, "r": r})


def Extrusion(
    px: float, py: float, pz: float,
    ox: float, oy: float, oz: float,
    s: float, e1: float, e2: float, b: float, w: float,
) -> Command:
    """An extrusion: plane orientation ``(px..pz, ox..oz)``, scale ``s``,
    extents ``e1, e2``, both-sides flag ``b`` and merge type ``w``."""
    return Command("Extrusion", {
        "px": px, "py": py, "pz": pz, "ox": ox, "oy": oy, "oz": oz,
        "s": s, "e1": e1, "e2": e2, "b": b, "w": w,
    })


def EOS() -> Command:
    """End-of-sequence marker."""
    return Command("<EOS>")


_MERGE_NAMES = {0: "a new body", 1: "joining", 2: "cutting", 3: "intersecting"}
_EXTRUDE_NAMES = {0: "one-sided", 1: "symmetric", 2: "two-sided"}


def _fmt(v: float) -> str:
    """Render a numeric parameter deterministically (ints stay int-looking)."""
    if float(v).is_integer():
        return str(int(v))
    return f"{v:g}"


def command_line(cmd: Command) -> str:
    """The intermediate one-line, comma-joined form of a command (paper Fig. 2b)."""
    if cmd.kind in ("<SOL>", "<EOS>"):
        return cmd.kind
    parts = [cmd.kind]
    # emit params in the canonical PARAMETERS order for determinism
    for key in PARAMETERS:
        if key in cmd.params:
            parts.append(f"{key}={_fmt(cmd.params[key])}")
    return ", ".join(parts)


def command_lines(sequence: Sequence[Command]) -> List[str]:
    """Every command rendered as a Command Line, in order."""
    return [command_line(c) for c in sequence]


def _render_one(cmd: Command) -> str:
    p = cmd.params
    if cmd.kind == "<SOL>":
        return "Start a new closed loop."
    if cmd.kind == "Line":
        return f"Draw a line to point ({_fmt(p['x'])}, {_fmt(p['y'])})."
    if cmd.kind == "Arc":
        ccw = "counter-clockwise" if p.get("c", 0) else "clockwise"
        return (f"Draw a {ccw} arc to point ({_fmt(p['x'])}, {_fmt(p['y'])}) "
                f"sweeping {_fmt(p['alpha'])} degrees.")
    if cmd.kind == "Circle":
        return (f"Draw a circle centred at ({_fmt(p['x'])}, {_fmt(p['y'])}) "
                f"with radius {_fmt(p['r'])}.")
    if cmd.kind == "Extrusion":
        merge = _MERGE_NAMES.get(int(p.get("w", 0)), "a new body")
        ext = _EXTRUDE_NAMES.get(int(p.get("b", 0)), "one-sided")
        return (f"Extrude the sketch ({ext}) by distances "
                f"{_fmt(p['e1'])} and {_fmt(p['e2'])} at scale {_fmt(p['s'])}, "
                f"merging as {merge}.")
    if cmd.kind == "<EOS>":
        return "End of the construction sequence."
    raise ValueError(f"cannot render command {cmd.kind!r}")


def render_prompt(sequence: Sequence[Command], numbered: bool = True) -> str:
    """Render a whole construction sequence into a deterministic text prompt.

    ``numbered`` prefixes each step with ``1.``, ``2.`` etc.; otherwise the steps
    are joined with single spaces into one paragraph.
    """
    lines = [_render_one(c) for c in sequence]
    if numbered:
        return "\n".join(f"{i}. {line}" for i, line in enumerate(lines, 1))
    return " ".join(lines)
