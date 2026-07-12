"""Template-based multi-level prompt generator for Text2CAD (Khan et al., 2024).

Text2CAD annotates each DeepCAD model with four prompts of increasing detail (see
:mod:`dataengine.text2cad_prompt_levels`). The paper produces these with an LLM, but
the *structure* of each level is deterministic (Fig. 4): the abstract/beginner
levels are dominated by a shape phrase, while the intermediate/expert levels layer
on sketch and extrusion parametrics. This module is the LLM-free, template-based
generator: given a DeepCAD command sequence (see
:mod:`reconstruction.deepcad_command_spec`) it extracts the structured design
aspects and renders a prompt at any requested level.

  * L0 Abstract     -- the supplied VLM shape phrase, verbatim.
  * L1 Beginner     -- shape phrase + plain construction steps, no numbers.
  * L2 Intermediate -- shape phrase + generalized sketch/extrusion description.
  * L3 Expert       -- coordinate-system setup + precise per-curve geometry with
                       relative (quantized) values + precise extrusion parameters.

Everything is pure and deterministic. The generator never invents geometry: numbers
in L3 come straight from the command parameters, rounded to a fixed number of
decimals for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass

from dataengine.text2cad_prompt_levels import level
from reconstruction.deepcad_command_spec import (
    ARC,
    CIRCLE,
    EXT,
    LINE,
    SOL,
    Command,
)

# Boolean-operation names by index (DeepCAD: New/Cut/Join/Intersect, paper Sec. 9).
_BOOLEAN_NAMES = ("new body", "cut", "join", "intersect")


class PromptGeneratorError(ValueError):
    """Raised for malformed command sequences or unsupported levels."""


@dataclass(frozen=True)
class LoopSummary:
    """Primitive counts within a single sketch loop."""

    lines: int
    arcs: int
    circles: int

    @property
    def total(self) -> int:
        return self.lines + self.arcs + self.circles


@dataclass(frozen=True)
class DesignAspects:
    """Structured, level-agnostic summary of a command sequence."""

    loops: tuple[LoopSummary, ...]
    lines: int
    arcs: int
    circles: int
    n_extrusions: int
    extrude_distances: tuple[tuple[float, float], ...]  # (e1, e2) per extrusion
    booleans: tuple[str, ...]                            # boolean op name per extrusion

    @property
    def n_loops(self) -> int:
        return len(self.loops)

    @property
    def n_primitives(self) -> int:
        return self.lines + self.arcs + self.circles


def _round(value: float, decimals: int) -> float:
    return round(float(value), decimals)


def extract_aspects(commands: list[Command]) -> DesignAspects:
    """Summarise a DeepCAD command sequence into structured design aspects."""
    loops: list[LoopSummary] = []
    cur_line = cur_arc = cur_circle = 0
    have_loop = False
    lines = arcs = circles = 0
    n_ext = 0
    distances: list[tuple[float, float]] = []
    booleans: list[str] = []

    def _flush() -> None:
        nonlocal cur_line, cur_arc, cur_circle, have_loop
        if have_loop:
            loops.append(LoopSummary(cur_line, cur_arc, cur_circle))
        cur_line = cur_arc = cur_circle = 0
        have_loop = False

    for cmd in commands:
        if cmd.type == SOL:
            _flush()
            have_loop = True
        elif cmd.type == LINE:
            cur_line += 1
            lines += 1
            have_loop = True
        elif cmd.type == ARC:
            cur_arc += 1
            arcs += 1
            have_loop = True
        elif cmd.type == CIRCLE:
            cur_circle += 1
            circles += 1
            have_loop = True
        elif cmd.type == EXT:
            _flush()
            n_ext += 1
            e1 = cmd.get("e1", 0.0)
            e2 = cmd.get("e2", 0.0)
            distances.append((_round(e1, 4), _round(e2, 4)))
            b_idx = int(cmd.get("b", 0.0))
            b_idx = b_idx if 0 <= b_idx < len(_BOOLEAN_NAMES) else 0
            booleans.append(_BOOLEAN_NAMES[b_idx])
    _flush()

    return DesignAspects(
        loops=tuple(loops),
        lines=lines,
        arcs=arcs,
        circles=circles,
        n_extrusions=n_ext,
        extrude_distances=tuple(distances),
        booleans=tuple(booleans),
    )


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" + ("" if count == 1 else "s")


def _primitive_phrase(aspects: DesignAspects) -> str:
    """Human phrase enumerating the sketch primitives, e.g. '2 circles and 4 lines'."""
    parts: list[str] = []
    if aspects.circles:
        parts.append(_plural(aspects.circles, "circle"))
    if aspects.arcs:
        parts.append(_plural(aspects.arcs, "arc"))
    if aspects.lines:
        parts.append(_plural(aspects.lines, "line"))
    if not parts:
        return "no primitives"
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _extrusion_phrase(aspects: DesignAspects, *, precise: bool, decimals: int) -> str:
    if aspects.n_extrusions == 0:
        return ""
    if precise:
        segs = []
        for (e1, e2), op in zip(aspects.extrude_distances, aspects.booleans):
            segs.append(
                f"extrude the sketch along the normal by {round(e1, decimals)} units "
                f"and {round(e2, decimals)} units in the opposite direction "
                f"as a {op} operation"
            )
        return "; ".join(segs)
    return ("extrude the sketch along the normal to form a solid body"
            if aspects.n_extrusions == 1
            else f"perform {aspects.n_extrusions} extrusions to form the solid")


def generate_prompt(
    commands: list[Command],
    level_code: str,
    *,
    shape_description: str | None = None,
    decimals: int = 2,
) -> str:
    """Render a Text2CAD prompt for ``commands`` at the requested level.

    ``shape_description`` is the VLM shape phrase; it is required for L0/L1 (which are
    shape-dominated) and optional as a lead-in for L2/L3.
    """
    lv = level(level_code)
    aspects = extract_aspects(commands)

    if lv.code in ("L0", "L1") and not shape_description:
        raise PromptGeneratorError(
            f"level {lv.code} requires a shape_description")

    if lv.code == "L0":
        return shape_description.strip()

    if lv.code == "L1":
        prims = _primitive_phrase(aspects)
        steps = (f"Start by sketching {prims}, then turn the sketch into a "
                 f"solid shape.")
        return f"{shape_description.strip()} {steps}"

    if lv.code == "L2":
        lead = f"{shape_description.strip()} " if shape_description else ""
        prims = _primitive_phrase(aspects)
        ext = _extrusion_phrase(aspects, precise=False, decimals=decimals)
        body = (f"Sketch {prims} across {_plural(aspects.n_loops, 'loop')}, "
                f"then {ext}." if ext
                else f"Sketch {prims} across {_plural(aspects.n_loops, 'loop')}.")
        return f"{lead}{body}"

    if lv.code == "L3":
        lead = f"{shape_description.strip()} " if shape_description else ""
        prims = _primitive_phrase(aspects)
        ext = _extrusion_phrase(aspects, precise=True, decimals=decimals)
        body = (
            f"First set up a coordinate system. On the sketch plane, draw {prims} "
            f"distributed over {_plural(aspects.n_loops, 'loop')}. Then {ext}."
            if ext else
            f"First set up a coordinate system. On the sketch plane, draw {prims} "
            f"distributed over {_plural(aspects.n_loops, 'loop')}."
        )
        return f"{lead}{body}"

    raise PromptGeneratorError(f"unsupported level: {level_code!r}")


def generate_all_levels(
    commands: list[Command],
    shape_description: str,
    *,
    decimals: int = 2,
) -> dict[str, str]:
    """Render all four prompts (L0..L3) for a command sequence."""
    return {
        code: generate_prompt(
            commands, code, shape_description=shape_description, decimals=decimals)
        for code in ("L0", "L1", "L2", "L3")
    }
