"""OpenMC-style CSG script (de)serialisation, dataset pipeline and metrics.

From *Don't Mesh with Me* (Mews et al., 2024). Sec. 3.1 describes how each CSG
model becomes a **Python script** (using OpenMC-style constructions): "the first
line of the input presents the surfaces that need to be reused to create the
output geometry. The rest of the script defines hyper-planes and cylinders as
variables and uses them to generate cells." The training data is then built by:

  * **Input-output splitting** -- "each sequence is divided into model input and
    target output components. The sequences are split at every possible point
    between cells to maximize the amount of training data."
  * **Input surface determination** -- "for each input-output split, the
    surfaces required to complete the input are identified. These surfaces are
    given as a positional input so that the model can learn to generate 3D
    geometry in a specific location."
  * **Augmentation** -- vary the cut point and the cell order (Sec. 4.1).

This module implements all of the deterministic pieces:

  * ``serialize`` a ``CSGModel`` to a canonical script (surfaces as reusable
    ``s0, s1, ...`` variables, cells as signed surface lists) and ``parse`` it
    back with a restricted, non-``exec`` line parser;
  * ``split_sequence`` / ``input_surfaces`` / ``build_training_pairs`` for the
    dataset pipeline, including cut and order augmentation;
  * plausibility metrics on scripts -- ``correct_syntax`` (does it parse),
    ``structural_signature`` (structure ignoring parameters -> the paper's
    "Same Syntax, Different Params") and parameter-exact equality ("Same Syntax,
    Same Params"), plus ``same_cell_count``.

The learned code-generation model (DeepSeek-Coder fine-tune) that *predicts* the
output script is external and not built here. Pure stdlib; deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from harnesscad.domain.geometry.sdf.halfspace_csg import (
    Cell,
    CSGModel,
    Cylinder,
    HalfSpace,
    Plane,
)


def _fmt(x: float) -> str:
    if x == 0:
        x = 0.0
    if float(x).is_integer():
        return str(int(x))
    return repr(round(float(x), 6))


# --------------------------------------------------------------------------
# Surface collection / naming.
# --------------------------------------------------------------------------

def collect_surfaces(model: CSGModel) -> List[object]:
    """Unique surfaces in order of first appearance across the model's cells."""
    seen: Dict[object, None] = {}
    for cell in model.cells:
        for hs in cell.half_spaces:
            if hs.surface not in seen:
                seen[hs.surface] = None
    return list(seen.keys())


def _surface_names(surfaces: Sequence[object]) -> Dict[object, str]:
    return {s: "s%d" % i for i, s in enumerate(surfaces)}


def _surface_line(name: str, surf: object) -> str:
    if isinstance(surf, Plane):
        return "%s = plane(%s, %s, %s, %s)" % (
            name,
            _fmt(surf.a),
            _fmt(surf.b),
            _fmt(surf.c),
            _fmt(surf.d),
        )
    if isinstance(surf, Cylinder):
        return "%s = cylinder(%s, %s, %s, %s)" % (
            name,
            surf.axis,
            _fmt(surf.u),
            _fmt(surf.v),
            _fmt(surf.radius),
        )
    raise TypeError("unknown surface type: %r" % (surf,))


# --------------------------------------------------------------------------
# Serialisation.
# --------------------------------------------------------------------------

def serialize(model: CSGModel, input_surface_names: Optional[Sequence[str]] = None) -> str:
    """Render ``model`` as a canonical OpenMC-style script.

    ``input_surface_names`` (if given) become the leading reuse-declaration
    line, mirroring the paper's script input (Fig. 3a).
    """
    surfaces = collect_surfaces(model)
    names = _surface_names(surfaces)
    lines: List[str] = []
    reuse = list(input_surface_names) if input_surface_names else []
    lines.append("# input_surfaces: " + " ".join(reuse))
    for surf in surfaces:
        lines.append(_surface_line(names[surf], surf))
    for ci, cell in enumerate(model.cells):
        terms = []
        for hs in cell.half_spaces:
            sign = "+" if hs.sense > 0 else "-"
            terms.append(sign + names[hs.surface])
        lines.append("cell%d = [%s]" % (ci, ", ".join(terms)))
    lines.append("model = union(%s)" % ", ".join("cell%d" % i for i in range(len(model.cells))))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Parsing (restricted, no exec).
# --------------------------------------------------------------------------

class ScriptSyntaxError(ValueError):
    pass


def _parse_surface_rhs(rhs: str) -> object:
    rhs = rhs.strip()
    if rhs.startswith("plane(") and rhs.endswith(")"):
        args = [a.strip() for a in rhs[len("plane(") : -1].split(",")]
        if len(args) != 4:
            raise ScriptSyntaxError("plane needs 4 args: %r" % rhs)
        return Plane(*[float(a) for a in args])
    if rhs.startswith("cylinder(") and rhs.endswith(")"):
        args = [a.strip() for a in rhs[len("cylinder(") : -1].split(",")]
        if len(args) != 4:
            raise ScriptSyntaxError("cylinder needs 4 args: %r" % rhs)
        axis = args[0]
        if axis not in ("x", "y", "z"):
            raise ScriptSyntaxError("bad cylinder axis: %r" % axis)
        return Cylinder(axis, float(args[1]), float(args[2]), float(args[3]))
    raise ScriptSyntaxError("unknown surface rhs: %r" % rhs)


@dataclass
class ParsedScript:
    model: CSGModel
    input_surfaces: List[str]


def parse(script: str) -> ParsedScript:
    """Parse a script produced by :func:`serialize` back into a model.

    Raises :class:`ScriptSyntaxError` on malformed input. This is the basis of
    the paper's "Correct Syntax" plausibility metric.
    """
    surfaces: Dict[str, object] = {}
    cells: List[Cell] = []
    input_surface_names: List[str] = []
    cell_names: Dict[str, int] = {}
    for raw in script.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("# input_surfaces:"):
            rest = line[len("# input_surfaces:") :].strip()
            input_surface_names = rest.split() if rest else []
            continue
        if line.startswith("#"):
            continue
        if "=" not in line:
            raise ScriptSyntaxError("no assignment: %r" % line)
        lhs, rhs = (p.strip() for p in line.split("=", 1))
        if lhs.startswith("s") and rhs.split("(", 1)[0] in ("plane", "cylinder"):
            surfaces[lhs] = _parse_surface_rhs(rhs)
        elif lhs.startswith("cell"):
            rhs = rhs.strip()
            if not (rhs.startswith("[") and rhs.endswith("]")):
                raise ScriptSyntaxError("cell rhs must be a list: %r" % rhs)
            body = rhs[1:-1].strip()
            hs_list: List[HalfSpace] = []
            if body:
                for term in body.split(","):
                    term = term.strip()
                    if not term or term[0] not in "+-":
                        raise ScriptSyntaxError("term needs +/- sign: %r" % term)
                    sense = 1 if term[0] == "+" else -1
                    sname = term[1:]
                    if sname not in surfaces:
                        raise ScriptSyntaxError("undefined surface: %r" % sname)
                    hs_list.append(HalfSpace(surfaces[sname], sense))
            cell_names[lhs] = len(cells)
            cells.append(Cell(tuple(hs_list), lhs))
        elif lhs == "model":
            continue
        else:
            raise ScriptSyntaxError("unrecognised line: %r" % line)
    return ParsedScript(CSGModel(tuple(cells)), input_surface_names)


# --------------------------------------------------------------------------
# Dataset pipeline.
# --------------------------------------------------------------------------

def split_sequence(
    cells: Sequence[Cell], cut: int
) -> Tuple[Tuple[Cell, ...], Tuple[Cell, ...]]:
    """Split a cell sequence at ``cut`` (1 <= cut <= len-1) into (input, output).

    The paper splits "at every possible point between cells"; a valid split
    keeps at least one cell on each side (a completion model needs >= 2 cells).
    """
    n = len(cells)
    if not (1 <= cut <= n - 1):
        raise ValueError("cut must be between 1 and len-1")
    return tuple(cells[:cut]), tuple(cells[cut:])


def all_splits(cells: Sequence[Cell]) -> List[Tuple[Tuple[Cell, ...], Tuple[Cell, ...]]]:
    """Every valid input/output split of a sequence (cut augmentation)."""
    return [split_sequence(cells, k) for k in range(1, len(cells))]


def input_surfaces(
    input_cells: Sequence[Cell], output_cells: Sequence[Cell]
) -> List[object]:
    """Surfaces reused by the output that already appear in the input.

    These are the paper's positional-input surfaces: the surfaces "required to
    complete the input" so the model places new geometry in a specific location.
    Returned in input order of first appearance.
    """
    in_surfaces: List[object] = []
    seen = set()
    for cell in input_cells:
        for hs in cell.half_spaces:
            if hs.surface not in seen:
                seen.add(hs.surface)
                in_surfaces.append(hs.surface)
    out_set = set()
    for cell in output_cells:
        for hs in cell.half_spaces:
            out_set.add(hs.surface)
    return [s for s in in_surfaces if s in out_set]


@dataclass
class TrainingPair:
    """One (input script, target output script) example."""

    input_script: str
    output_script: str
    input_cells: Tuple[Cell, ...]
    output_cells: Tuple[Cell, ...]
    reused_surface_names: Tuple[str, ...]


def build_training_pair(
    ordered_cells: Sequence[Cell], cut: int
) -> TrainingPair:
    """Build one training pair for a given ordering and cut point."""
    in_cells, out_cells = split_sequence(ordered_cells, cut)
    reused = input_surfaces(in_cells, out_cells)
    # Name reused surfaces by their position in the *input* model's surface list.
    in_model = CSGModel(tuple(in_cells))
    names = _surface_names(collect_surfaces(in_model))
    reused_names = tuple(names[s] for s in reused)
    in_script = serialize(in_model, reused_names)
    out_script = serialize(CSGModel(tuple(out_cells)))
    return TrainingPair(in_script, out_script, tuple(in_cells), tuple(out_cells), reused_names)


def build_training_pairs(
    orderings: Sequence[Sequence[Cell]],
) -> List[TrainingPair]:
    """All (order x cut) training pairs -- the full augmentation.

    ``orderings`` is a list of cell sequences (e.g. from
    ``reconstruction.dontmesh_cell_graph.plausible_sequences`` mapped through the
    model's cells). Every ordering is split at every valid cut.
    """
    pairs: List[TrainingPair] = []
    for order in orderings:
        for cut in range(1, len(order)):
            pairs.append(build_training_pair(order, cut))
    return pairs


# --------------------------------------------------------------------------
# Plausibility metrics on scripts / models.
# --------------------------------------------------------------------------

def correct_syntax(script: str) -> bool:
    """True if the script parses (the paper's "Correct Syntax" metric)."""
    try:
        parse(script)
        return True
    except ScriptSyntaxError:
        return False


def _cell_structure(cell: Cell) -> Tuple[Tuple[str, int], ...]:
    return tuple(
        sorted((hs.surface.kind(), hs.sense) for hs in cell.half_spaces)
    )


def structural_signature(model: CSGModel) -> Tuple[Tuple[Tuple[str, int], ...], ...]:
    """Canonical structure ignoring parameters: per cell, the multiset of
    (surface-kind, sense) pairs; cells sorted. Two models with this signature
    equal are the paper's "Same Syntax, Different Params"."""
    return tuple(sorted(_cell_structure(c) for c in model.cells))


def _cell_full(cell: Cell, ndigits: int) -> Tuple[Tuple[str, int, Tuple[float, ...]], ...]:
    items = []
    for hs in cell.half_spaces:
        params = tuple(round(p, ndigits) for p in hs.surface.params())
        items.append((hs.surface.kind(), hs.sense, params))
    return tuple(sorted(items))


def parameter_signature(
    model: CSGModel, ndigits: int = 6
) -> Tuple[Tuple[Tuple[str, int, Tuple[float, ...]], ...], ...]:
    """Structure *and* rounded parameters ("Same Syntax, Same Params")."""
    return tuple(sorted(_cell_full(c, ndigits) for c in model.cells))


def same_structure(a: CSGModel, b: CSGModel) -> bool:
    return structural_signature(a) == structural_signature(b)


def same_structure_and_params(a: CSGModel, b: CSGModel, ndigits: int = 6) -> bool:
    return parameter_signature(a, ndigits) == parameter_signature(b, ndigits)


def same_cell_count(a: CSGModel, b: CSGModel) -> bool:
    return len(a.cells) == len(b.cells)


def cells_of(model: CSGModel, order: Sequence[int]) -> List[Cell]:
    """Reorder a model's cells by an index sequence (bridges the cell graph's
    integer orderings to concrete cell lists)."""
    return [model.cells[i] for i in order]
