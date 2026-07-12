"""DeepCAD's on-disk vector layout: the 17-column ``(1 + N_ARGS)`` command matrix.

Source: ``cadlib/macro.py`` plus the ``to_vector``/``from_vector`` methods of
``Loop``, ``Profile``, ``Extrude`` and ``CADSequence`` in the DeepCAD reference code
(Wu, Xiao & Zheng, ICCV 2021). This is the exact layout of the released
``cad_vec`` arrays in the DeepCAD ``.h5`` dataset.

Why this is not ``reconstruction.deepcad_command_spec``
-------------------------------------------------------
That module models the *paper's* presentation: six command types in the order
``(SOL, Line, Arc, Circle, Ext, EOS)``, a named 16-slot parameter dict, and one-hot
encodings. The released *code and data* use a different, load-bearing convention that
any consumer of the real ``.h5`` files must match:

* command indices ``Line=0, Arc=1, Circle=2, EOS=3, SOL=4, Ext=5`` -- **not** the
  paper's ordering; a mismatch silently permutes every command;
* one flat row ``[cmd, x, y, alpha, f, r, theta, phi, gamma, px, py, pz, s,
  e1, e2, b, u]`` of width ``1 + 16 = 17``, unused slots holding ``PAD_VAL = -1``;
* ``CMD_ARGS_MASK``, the 6x16 0/1 matrix saying which slots each command actually
  uses -- the mask that gates the official accuracy metric;
* the structural budgets ``MAX_N_EXT=10``, ``MAX_N_LOOPS=6``, ``MAX_N_CURVES=15``,
  ``MAX_TOTAL_LEN=60`` and the ``EOS``-padding rules (a loop is ``SOL`` + curves; a
  profile concatenates loops and ends with one ``EOS``; an extrude appends its ``Ext``
  row *before* that ``EOS``; a sequence concatenates extrudes and ends with one
  ``EOS``, padded to ``MAX_TOTAL_LEN``).

Everything here is plain lists of ints/floats. Deterministic, stdlib only.
"""

from __future__ import annotations

from typing import Sequence

# --- command vocabulary, in the REFERENCE CODE's order (macro.py) -----------
ALL_COMMANDS: tuple[str, ...] = ("Line", "Arc", "Circle", "EOS", "SOL", "Ext")
LINE_IDX = 0
ARC_IDX = 1
CIRCLE_IDX = 2
EOS_IDX = 3
SOL_IDX = 4
EXT_IDX = 5

EXTRUDE_OPERATIONS: tuple[str, ...] = (
    "NewBodyFeatureOperation", "JoinFeatureOperation",
    "CutFeatureOperation", "IntersectFeatureOperation",
)
EXTENT_TYPE: tuple[str, ...] = (
    "OneSideFeatureExtentType", "SymmetricFeatureExtentType",
    "TwoSidesFeatureExtentType",
)

PAD_VAL = -1

N_ARGS_SKETCH = 5       # x, y, alpha, f, r
N_ARGS_PLANE = 3        # theta, phi, gamma
N_ARGS_TRANS = 4        # px, py, pz, s
N_ARGS_EXT_PARAM = 4    # e1, e2, b, u
N_ARGS_EXT = N_ARGS_PLANE + N_ARGS_TRANS + N_ARGS_EXT_PARAM   # == 11
N_ARGS = N_ARGS_SKETCH + N_ARGS_EXT                           # == 16
ROW_LEN = 1 + N_ARGS                                          # == 17

ARG_NAMES: tuple[str, ...] = (
    "x", "y", "alpha", "f", "r",
    "theta", "phi", "gamma",
    "px", "py", "pz", "s",
    "e1", "e2", "b", "u",
)

SOL_VEC: tuple[int, ...] = (SOL_IDX,) + (PAD_VAL,) * N_ARGS
EOS_VEC: tuple[int, ...] = (EOS_IDX,) + (PAD_VAL,) * N_ARGS

#: 6 x 16 mask: which argument slots each command type actually uses.
CMD_ARGS_MASK: tuple[tuple[int, ...], ...] = (
    (1, 1, 0, 0, 0) + (0,) * N_ARGS_EXT,   # Line   -> x, y
    (1, 1, 1, 1, 0) + (0,) * N_ARGS_EXT,   # Arc    -> x, y, alpha, f
    (1, 1, 0, 0, 1) + (0,) * N_ARGS_EXT,   # Circle -> x, y, r
    (0, 0, 0, 0, 0) + (0,) * N_ARGS_EXT,   # EOS
    (0, 0, 0, 0, 0) + (0,) * N_ARGS_EXT,   # SOL
    (0,) * N_ARGS_SKETCH + (1,) * N_ARGS_EXT,  # Ext -> all 11 extrusion args
)

NORM_FACTOR = 0.75
MAX_N_EXT = 10
MAX_N_LOOPS = 6
MAX_N_CURVES = 15
MAX_TOTAL_LEN = 60
ARGS_DIM = 256

Row = tuple[int, ...]


# --- row constructors -------------------------------------------------------
def _row(cmd: int, **args) -> Row:
    values = [PAD_VAL] * N_ARGS
    for name, value in args.items():
        values[ARG_NAMES.index(name)] = value
    return (cmd, *values)


def line_row(x, y) -> Row:
    """``Line`` row: only the end point is stored (start = previous end)."""
    return _row(LINE_IDX, x=x, y=y)


def arc_row(x, y, alpha, f) -> Row:
    """``Arc`` row: end point, sweep angle, counter-clockwise flag."""
    return _row(ARC_IDX, x=x, y=y, alpha=alpha, f=f)


def circle_row(x, y, r) -> Row:
    """``Circle`` row: centre and radius (a circle is a whole loop by itself)."""
    return _row(CIRCLE_IDX, x=x, y=y, r=r)


def ext_row(theta, phi, gamma, px, py, pz, s, e1, e2, b, u) -> Row:
    """``Ext`` row: the 11 extrusion arguments; all sketch slots are ``-1``."""
    return _row(EXT_IDX, theta=theta, phi=phi, gamma=gamma, px=px, py=py, pz=pz,
                s=s, e1=e1, e2=e2, b=b, u=u)


def sol_row() -> Row:
    return SOL_VEC


def eos_row() -> Row:
    return EOS_VEC


# --- masks ------------------------------------------------------------------
def arg_mask(command: int) -> tuple[int, ...]:
    """The 16-wide 0/1 mask of the argument slots ``command`` uses."""
    if not 0 <= command < len(ALL_COMMANDS):
        raise ValueError(f"unknown command index: {command}")
    return CMD_ARGS_MASK[command]


def used_args(row: Sequence[int]) -> dict:
    """The named arguments a row actually carries (mask applied)."""
    mask = arg_mask(row[0])
    return {name: row[1 + i] for i, name in enumerate(ARG_NAMES) if mask[i]}


def validate_row(row: Sequence[int]) -> None:
    """Raise unless ``row`` has width 17 and holds ``-1`` in every unused slot."""
    if len(row) != ROW_LEN:
        raise ValueError(f"row must have {ROW_LEN} entries, got {len(row)}")
    mask = arg_mask(row[0])
    for i, flag in enumerate(mask):
        if not flag and row[1 + i] != PAD_VAL:
            raise ValueError(
                f"{ALL_COMMANDS[row[0]]} row carries a value in unused slot "
                f"{ARG_NAMES[i]!r}")


# --- assembly ---------------------------------------------------------------
def loop_vector(curves: Sequence[Row], add_sol: bool = True,
                add_eos: bool = True, max_len: int | None = None) -> list[Row] | None:
    """``[SOL] + curves [+ EOS]``, EOS-padded to ``max_len``; ``None`` if too long."""
    rows: list[Row] = []
    if add_sol:
        rows.append(SOL_VEC)
    rows.extend(tuple(c) for c in curves)
    if add_eos:
        rows.append(EOS_VEC)
    if max_len is None:
        return rows
    if len(rows) > max_len:
        return None
    return rows + [EOS_VEC] * (max_len - len(rows))


def profile_vector(loops: Sequence[Sequence[Row]], max_n_loops: int = MAX_N_LOOPS,
                   max_len_loop: int = MAX_N_CURVES,
                   pad: bool = True) -> list[Row] | None:
    """Concatenate ``SOL``-prefixed loops, terminate with one ``EOS``.

    ``None`` when the profile exceeds ``max_n_loops`` loops or any loop exceeds
    ``max_len_loop`` rows (``SOL`` included) -- the reference's rejection rule for
    over-budget models. With ``pad`` the result is EOS-padded to
    ``max_n_loops * max_len_loop`` rows.
    """
    if len(loops) > max_n_loops:
        return None
    rows: list[Row] = []
    for loop in loops:
        loop_rows = loop_vector(loop, add_sol=True, add_eos=False)
        if len(loop_rows) > max_len_loop:
            return None
        rows.extend(loop_rows)
    rows.append(EOS_VEC)
    if pad:
        rows.extend([EOS_VEC] * (max_n_loops * max_len_loop - len(rows)))
    return rows


def extrude_vector(loops: Sequence[Sequence[Row]], extrusion: Row,
                   max_n_loops: int = MAX_N_LOOPS,
                   max_len_loop: int = MAX_N_CURVES,
                   pad: bool = True) -> list[Row] | None:
    """``[SOL ... loops ..., Ext, EOS]`` -- the Ext row goes *before* the EOS."""
    profile = profile_vector(loops, max_n_loops, max_len_loop, pad=False)
    if profile is None:
        return None
    rows = profile[:-1] + [tuple(extrusion)] + [EOS_VEC]
    if pad:
        rows.extend([EOS_VEC] * (max_n_loops * max_len_loop - len(rows)))
    return rows


def cad_vector(extrudes: Sequence[Sequence[Sequence[Row]]],
               max_n_ext: int = MAX_N_EXT, max_total_len: int = MAX_TOTAL_LEN,
               pad: bool = True) -> list[Row] | None:
    """Full sequence: each extrude's rows minus its trailing EOS, then one EOS.

    ``extrudes`` is a sequence of ``(loops, ext_row)`` pairs. ``None`` if the model
    has more than ``max_n_ext`` extrusions or any extrude is over budget.
    """
    if len(extrudes) > max_n_ext:
        return None
    rows: list[Row] = []
    for loops, extrusion in extrudes:
        vec = extrude_vector(loops, extrusion, pad=False)
        if vec is None:
            return None
        rows.extend(vec[:-1])      # drop the per-extrude EOS
    rows.append(EOS_VEC)
    if pad and len(rows) < max_total_len:
        rows.extend([EOS_VEC] * (max_total_len - len(rows)))
    return rows


# --- disassembly ------------------------------------------------------------
def split_extrudes(vec: Sequence[Sequence[int]]) -> list[list[Row]]:
    """Split a sequence into per-extrude slices, each ending at its ``Ext`` row."""
    out: list[list[Row]] = []
    start = 0
    for i, row in enumerate(vec):
        if row[0] == EXT_IDX:
            out.append([tuple(r) for r in vec[start:i + 1]])
            start = i + 1
    return out


def split_loops(vec: Sequence[Sequence[int]]) -> list[list[Row]]:
    """Split a profile / extrude slice into ``SOL``-started loops.

    Stops at the first ``EOS`` or ``Ext`` row. Loops with no curve rows are dropped
    (the reference's ``SOL`` followed immediately by ``SOL``/``EOS`` guard).
    """
    loops: list[list[Row]] = []
    current: list[Row] | None = None
    for row in vec:
        cmd = row[0]
        if cmd in (EOS_IDX, EXT_IDX):
            break
        if cmd == SOL_IDX:
            if current is not None and len(current) > 1:
                loops.append(current)
            current = [tuple(row)]
        elif current is not None:
            current.append(tuple(row))
    if current is not None and len(current) > 1:
        loops.append(current)
    return loops


def trim_eos(vec: Sequence[Sequence[int]]) -> list[Row]:
    """Everything strictly before the first ``EOS`` row (the padding stripper)."""
    out: list[Row] = []
    for row in vec:
        if row[0] == EOS_IDX:
            break
        out.append(tuple(row))
    return out


def sequence_length(vec: Sequence[Sequence[int]]) -> int:
    """Number of rows before the first ``EOS`` (the model's true command count)."""
    return len(trim_eos(vec))
