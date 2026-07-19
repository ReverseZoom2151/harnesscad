"""Exact canonical CAD command-sequence specification.

This is the canonical CAD-sequence representation. It describes a CAD model
``M = [C1, ..., C_Nc]`` as a sequence of commands, each ``Ci = (ti, pi)`` with a
command *type* ``ti`` and a **fixed 16-dimensional** parameter vector ``pi``. This
module is the deterministic, network-agnostic part of that representation, faithful
to its exact conventions:

Command types (exactly six)::

    SOL   start of a loop            (no parameters)
    Line  x, y                        line end-point
    Arc   x, y, alpha, f             arc end-point, sweep angle, ccw flag
    Circle x, y, r                    centre, radius
    Ext   theta, phi, gamma,          sketch-plane orientation
          px, py, pz,                 sketch-plane origin
          s,                          scale of associated sketch profile
          e1, e2,                     extrude distances toward both sides
          b, u                        boolean type, extrude type
    EOS   end of the whole sequence  (no parameters)

The exact 16-slot parameter vector ordering::

    pi = [x, y, alpha, f, r, theta, phi, gamma, px, py, pz, s, e1, e2, b, u]

Every command stacks *its* parameters into this common 16-vector; slots a command
does not use hold the sentinel ``-1``. The full sequence is padded with the empty
command ``EOS`` to the fixed length ``Nc = 60`` (the maximal command-sequence
length in the reference dataset).

Continuous parameters are normalised into a 2x2x2 cube and quantised to **256
levels** (8-bit); the one-hot width is ``2**8 + 1 = 257`` (the extra index encodes
the -1 sentinel), matching the specification exactly.

This differs from :mod:`reconstruction.cadparser_schema` (a *19*-slot, 12-type
B-Rep-workflow variant) and from ``bench/contrastcad_recon_accuracy``'s
differently-named 16-slot vector: this is the canonical spec with the exact
six types and the exact slot ordering above. Pure and deterministic; the learned
encoder/decoder are out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- command vocabulary (exactly six types) -----------------------
SOL = "SOL"      # start of a loop
LINE = "Line"
ARC = "Arc"
CIRCLE = "Circle"
EXT = "Ext"      # extrusion
EOS = "EOS"      # end of sequence / empty padding command

# Ordered vocabulary; index is the one-hot position of the command-type indicator.
COMMAND_TYPES: tuple[str, ...] = (SOL, LINE, ARC, CIRCLE, EXT, EOS)
COMMAND_INDEX: dict[str, int] = {name: i for i, name in enumerate(COMMAND_TYPES)}
N_COMMAND_TYPES = len(COMMAND_TYPES)  # == 6

# Commands carrying no parameters (whole 16-vector is the sentinel).
_NO_PARAM = frozenset({SOL, EOS})


# --- exact 16-slot parameter layout ----------------------------------------
PARAM_SLOTS: tuple[str, ...] = (
    "x", "y", "alpha", "f", "r",              # curve params            (0..4)
    "theta", "phi", "gamma",                  # sketch-plane orientation (5..7)
    "px", "py", "pz",                         # sketch-plane origin      (8..10)
    "s",                                      # profile scale            (11)
    "e1", "e2",                               # extrude distances        (12..13)
    "b", "u",                                 # boolean type, extrude type (14..15)
)
PARAM_INDEX: dict[str, int] = {name: i for i, name in enumerate(PARAM_SLOTS)}
PARAM_LEN = len(PARAM_SLOTS)  # == 16

UNUSED = -1.0

# Which named slots each command type populates.
_COMMAND_PARAMS: dict[str, tuple[str, ...]] = {
    LINE: ("x", "y"),
    ARC: ("x", "y", "alpha", "f"),
    CIRCLE: ("x", "y", "r"),
    EXT: ("theta", "phi", "gamma", "px", "py", "pz", "s", "e1", "e2", "b", "u"),
}

# Fixed sequence length (Nc = 60, the maximal length in the dataset).
NC_DEFAULT = 60


def param_names(command_type: str) -> tuple[str, ...]:
    """Ordered names of the parameter slots a command type populates."""
    if command_type in _NO_PARAM:
        return ()
    if command_type not in _COMMAND_PARAMS:
        raise ValueError(f"unknown command type: {command_type!r}")
    return _COMMAND_PARAMS[command_type]


@dataclass(frozen=True)
class Command:
    """A single CAD command ``Ci = (ti, pi)``.

    ``params`` maps a subset of :data:`PARAM_SLOTS` names to values; unlisted slots
    become the sentinel ``-1`` when packed into the fixed 16-vector.
    """

    type: str
    params: tuple[tuple[str, float], ...] = ()

    def __post_init__(self):
        if self.type not in COMMAND_INDEX:
            raise ValueError(f"unknown command type: {self.type!r}")
        allowed = set(param_names(self.type))
        for name, _ in self.params:
            if name not in allowed:
                raise ValueError(
                    f"command {self.type!r} does not accept parameter {name!r}")

    def get(self, name: str, default: float = UNUSED) -> float:
        for key, value in self.params:
            if key == name:
                return value
        return default


def command(type: str, **params: float) -> Command:
    """Builder ordering params by the command's slot order."""
    ordered = tuple((n, float(params[n])) for n in param_names(type) if n in params)
    return Command(type, ordered)


def to_vector(cmd: Command) -> tuple[float, ...]:
    """Pack a command's parameters into the fixed 16-vector (unused slots = -1)."""
    vector = [UNUSED] * PARAM_LEN
    for name in param_names(cmd.type):
        vector[PARAM_INDEX[name]] = float(cmd.get(name, UNUSED))
    return tuple(vector)


def from_vector(command_type: str, vector: tuple[float, ...]) -> Command:
    """Inverse of :func:`to_vector`: read named params back out of the 16-vector."""
    if len(vector) != PARAM_LEN:
        raise ValueError(f"expected {PARAM_LEN}-length vector, got {len(vector)}")
    params = tuple(
        (name, float(vector[PARAM_INDEX[name]])) for name in param_names(command_type))
    return Command(command_type, params)


# --- fixed-length sequence packing -----------------------------------------
def pad_sequence(commands: list[Command], nc: int = NC_DEFAULT) -> tuple[Command, ...]:
    """Pad a command list with the empty ``EOS`` command to fixed length ``Nc``.

    The sequence is padded with the empty command ``EOS`` until the length
    reaches ``Nc``. Raises when the content itself exceeds ``nc``.
    """
    seq = list(commands)
    if len(seq) > nc:
        raise ValueError(f"sequence length {len(seq)} exceeds Nc={nc}")
    seq += [Command(EOS)] * (nc - len(seq))
    return tuple(seq)


def vector_representation(commands: list[Command], nc: int = NC_DEFAULT):
    """Vector <-> CAD-operation conversion: return ``(types, param_matrix)``.

    ``types`` is the per-step command-type index; ``param_matrix`` is an
    ``nc x 16`` tuple-of-tuples of packed parameters. This is the
    network-friendly regularised representation of a whole model.
    """
    seq = pad_sequence(commands, nc)
    types = tuple(COMMAND_INDEX[c.type] for c in seq)
    matrix = tuple(to_vector(c) for c in seq)
    return types, matrix


def commands_from_vectors(types, param_matrix) -> list[Command]:
    """Inverse of :func:`vector_representation` (drops trailing ``EOS`` padding)."""
    if len(types) != len(param_matrix):
        raise ValueError("types and param_matrix must have equal length")
    out: list[Command] = []
    for t, vec in zip(types, param_matrix):
        name = COMMAND_TYPES[t]
        out.append(from_vector(name, tuple(vec)))
    # Strip the trailing run of EOS padding but keep any interior EOS.
    while out and out[-1].type == EOS:
        out.pop()
    return out


# --- one-hot / quantisation (feeds the learned embeddings) -----------------
def command_onehot(command_type: str) -> tuple[int, ...]:
    """Six-way command-type one-hot (the indicator vector ``c_i``)."""
    vector = [0] * N_COMMAND_TYPES
    vector[COMMAND_INDEX[command_type]] = 1
    return tuple(vector)


N_QUANT_LEVELS = 256
PARAM_ONEHOT_DIM = N_QUANT_LEVELS + 1  # == 257 == 2**8 + 1
_UNUSED_INDEX = 0                      # index reserved for the -1 sentinel


def quantize(value: float, n_levels: int = N_QUANT_LEVELS,
             low: float = -1.0, high: float = 1.0) -> int:
    """Quantise a continuous value in ``[low, high]`` to ``0..n_levels-1``."""
    clamped = min(high, max(low, value))
    return int(round((clamped - low) / (high - low) * (n_levels - 1)))


def dequantize(level: int, n_levels: int = N_QUANT_LEVELS,
               low: float = -1.0, high: float = 1.0) -> float:
    """Inverse of :func:`quantize`: bin index to a representative value."""
    return low + level * (high - low) / (n_levels - 1)


def param_index(value: float, n_levels: int = N_QUANT_LEVELS,
                low: float = -1.0, high: float = 1.0) -> int:
    """Map a param value (or -1 sentinel) to its 0..256 embedding index.

    Sentinel ``-1`` maps to index 0; a quantised level ``q`` maps to ``q + 1``
    (one-hot dimension ``2**8 + 1 = 257``).
    """
    if value == UNUSED:
        return _UNUSED_INDEX
    return quantize(value, n_levels, low, high) + 1


def param_onehot(value: float, n_levels: int = N_QUANT_LEVELS) -> tuple[int, ...]:
    """One-hot of :func:`param_index` with width :data:`PARAM_ONEHOT_DIM` (257)."""
    vector = [0] * (n_levels + 1)
    vector[param_index(value, n_levels)] = 1
    return tuple(vector)
