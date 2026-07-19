"""B-Rep construction command / token schema.

This schema models a B-Rep construction workflow as a fixed-width command sequence
``S = (C1, C2, ..., C_Nc)`` where each command ``Ci = (ti, pi)`` carries a command
*type* ``ti`` and a fixed-length parameter vector ``pi``. This module is the
deterministic, network-agnostic part of that representation: the token vocabulary,
the fixed ``PARAM_LEN``-slot parameter layout, the "stack all parameters into one
vector, unused slots = -1" packing rule, fixed-length ``NC`` padding, and the
one-hot index construction feeding the (learned) embedding matrices.

The learned embeddings, encoder and decoder are out of scope. Everything here is
pure and deterministic so a workflow round-trips exactly.

Command types and their parameters::

    <SOS>                                   (no params)
    L (Line)         x, y                    line endpoint
    A (Arc)          x, y, alpha, f          arc endpoint, sweep angle, cw flag
    C (Circle)       x, y, r                 centre, radius
    E (Extrusion)    tx,ty,tz, theta,gamma,delta, s, e1,e2
    Ec (ExtrusionCut)  = E
    Ax (RevolutionAxis) tx,ty,tz, theta,gamma,delta
    R (Revolution)   tx,ty,tz, theta,gamma,delta, s, rev_angle
    Rc (RevolutionCut) = R
    F (Fillet)       px, py, pz              3D point on the filleted edge
    Cf (Chamfer)     = F
    <PAD>                                    (no params)
    <EOS>                                    (no params)

The parameter vector is an R^19 vector paired with a 12-way command one-hot; the
union of named slots below honours both invariants (a fixed 19-length vector with
unused = -1, and a stable command-type index) exactly.
"""

from __future__ import annotations

from dataclasses import dataclass


# --- command vocabulary -----------------------------------------------------
SOS = "<SOS>"
EOS = "<EOS>"
PAD = "<PAD>"

# Ordered command vocabulary. Index in this tuple is the one-hot position used to
# build the command-type indicator vector.
COMMAND_TYPES: tuple[str, ...] = (
    SOS, "L", "A", "C", "E", "Ec", "Ax", "R", "Rc", "F", "Cf", PAD, EOS,
)
COMMAND_INDEX: dict[str, int] = {name: i for i, name in enumerate(COMMAND_TYPES)}
N_COMMAND_TYPES = len(COMMAND_TYPES)

# Commands that carry no parameters (all slots are -1).
_NO_PARAM = frozenset({SOS, EOS, PAD})

# Commands whose parameter footprint is defined to equal another command's
# ("Ec = E", "Rc = R", "Cf = F"); they differ only in command type.
COMMAND_ALIASES: dict[str, str] = {"Ec": "E", "Rc": "R", "Cf": "F"}


# --- fixed 19-slot parameter layout ----------------------------------------
# Every command's parameters are stacked into a single fixed-length vector; a
# slot a command does not use holds the sentinel -1.
PARAM_SLOTS: tuple[str, ...] = (
    "x", "y", "alpha", "f", "r",              # sketch-curve params (0..4)
    "tx", "ty", "tz",                         # sketch-plane translation (5..7)
    "theta", "gamma", "delta",                # sketch-plane rotation (8..10)
    "s",                                      # sketch scale (11)
    "e1", "e2",                               # extrude distances (12..13)
    "rev_angle",                              # revolution angle (14)
    "px", "py", "pz",                         # fillet/chamfer edge point (15..17)
    "reserved",                               # reserved -> R^19 (18)
)
PARAM_INDEX: dict[str, int] = {name: i for i, name in enumerate(PARAM_SLOTS)}
PARAM_LEN = len(PARAM_SLOTS)  # == 19

UNUSED = -1.0

# Which named slots each command populates.
_COMMAND_PARAMS: dict[str, tuple[str, ...]] = {
    "L": ("x", "y"),
    "A": ("x", "y", "alpha", "f"),
    "C": ("x", "y", "r"),
    "E": ("tx", "ty", "tz", "theta", "gamma", "delta", "s", "e1", "e2"),
    "Ax": ("tx", "ty", "tz", "theta", "gamma", "delta"),
    "R": ("tx", "ty", "tz", "theta", "gamma", "delta", "s", "rev_angle"),
    "F": ("px", "py", "pz"),
}


def param_names(command_type: str) -> tuple[str, ...]:
    """Ordered names of the parameter slots a command type populates."""
    base = COMMAND_ALIASES.get(command_type, command_type)
    if base in _NO_PARAM:
        return ()
    if base not in _COMMAND_PARAMS:
        raise ValueError(f"unknown command type: {command_type!r}")
    return _COMMAND_PARAMS[base]


@dataclass(frozen=True)
class Command:
    """A single construction command ``Ci = (ti, pi)``.

    ``params`` maps a subset of :data:`PARAM_SLOTS` names to values; unlisted
    slots are the sentinel -1 when packed into the fixed vector.
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
    """Convenience builder that orders params by the command's slot order."""
    ordered = tuple((n, float(params[n])) for n in param_names(type) if n in params)
    return Command(type, ordered)


def to_vector(cmd: Command) -> tuple[float, ...]:
    """Pack a command's parameters into the fixed ``PARAM_LEN`` vector.

    Unused slots hold :data:`UNUSED` (-1). This is the "stack the whole of the
    parameters from all the CAD commands into one vector" rule.
    """
    vector = [UNUSED] * PARAM_LEN
    for name in param_names(cmd.type):
        vector[PARAM_INDEX[name]] = float(cmd.get(name, UNUSED))
    return tuple(vector)


def from_vector(command_type: str, vector: tuple[float, ...]) -> Command:
    """Inverse of :func:`to_vector`: read a command's named params back out."""
    if len(vector) != PARAM_LEN:
        raise ValueError(f"expected {PARAM_LEN}-length vector, got {len(vector)}")
    params = tuple(
        (name, float(vector[PARAM_INDEX[name]])) for name in param_names(command_type))
    return Command(command_type, params)


# --- fixed-length sequence packing -----------------------------------------
NC_DEFAULT = 32  # NC = 32, from dataset statistics


def pad_sequence(commands: list[Command], nc: int = NC_DEFAULT,
                 add_terminators: bool = True) -> tuple[Command, ...]:
    """Build the fixed-length ``NC`` command sequence for a workflow.

    With ``add_terminators`` the sequence is ``<SOS> c... <EOS> <PAD>...`` padded
    to length ``nc``. Raises if the content does not fit.
    """
    body = list(commands)
    if add_terminators:
        seq = [Command(SOS)] + body + [Command(EOS)]
    else:
        seq = body
    if len(seq) > nc:
        raise ValueError(f"sequence length {len(seq)} exceeds NC={nc}")
    seq += [Command(PAD)] * (nc - len(seq))
    return tuple(seq)


def sequence_matrix(commands: list[Command], nc: int = NC_DEFAULT,
                    add_terminators: bool = True):
    """Return ``(types, param_matrix)`` for a padded workflow.

    ``types`` is the per-step command-type index; ``param_matrix`` is an
    ``nc x PARAM_LEN`` tuple-of-tuples of packed parameters.
    """
    seq = pad_sequence(commands, nc, add_terminators)
    types = tuple(COMMAND_INDEX[c.type] for c in seq)
    matrix = tuple(to_vector(c) for c in seq)
    return types, matrix


# --- one-hot / index construction (feeds the learned embeddings) -----------
def command_onehot(command_type: str) -> tuple[int, ...]:
    """12/13-way command-type one-hot (the delta_ic indicator vector)."""
    vector = [0] * N_COMMAND_TYPES
    vector[COMMAND_INDEX[command_type]] = 1
    return tuple(vector)


# Continuous params are quantised to 256 levels; a 257th index encodes the
# sentinel -1 (embedding dimension 2^8 + 1 = 257).
N_QUANT_LEVELS = 256
PARAM_ONEHOT_DIM = N_QUANT_LEVELS + 1  # == 257
_UNUSED_INDEX = 0  # index reserved for the -1 sentinel


def param_index(value: float, n_levels: int = N_QUANT_LEVELS,
                low: float = -1.0, high: float = 1.0) -> int:
    """Map a (possibly -1 sentinel) param value to its 0..256 one-hot index.

    The sentinel -1 maps to index 0; a quantised level ``q in 0..n_levels-1`` maps
    to ``q + 1``. Continuous values are clamped to ``[low, high]`` and quantised
    into ``n_levels`` uniform bins (a 2x2x2-cube normalisation, 256 levels).
    """
    if value == UNUSED:
        return _UNUSED_INDEX
    clamped = min(high, max(low, value))
    q = round((clamped - low) / (high - low) * (n_levels - 1))
    return int(q) + 1


def param_onehot(value: float, n_levels: int = N_QUANT_LEVELS) -> tuple[int, ...]:
    """One-hot of :func:`param_index` with width :data:`PARAM_ONEHOT_DIM` (257)."""
    vector = [0] * (n_levels + 1)
    vector[param_index(value, n_levels)] = 1
    return tuple(vector)


def quantize(value: float, n_levels: int = N_QUANT_LEVELS,
             low: float = -1.0, high: float = 1.0) -> int:
    """Quantise a continuous value into ``0..n_levels-1`` (no sentinel handling)."""
    clamped = min(high, max(low, value))
    return int(round((clamped - low) / (high - low) * (n_levels - 1)))


def dequantize(level: int, n_levels: int = N_QUANT_LEVELS,
               low: float = -1.0, high: float = 1.0) -> float:
    """Inverse of :func:`quantize`: bin index back to a representative value."""
    return low + level * (high - low) / (n_levels - 1)
