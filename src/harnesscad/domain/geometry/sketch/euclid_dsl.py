"""Construction-step DSL: token schema and quantisation.

Implements the tokenisation scheme of Li et al., "Draw It Like Euclid"
(Appendix A). Scalars, points and directed infinite lines are quantised onto
odd-sized grids so that the important values (-1, 0, 1 for lengths; the domain
centre for points; 0, pi, pi/2, pi/3 for angles) are recovered exactly when
de-quantised. The module also declares the construction-step schema (operation
type, input arity, output arity) and provides a deterministic tokenizer /
detokenizer for construction sequences.

This is the deterministic representation only -- the transformer that would be
trained on these token streams is out of scope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

TWO_PI = 2.0 * math.pi

# ---------------------------------------------------------------------------
# Scalar / entity quantisers (Table 4)
# ---------------------------------------------------------------------------
LENGTH_BINS = 127          # signed lengths in [-1, 1]; odd -> -1,0,1 exact
POINT_GRID = 127           # 127 x 127 grid; centre exact
ANGLE_STEPS = 120          # 121 values with wraparound; 120 = 2*2*2*3*5
POINT_LO, POINT_HI = -0.5, 0.5


def quantize_length(value: float, bins: int = LENGTH_BINS) -> int:
    """Quantise a signed length in ``[-1, 1]`` to a bin index ``0..bins-1``."""
    v = max(-1.0, min(1.0, value))
    return int(round((v + 1.0) / 2.0 * (bins - 1)))


def dequantize_length(index: int, bins: int = LENGTH_BINS) -> float:
    index = max(0, min(bins - 1, int(index)))
    return index / (bins - 1) * 2.0 - 1.0


def quantize_point_coord(value: float, grid: int = POINT_GRID) -> int:
    """Quantise a coordinate in ``[POINT_LO, POINT_HI]`` to a grid index."""
    v = max(POINT_LO, min(POINT_HI, value))
    frac = (v - POINT_LO) / (POINT_HI - POINT_LO)
    return int(round(frac * (grid - 1)))


def dequantize_point_coord(index: int, grid: int = POINT_GRID) -> float:
    index = max(0, min(grid - 1, int(index)))
    return POINT_LO + index / (grid - 1) * (POINT_HI - POINT_LO)


def quantize_angle(value: float, steps: int = ANGLE_STEPS) -> int:
    """Quantise an angle in ``[0, 2*pi)`` to an index ``0..steps`` (wrapping)."""
    a = math.fmod(value, TWO_PI)
    if a < 0.0:
        a += TWO_PI
    idx = int(round(a / TWO_PI * steps))
    if idx >= steps:
        idx = 0  # 2*pi wraps back to bin 0
    return idx


def dequantize_angle(index: int, steps: int = ANGLE_STEPS) -> float:
    index = int(index) % steps
    return index / steps * TWO_PI


def quantize_point(x: float, y: float, grid: int = POINT_GRID) -> Tuple[int, int]:
    return (quantize_point_coord(x, grid), quantize_point_coord(y, grid))


def dequantize_point(ix: int, iy: int, grid: int = POINT_GRID) -> Tuple[float, float]:
    return (dequantize_point_coord(ix, grid), dequantize_point_coord(iy, grid))


def quantize_infline(phi: float, rho: float,
                     steps: int = ANGLE_STEPS,
                     bins: int = LENGTH_BINS) -> Tuple[int, int]:
    """Quantise a directed infinite line (Hessian form) to ``(angle, dist)``."""
    return (quantize_angle(phi, steps), quantize_length(rho, bins))


def dequantize_infline(iphi: int, irho: int,
                       steps: int = ANGLE_STEPS,
                       bins: int = LENGTH_BINS) -> Tuple[float, float]:
    return (dequantize_angle(iphi, steps), dequantize_length(irho, bins))


# ---------------------------------------------------------------------------
# Construction-step schema (Tables 1 and 7)
# ---------------------------------------------------------------------------
# Each spec: operation name -> (input entity types, output entity types,
# number of scalar params). Entity type strings: "point", "line", "circle",
# "arc". Scalar params are Length or Angle values (declared via UseParameterN).
@dataclass(frozen=True)
class StepSpec:
    name: str
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    n_params: int = 0
    param_kinds: Tuple[str, ...] = ()  # "length" or "angle" per param


STEP_SPECS: Dict[str, StepSpec] = {
    s.name: s
    for s in [
        StepSpec("CircleOffsetCircle", ("circle",), ("circle",), 1, ("length",)),
        StepSpec("LineXLine", ("line", "line"), ("point",)),
        StepSpec("LineOffsetLine", ("line",), ("line",), 1, ("length",)),
        StepSpec("LineXCircle", ("line", "circle"), ("point",)),
        StepSpec("CircleReverseCircle", ("circle",), ("circle",)),
        StepSpec("CirclePointPointArc", ("circle", "point", "point"), ("arc",)),
        StepSpec("LineDatumParallelLine", ("line", "point"), ("line",)),
        StepSpec("LineLineFillet", ("line", "line"), ("arc",), 1, ("length",)),
        StepSpec("LineCircleParallelLine", ("line", "circle"), ("line",)),
        StepSpec("LineSymLineLine", ("line", "line"), ("line",)),
        StepSpec("PointLineSymPoint", ("point", "line"), ("point",)),
        StepSpec("LineReverseLine", ("line",), ("line",)),
        StepSpec("LineAxisRotatedLine", ("line", "point"), ("line",), 1, ("angle",)),
        StepSpec("PointRadiusCircle", ("point",), ("circle",), 1, ("length",)),
        StepSpec("SymlineOffsetLineLine", ("line",), ("line", "line"), 1, ("length",)),
    ]
}

MAX_PARAMETERS = 32  # UseParameter0 .. UseParameter31 (Appendix A.4)


# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------
START_OF_CONSTRUCTION = "StartOfConstruction"
END_OF_CONSTRUCTION = "EndOfConstruction"
CREATED_CURVE = "CreatedCurve"


@dataclass(frozen=True)
class Step:
    """One construction step: an op, input geometry ids, output ids, params.

    ``inputs`` and ``outputs`` are symbolic string ids referencing entries in a
    geometry environment. ``param_indices`` gives the UseParameterN indices for
    the scalar arguments (in order); their values are held in a separate
    parameter table so the sequence can be replayed with edited parameters.
    ``creates_curve`` marks a step whose (last) output is emitted into the final
    profile via a CreatedCurve token.
    """

    op: str
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    param_indices: Tuple[int, ...] = ()
    creates_curve: bool = False

    def spec(self) -> StepSpec:
        return STEP_SPECS[self.op]


@dataclass
class ConstructionSequence:
    """A construction sequence with its parameter table.

    ``parameters`` maps a parameter index to its scalar value. Steps reference
    parameters by index so a single value can drive several steps (e.g. a shared
    fillet radius).
    """

    steps: List[Step] = field(default_factory=list)
    parameters: Dict[int, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tokenizer / detokenizer (structured token stream)
# ---------------------------------------------------------------------------
def tokenize(seq: ConstructionSequence) -> List[Tuple]:
    """Serialise a construction sequence into a flat list of structured tokens.

    Tokens are tuples whose first element is a kind string. The stream is:
    StartOfConstruction, then for each step: optional UseParameterN + value
    tokens, the op token with input/output id lists, an optional CreatedCurve
    marker, and finally EndOfConstruction.
    """
    out: List[Tuple] = [(START_OF_CONSTRUCTION,)]
    for step in seq.steps:
        spec = step.spec()
        for k, pidx in enumerate(step.param_indices):
            out.append(("UseParameter", pidx))
            value = seq.parameters.get(pidx, 0.0)
            kind = spec.param_kinds[k] if k < len(spec.param_kinds) else "length"
            if kind == "angle":
                out.append(("Angle", quantize_angle(value)))
            else:
                out.append(("Length", quantize_length(value)))
        out.append(("Op", step.op, tuple(step.inputs), tuple(step.outputs)))
        if step.creates_curve:
            out.append((CREATED_CURVE,))
    out.append((END_OF_CONSTRUCTION,))
    return out


def detokenize(tokens: Sequence[Tuple]) -> ConstructionSequence:
    """Inverse of :func:`tokenize`. Raises ValueError on a malformed stream."""
    if not tokens or tokens[0][0] != START_OF_CONSTRUCTION:
        raise ValueError("stream must start with StartOfConstruction")
    if tokens[-1][0] != END_OF_CONSTRUCTION:
        raise ValueError("stream must end with EndOfConstruction")
    seq = ConstructionSequence()
    pending_params: List[int] = []
    i = 1
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        kind = tok[0]
        if kind == END_OF_CONSTRUCTION:
            break
        if kind == "UseParameter":
            pidx = tok[1]
            if not (0 <= pidx < MAX_PARAMETERS):
                raise ValueError("parameter index out of range: %r" % (pidx,))
            pending_params.append(pidx)
            i += 1
            if i >= n or tokens[i][0] not in ("Length", "Angle"):
                raise ValueError("UseParameter must be followed by a value token")
            vtok = tokens[i]
            if vtok[0] == "Angle":
                seq.parameters[pidx] = dequantize_angle(vtok[1])
            else:
                seq.parameters[pidx] = dequantize_length(vtok[1])
            i += 1
            continue
        if kind == "Op":
            op, inputs, outputs = tok[1], tok[2], tok[3]
            if op not in STEP_SPECS:
                raise ValueError("unknown op: %r" % (op,))
            spec = STEP_SPECS[op]
            if len(pending_params) != spec.n_params:
                raise ValueError(
                    "op %s expects %d params, got %d"
                    % (op, spec.n_params, len(pending_params)))
            creates = i + 1 < n and tokens[i + 1][0] == CREATED_CURVE
            seq.steps.append(Step(op, tuple(inputs), tuple(outputs),
                                  tuple(pending_params), creates))
            pending_params = []
            i += 2 if creates else 1
            continue
        raise ValueError("unexpected token kind: %r" % (kind,))
    return seq


def build_vocabulary() -> Dict[str, int]:
    """A deterministic flat integer vocabulary over all token kinds/values.

    Useful as a stand-in embedding table: special tokens, every op name, every
    UseParameterN, and the quantised value ranges each get a contiguous id
    block. Ordering is fixed so the mapping is reproducible.
    """
    vocab: Dict[str, int] = {}

    def add(name: str) -> None:
        vocab[name] = len(vocab)

    for tok in (START_OF_CONSTRUCTION, END_OF_CONSTRUCTION, CREATED_CURVE):
        add(tok)
    for op in STEP_SPECS:  # dict preserves insertion order
        add("Op:" + op)
    for k in range(MAX_PARAMETERS):
        add("UseParameter%d" % k)
    for k in range(LENGTH_BINS):
        add("Length:%d" % k)
    for k in range(ANGLE_STEPS):
        add("Angle:%d" % k)
    return vocab
