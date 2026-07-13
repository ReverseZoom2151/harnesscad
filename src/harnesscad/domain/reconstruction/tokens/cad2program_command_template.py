"""cad2program_command_template — fixed-slot command template + quantization.

CAD2PROGRAM (Wang et al., AAAI 2025, Sec. 3.1 / 4.3) contrasts its free-text
shape program with the *domain-specific command template* used by prior sequence
models (DeepCAD, PlankAssembly, ...).  A command template aggregates all common
parameters into a fixed-length slot vector, and every continuous value is
*quantized* into discrete tokens by a domain tokenizer:

  * each position/size parameter -> one of **1500 bins at 3 mm resolution**;
  * the rotation angle -> one of **4 bins** (0/90/180/270 degrees);
  * the model ID -> a special token.

The paper's key argument for text output is that this quantization introduces an
*inherent quantization error* that a text representation avoids.  Both the
quantizer and the resulting error are deterministic, so this module implements:

  * :func:`quantize_value` / :func:`dequantize_value` — the uniform binning and
    its bin-center inverse;
  * :func:`encode_command` / :func:`decode_command` — the fixed-slot vector for a
    single primitive (model ID slot + 6 quantized common params + angle bin);
  * :func:`quantization_error` — the max absolute round-trip error a template
    incurs on a shape program, which is exactly what the text representation
    drives to zero.

This is the *baseline* the paper measures against; the VLM predicting the tokens
is external.  Pure stdlib, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from harnesscad.domain.reconstruction.translate.cad2program_shape_program import (
    Bbox, PrimitiveInstance, ShapeProgram,
)

# Paper's defaults (Sec. 4.3, first variant).
DEFAULT_RESOLUTION = 3.0      # mm per bin
DEFAULT_N_BINS = 1500         # position/size bins
DEFAULT_ANGLE_BINS = 4        # rotation bins (0/90/180/270)

# The six common continuous parameters, in slot order.
COMMON_FIELDS: Tuple[str, ...] = (
    "position_x", "position_y", "position_z",
    "scale_x", "scale_y", "scale_z",
)


def quantize_value(value: float, resolution: float = DEFAULT_RESOLUTION,
                   n_bins: int = DEFAULT_N_BINS) -> int:
    """Quantize a continuous value to a bin index in ``[0, n_bins)``.

    Uniform binning of width ``resolution`` starting at 0; values are clamped to
    the representable range ``[0, (n_bins-1)*resolution]``.
    """
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    idx = int(round(value / resolution))
    if idx < 0:
        idx = 0
    elif idx > n_bins - 1:
        idx = n_bins - 1
    return idx


def dequantize_value(index: int, resolution: float = DEFAULT_RESOLUTION) -> float:
    """Inverse of :func:`quantize_value`: bin index -> bin-center value."""
    return index * resolution


def quantize_angle(angle: float, n_bins: int = DEFAULT_ANGLE_BINS) -> int:
    """Quantize a z-rotation (degrees) into ``n_bins`` equal sectors of 360deg."""
    step = 360.0 / n_bins
    return int(round((angle % 360) / step)) % n_bins


def dequantize_angle(index: int, n_bins: int = DEFAULT_ANGLE_BINS) -> float:
    step = 360.0 / n_bins
    return (index % n_bins) * step


@dataclass(frozen=True)
class Command:
    """A fixed-slot command: model ID token + 6 param bins + 1 angle bin."""

    model_id: str
    param_bins: Tuple[int, ...]   # length 6, order = COMMON_FIELDS
    angle_bin: int


def encode_command(inst: PrimitiveInstance,
                   resolution: float = DEFAULT_RESOLUTION,
                   n_bins: int = DEFAULT_N_BINS,
                   angle_bins: int = DEFAULT_ANGLE_BINS) -> Command:
    """Encode one primitive's common parameters into a fixed-slot command.

    Model-specific parameters (``P_i``) have no slot in this template — the very
    limitation the paper cites — so they are dropped here.
    """
    b = inst.bbox
    bins = tuple(quantize_value(getattr(b, f), resolution, n_bins)
                 for f in COMMON_FIELDS)
    return Command(inst.model_id, bins, quantize_angle(b.angle_z, angle_bins))


def decode_command(cmd: Command, resolution: float = DEFAULT_RESOLUTION,
                   angle_bins: int = DEFAULT_ANGLE_BINS) -> PrimitiveInstance:
    """Decode a command back to a primitive (bin centers; params lost)."""
    vals = [dequantize_value(i, resolution) for i in cmd.param_bins]
    box = Bbox(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5],
               dequantize_angle(cmd.angle_bin, angle_bins))
    return PrimitiveInstance(cmd.model_id, box, ())


def encode_program(program: ShapeProgram,
                   resolution: float = DEFAULT_RESOLUTION,
                   n_bins: int = DEFAULT_N_BINS,
                   angle_bins: int = DEFAULT_ANGLE_BINS) -> List[Command]:
    return [encode_command(inst, resolution, n_bins, angle_bins)
            for inst in program.instances]


def quantization_error(program: ShapeProgram,
                       resolution: float = DEFAULT_RESOLUTION,
                       n_bins: int = DEFAULT_N_BINS) -> float:
    """Maximum absolute round-trip error over all common params of a program.

    This is the error the fixed-slot command template incurs and that the paper's
    text representation avoids (returns 0 exactly when every value already lies on
    a bin center within the representable range).
    """
    worst = 0.0
    for inst in program.instances:
        b = inst.bbox
        for f in COMMON_FIELDS:
            v = getattr(b, f)
            r = dequantize_value(quantize_value(v, resolution, n_bins), resolution)
            worst = max(worst, abs(v - r))
    return worst
