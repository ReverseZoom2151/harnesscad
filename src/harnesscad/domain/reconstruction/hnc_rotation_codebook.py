"""HNC-CAD's 25-frame discrete extrude-orientation codebook (Xu et al., ICML 2023).

This is an *implementation-level* detail of "Hierarchical Neural Coding for
Controllable CAD Model Generation" that the paper review missed. The paper (and
:mod:`reconstruction.hnc_spl_tree`) describe the S-P-L code tree and the 6-bit
coordinate quantization, but say nothing about how the *orientation* of the
extrude sketch-plane is encoded. The reference data pipeline
(``data_process/utils.py::process_model``, plus the module-level ``ROT`` table)
reveals a distinctive scheme that differs sharply from the rest of the
DeepCAD family:

**The DeepCAD family encodes a sketch-plane rotation as continuous angles**
(:mod:`reconstruction.deepcad_command_spec` / :mod:`reconstruction.deepcad_sketch_plane`
store ``(theta, phi, gamma)`` ZYZ Euler angles, each quantized to 256 levels),
or, in SkexGen (:mod:`reconstruction.skexgen_extrude_tokens`), as nine
independently-rounded 3x3 matrix components.

**HNC-CAD does neither.** It reads the sketch plane's three axis vectors
``t_x, t_y, t_z`` (each a 3-vector), rounds-then-clips every component into
``{-1, 0, 1}`` and concatenates them into a length-9 pattern. That pattern is
then required to *exactly match* one of **exactly 25** canonical patterns
observed across the DeepCAD dataset (the ``ROT`` table). Orientation therefore
collapses to a single categorical index in ``[0, 25)`` -- one extra token, no
per-axis quantization at all. The reference code enforces this with
``assert (ROT == ext_R).all(axis=1).sum() == 1``: an orientation that does not
clip onto a known frame is rejected, so this table also doubles as a validator
for in-distribution extrude orientations.

Note the 25 patterns are *not* the 24 proper axis-aligned rotations and are not
orthonormal -- e.g. frame 0's x-axis is ``(-1, -1, 0)`` (norm sqrt(2)). They are
the empirical set of clipped axis triples, reproduced here faithfully so an index
round-trips bit-for-bit with the reference pipeline.

Pure stdlib, deterministic. No numpy, no learned components.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

Vec3 = Tuple[int, int, int]
Frame9 = Tuple[int, int, int, int, int, int, int, int, int]

# The reference ``ROT`` table: 25 canonical clipped-orientation patterns. Each
# row is ``(t_x[0..2], t_y[0..2], t_z[0..2])`` with entries in {-1, 0, 1}.
ROTATION_FRAMES: Tuple[Frame9, ...] = (
    (-1, -1, 0, 0, 0, 1, -1, 1, 0),
    (-1, 0, 0, 0, -1, 1, 0, 1, 1),
    (-1, 0, 0, 0, 0, 1, 0, 1, 0),
    (-1, 0, 0, 0, 1, 1, 0, 1, -1),
    (-1, 1, 0, 0, 0, 1, 1, 1, 0),
    (0, -1, 0, -1, 0, 1, -1, 0, -1),
    (0, -1, 0, 0, 0, 1, -1, 0, 0),
    (0, -1, 0, 1, 0, 1, -1, 0, 1),
    (0, 1, 0, -1, 0, 1, 1, 0, 1),
    (0, 1, 0, 0, 0, 1, 1, 0, 0),
    (0, 1, 0, 1, 0, 1, 1, 0, -1),
    (1, -1, 0, 0, 0, 1, -1, -1, 0),
    (1, 0, -1, 0, -1, 0, -1, 0, -1),
    (1, 0, -1, 0, 1, 0, 1, 0, 1),
    (1, 0, 0, 0, -1, -1, 0, 1, -1),
    (1, 0, 0, 0, -1, 0, 0, 0, -1),
    (1, 0, 0, 0, -1, 1, -1, -1, -1),
    (1, 0, 0, 0, -1, 1, 0, -1, -1),
    (1, 0, 0, 0, 0, 1, 0, -1, 0),
    (1, 0, 0, 0, 1, -1, 0, 1, 1),
    (1, 0, 0, 0, 1, 0, 0, 0, 1),
    (1, 0, 0, 0, 1, 1, 0, -1, 1),
    (1, 0, 1, 0, -1, 0, 1, 0, -1),
    (1, 0, 1, 0, 1, 0, -1, 0, 1),
    (1, 1, 0, 0, 0, 1, 1, -1, 0),
)

NUM_FRAMES = len(ROTATION_FRAMES)  # == 25

# Reverse index for O(1) exact-match lookup.
_FRAME_TO_INDEX = {frame: i for i, frame in enumerate(ROTATION_FRAMES)}

assert NUM_FRAMES == 25
assert len(_FRAME_TO_INDEX) == 25, "ROTATION_FRAMES must be distinct"


def _rint(value: float) -> int:
    """Round-half-to-even (matches ``numpy.rint`` used by the reference)."""
    # Python's built-in round() is banker's rounding, exactly numpy.rint.
    return int(round(float(value)))


def clip_axis(axis: Sequence[float]) -> Vec3:
    """Round-then-clip one 3-vector axis into ``{-1, 0, 1}`` component-wise.

    Faithful to ``np.clip(np.rint(axis).astype(int), -1, 1)``.
    """
    if len(axis) != 3:
        raise ValueError("axis must have length 3")
    out = []
    for c in axis:
        v = _rint(c)
        if v < -1:
            v = -1
        elif v > 1:
            v = 1
        out.append(v)
    return (out[0], out[1], out[2])


def clip_orientation(t_x: Sequence[float], t_y: Sequence[float],
                     t_z: Sequence[float]) -> Frame9:
    """Concatenate the three clipped axis vectors into a length-9 pattern."""
    x = clip_axis(t_x)
    y = clip_axis(t_y)
    z = clip_axis(t_z)
    return (x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2])


def quantize_orientation(t_x: Sequence[float], t_y: Sequence[float],
                         t_z: Sequence[float]) -> int:
    """Map a sketch-plane frame ``(t_x, t_y, t_z)`` to its codebook index.

    Reproduces ``process_model``'s ``ext_R_idx`` computation. Raises
    :class:`ValueError` if the clipped orientation is not one of the 25 known
    frames (the reference's ``assert ... .sum() == 1`` -- an out-of-distribution
    orientation).
    """
    pattern = clip_orientation(t_x, t_y, t_z)
    idx = _FRAME_TO_INDEX.get(pattern)
    if idx is None:
        raise ValueError(f"orientation {pattern} is not a known HNC frame")
    return idx


def is_known_orientation(t_x: Sequence[float], t_y: Sequence[float],
                         t_z: Sequence[float]) -> bool:
    """True iff the frame clips onto one of the 25 canonical patterns."""
    return clip_orientation(t_x, t_y, t_z) in _FRAME_TO_INDEX


def frame_pattern(index: int) -> Frame9:
    """Inverse lookup: the length-9 clipped pattern for a codebook index."""
    if not 0 <= index < NUM_FRAMES:
        raise IndexError(f"frame index {index} out of range [0, {NUM_FRAMES})")
    return ROTATION_FRAMES[index]


def frame_axes(index: int) -> Tuple[Vec3, Vec3, Vec3]:
    """Inverse lookup returning the three ``(t_x, t_y, t_z)`` axis vectors."""
    p = frame_pattern(index)
    return ((p[0], p[1], p[2]), (p[3], p[4], p[5]), (p[6], p[7], p[8]))


def frame_matrix(index: int) -> List[List[int]]:
    """The frame as a 3x3 row-major matrix whose rows are ``t_x, t_y, t_z``."""
    x, y, z = frame_axes(index)
    return [list(x), list(y), list(z)]
