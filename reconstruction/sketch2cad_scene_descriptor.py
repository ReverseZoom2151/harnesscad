"""Scene-descriptor token codec (Yang, "Sketch2CAD: 3D CAD Model Reconstruction
from 2D Sketch using Visual Transformer", EPFL 2023).

Sketch2CAD reframes single-image 3D reconstruction as a Pix2Seq-style *text
generation* task: a visual transformer emits a sequence of tokens describing a
"scene descriptor" -- the list of objects in the scene and their shape parameters.
This module implements the deterministic *target parameterisation* (Sec. III-B),
independent of the learned encoder/decoder, which are out of scope.

Per the paper each object is serialised as the fixed 9-slot row (Sec. III-B-2)::

    [shape-type, position-x, position-y, position-z, yaw, pitch,
     size-x, size-y, size-z]

and a whole scene is the camera pose ID followed by the concatenation of every
object's row (the camera pose "is encapsulated at the beginning of the sequence").

Continuous parameters are tokenised exactly as the paper's Eq. (Sec. III-B-2):

    Q_i = round( (x_i - min(X)) / (max(X) - min(X)) * (n_bins - 1) )

i.e. uniform discretisation of a value into an integer in ``[0, n_bins - 1]``. The
paper uses *different* bin counts per property (e.g. complex dataset: position 200,
size 60, rotation 4) and shares the vocabulary *only between axes of the same
property*. The flat vocabulary therefore has the layout / total size (Sec. III-B-2)::

    total = n_cam_pose + n_shape_type + n_bin_pos + n_bin_rot + n_bin_size

This module lays those five blocks out at fixed offsets so a scene maps to a flat
list of integer token ids and back.

This is distinct from :mod:`reconstruction.ppa_quantization` (2D sketch-primitive
[0,1] normalisation) and :mod:`reconstruction.deepcad_command_spec` (SOL/Line/Arc/
Circle/Ext command vector): this is a 3D-object *scene* serialisation with a
camera-pose prefix and per-property shared vocabularies. Pure stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The seven architectural shapes the paper uses (Sec. III-A-1, Fig. 4).
SHAPE_TYPES: tuple[str, ...] = (
    "cube",
    "cylinder",
    "pyramid",
    "shed",
    "hip",
    "aframe",
    "mansard",
)
SHAPE_INDEX: dict[str, int] = {s: i for i, s in enumerate(SHAPE_TYPES)}


def quantize(value: float, lo: float, hi: float, n_bins: int) -> int:
    """Paper Eq.: uniformly discretise ``value`` in ``[lo, hi]`` to ``[0, n_bins-1]``.

    ``Q = round((value - lo) / (hi - lo) * (n_bins - 1))``, clamped to the range.
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if n_bins == 1:
        return 0
    if hi <= lo:
        raise ValueError("hi must exceed lo")
    frac = (value - lo) / (hi - lo)
    q = int(_round_half_up(frac * (n_bins - 1)))
    if q < 0:
        return 0
    if q > n_bins - 1:
        return n_bins - 1
    return q


def dequantize(level: int, lo: float, hi: float, n_bins: int) -> float:
    """Inverse of :func:`quantize`: the bin centre value for ``level``."""
    if n_bins <= 1:
        return lo
    level = max(0, min(n_bins - 1, int(level)))
    return lo + (hi - lo) * level / (n_bins - 1)


def _round_half_up(x: float) -> int:
    """Deterministic round-half-up (avoids banker's rounding of ``round``)."""
    import math

    return math.floor(x + 0.5)


@dataclass(frozen=True)
class SceneObject:
    """One object in the scene: a shape type plus 3D pose/size (world units)."""

    shape: str
    position: tuple[float, float, float]
    rotation: tuple[float, float]  # (yaw, pitch) in degrees
    size: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.shape not in SHAPE_INDEX:
            raise ValueError(f"unknown shape {self.shape!r}")
        if len(self.position) != 3 or len(self.size) != 3 or len(self.rotation) != 2:
            raise ValueError("position/size must be 3-tuples, rotation a 2-tuple")


@dataclass(frozen=True)
class DescriptorConfig:
    """Bin counts and value ranges for the five vocabulary blocks (Sec. III-B-2)."""

    n_cam_pose: int = 60
    n_bin_pos: int = 200
    n_bin_rot: int = 4
    n_bin_size: int = 60
    pos_range: tuple[float, float] = (0.0, 200.0)
    rot_range: tuple[float, float] = (0.0, 360.0)
    size_range: tuple[float, float] = (0.0, 60.0)
    n_shape_type: int = field(default=len(SHAPE_TYPES))

    # --- flat vocabulary block offsets (paper's block order) -------------
    @property
    def off_cam(self) -> int:
        return 0

    @property
    def off_shape(self) -> int:
        return self.n_cam_pose

    @property
    def off_pos(self) -> int:
        return self.off_shape + self.n_shape_type

    @property
    def off_rot(self) -> int:
        return self.off_pos + self.n_bin_pos

    @property
    def off_size(self) -> int:
        return self.off_rot + self.n_bin_rot

    @property
    def vocab_size(self) -> int:
        """total = n_cam_pose + n_shape_type + n_bin_pos + n_bin_rot + n_bin_size."""
        return self.off_size + self.n_bin_size


# Number of tokens per object row: [shape, px, py, pz, yaw, pitch, sx, sy, sz].
TOKENS_PER_OBJECT = 9


class SceneDescriptorCodec:
    """Encode/decode a scene <-> a flat list of vocabulary token ids."""

    def __init__(self, config: DescriptorConfig | None = None) -> None:
        self.config = config or DescriptorConfig()

    # --- single-token helpers -------------------------------------------
    def cam_token(self, pose_id: int) -> int:
        c = self.config
        if not 0 <= pose_id < c.n_cam_pose:
            raise ValueError(f"pose_id out of range: {pose_id}")
        return c.off_cam + pose_id

    def shape_token(self, shape: str) -> int:
        return self.config.off_shape + SHAPE_INDEX[shape]

    def _pos_token(self, v: float) -> int:
        c = self.config
        return c.off_pos + quantize(v, c.pos_range[0], c.pos_range[1], c.n_bin_pos)

    def _rot_token(self, v: float) -> int:
        c = self.config
        return c.off_rot + quantize(v, c.rot_range[0], c.rot_range[1], c.n_bin_rot)

    def _size_token(self, v: float) -> int:
        c = self.config
        return c.off_size + quantize(v, c.size_range[0], c.size_range[1], c.n_bin_size)

    # --- object / scene encoding ----------------------------------------
    def encode_object(self, obj: SceneObject) -> list[int]:
        """The 9-token row ``[shape, px,py,pz, yaw,pitch, sx,sy,sz]``."""
        row = [self.shape_token(obj.shape)]
        row += [self._pos_token(v) for v in obj.position]
        row += [self._rot_token(v) for v in obj.rotation]
        row += [self._size_token(v) for v in obj.size]
        return row

    def encode_scene(self, pose_id: int, objects) -> list[int]:
        """Camera-pose token followed by every object's row (paper's serialisation)."""
        seq = [self.cam_token(pose_id)]
        for obj in objects:
            seq += self.encode_object(obj)
        return seq

    # --- decoding --------------------------------------------------------
    def decode_object(self, row) -> SceneObject:
        c = self.config
        if len(row) != TOKENS_PER_OBJECT:
            raise ValueError(f"object row must have {TOKENS_PER_OBJECT} tokens")
        shape_idx = row[0] - c.off_shape
        if not 0 <= shape_idx < c.n_shape_type:
            raise ValueError("first token is not a shape-type token")
        shape = SHAPE_TYPES[shape_idx]
        px, py, pz = (
            dequantize(row[i] - c.off_pos, c.pos_range[0], c.pos_range[1], c.n_bin_pos)
            for i in (1, 2, 3)
        )
        yaw, pitch = (
            dequantize(row[i] - c.off_rot, c.rot_range[0], c.rot_range[1], c.n_bin_rot)
            for i in (4, 5)
        )
        sx, sy, sz = (
            dequantize(row[i] - c.off_size, c.size_range[0], c.size_range[1], c.n_bin_size)
            for i in (6, 7, 8)
        )
        return SceneObject(shape, (px, py, pz), (yaw, pitch), (sx, sy, sz))

    def decode_scene(self, seq):
        """Return ``(pose_id, [SceneObject, ...])`` from a flat token sequence."""
        if not seq:
            raise ValueError("empty sequence")
        c = self.config
        pose_id = seq[0] - c.off_cam
        if not 0 <= pose_id < c.n_cam_pose:
            raise ValueError("first token is not a camera-pose token")
        body = seq[1:]
        if len(body) % TOKENS_PER_OBJECT != 0:
            raise ValueError("object token block is not a multiple of 9")
        objs = [
            self.decode_object(body[i : i + TOKENS_PER_OBJECT])
            for i in range(0, len(body), TOKENS_PER_OBJECT)
        ]
        return pose_id, objs


def serialize_scene(codec: SceneDescriptorCodec, pose_id: int, objects, rng=None):
    """Encode a scene using the paper's *random ordering* strategy (Sec. III-B-2).

    Pix2Seq showed a random object order outperforms a fixed deterministic one, so
    the objects are shuffled by ``rng`` (a ``random.Random`` for reproducibility)
    before serialisation. The camera-pose token always stays first.
    """
    objs = list(objects)
    if rng is not None:
        rng.shuffle(objs)
    return codec.encode_scene(pose_id, objs)
