"""Hierarchy-aware masking scheme for FlexCAD (Zhang et al. 2024, Sec. 3.2).

FlexCAD's single unified controllable-generation model is trained by, at each epoch,
*uniformly sampling one construction hierarchy* and replacing that hierarchy-aware
field in the CAD text with a mask token, then asking the LLM to predict the masked
field. The seven addressable hierarchies are::

    cad | sketch-extrusion | sketch | extrusion | face | loop | curve

Mask-token design (paper Fig. 4):

* **cad**: mask *every* internal SE with ``[sketch-extrusionmask]`` -- only the SE
  count is preserved, letting inference freely set model complexity.
* **sketch-extrusion / sketch / extrusion**: replace the field with
  ``[sketch-extrusionmask]`` / ``[sketchmask]`` / ``[extrusionmask]``.
* **face / loop**: mask a single face (loop); when several faces (loops) belong to
  the same sketch (face) they can be masked *all at once* with that many mask tokens.
* **curve**: all curves of one loop are masked with **typed** tokens
  (``[linemask]`` / ``[arcmask]`` / ``[circlemask]``) -- controlling curve topology
  (type & number) and geometry.

This module is the deterministic masking scheme layered on
:mod:`reconstruction.flexcad_text`: it (a) enumerates every maskable field of a
model, (b) samples a hierarchy uniformly with a seeded ``random.Random`` (no wall
clock), and (c) realises single- and multi-field masks. The learned infill model is
out of scope.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from reconstruction.flexcad_text import (
    CADModel,
    LEVELS,
    LEVEL_CAD,
    LEVEL_SE,
    LEVEL_SKETCH,
    LEVEL_EXTRUSION,
    LEVEL_FACE,
    LEVEL_LOOP,
    LEVEL_CURVE,
    MaskResult,
    MaskTarget,
    SE_MASK,
    FACE_MASK,
    LOOP_MASK,
    CURVE_MASK,
    mask_field,
    tokenize,
    _emit_face,
    _emit_loop,
    _emit_sketch,
    _emit_se,
    _curve_run_tokens,
)

# Uniform-sampling pool of the seven hierarchies (paper Sec. 3.2 "Unified Training").
SAMPLING_LEVELS: tuple[str, ...] = LEVELS


def enumerate_fields(m: CADModel) -> list[MaskTarget]:
    """Every individually maskable field of ``m``, coarse-to-fine.

    Includes one :class:`MaskTarget` per SE (se/sketch/extrusion levels), per face,
    per loop, and per loop for the curve level, plus a single whole-model CAD-level
    target. This is the ground set the unified trainer samples over.
    """
    fields: list[MaskTarget] = [MaskTarget(LEVEL_CAD)]
    for i, s in enumerate(m.ses):
        fields.append(MaskTarget(LEVEL_SE, se=i))
        fields.append(MaskTarget(LEVEL_SKETCH, se=i))
        fields.append(MaskTarget(LEVEL_EXTRUSION, se=i))
        for j, f in enumerate(s.sketch.faces):
            fields.append(MaskTarget(LEVEL_FACE, se=i, face=j))
            for k in range(len(f.loops)):
                fields.append(MaskTarget(LEVEL_LOOP, se=i, face=j, loop=k))
                fields.append(MaskTarget(LEVEL_CURVE, se=i, face=j, loop=k))
    return fields


def fields_at_level(m: CADModel, level: str) -> list[MaskTarget]:
    """All maskable fields of ``m`` at one hierarchy ``level``."""
    if level not in LEVELS:
        raise ValueError(f"unknown level: {level!r}")
    return [t for t in enumerate_fields(m) if t.level == level]


def sample_level(rng: random.Random) -> str:
    """Uniformly sample one of the seven hierarchies (deterministic given ``rng``)."""
    return rng.choice(SAMPLING_LEVELS)


def sample_target(m: CADModel, rng: random.Random,
                  level: str | None = None) -> MaskTarget:
    """Sample a maskable field: a uniform level, then a uniform field within it.

    If ``level`` is given, only fields at that level are considered. Falls back to
    resampling the level when a model has no field at the drawn level (e.g. it has
    no faces to mask -- impossible for a valid model, but guarded).
    """
    if level is not None:
        candidates = fields_at_level(m, level)
        if not candidates:
            raise ValueError(f"model has no field at level {level!r}")
        return rng.choice(candidates)
    for _ in range(len(SAMPLING_LEVELS) * 4):
        lvl = sample_level(rng)
        candidates = fields_at_level(m, lvl)
        if candidates:
            return rng.choice(candidates)
    # Deterministic fallback: the always-present CAD-level field.
    return MaskTarget(LEVEL_CAD)


def mask(m: CADModel, target: MaskTarget) -> MaskResult:
    """Mask a single addressed field (thin re-export of the text-level masker)."""
    return mask_field(m, target)


# --- multi-field ("all at once") masks (paper Fig. 4(d)) -------------------
def mask_all_faces_of_sketch(m: CADModel, se: int) -> MaskResult:
    """Mask *every* face of one sketch at once (one ``[facemask]`` per face)."""
    tokens = tokenize(m)
    base = _se_base_offset(m, se)
    s = m.ses[se]
    # The faces span the sketch minus its trailing <sketch_end>.
    sk_buf: list[str] = []
    _emit_sketch(sk_buf, s.sketch)
    faces_len = len(sk_buf) - 1  # drop <sketch_end>
    span = (base, base + faces_len)
    mask_toks = tuple(FACE_MASK for _ in s.sketch.faces)
    return _splice(tokens, span, mask_toks, LEVEL_FACE)


def mask_all_loops_of_face(m: CADModel, se: int, face: int) -> MaskResult:
    """Mask *every* loop of one face at once (one ``[loopmask]`` per loop)."""
    tokens = tokenize(m)
    base = _face_base_offset(m, se, face)
    f = m.ses[se].sketch.faces[face]
    fb: list[str] = []
    _emit_face(fb, f)
    loops_len = len(fb) - 1  # drop <face_end>
    span = (base, base + loops_len)
    mask_toks = tuple(LOOP_MASK for _ in f.loops)
    return _splice(tokens, span, mask_toks, LEVEL_LOOP)


def mask_curves_of_loop(m: CADModel, se: int, face: int, loop: int) -> MaskResult:
    """Mask all curves of one loop with typed masks (paper Fig. 4(e))."""
    return mask_field(m, MaskTarget(LEVEL_CURVE, se=se, face=face, loop=loop))


# --- offset helpers (mirror the serialiser's ordering) ---------------------
def _se_base_offset(m: CADModel, se: int) -> int:
    off = 0
    for k in range(se):
        buf: list[str] = []
        _emit_se(buf, m.ses[k])
        off += len(buf)
    return off


def _face_base_offset(m: CADModel, se: int, face: int) -> int:
    off = _se_base_offset(m, se)
    s = m.ses[se]
    for j in range(face):
        fb: list[str] = []
        _emit_face(fb, s.sketch.faces[j])
        off += len(fb)
    return off


def _splice(tokens: list[str], span: tuple[int, int],
            mask_toks: tuple[str, ...], level: str) -> MaskResult:
    prefix = tokens[: span[0]]
    answer = tokens[span[0]: span[1]]
    suffix = tokens[span[1]:]
    instruction = tuple(prefix) + mask_toks + tuple(suffix)
    return MaskResult(instruction, tuple(answer), mask_toks, level)
