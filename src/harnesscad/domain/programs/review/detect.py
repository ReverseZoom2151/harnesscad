"""Error detector -- locate the wrong block and name the error type.

The task is: given a (possibly erroneous) CAD program and a reference design,
produce the error *type* (one of the eight scenarios in
:mod:`cadreview_taxonomy`) AND the ID of the offending code block. A
vision-grounded variant would learn this with a multimodal model that aligns a
rendered image with the program; that learned detector is out of scope here.

This module implements the deterministic, *reference-program*-grounded half of
the same task: when a known-correct reference program is available (which is how
the benchmark itself is constructed -- every erroneous sample is a mutation of a
correct one), the discrepancy can be recovered exactly by a structural block
diff, with no model. :func:`detect` segments both programs
(:mod:`cadreview_blocks`), aligns their blocks by signature, and classifies the
difference:

  * a block present in the reference but absent from the candidate -> Missing
    block; the reverse -> Redundant block;
  * an aligned pair that differs -> Primitive / Rotation / Position / Size /
    Logic / Constant error, chosen by which construct actually changed.

It returns the primary discrepancy (samples carry one injected error) plus
every discrepancy found, each as a :class:`Detection` naming the error type and
block ID -- exactly the pair the accuracy metric ("Acc") scores. Pure stdlib.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import List, Optional

from harnesscad.domain.programs.review.blocks import Block, segment
from harnesscad.domain.programs.review.taxonomy import (
    CONSTANT_ERROR, ErrorType, LOGIC_ERROR, MISSING_BLOCK, NO_ERROR,
    POSITION_ERROR, PRIMITIVE_ERROR, REDUNDANT_BLOCK, ROTATION_ERROR,
    SIZE_ERROR,
)

_PRIMS = {"cube", "sphere", "cylinder", "polyhedron", "square", "circle",
          "polygon", "text"}
_NUM_RE = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")


@dataclass
class Detection:
    """A single detected discrepancy.

    ``error_type`` is the classified :class:`ErrorType`; ``block_id`` is the ID
    of the offending block (in the candidate program, or the reference block ID
    for a Missing block); ``detail`` explains the evidence."""

    error_type: ErrorType
    block_id: Optional[int]
    detail: str

    def to_dict(self) -> dict:
        return {
            "error_type": self.error_type.label,
            "error_id": self.error_type.id,
            "block_id": self.block_id,
            "detail": self.detail,
        }


@dataclass
class Review:
    """The outcome of reviewing a candidate against a reference."""

    primary: Detection
    detections: List[Detection] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.primary.error_type.id == NO_ERROR.id

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "primary": self.primary.to_dict(),
            "detections": [d.to_dict() for d in self.detections],
        }


def _sig(b: Block) -> tuple:
    """Order-insensitive structural signature used to align blocks."""
    return (b.kind, b.head, tuple(b.calls))


def _call_args(text: str, name: str) -> Optional[str]:
    """The raw argument string of the first ``name(...)`` call, paren-balanced."""
    m = re.search(r"\b" + re.escape(name) + r"\s*\(", text)
    if not m:
        return None
    i = m.end() - 1
    depth = 0
    for j in range(i, len(text)):
        c = text[j]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def _nums(s: Optional[str]) -> List[float]:
    if not s:
        return []
    return [round(float(x), 6) for x in _NUM_RE.findall(s)]


def _prims(b: Block) -> List[str]:
    return [c for c in b.calls if c in _PRIMS]


def _classify_pair(ref: Block, cand: Block) -> Optional[Detection]:
    """Classify a difference between an aligned reference/candidate block pair,
    or None if they are equivalent. Priority mirrors the taxonomy's construct
    focus: constant -> logic -> primitive -> rotation -> position -> size."""
    if ref.text.strip() == cand.text.strip():
        return None

    # Constant error: leading assignment block values changed.
    if ref.kind == "assignment" and cand.kind == "assignment":
        if _nums(ref.text) != _nums(cand.text):
            return Detection(CONSTANT_ERROR, cand.id,
                             "global constant / macro value changed vs reference")

    # Logic error: control-flow condition (for range / if test) changed.
    if ref.kind == "control_flow" or cand.kind == "control_flow":
        for name in ("for", "if", "intersection_for", "let"):
            if _nums(_call_args(ref.text, name)) != _nums(_call_args(cand.text, name)):
                return Detection(LOGIC_ERROR, cand.id,
                                 f"control-flow condition ({name}) changed vs reference")

    # Primitive error: the geometric primitive kind changed.
    rp, cp = set(_prims(ref)), set(_prims(cand))
    if rp != cp and rp and cp:
        return Detection(PRIMITIVE_ERROR, cand.id,
                         f"primitive changed {sorted(rp)} -> {sorted(cp)}")

    # Rotation error: rotate() presence or angle changed.
    if _nums(_call_args(ref.text, "rotate")) != _nums(_call_args(cand.text, "rotate")):
        return Detection(ROTATION_ERROR, cand.id,
                         "rotate() added / removed or angle changed vs reference")

    # Position error: translate() presence or offset changed.
    if _nums(_call_args(ref.text, "translate")) != _nums(_call_args(cand.text, "translate")):
        return Detection(POSITION_ERROR, cand.id,
                         "translate() added / removed or offset changed vs reference")

    # Size error: same primitive kind but numeric dimensions changed.
    for prim in cp & rp:
        if _nums(_call_args(ref.text, prim)) != _nums(_call_args(cand.text, prim)):
            return Detection(SIZE_ERROR, cand.id,
                             f"{prim} dimension changed vs reference")

    # Fell through: some numeric difference we could not attribute more precisely.
    if _nums(ref.text) != _nums(cand.text):
        return Detection(SIZE_ERROR, cand.id,
                         "numeric parameter changed vs reference")
    return Detection(PRIMITIVE_ERROR, cand.id, "block content changed vs reference")


def detect(candidate_src: str, reference_src: str) -> Review:
    """Review ``candidate_src`` against ``reference_src`` and classify the error.

    Returns a :class:`Review`; ``primary`` is the highest-priority discrepancy
    (or ``No error`` when the programs are block-equivalent)."""
    ref_blocks = segment(reference_src)
    cand_blocks = segment(candidate_src)
    ref_sigs = [_sig(b) for b in ref_blocks]
    cand_sigs = [_sig(b) for b in cand_blocks]

    sm = difflib.SequenceMatcher(a=ref_sigs, b=cand_sigs, autojunk=False)
    dets: List[Detection] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            # Signatures match but text may still differ (size/const/etc).
            for k in range(i2 - i1):
                d = _classify_pair(ref_blocks[i1 + k], cand_blocks[j1 + k])
                if d is not None:
                    dets.append(d)
        elif tag == "replace":
            n = min(i2 - i1, j2 - j1)
            for k in range(n):
                d = _classify_pair(ref_blocks[i1 + k], cand_blocks[j1 + k])
                if d is not None:
                    dets.append(d)
            # Surplus reference blocks are missing; surplus candidate are redundant.
            for k in range(n, i2 - i1):
                dets.append(Detection(MISSING_BLOCK, ref_blocks[i1 + k].id,
                                      "reference block absent from candidate"))
            for k in range(n, j2 - j1):
                dets.append(Detection(REDUNDANT_BLOCK, cand_blocks[j1 + k].id,
                                      "candidate block absent from reference"))
        elif tag == "delete":
            for k in range(i1, i2):
                dets.append(Detection(MISSING_BLOCK, ref_blocks[k].id,
                                      "reference block absent from candidate"))
        elif tag == "insert":
            for k in range(j1, j2):
                dets.append(Detection(REDUNDANT_BLOCK, cand_blocks[k].id,
                                      "candidate block absent from reference"))

    if not dets:
        return Review(primary=Detection(NO_ERROR, None,
                                        "candidate matches the reference design"),
                      detections=[])
    return Review(primary=dets[0], detections=dets)
