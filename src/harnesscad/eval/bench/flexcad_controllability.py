"""Prediction-validity & controllability metrics for FlexCAD (Zhang et al. 2024).

FlexCAD is evaluated on both *generation quality* and *controllability*. Two of its
deterministic, learning-free metrics are implemented here:

* **Prediction Validity (PV)** (paper Sec. 4.1 "Metrics"): the fraction of predicted
  CAD texts that can be rendered into a 3D shape -- "rather than just 2D sketches or
  nothing". Here a prediction is *valid* when it parses and is structurally complete
  enough to extrude to a solid: at least one SE; every sketch has >=1 face; every
  face has >=1 loop (its first being the outer loop); every loop has >=1 curve; and
  every extrusion has a non-degenerate extent (not all attributes zero).

* **Controllability**: FlexCAD's defining property is that only the *masked* field
  changes while every unmasked element stays intact (paper Sec. 4.2; the baselines
  fail this -- SkexGen cannot target a specific SE, Hnc-cad cannot preserve unmasked
  elements). Given the original model, the mask target, and a predicted model, we
  check that the surrounding tokens are preserved verbatim and report whether the
  masked field actually changed (an *edit*).

Pure stdlib, deterministic. Consumes :mod:`reconstruction.flexcad_text` structures.
"""

from __future__ import annotations

from dataclasses import dataclass

from harnesscad.domain.reconstruction.flexcad_text import (
    CADModel,
    MaskResult,
    MaskTarget,
    ParseError,
    mask_field,
    parse,
    tokenize,
)


# --- Prediction Validity (PV) ---------------------------------------------
@dataclass(frozen=True)
class ValidityReport:
    """Why a predicted CAD text is / is not renderable to a 3D solid."""

    valid: bool
    reason: str


def _extrusion_is_degenerate(params: tuple[int, ...]) -> bool:
    """An extrusion with no attributes or all-zero attributes has zero extent."""
    return len(params) == 0 or all(p == 0 for p in params)


def model_validity(m: CADModel) -> ValidityReport:
    """Structural 3D-renderability check on a parsed model (PV predicate)."""
    if not m.ses:
        return ValidityReport(False, "no sketch-extrusion")
    for i, s in enumerate(m.ses):
        if not s.sketch.faces:
            return ValidityReport(False, f"se{i}: sketch has no face")
        for j, f in enumerate(s.sketch.faces):
            if not f.loops:
                return ValidityReport(False, f"se{i} face{j}: no loop")
            for k, lp in enumerate(f.loops):
                if not lp.curves:
                    return ValidityReport(False, f"se{i} face{j} loop{k}: no curve")
        if _extrusion_is_degenerate(s.extrusion.params):
            return ValidityReport(False, f"se{i}: degenerate extrusion (zero extent)")
    return ValidityReport(True, "ok")


def text_validity(text: str) -> ValidityReport:
    """PV predicate on a raw predicted CAD text: parses AND renders to 3D."""
    try:
        m = parse(text)
    except (ParseError, ValueError) as exc:
        return ValidityReport(False, f"parse error: {exc}")
    return model_validity(m)


def prediction_validity(predictions: list[str]) -> float:
    """PV: fraction of predicted CAD texts that render to a 3D shape (0..1)."""
    if not predictions:
        return 0.0
    good = sum(1 for t in predictions if text_validity(t).valid)
    return good / len(predictions)


# --- Controllability -------------------------------------------------------
@dataclass(frozen=True)
class ControllabilityReport:
    """Result of comparing an edit against its mask target.

    ``preserved`` -- every token *outside* the masked field is byte-identical.
    ``changed``   -- the masked field itself differs (a genuine edit was made).
    ``controllable`` -- the FlexCAD ideal: preserved AND changed.
    """

    preserved: bool
    changed: bool
    valid: bool

    @property
    def controllable(self) -> bool:
        return self.preserved and self.changed


def _split_around_mask(result: MaskResult) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the (prefix, suffix) of an instruction around its mask run."""
    instr = list(result.instruction)
    n = len(result.mask)
    for i in range(len(instr) - n + 1):
        if tuple(instr[i:i + n]) == tuple(result.mask):
            return tuple(instr[:i]), tuple(instr[i + n:])
    raise ParseError("mask run not found")


def controllability(original: CADModel, target: MaskTarget,
                    predicted: CADModel) -> ControllabilityReport:
    """Assess whether ``predicted`` edits only the ``target`` field of ``original``.

    Masks both models at the *same* target and compares the surrounding tokens: if
    the prefix and suffix match, everything outside the masked field was preserved.
    ``changed`` reports whether the masked field's content actually differs.
    """
    valid = model_validity(predicted).valid
    try:
        orig_mask = mask_field(original, target)
        pred_mask = mask_field(predicted, target)
    except (IndexError, ValueError):
        # Target field does not exist in the prediction (e.g. SE was dropped).
        return ControllabilityReport(False, True, valid)

    try:
        o_pre, o_suf = _split_around_mask(orig_mask)
        p_pre, p_suf = _split_around_mask(pred_mask)
    except ParseError:
        return ControllabilityReport(False, True, valid)

    preserved = (o_pre == p_pre) and (o_suf == p_suf)
    changed = orig_mask.answer != pred_mask.answer
    return ControllabilityReport(preserved, changed, valid)


def controllability_rate(cases: list[tuple[CADModel, MaskTarget, CADModel]]) -> dict:
    """Aggregate controllability over many (original, target, predicted) cases.

    Returns preserved-rate, edit-rate (changed), controllable-rate (preserved AND
    changed), and PV over the predictions. All deterministic.
    """
    if not cases:
        return {"preserved": 0.0, "changed": 0.0, "controllable": 0.0, "pv": 0.0}
    reports = [controllability(o, t, p) for (o, t, p) in cases]
    n = len(reports)
    return {
        "preserved": sum(r.preserved for r in reports) / n,
        "changed": sum(r.changed for r in reports) / n,
        "controllable": sum(r.controllable for r in reports) / n,
        "pv": sum(r.valid for r in reports) / n,
    }
