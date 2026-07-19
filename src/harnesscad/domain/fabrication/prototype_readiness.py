"""Looks-like prototyping-readiness gate for image-to-3D output.

A sketch-to-image-to-3D pipeline has concrete, recurring failure modes that stop
a generated concept from being a printable "looks-like" prototype:

  * Image-to-3D meshes are often non-smooth, FRAGMENTED, and sometimes
    non-manufacturable; postprocessing is required either to smooth surfaces,
    FILL HOLES, or remove unmanufacturable parts.
  * A generated image may contain TEXT, which
    negatively affects the generation quality when converting from image to 3D,
    so text-bearing images are excluded up front.
  * A sketch-to-text prompt can be rejected as an UNSAFE prompt by the image
    generator, which blocks generation entirely.
  * Image-to-3D outputs are exported as STL for printing; a readiness
    check gates whether that export is even worth attempting.

This module turns those modes into a deterministic checklist over a
lightweight prototype descriptor (a dict), producing advisory findings with a
severity model (``ok`` / ``warning`` / ``error``) and an overall readiness gate.
It is distinct from ``fabrication/fabworkflow_feasibility`` (machine envelope /
print time / material stock rules keyed on a fabrication paradigm) and from
``verifiers/dfm`` (per-solid wall/draft checks on an OCCT solid): this operates
on the image-to-3D *stage output* and its upstream prompt/image guards, with
no geometry kernel involved.

Standard library only; fully deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass


OK = "ok"
WARNING = "warning"
ERROR = "error"

_SEVERITY_ORDER = {OK: 0, WARNING: 1, ERROR: 2}


@dataclass(frozen=True)
class Finding:
    """One readiness check result."""
    check: str
    severity: str
    message: str


def _f(check, severity, message):
    return Finding(check, severity, message)


def check_prompt_safety(descriptor):
    """Error if the sketch-to-text prompt was flagged unsafe (blocks generation)."""
    if descriptor.get("prompt_flagged_unsafe", False):
        return _f(
            "prompt_safety", ERROR,
            "sketch-to-text prompt flagged unsafe; text-to-image generation blocked",
        )
    return _f("prompt_safety", OK, "prompt accepted")


def check_source_image_text(descriptor):
    """Warn if the source image contains rendered text before image-to-3D.

    Text-bearing images are excluded because text degrades the
    image-to-3D conversion.
    """
    if descriptor.get("image_contains_text", False):
        return _f(
            "source_image_text", WARNING,
            "source image contains rendered text; exclude before image-to-3D",
        )
    return _f("source_image_text", OK, "source image free of rendered text")


def check_fragmentation(descriptor):
    """Error on a fragmented mesh (more than one connected component)."""
    n = int(descriptor.get("component_count", 1))
    if n < 1:
        raise ValueError("component_count must be >= 1")
    if n > 1:
        return _f(
            "fragmentation", ERROR,
            "mesh fragmented into %d components; expected a single connected part" % n,
        )
    return _f("fragmentation", OK, "mesh is a single connected component")


def check_watertight(descriptor):
    """Warn on open boundary holes; a watertight mesh is needed for printing.

    ``hole_count`` counts unfilled boundary loops. Non-watertight meshes are
    printable only after the hole-filling post-process step.
    """
    holes = int(descriptor.get("hole_count", 0))
    if holes < 0:
        raise ValueError("hole_count must be >= 0")
    if holes > 0:
        return _f(
            "watertight", WARNING,
            "mesh has %d unfilled hole(s); fill before export" % holes,
        )
    return _f("watertight", OK, "mesh is watertight")


def check_surface_smoothness(descriptor):
    """Warn on non-smooth surfaces (image-to-3D often yields uneven surfaces)."""
    if not descriptor.get("surface_smooth", True):
        return _f(
            "surface_smoothness", WARNING,
            "surfaces non-smooth; apply smoothing post-process",
        )
    return _f("surface_smoothness", OK, "surfaces acceptably smooth")


def check_manufacturable_volume(descriptor):
    """Error on a degenerate/sparse mesh with non-positive printable volume.

    Unprocessed-sketch meshes are typically sparse and unmanufacturable.
    """
    if "volume" not in descriptor:
        return _f("manufacturable_volume", OK, "volume not provided; skipped")
    vol = float(descriptor["volume"])
    if vol <= 0.0:
        return _f(
            "manufacturable_volume", ERROR,
            "mesh volume %.6g is non-positive; sparse/unmanufacturable" % vol,
        )
    return _f("manufacturable_volume", OK, "positive printable volume")


ALL_CHECKS = (
    check_prompt_safety,
    check_source_image_text,
    check_fragmentation,
    check_watertight,
    check_surface_smoothness,
    check_manufacturable_volume,
)


def evaluate(descriptor):
    """Run every readiness check over a prototype descriptor, in a fixed order.

    Returns a tuple of :class:`Finding`. Deterministic: the order of findings is
    always ``ALL_CHECKS`` order regardless of descriptor contents.
    """
    return tuple(check(descriptor) for check in ALL_CHECKS)


def worst_severity(findings):
    """Return the most severe severity among findings (``ok`` if none)."""
    worst = OK
    for f in findings:
        if _SEVERITY_ORDER[f.severity] > _SEVERITY_ORDER[worst]:
            worst = f.severity
    return worst


def is_ready(descriptor, allow_warnings=True):
    """True if the prototype is print-ready.

    With ``allow_warnings`` (default) a prototype is ready when no check reports
    an ERROR -- warnings (holes, non-smooth surfaces, text-in-image) are
    resolvable by the documented post-process step. Set ``allow_warnings=False``
    to require a clean bill (no warnings either).
    """
    worst = worst_severity(evaluate(descriptor))
    if worst == ERROR:
        return False
    if worst == WARNING and not allow_warnings:
        return False
    return True


def readiness_report(descriptor, allow_warnings=True):
    """Deterministic summary dict of the readiness evaluation.

    Keys: ``ready`` (bool), ``worst_severity``, ``findings`` (tuple of dicts
    with check/severity/message), ``error_count``, ``warning_count``.
    """
    findings = evaluate(descriptor)
    errors = sum(1 for f in findings if f.severity == ERROR)
    warnings = sum(1 for f in findings if f.severity == WARNING)
    return {
        "ready": is_ready(descriptor, allow_warnings=allow_warnings),
        "worst_severity": worst_severity(findings),
        "error_count": errors,
        "warning_count": warnings,
        "findings": tuple(
            {"check": f.check, "severity": f.severity, "message": f.message}
            for f in findings
        ),
    }
