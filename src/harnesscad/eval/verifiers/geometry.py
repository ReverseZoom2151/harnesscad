"""Geometry-level verifiers that read a real B-rep backend.

``BRepValidityCheck`` is the real topology check the blueprint reserves for the
CadQuery/OCCT backend (the stub's ``SolidPresenceCheck`` is only a placeholder).
It reads ``backend.query('validity')`` and errors when a solid exists but is not
manifold / watertight / valid.

Standalone by design: this is NOT wired into ``verify.default_verifiers`` here —
it is added to the default set at integration time (verify.py is untouched).
"""

from __future__ import annotations

from typing import List

from harnesscad.eval.verifiers.verify import Diagnostic, Severity, VerifyReport


class BRepValidityCheck:
    """Error when a solid is present but topologically invalid.

    Reads the backend's real OCCT validity report. If no solid is present the
    check is a no-op (solid *presence* is another verifier's concern).
    """

    name = "brep-validity"

    def check(self, backend, opdag) -> VerifyReport:
        v = backend.query("validity")
        diags: List[Diagnostic] = []
        if v.get("solid_present"):
            if not (v.get("manifold") and v.get("watertight") and v.get("is_valid")):
                diags.append(Diagnostic(
                    Severity.ERROR, "invalid-brep",
                    "solid is present but not manifold/watertight/valid "
                    f"(manifold={v.get('manifold')}, "
                    f"watertight={v.get('watertight')}, "
                    f"is_valid={v.get('is_valid')})"))
        return VerifyReport(diags)
