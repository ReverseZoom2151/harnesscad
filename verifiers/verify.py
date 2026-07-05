"""Verification layer — the plural verifier the harness is built around.

Per the blueprint, verification is not one check but several independent ones
whose diagnostics feed back into the loop (block-and-correct / recycling):
  - constraint solver (sketch DOF)
  - B-rep validity (topology) — real check arrives with the CadQuery backend
  - (later) assembly solver (mates / DOF / collision)

A verifier reads the backend (never mutates) and returns a VerifyReport. Any
ERROR-severity diagnostic makes the report `ok == False`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Protocol


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Diagnostic:
    severity: Severity
    code: str
    message: str
    where: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "where": self.where,
        }


@dataclass
class VerifyReport:
    diagnostics: List[Diagnostic] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(d.severity is Severity.ERROR for d in self.diagnostics)


class Verifier(Protocol):
    name: str

    def check(self, backend, opdag) -> VerifyReport: ...


class SketchConstraintCheck:
    """Flag over-constrained (error) and under-constrained (warning) sketches."""

    name = "sketch-constraint"

    def check(self, backend, opdag) -> VerifyReport:
        diags: List[Diagnostic] = []
        for sid, dof in backend.query("sketch_dof").items():
            if dof < 0:
                diags.append(Diagnostic(
                    Severity.ERROR, "over-constrained",
                    f"sketch {sid} is over-constrained (dof={dof})", sid))
            elif dof > 0:
                diags.append(Diagnostic(
                    Severity.WARNING, "under-constrained",
                    f"sketch {sid} is under-constrained (dof={dof})", sid))
        return VerifyReport(diags)


class SolidPresenceCheck:
    """Once a feature has run, there must be a non-empty solid.

    Placeholder for the real B-rep manifold/watertight/self-intersection check
    that the CadQuery/OCCT backend will provide.
    """

    name = "solid-presence"

    def check(self, backend, opdag) -> VerifyReport:
        summary = backend.query("summary")
        if summary["feature_count"] > 0 and not summary["solid_present"]:
            return VerifyReport([Diagnostic(
                Severity.ERROR, "empty-solid",
                "features exist but no solid is present (degenerate build)")])
        return VerifyReport([])


def default_verifiers() -> List[Verifier]:
    # Lazy local import: checks_geometry imports names from this module, so a
    # top-level import here would be circular. BRepValidityCheck is a no-op when
    # the backend reports no solid/validity (e.g. the stub backend), so adding it
    # to the default set is safe for every backend.
    from verifiers.geometry import BRepValidityCheck
    return [SketchConstraintCheck(), SolidPresenceCheck(), BRepValidityCheck()]
