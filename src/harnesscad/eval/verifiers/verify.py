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
    #: Soundness tier of the rule that produced this diagnostic -- "proven",
    #: "measured" or "heuristic" (harnesscad.eval.verifiers.soundness). Stamped
    #: by the fleet dispatcher, which knows the emitting verifier. `None` means
    #: "not stamped"; soundness.tier_of then falls back to the code index and,
    #: failing that, to HEURISTIC. Only PROVEN/MEASURED diagnostics are fed back
    #: into a model's retry prompt: a wrong instruction is worse than none.
    #:
    #: Deliberately absent from `to_dict`: the wire format is what the pressure
    #: experiment recorded, and it stays byte-identical.
    soundness: Optional[str] = None

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
    """Flag over-constrained (error) and under-constrained (warning) sketches.

    Soundness: PROVEN for ``over-constrained`` (a negative DOF count means more
    independent constraints than degrees of freedom, so the system has no
    solution -- a theorem, not a guess), MEASURED for ``under-constrained`` (it
    reports the solver's own DOF count).

    Both messages state the observation and its evidence; the imperative, if
    any, is a trailing SUGGESTION. See `verifiers.soundness.observe`.
    """

    name = "sketch-constraint"

    def check(self, backend, opdag) -> VerifyReport:
        from harnesscad.eval.verifiers.soundness import observe

        diags: List[Diagnostic] = []
        for sid, dof in backend.query("sketch_dof").items():
            if dof < 0:
                diags.append(Diagnostic(
                    Severity.ERROR, "over-constrained",
                    observe(
                        f"sketch {sid} is over-constrained (dof={dof})",
                        f"the solver reports {-dof} more independent constraint(s) "
                        f"than the sketch has degrees of freedom, so no assignment "
                        f"of the sketch variables satisfies all of them",
                        "remove or relax the redundant constraint(s) until dof >= 0"),
                    sid))
            elif dof > 0:
                diags.append(Diagnostic(
                    Severity.WARNING, "under-constrained",
                    observe(
                        f"sketch {sid} is under-constrained (dof={dof})",
                        f"the solver reports {dof} unpinned degree(s) of freedom; "
                        f"the geometry built is one of infinitely many solutions"),
                    sid))
        return VerifyReport(diags)


class SolidPresenceCheck:
    """Once a feature has run, there must be a non-empty solid.

    Placeholder for the real B-rep manifold/watertight/self-intersection check
    that the CadQuery/OCCT backend will provide.

    Soundness: MEASURED. It relays what the backend answered -- features were
    applied, and the backend reports no solid. It infers nothing.
    """

    name = "solid-presence"

    def check(self, backend, opdag) -> VerifyReport:
        from harnesscad.eval.verifiers.soundness import observe

        summary = backend.query("summary")
        if summary["feature_count"] > 0 and not summary["solid_present"]:
            return VerifyReport([Diagnostic(
                Severity.ERROR, "empty-solid",
                observe(
                    "features exist but no solid is present (degenerate build)",
                    f"{summary['feature_count']} feature(s) were applied and the "
                    f"backend reports solid_present=False: the build produced "
                    f"empty geometry"))])
        return VerifyReport([])


def default_verifiers() -> List[Verifier]:
    # Lazy local import: checks_geometry imports names from this module, so a
    # top-level import here would be circular. BRepValidityCheck is a no-op when
    # the backend reports no solid/validity (e.g. the stub backend), so adding it
    # to the default set is safe for every backend.
    from harnesscad.eval.verifiers.geometry import BRepValidityCheck
    return [SketchConstraintCheck(), SolidPresenceCheck(), BRepValidityCheck()]
