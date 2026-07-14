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

from harnesscad.eval.verifiers import soundness as _soundness

#: The diagnostic wire format is VERSIONED, explicitly, because it crosses four
#: JSON boundaries (MCP, A2A, the JSONL tracer, the pressure experiment's
#: results file) and one of them is frozen.
#:
#: v1 -- severity/code/message/where. What `assets/pressure/results.json`
#:       recorded. Reproduce it with :meth:`Diagnostic.to_dict_v1`.
#: v2 -- v1 + `soundness`, the RESOLVED tier of the rule that spoke.
#:
#: v1 omitted `soundness`, and that omission was a bug in a fix: the whole point
#: of soundness tiering is that only PROVEN/MEASURED diagnostics may instruct a
#: model, and the tier evaporated at every serialization boundary. A remote MCP
#: client could not tell a theorem from a guess. v2 is the default because a
#: correct tier on the wire is worth more than a byte-identical wire.
DIAGNOSTIC_WIRE_VERSION = 2
DIAGNOSTIC_WIRE_V1_KEYS = ("severity", "code", "message", "where")


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
    soundness: Optional[str] = None

    def to_dict(self) -> dict:
        """The v2 wire form: v1 plus the RESOLVED soundness tier.

        `soundness` is never None on the wire. An unstamped diagnostic is
        resolved through `soundness.tier_of`, which falls back to the code index
        and then to HEURISTIC -- failing closed. A consumer on the far side of a
        JSON boundary can therefore apply the same gate the in-process planner
        applies, which is the whole point of the tier.
        """
        d = self.to_dict_v1()
        d["soundness"] = _soundness.tier_of(self)
        return d

    def to_dict_v1(self) -> dict:
        """The FROZEN v1 wire form, byte-identical to what the pressure run recorded.

        Kept so a re-run of `eval/pressure` can be compared against
        `assets/pressure/results.json` key-for-key. This is the only place the
        old format is promised; the type no longer holds the experiment hostage.
        """
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
                # INFO, not WARNING. It is a true fact (MEASURED) and it is not
                # a defect: an op stream that emits no `constrain` ops -- which
                # is nearly every op stream a model writes -- builds EXACTLY the
                # geometry it specified, with the sketch left unpinned. This
                # fired on 7 of 7 known-good parts. A diagnostic that flags every
                # correct part carries no information about correctness, and at
                # WARNING severity it reached the model and told it to go and
                # constrain a sketch that was already right.
                diags.append(Diagnostic(
                    Severity.INFO, "under-constrained",
                    observe(
                        f"sketch {sid} is under-constrained (dof={dof})",
                        f"the solver reports {dof} unpinned degree(s) of freedom; "
                        f"the geometry built is one of infinitely many solutions. "
                        f"This is a note, not a defect: the built geometry is the "
                        f"one the ops specified"),
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
