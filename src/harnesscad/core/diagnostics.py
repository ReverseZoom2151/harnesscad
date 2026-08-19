"""Diagnostic protocol types -- the op contract's failure vocabulary.

``Severity``, ``Diagnostic`` and ``VerifyReport`` are PROTOCOL types, not
evaluators. A diagnostic is part of what an op application *returns*: the CISP
protocol, the contract, the environment, the harness and the loop all speak it,
and none of them scores anything. They lived in
``harnesscad.eval.verifiers.verify`` for historical reasons only -- because the
first producer of a diagnostic happened to be a verifier -- and that placement
put five inward-layer modules on the wrong side of the layer line
(``tests/test_layering.py``).

They now live here, in ``core``, next to the rest of the op contract.
``harnesscad.eval.verifiers.verify`` re-exports them unchanged, so every
existing ``from harnesscad.eval.verifiers.verify import Diagnostic`` keeps
working. The verifier LOGIC stays in ``eval`` -- only the vocabulary moved.

Stdlib-only, deterministic, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Protocol

__all__ = [
    "DIAGNOSTIC_WIRE_VERSION",
    "DIAGNOSTIC_WIRE_V1_KEYS",
    "Severity",
    "Diagnostic",
    "VerifyReport",
    "Verifier",
]

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

#: The tier an unresolvable diagnostic takes. Mirrors
#: ``eval.verifiers.soundness.HEURISTIC`` and exists only so this module can
#: fail CLOSED when the eval layer is not installed at all (a core-only
#: distribution): an untiered diagnostic must never be mistaken for a trusted
#: one. When eval IS present -- which is every in-tree run -- the real resolver
#: is used and this constant is never reached.
_UNTRUSTED_TIER = "heuristic"


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

        The resolver lives in the eval layer (it is a property of the verifier
        fleet, not of the protocol), so it is imported LAZILY here: core must not
        depend on eval at import time. If eval is absent the tier fails closed to
        HEURISTIC, exactly as an unrecognised code does.
        """
        d = self.to_dict_v1()
        try:
            from harnesscad.eval.verifiers.soundness import tier_of
        except ImportError:  # pragma: no cover - core installed without eval
            d["soundness"] = self.soundness or _UNTRUSTED_TIER
        else:
            d["soundness"] = tier_of(self)
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
    """The SHAPE of a verifier -- a structural type, carrying no logic.

    It lives with the diagnostic vocabulary because it is the other half of the
    same contract: a verifier is anything that reads a backend and answers with
    a :class:`VerifyReport`. The verifier IMPLEMENTATIONS (and the default
    fleet) stay in ``harnesscad.eval.verifiers``; core only needs to be able to
    name the shape of the collaborators it is handed.
    """

    name: str

    def check(self, backend, opdag) -> VerifyReport: ...
