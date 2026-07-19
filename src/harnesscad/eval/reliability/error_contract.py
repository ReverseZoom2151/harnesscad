"""Structured tool-error contract feeding the repair loop's retry-vs-abstain call.

A ``ToolError{error_type, message, recoverable, suggested_action}`` contract, a
``normalize_error_type`` that disambiguates by MESSAGE TOKENS (the same raw
exception name can mean "unsupported geometry" or "ambiguous request" depending
on what the message says), and a ``suggested_action`` table of terse recovery
hints per public error type.

The point of the contract is the ``recoverable`` bit: the repair loop
(``eval/reliability/repair_loop.py``) iterates detect -> repair -> re-check,
but nothing today tells it which failures are WORTH re-entering the loop for
and which should abstain immediately (backend missing, ambiguous brief,
artifact I/O). This module supplies that decision:

  * :func:`normalize_error_type` maps internal exception names + message
    tokens to a stable public error type;
  * :func:`suggested_action_for_error` returns the recovery hint;
  * :func:`recoverability_for_error` marks types retryable or not;
  * :func:`repair_decision` collapses a ``ToolError`` to ``"retry"`` /
    ``"abstain"`` for the loop driver;
  * :func:`from_code_error` EXTENDS (imports, does not rewrite) the harness's
    tiny ``code_error.CodeError`` normalizer, lifting its 4 categories into
    the full contract.

The contract is a stdlib dataclass keyed by message tokens over the CAD
harness's vocabulary. Pure stdlib, deterministic; no kernel, no model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Sequence

from harnesscad.eval.reliability.code_error import CodeError

#: The public contract error types.
STANDARD_ERROR_TYPES: tuple = (
    "UnsupportedObjectError",
    "UnsupportedGeometryError",
    "InvalidParameterError",
    "ValidationFailedError",
    "CadBackendUnavailableError",
    "CadGenerationError",
    "EditRejectedError",
    "AmbiguousRequestError",
    "ArtifactError",
    "LLMProviderUnavailableError",
    "InternalError",
)

#: Message tokens that flip a rejection into "the request itself is ambiguous"
#: (UnsupportedEditError disambiguation).
_AMBIGUITY_TOKENS = ("measurable", "ambiguous", "optimize", "better",
                     "stronger", "cheaper", "under-specified", "unclear")

#: Message tokens that flip a rejection into "geometry class unsupported".
_GEOMETRY_TOKENS = ("curved", "adjustable", "sheet-metal", "freeform",
                    "coordinates", "non-manifold", "self-intersect")


@dataclass(frozen=True)
class ToolError:
    """Structured external error object for tool/API responses."""

    error_type: str
    message: str
    recoverable: bool = True
    suggested_action: str = ""

    def to_dict(self) -> dict:
        return {
            "error_type": self.error_type,
            "message": self.message,
            "recoverable": self.recoverable,
            "suggested_action": self.suggested_action,
        }


def normalize_error_type(error_type: str, message: str = "") -> str:
    """Map internal exception names/messages to public contract error types.

    A token-disambiguation ladder, with the CAD-harness kernel vocabulary added
    (OCCT/StdFail names -> generation errors).
    """
    lowered = (message or "").lower()
    if error_type in {"CadQueryUnavailableError", "ImportError",
                      "ModuleNotFoundError"}:
        return "CadBackendUnavailableError"
    if error_type == "LLMProviderUnavailableError":
        return "LLMProviderUnavailableError"
    if error_type == "UnsupportedObjectError":
        if any(token in lowered for token in _GEOMETRY_TOKENS):
            return "UnsupportedGeometryError"
        return "UnsupportedObjectError"
    if error_type == "UnsupportedEditError":
        if any(token in lowered for token in _AMBIGUITY_TOKENS):
            return "AmbiguousRequestError"
        if any(token in lowered for token in _GEOMETRY_TOKENS):
            return "UnsupportedGeometryError"
        return "EditRejectedError"
    if error_type in {"EditRejected", "EditRejectedError"}:
        return "EditRejectedError"
    if error_type in {"ValueError", "TypeError"}:
        return "InvalidParameterError"
    if error_type in {"FileNotFoundError", "OSError", "JSONDecodeError"}:
        return "ArtifactError"
    if error_type == "ValidationFailedError":
        return "ValidationFailedError"
    # Already-public types pass through unchanged (idempotent normalize).
    if error_type in STANDARD_ERROR_TYPES:
        return error_type
    # CAD-harness extension: kernel construction failures are generation
    # errors (retryable), not internal faults.
    if error_type in {"StdFail_NotDone", "BooleanKernelError", "KernelError",
                      "GeometryError"} or "brep_api" in lowered:
        return "CadGenerationError"
    return "InternalError" if error_type.endswith("Error") else error_type


def suggested_action_for_error(error_type: str, message: str = "") -> str:
    """Terse recovery hint per public error type (CAD hint table)."""
    normalized = normalize_error_type(error_type, message)
    table = {
        "CadBackendUnavailableError":
            "Install or configure the CAD backend before retrying; the "
            "repair loop cannot fix a missing kernel.",
        "UnsupportedObjectError":
            "Use one of the supported model families for this backend.",
        "UnsupportedGeometryError":
            "Restrict the request to the supported deterministic geometry "
            "options for the selected model family.",
        "InvalidParameterError":
            "Provide numeric dimensions and constraints within the supported "
            "parameter ranges.",
        "AmbiguousRequestError":
            "Ask a clarifying question or request a measurable parameter; "
            "do not guess a dimension.",
        "EditRejectedError":
            "Revise the edit so it preserves supported constraints and "
            "feature intent.",
        "ValidationFailedError":
            "Review the failed validation checks and adjust parameters "
            "before exporting CAD.",
        "ArtifactError":
            "Check that the requested artifact or run metadata path exists "
            "and is readable.",
        "LLMProviderUnavailableError":
            "Configure an LLM provider or use the deterministic mock "
            "provider for local testing.",
        "CadGenerationError":
            "Adjust the failing operation's parameters (overlap, radius, "
            "tolerance) and re-enter the repair loop.",
    }
    return table.get(normalized,
                     "Inspect the response metadata and retry with a "
                     "narrower supported request.")


#: Which public types the repair loop should re-enter for. Environment and
#: intent problems abstain; parameter/validation/generation problems retry.
_RECOVERABLE = {
    "InvalidParameterError": True,
    "ValidationFailedError": True,
    "EditRejectedError": True,
    "CadGenerationError": True,
    "UnsupportedObjectError": False,
    "UnsupportedGeometryError": False,
    "CadBackendUnavailableError": False,
    "AmbiguousRequestError": False,
    "ArtifactError": False,
    "LLMProviderUnavailableError": False,
    "InternalError": False,
}


def recoverability_for_error(error_type: str, message: str = "") -> bool:
    return _RECOVERABLE.get(normalize_error_type(error_type, message), False)


def tool_error(
    error_type: str,
    message: str,
    recoverable: Optional[bool] = None,
    suggested_action: Optional[str] = None,
) -> ToolError:
    """Build a fully-populated structured error object."""
    normalized = normalize_error_type(error_type, message)
    return ToolError(
        error_type=normalized,
        message=message or normalized,
        recoverable=(recoverable if recoverable is not None
                     else recoverability_for_error(error_type, message)),
        suggested_action=(suggested_action if suggested_action is not None
                          else suggested_action_for_error(error_type, message)),
    )


def from_exception(exc: BaseException, message: str = "") -> ToolError:
    """Contract object straight from a raised exception."""
    return tool_error(type(exc).__name__, message or str(exc))


def from_code_error(err: CodeError) -> ToolError:
    """Lift the harness's 4-category ``code_error.CodeError`` into the contract.

    ``code_error.normalize`` buckets exceptions as type/syntax/value/kernel;
    this maps those buckets onto the public contract WITHOUT rewriting the
    existing normalizer.
    """
    category_to_type = {
        "type": "InvalidParameterError",
        "value": "InvalidParameterError",
        "syntax": "CadGenerationError",
        "kernel": "CadGenerationError",
    }
    etype = category_to_type.get(err.category, "InternalError")
    msg_bits = [f"{err.category} error"]
    if err.operation:
        msg_bits.append(f"in {err.operation}")
    if err.expected:
        msg_bits.append(f"(expected {err.expected})")
    message = " ".join(msg_bits)
    action = err.hint or suggested_action_for_error(etype, message)
    return ToolError(
        error_type=etype,
        message=message,
        recoverable=recoverability_for_error(etype, message),
        suggested_action=action,
    )


def repair_decision(error: ToolError) -> str:
    """Collapse a contract error to the repair loop's branch: retry or abstain.

    ``"retry"`` means re-enter ``repair_loop.repair_until_feasible`` (the
    failure is parameter/geometry-shaped and a repair iteration can change
    the outcome); ``"abstain"`` means stop and surface the suggested_action
    (environment, intent, or I/O problems that iteration cannot fix).
    """
    return "retry" if error.recoverable else "abstain"


# --------------------------------------------------------------------------- #
# selfcheck
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Structured ToolError contract + retry-vs-abstain "
                    "decision (IntentForge contracts/errors.py port).",
    )
    parser.add_argument("--selfcheck", action="store_true",
                        help="exercise normalization, token disambiguation, "
                             "the hint table, and the repair decision.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0

    # 1. Message-token disambiguation (the source's signature behavior).
    assert normalize_error_type("UnsupportedEditError",
                                "make it stronger") == "AmbiguousRequestError"
    assert normalize_error_type("UnsupportedEditError",
                                "curved flange") == "UnsupportedGeometryError"
    assert normalize_error_type("UnsupportedEditError",
                                "shrink base") == "EditRejectedError"
    assert normalize_error_type("ModuleNotFoundError",
                                "") == "CadBackendUnavailableError"
    assert normalize_error_type("ValueError", "") == "InvalidParameterError"
    assert normalize_error_type("StdFail_NotDone",
                                "boolean failed") == "CadGenerationError"
    assert normalize_error_type("WeirdError", "") == "InternalError"
    assert normalize_error_type("not_an_error", "") == "not_an_error"
    print("[selfcheck] normalize_error_type: token disambiguation OK")

    # 2. Every standard type has a non-default suggested action.
    fallback = suggested_action_for_error("not_an_error")
    for etype in STANDARD_ERROR_TYPES:
        hint = suggested_action_for_error(etype)
        assert hint, etype
        if etype != "InternalError":
            assert hint != fallback, etype
    print(f"[selfcheck] suggested_action table covers "
          f"{len(STANDARD_ERROR_TYPES)} types")

    # 3. Retry-vs-abstain feeding the repair loop.
    retry = tool_error("ValueError", "radius must be positive")
    assert retry.recoverable and repair_decision(retry) == "retry"
    abstain = tool_error("ModuleNotFoundError", "no backend")
    assert not abstain.recoverable and repair_decision(abstain) == "abstain"
    ambiguous = tool_error("UnsupportedEditError", "make it better")
    assert repair_decision(ambiguous) == "abstain"
    print("[selfcheck] repair_decision: retry/abstain split OK")

    # 4. from_code_error lifts the existing normalizer without rewriting it.
    from harnesscad.eval.reliability.code_error import normalize as ce_normalize
    ce = ce_normalize(ValueError("bad"), operation="extrude",
                      signature="extrude(distance: float)")
    lifted = from_code_error(ce)
    assert lifted.error_type == "InvalidParameterError", lifted
    assert "extrude" in lifted.message
    assert lifted.suggested_action == ce.hint  # existing hint is preserved
    assert repair_decision(lifted) == "retry"
    kernel = from_code_error(ce_normalize(RuntimeError("BRep_API"), "fillet"))
    assert kernel.error_type == "CadGenerationError"
    print("[selfcheck] from_code_error extends code_error via import")

    # 5. from_exception round-trip and determinism.
    e1 = from_exception(TypeError("wrong arity")).to_dict()
    e2 = from_exception(TypeError("wrong arity")).to_dict()
    assert e1 == e2 and e1["error_type"] == "InvalidParameterError"
    print("[selfcheck] deterministic contract objects")
    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
