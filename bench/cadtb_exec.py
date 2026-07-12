"""String-based CADTEST execution harness (Mallis et al., "Text-to-CAD
Evaluation with CADTESTS"; implementation-level pieces from the reference
``cadtestbench.metrics.cadtest`` module).

The paper-level runner (:mod:`bench.cadtests_runner`) evaluates CADTESTS that are
already Python callables over an in-memory B-rep model. The *reference
implementation*, however, ships each CADTEST as an opaque **source-code string**
that is executed against a live model recovered from generated CAD code. Turning
that representation into a deterministic outcome requires four implementation
pieces the callable-predicate runner never needed, all reproduced here in
stdlib-only, deterministic form:

  * :func:`extract_model_var_name` -- static (AST) recovery of the variable name
    handed to ``cq.exporters.export(<var>, ...)`` inside generated model code, so
    the produced model object can be located after execution without running the
    exporter.
  * :func:`strip_export_calls` -- remove the exporter side-effect lines (which
    would write mesh files) before executing the model source.
  * :func:`execute_cadtest` -- run one CADTEST *code string* against a namespace
    using a ``check(condition, pass_msg, fail_msg)`` assertion primitive that
    yields an interpretable message in both the pass and fail cases, capturing an
    ``AssertionError`` as a clean failure and any other exception as an
    ``exception``-tagged failure (never propagating).
  * :func:`run_cadtest_block` -- execute a model's source in a caller-supplied
    namespace, recover the model variable, expose it as ``final_result`` and run
    every CADTEST string against it; a model that fails to execute fails every
    CADTEST (invalid generation, Sec. 5).

:func:`build_replay_script` additionally reproduces the reference debug bundle: a
single self-contained script that re-runs the model then each CADTEST under a
results tracker.

The harness is model-kernel agnostic: the caller supplies the execution
namespace (e.g. ``{"cq": cadquery}`` or, in tests, a lightweight stand-in
object), exactly mirroring the injected-model design of
:mod:`bench.cadtests_model`. Deterministic, stdlib-only: no wall clock, no
randomness, no file writes.
"""

from __future__ import annotations

import ast
import contextlib
import io
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# The check() assertion primitive and its exec preamble.
# ---------------------------------------------------------------------------
# A CADTEST string calls ``check(condition, pass_msg, fail_msg)``. The primitive
# records the pass message on success and raises AssertionError(fail_msg) on
# failure, so a single executor path produces an interpretable message either
# way. ``math`` is made available because generated CADTESTS routinely use it.
ASSERTION_EXEC_PREAMBLE = (
    "import math\n"
    "_check_pass_msg = None\n"
    "def check(condition, pass_msg, fail_msg):\n"
    "    global _check_pass_msg\n"
    "    if not condition:\n"
    "        raise AssertionError(fail_msg)\n"
    "    _check_pass_msg = pass_msg\n"
)

_DEFAULT_MODEL_VAR = "final_result"


def make_check(record: Dict[str, Any]) -> Callable[[Any, Any, Any], None]:
    """Return a native ``check(condition, pass_msg, fail_msg)`` primitive.

    The last successful pass-message is stored under ``record['pass_msg']``. On a
    false condition an :class:`AssertionError` carrying ``fail_msg`` is raised.
    This is the callable equivalent of :data:`ASSERTION_EXEC_PREAMBLE` for
    callers that would rather inject the primitive than prepend source.
    """
    def check(condition, pass_msg, fail_msg):
        if not condition:
            raise AssertionError(fail_msg)
        record["pass_msg"] = pass_msg
    return check


# ---------------------------------------------------------------------------
# Static recovery of the exported model variable.
# ---------------------------------------------------------------------------
def extract_model_var_name(code: str, filename: str = "<generated_model>",
                           default: str = _DEFAULT_MODEL_VAR) -> str:
    """Return the variable name passed as the first argument to
    ``cq.exporters.export(<var>, ...)`` in ``code``.

    The whole model source is parsed once and every call node inspected for the
    ``cq.exporters.export`` attribute chain; the ``id`` of the first ``Name``
    first-argument is returned. When no such call is present (or its first
    argument is not a bare name), ``default`` is returned. Raises
    :class:`SyntaxError` only if ``code`` does not parse.
    """
    tree = ast.parse(code, filename)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (isinstance(func, ast.Attribute)
                and func.attr == "export"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "exporters"
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "cq"):
            if node.args and isinstance(node.args[0], ast.Name):
                return node.args[0].id
    return default


def strip_export_calls(code: str, marker: str = "cq.exporters.export") -> str:
    """Drop every source line containing ``marker`` (default the exporter call).

    Executing generated model code verbatim would trigger the exporter and write
    a mesh file; removing those lines makes execution a pure, side-effect-free
    reconstruction of the model object. Line-based to match the reference
    implementation exactly.
    """
    return "\n".join(line for line in code.splitlines() if marker not in line)


# ---------------------------------------------------------------------------
# Per-CADTEST execution.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CadTestOutcome:
    """Structured outcome of executing one CADTEST string.

    ``status``     ``"pass"`` or ``"fail"``.
    ``message``    interpretable log line (pass message, assertion message, or
                   error text).
    ``exception``  the exception type name when a non-assertion error occurred,
                   else ``None``.
    ``cadtest_id`` optional identifier carried through from the caller.
    """
    status: str
    message: Optional[str]
    exception: Optional[str] = None
    cadtest_id: Any = None

    @property
    def passed(self) -> bool:
        return self.status == "pass"


def execute_cadtest(code: str, env: Dict[str, Any], *,
                    cadtest_id: Any = None) -> CadTestOutcome:
    """Execute one CADTEST ``code`` string against namespace ``env``.

    ``env`` must already expose the model (conventionally as ``final_result``).
    The :data:`ASSERTION_EXEC_PREAMBLE` is prepended so the CADTEST may call
    ``check(...)``. A raised :class:`AssertionError` becomes a clean ``fail``; any
    other exception becomes a ``fail`` tagged with its type name. On success the
    captured ``check`` pass-message is returned. Never propagates.
    """
    env["_check_pass_msg"] = None
    try:
        exec(ASSERTION_EXEC_PREAMBLE + code, env)
        return CadTestOutcome("pass", env.get("_check_pass_msg"),
                              None, cadtest_id)
    except AssertionError as exc:
        return CadTestOutcome("fail", str(exc) or "cadtest failed",
                              "AssertionError", cadtest_id)
    except Exception as exc:  # noqa: BLE001 -- any runtime error => fail
        return CadTestOutcome("fail", "%s: %s" % (type(exc).__name__, exc),
                              type(exc).__name__, cadtest_id)


# ---------------------------------------------------------------------------
# Whole-block execution: model then every CADTEST.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BlockResult:
    """Outcome of running a model's source plus its CADTEST suite."""
    outcomes: Tuple[CadTestOutcome, ...]
    model_error: Optional[str]        # non-None iff the model failed to execute
    stdout: str                       # captured (not echoed) model/test output

    @property
    def model_failed(self) -> bool:
        return self.model_error is not None

    @property
    def num_passed(self) -> int:
        return sum(1 for o in self.outcomes if o.passed)

    @property
    def passed_all(self) -> bool:
        return bool(self.outcomes) and self.num_passed == len(self.outcomes)


def _normalize_cadtests(
    cadtests: Sequence[Any],
) -> List[Tuple[Any, str]]:
    """Normalise a CADTEST suite to ``(id, code)`` pairs.

    Each element may be a bare code ``str`` (id defaults to its position) or a
    mapping with ``cadtest_code`` (or ``code``) and optional ``cadtest_id`` (or
    ``id``), matching the reference dataset row shape.
    """
    pairs: List[Tuple[Any, str]] = []
    for i, item in enumerate(cadtests):
        if isinstance(item, str):
            pairs.append((i, item))
        elif isinstance(item, dict):
            code = item.get("cadtest_code")
            if code is None:
                code = item.get("code")
            cid = item.get("cadtest_id", item.get("id", i))
            pairs.append((cid, "" if code is None else str(code)))
        else:
            raise TypeError("cadtest must be str or dict, got %r" % (type(item),))
    return pairs


def run_cadtest_block(model_source: str, cadtests: Sequence[Any], *,
                      base_env: Optional[Dict[str, Any]] = None,
                      filename: str = "<generated_model>") -> BlockResult:
    """Execute ``model_source`` then run every CADTEST string against it.

    ``base_env`` seeds the execution namespace (e.g. ``{"cq": cadquery}`` or a
    stand-in object bound to ``cq``); it is shallow-copied so the caller's dict is
    never mutated. The exported model variable is recovered statically, the
    exporter lines are stripped, the sanitised source executed, and the recovered
    variable exposed as ``final_result`` before each CADTEST runs.

    If the model source fails to parse or execute, or produces no such variable,
    every CADTEST fails with a ``ModelExecError`` outcome (invalid generation).
    All model/test stdout is captured, never echoed. Deterministic.
    """
    pairs = _normalize_cadtests(cadtests)
    capture = io.StringIO()

    def _fail_all(reason: str) -> BlockResult:
        outs = tuple(
            CadTestOutcome("fail", reason, "ModelExecError", cid)
            for cid, _ in pairs
        )
        return BlockResult(outs, reason, capture.getvalue())

    try:
        model_var = extract_model_var_name(model_source, filename)
    except SyntaxError as exc:
        return _fail_all("model parse error: %s: %s"
                         % (type(exc).__name__, exc))

    env: Dict[str, Any] = dict(base_env) if base_env else {}
    env.setdefault("__name__", "__main__")

    sanitized = strip_export_calls(model_source)
    with contextlib.redirect_stdout(capture):
        try:
            exec(compile(sanitized, filename, "exec"), env)
        except Exception as exc:  # noqa: BLE001
            return _fail_all("model exec error: %s: %s"
                             % (type(exc).__name__, exc))

        if model_var not in env:
            return _fail_all(
                "model exec produced no variable named %r "
                "(was the export call removed?)" % model_var)

        env["final_result"] = env[model_var]
        outcomes = tuple(
            execute_cadtest(code, env, cadtest_id=cid) for cid, code in pairs
        )

    return BlockResult(outcomes, None, capture.getvalue())


# ---------------------------------------------------------------------------
# Self-contained replay-script generation.
# ---------------------------------------------------------------------------
def _indent_block(text: str, spaces: int = 4) -> str:
    pad = " " * spaces
    return "\n".join("%s%s" % (pad, line) for line in text.splitlines())


def build_replay_script(model_code: str, cadtests: Sequence[Any]) -> str:
    """Build a self-contained replay script (model + CADTESTS + tracker).

    The script executes the (export-stripped) model; on a model error it records
    every CADTEST as failed, otherwise it runs each CADTEST under its own
    ``try/except`` accumulating pass/fail ids per category into a
    ``cadtest_results_tracker`` dict. This mirrors the reference debug bundle so a
    failing sample can be re-run in isolation. Pure string construction.
    """
    pairs = _normalize_cadtests(cadtests)
    categories = []
    for item in cadtests:
        if isinstance(item, dict):
            categories.append(str(item.get("cadtest_type") or "uncategorized"))
        else:
            categories.append("uncategorized")

    body = strip_export_calls(model_code)
    try:
        model_var = extract_model_var_name(model_code)
        if model_var != _DEFAULT_MODEL_VAR:
            body = "%s\n%s = %s" % (body, _DEFAULT_MODEL_VAR, model_var)
    except SyntaxError:
        pass
    model_body = _indent_block(body)
    parts = [
        'cadtest_results_tracker = {"total_test": 0, "passed": [], '
        '"failed": [], "categories": {}, "model_compile_error": False}\n'
        "try:\n%s\nexcept Exception as e:" % model_body
    ]

    for (cid, _code), cat in zip(pairs, categories):
        parts.append(
            '\n    cadtest_results_tracker["total_test"] += 1'
            '\n    cadtest_results_tracker["categories"].setdefault(%r, '
            '{"passed": [], "failed": []})'
            '\n    cadtest_results_tracker["failed"].append(%r)'
            '\n    cadtest_results_tracker["categories"][%r]["failed"].append(%r)'
            % (cat, cid, cat, cid))

    parts.append('\n    cadtest_results_tracker["model_compile_error"] = '
                 'True\nelse:\n')

    for (cid, code), cat in zip(pairs, categories):
        parts.append(
            "\n    try:"
            '\n        cadtest_results_tracker["total_test"] += 1'
            '\n        cadtest_results_tracker["categories"].setdefault(%r, '
            '{"passed": [], "failed": []})'
            "\n%s"
            '\n        cadtest_results_tracker["passed"].append(%r)'
            '\n        cadtest_results_tracker["categories"][%r]["passed"]'
            ".append(%r)"
            "\n    except Exception:"
            '\n        cadtest_results_tracker["failed"].append(%r)'
            '\n        cadtest_results_tracker["categories"][%r]["failed"]'
            ".append(%r)\n"
            % (cat, _indent_block(code, spaces=8), cid, cat, cid, cid, cat,
               cid))

    return "".join(parts)
