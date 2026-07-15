"""End-to-end build pipeline: brief -> planner (LLM) -> session -> STEP.

`build` is the single entry point that wires the whole harness together:

    natural-language brief
        -> Planner (LLM turns the brief into validated CISP ops)
        -> HarnessSession over a geometry backend (apply -> verify -> checkpoint)
        -> verified real geometry
        -> STEP export

It is deliberately thin: every hard part already lives in a tested module
(`core.harness.AgentHarness` drives the correction loop, `HarnessSession` owns
the transactional spine, the backends produce geometry + digests). This module
just assembles those pieces and normalises the outcome into a plain result dict.

THE SHIPPING PATH IS THE GOOD PATH. This module used to construct the weakest of
the repository's agent loops (`agent.runner.run`: no loop detection, no context
pre-flight, no write gate, no contract, no trajectory) while the fully-configured
`AgentHarness` was reachable only through the ACP editor integration. It now
builds the same harness every other surface builds: loop detection, the
guardrail + human-approval write gate, the soundness feedback gate, context
pre-flight and a replayable trajectory, all on by default.

Provider keys are never read or hardcoded here beyond passing a model name to
`LiteLLMClient`. When no `llm` is injected and no API key is present in the
environment, `build` raises a clear, actionable error instead of failing deep
inside a provider call.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from harnesscad.agents.agent.planner import Planner
from harnesscad.agents.agent.trace_reward import (
    first_divergence,
    reward_for_session,
    step_accuracy,
)
from harnesscad.agents.context.manager import ContextManager
from harnesscad.core.harness import AgentHarness
from harnesscad.io.backends.stub import StubBackend
from harnesscad.agents.llm.base import LLM
from harnesscad.core.loop import HarnessSession
from harnesscad.core.trace import Tracer
from harnesscad.eval.reliability.loopdetect import LoopDetector


# The default model used only when the caller injects no `llm`. Kept as a
# module constant so it is obvious and easy to override via `model=`.
DEFAULT_MODEL = "anthropic/claude-sonnet-4-5"

# Env vars we consider proof that a live provider call can succeed. We only
# *check* for their presence; we never read their values (litellm does that).
_API_KEY_ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")

#: The context window the pre-flight budgets against. Conservative on purpose:
#: the counter is still a heuristic (see audit gap #8), so the budget it guards
#: is stated smaller than any model we route to.
CONTEXT_BUDGET_TOKENS = 100_000

#: Env var pointing at the persistent memory store, so a shipping run gets
#: session continuity without any edit to the CLI (it just sets/exports this).
#: An explicit ``memory_path=`` argument to ``build`` wins over it.
MEMORY_PATH_ENV = "HARNESSCAD_MEMORY"

_LOG = logging.getLogger(__name__)


class BuildError(RuntimeError):
    """Raised when the pipeline cannot even start (e.g. no LLM and no API key)."""


class _LazyLiteLLM(LLM):
    """An `LLM` that constructs its `LiteLLMClient` on first use.

    Constructing this costs nothing and imports no provider SDK; the real
    `LiteLLMClient` (and litellm itself) is built only if/when the planner
    actually calls `complete`/`stream`. That keeps `build` cheap and import-safe
    right up to the first live model call.
    """

    def __init__(self, model: str) -> None:
        self._model = model
        self._client: Optional[LLM] = None

    def _ensure(self) -> LLM:
        if self._client is None:
            from harnesscad.agents.llm.litellm_backend import LiteLLMClient  # lazy import
            self._client = LiteLLMClient(self._model)
        return self._client

    def complete(self, messages, tools=None, response_schema=None, **opts):
        return self._ensure().complete(
            messages, tools=tools, response_schema=response_schema, **opts)

    def stream(self, messages, tools=None, response_schema=None, **opts):
        return self._ensure().stream(
            messages, tools=tools, response_schema=response_schema, **opts)


def _make_backend(backend: str):
    """Return (backend_instance, resolved_name, note).

    `cadquery` is used when requested *and* importable; otherwise we fall back to
    the dependency-free stub and report why in `note`.
    """
    if backend == "cadquery":
        try:
            # CadQueryBackend itself is intentionally import-safe without the
            # optional kernel. Probe the dependency here so the pipeline does
            # not select a backend that can construct but cannot execute.
            import cadquery  # noqa: F401
            from harnesscad.io.backends.cadquery import CadQueryBackend  # type: ignore
            return CadQueryBackend(), "cadquery", None
        except Exception as exc:  # pragma: no cover - depends on optional dep
            return (StubBackend(), "stub",
                    f"cadquery backend unavailable ({exc}); fell back to stub")
    if backend == "onshape":
        try:
            # The OnshapeBackend actuates Onshape geometry over its signed REST
            # API. It needs ONSHAPE_ACCESS_KEY / ONSHAPE_SECRET_KEY in the
            # environment (read there only, never entered/printed); absent them
            # its constructor raises BackendUnavailable and we fall back to stub.
            from harnesscad.io.backends.onshape import OnshapeBackend  # type: ignore
            return OnshapeBackend(), "onshape", None
        except Exception as exc:  # pragma: no cover - depends on live credentials
            return (StubBackend(), "stub",
                    f"onshape backend unavailable ({exc}); fell back to stub")
    return StubBackend(), "stub", None


def _route(llm: LLM) -> LLM:
    """Wrap a DEFAULT-constructed LLM in ``routing.RoutingLLM`` so retry/fallback
    and the cost/usage tally (blueprint sec.11/13) are on the shipping path.

    It is transparent: every task class maps to the same underlying client, so
    the same model answers every request and the call interface is unchanged; the
    router adds the classify -> sequential-fallback -> cost-tally spine around it.
    Only ever applied to the client this module builds (`_LazyLiteLLM`, which
    honours the full ``complete(messages, tools, response_schema, **opts)``
    signature the router dispatches with). An injected caller `llm` is handed
    through untouched — its narrower signature is the caller's contract, not ours.

    Never fatal: if routing cannot be constructed the raw client is returned and
    the reason is logged, so the build degrades to the pre-routing path.
    """
    try:
        from harnesscad.core.routing import RoutingLLM, TaskClass
        return RoutingLLM({
            TaskClass.CHEAP: llm,
            TaskClass.STANDARD: llm,
            TaskClass.HARD: llm,
        })
    except Exception as exc:  # noqa: BLE001 - routing is an enhancement, never a gate
        _LOG.warning("routing unavailable; using the raw LLM (%s)", exc)
        return llm


def _resolve_llm(llm: Optional[LLM], model: Optional[str]) -> LLM:
    """Pick the LLM to plan with: the injected one, or a lazy LiteLLM client.

    Raises `BuildError` if no `llm` is given and no provider API key is set —
    failing fast with a clear message instead of deep inside a provider call.

    Returns the bare client; the routing wrap is applied by `build` (only to the
    client we construct ourselves), so this helper's contract is unchanged.
    """
    if llm is not None:
        return llm
    if not any(os.environ.get(name) for name in _API_KEY_ENV_VARS):
        raise BuildError(
            "no LLM was provided and no provider API key is set. Set one of "
            + " or ".join(_API_KEY_ENV_VARS)
            + " in the environment, or pass an `llm=` implementing llm.base.LLM."
        )
    return _LazyLiteLLM(model or DEFAULT_MODEL)


def _default_memory_path() -> str:
    """A persistent store path under the same temp-scratch convention the external
    backends use (`io/backends/external.cache_dir`): repo-scoped, off the source
    tree, writable without ceremony."""
    return os.path.join(tempfile.gettempdir(), "harnesscad-memory", "store.json")


def _resolve_memory_path(memory_path: Optional[str]) -> str:
    """Explicit argument wins, then the env var, then the temp-scratch default."""
    if memory_path:
        return memory_path
    return os.environ.get(MEMORY_PATH_ENV) or _default_memory_path()


def _build_memory(memory_path: Optional[str]):
    """Construct a persistent ``HarnessMemory`` for the shipping agent.

    Returns ``(memory, path, note)``. On ANY failure returns ``(None, None, why)``
    so the caller degrades to the historical cold path (no memory) with a logged
    reason and never crashes the build. A store file that exists is loaded so a
    prior session's oracle-verified parts and lessons are recalled; a missing or
    unreadable file starts a fresh in-memory store that is saved at the end.
    """
    try:
        from harnesscad.agents.memory.harness_memory import HarnessMemory
    except Exception as exc:  # noqa: BLE001 - memory is optional grounding, not a gate
        return None, None, f"memory module unavailable ({exc})"
    path = _resolve_memory_path(memory_path)
    try:
        if path and os.path.exists(path):
            return HarnessMemory.load(path), path, None
        return HarnessMemory(), path, None
    except Exception as exc:  # noqa: BLE001 - a corrupt store must not break the build
        try:
            return HarnessMemory(), path, f"could not load {path!r}, started cold ({exc})"
        except Exception as exc2:  # noqa: BLE001
            return None, None, f"memory construction failed ({exc2})"


def _save_memory(memory: Any, path: Optional[str]) -> None:
    """Persist the store so the next session continues from it. Never fatal."""
    if memory is None or not path:
        return
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        memory.save(path)
    except Exception as exc:  # noqa: BLE001 - persistence failure never breaks a build
        _LOG.warning("memory not persisted to %s (%s)", path, exc)


def _export_step(backend, ok: bool) -> Optional[str]:
    """Best-effort STEP export. Returns the STEP text, or None if unavailable."""
    if not ok:
        return None
    export = getattr(backend, "export", None)
    if export is None:
        return None
    try:
        return export("step")
    except Exception:  # noqa: BLE001 - export must never break the result dict
        return None


#: The loop strategies `build` can run.
#:
#: "refine"     -- plan -> apply -> feed the (soundness-gated) diagnostics back ->
#:                 replan. The harness's historical single strategy, and the one
#:                 that LOST the controlled experiment in assets/pressure/report.md
#:                 by 8.3 points to blind resampling.
#: "best-of-n"  -- draw N independent plans, apply each in a FRESH session, and let
#:                 the deterministic verifier pick the winner. No feedback channel
#:                 at all, therefore no poisoning surface. `P(success) = 1 - (1-p)^N`.
#:                 `eval/reliability/strategies/best_of_n.py` implemented this and
#:                 was an orphan: the budget went to typed feedback, which lost,
#:                 while the mechanism that scales with the oracle sat unreachable.
STRATEGIES = ("refine", "best-of-n")


def build(
    brief: str,
    *,
    llm: Optional[LLM] = None,
    backend: str = "cadquery",
    model: Optional[str] = None,
    max_iters: int = 5,
    tracer: Optional[Tracer] = None,
    strategy: str = "refine",
    n: int = 4,
    use_memory: bool = True,
    memory_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a part from a natural-language `brief`.

    Wires planner -> HarnessSession -> backend -> STEP export and returns a
    plain result dict:

        {
          "ok":          bool,   # did the loop reach a verified state?
          "applied":     int,    # number of ops accepted+verified
          "digest":      str,    # deterministic model hash
          "diagnostics": [dict], # verifier/backend diagnostics (last batch)
          "summary":     dict,   # backend `query("summary")` projection
          "step":        str|None,  # exported STEP text when ok (else None)
          "backend":     str,    # resolved backend ("cadquery" | "stub")
          "backend_note":str|None,  # e.g. why cadquery fell back to stub
        }

    Parameters
    ----------
    brief:
        The design request in natural language.
    llm:
        An `llm.base.LLM` to plan with. If omitted, a lazy `LiteLLMClient` is
        used (built only on first model call) and a provider API key must be set.
    backend:
        "cadquery" (real OCCT geometry when installed) or "stub".
    model:
        Model name for the default LiteLLM client (ignored when `llm` is given).
    max_iters:
        Max plan -> apply -> replan correction iterations.
    tracer:
        Optional `trace.Tracer` for structured loop events.
    use_memory:
        Ground the default agent in a persistent `HarnessMemory` (ON by default):
        the planner recalls prior oracle-verified parts/skills/lessons into the
        prompt and the harness writes new ones through the oracle gate. Set False
        to reproduce the historical cold-start path (no recall, no write).
    memory_path:
        Where to load/save the persistent store. Defaults to `$HARNESSCAD_MEMORY`
        or a temp-scratch file. Ignored when `use_memory` is False.

    Raises
    ------
    BuildError:
        If no `llm` is provided and no provider API key is present.
    """
    if strategy not in STRATEGIES:
        raise BuildError(
            f"unknown strategy {strategy!r}; expected one of {STRATEGIES!r}")
    resolved_llm = _resolve_llm(llm, model)
    # ROUTING ON THE DEFAULT PATH. When we built the client ourselves, wrap it in
    # `routing.RoutingLLM` (classify -> sequential fallback -> cost/usage tally)
    # instead of handing the raw `_LazyLiteLLM` straight through. An injected
    # caller `llm` is left exactly as given -- its call signature is the caller's
    # contract, not ours -- so no existing caller's behaviour changes.
    if llm is None:
        resolved_llm = _route(resolved_llm)

    if strategy == "best-of-n":
        return _build_best_of_n(brief, llm=resolved_llm, backend=backend,
                                n=n, tracer=tracer)

    backend_instance, backend_name, backend_note = _make_backend(backend)
    session = HarnessSession(backend_instance, tracer=tracer)

    # MEMORY, ON BY DEFAULT. The shipping agent used to start COLD: a bare
    # `Planner(llm)` and a memoryless harness, so it forgot every part it ever
    # built. It now shares ONE persistent `HarnessMemory` between the planner
    # (which RECALLS oracle-verified exemplars/skills/lessons into the prompt)
    # and the harness (which WRITES new ones through the oracle gate) -- the
    # read-act-reflect-write loop. Both seams already existed; this wires them.
    # Construction is fully defensive: a failure degrades to the old cold path
    # (memory=None reproduces the prior behaviour byte-for-byte) with a logged
    # reason, never a crashed build.
    memory = None
    resolved_memory_path: Optional[str] = None
    if use_memory:
        memory, resolved_memory_path, memory_note = _build_memory(memory_path)
        if memory_note:
            _LOG.info("memory: %s", memory_note)

    planner = Planner(resolved_llm, memory=memory)

    # THE ONE LOOP, fully configured. `executor=None` means the harness mints its
    # own SessionToolExecutor: guardrail hard gate -> human-approval tier ->
    # apply-with-retry/backoff -> output truncation, ahead of the session's own
    # block-and-correct + transactional verify. The feedback channel is gated on
    # soundness at the harness boundary. Both were absent from this path.
    harness = AgentHarness(
        session,
        planner,
        context=ContextManager(budget=CONTEXT_BUDGET_TOKENS),
        loop_detector=LoopDetector(),
        tracer=tracer,
        max_iterations=max_iters,
        memory=memory,
    )
    run_result = harness.run(brief)

    # Persist the (now possibly enriched) store so the NEXT session continues
    # from this one. Best-effort: a save failure never touches the result.
    _save_memory(memory, resolved_memory_path)

    diagnostics: List[dict] = list(run_result.diagnostics)
    step = _export_step(backend_instance, run_result.ok)

    return {
        "ok": run_result.ok,
        "applied": run_result.applied,
        "digest": run_result.digest,
        "diagnostics": diagnostics,
        "summary": session.summary(),
        "step": step,
        "backend": backend_name,
        "backend_note": backend_note,
        # The replayable audit trail of the run: per-iteration ops, dispatch
        # verdicts, diagnostics and digests. Previously the shipping path
        # produced no trajectory at all.
        "trajectory": run_result.trajectory,
        "stop_reason": run_result.stop_reason,
        "run_id": run_result.run_id,
        # The live session, so a caller that means to WRITE the STEP can put it
        # through the output gate (harnesscad.io.gate) against the geometry the
        # backend actually built, rather than trusting the text above.
        "session": session,
        "strategy": "refine",
        # PROCESS SUPERVISION. The loop used to hand back one scalar about the
        # finished solid and attribute nothing: a six-op plan that failed
        # condemned ops 1-5 equally. `step_rewards` is the per-op vector,
        # `first_divergence` is the single op that broke the trajectory, and
        # `reward` is the full R = a*R_ORM + b*mean(R_step) + c*R_format from
        # `agents/agent/tool_reward.py` -- which was implemented, tested, and
        # imported by nothing but a dispatch table.
        #
        # `orm_verdict` here is the LOOP's verdict (did every op apply and
        # verify), not an oracle's. A caller with an oracle -- selftest.golden,
        # selftest.differential, or a grader -- should re-score with
        # `trace_reward.reward_for_session(session, orm_verdict=<oracle>)`. The
        # fleet must never be used as the ORM: its false-positive rate is
        # measured and non-zero.
        "step_rewards": list(session.step_rewards),
        "first_divergence": first_divergence(session.step_rewards),
        "step_accuracy": step_accuracy(session.step_rewards),
        "reward": reward_for_session(session, orm_verdict=run_result.ok).__dict__,
    }


def _build_best_of_n(
    brief: str,
    *,
    llm: LLM,
    backend: str,
    n: int,
    tracer: Optional[Tracer] = None,
) -> Dict[str, Any]:
    """BEST-OF-N: draw N plans, apply each in a fresh session, let the verifier pick.

    The audit's finding, in one sentence: *we spent the budget on typed feedback
    and lost, while Best-of-N sat orphaned.* This is the wiring. There is NO
    feedback channel here — a candidate is drawn, applied and scored, and the
    selector is the deterministic verifier, not a diagnostic the model has to
    believe. A wrong rule cannot poison a loop that never speaks to the model.

    Candidates are generated sequentially and are side-effect isolated (a fresh
    backend + session each), so a caller may trivially parallelise them.
    """
    from harnesscad.eval.reliability.strategies.best_of_n import best_of_n

    if n < 1:
        raise BuildError(f"best-of-n needs n >= 1 (got {n})")

    planner = Planner(llm)
    sessions: List[HarnessSession] = []
    names: List[str] = []
    notes: List[Optional[str]] = []

    def session_factory() -> HarnessSession:
        inst, name, note = _make_backend(backend)
        sess = HarnessSession(inst, tracer=tracer)
        sessions.append(sess)
        names.append(name)
        notes.append(note)
        return sess

    outcome = best_of_n(planner, session_factory, brief, n)
    idx = outcome.winner_index if outcome.winner_index >= 0 else 0
    session = sessions[idx]
    backend_instance = session.backend
    result = outcome.winner.result if outcome.winner else None
    ok = bool(result and result.ok)

    return {
        "ok": ok,
        "applied": result.applied if result else 0,
        "digest": result.digest if result else session.digest(),
        "diagnostics": [d.to_dict() if hasattr(d, "to_dict") else d
                        for d in (result.diagnostics if result else [])],
        "summary": session.summary(),
        "step": _export_step(backend_instance, ok),
        "backend": names[idx],
        "backend_note": notes[idx],
        "trajectory": [
            {"index": c.index, "ok": c.ok, "error": c.error,
             "applied": c.result.applied if c.result else 0,
             "n_diagnostics": len(c.result.diagnostics) if c.result else 0,
             # Per-op credit for every candidate, not just the winner: which op
             # broke each losing draw is the only thing that tells you WHY N
             # had to be this large.
             "step_rewards": list(sessions[c.index].step_rewards)}
            for c in outcome.candidates
        ],
        "stop_reason": "best-of-n",
        "run_id": None,
        "session": session,
        "strategy": "best-of-n",
        "n": n,
        "winner_index": idx,
    }
