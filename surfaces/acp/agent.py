"""ACPAgent — the Zed Agent Client Protocol agent for HarnessCAD text-to-CAD.

Implements the ACP *agent* method set (protocol v1, JSON-RPC 2.0) so any ACP
client (Zed, etc.) can drive HarnessCAD in-editor:

  initialize        -> handshake: protocolVersion, agentCapabilities, agentInfo.
                       Reads clientCapabilities.fs to decide STEP delivery.
  session/new       -> build a HarnessSession(backend), wire a ToolExecutor whose
                       approval callback issues a blocking session/request_permission
                       round-trip, register a sessionId.
  session/prompt    -> run AgentHarness.run(brief); stream UIEvents/trace as
                       session/update notifications; return a mapped {stopReason}.
  session/cancel    -> set a cancel flag the loop checks; run ends "cancelled".

This module COMPOSES the existing harness (harness.py / loop.py /
reliability.executor / surfaces.ui.*); it imports them read-only. The only
new machinery is the ACP protocol translation, which lives here and in
``bridge.py`` / ``jsonrpc.py``.

stopReason mapping (HarnessRun -> ACP):
  cancelled flag set                 -> "cancelled"
  run.ok (stop_reason "converged")   -> "end_turn"
  stop_reason "max_iterations"       -> "max_turn_requests"
  otherwise (e.g. "loop")            -> "refusal"
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, List, Optional

from cisp.protocol import ApplyOpsResult
from harness import AgentHarness
from llm.structured import ParsedOps
from loop import HarnessSession
from reliability.executor import ToolExecutor
from surfaces.acp.bridge import ACPBridge
from surfaces.acp.jsonrpc import (
    Connection, INTERNAL_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND,
)
from surfaces.server import _make_backend
from surfaces.ui.approval import op_name

AGENT_NAME = "harnesscad"
AGENT_VERSION = "0.2.1"
PROTOCOL_VERSION = 1


class PromptCancelled(Exception):
    """Raised out of the harness dispatch seam when a prompt is cancelled."""


class _NullPlanner:
    """Fallback planner when no LLM is configured (offline default).

    Produces no ops and a re-promptable error, so a bare ``python -m surfaces.acp``
    with no model still speaks ACP correctly (the run just yields no geometry).
    A real deployment injects an LLM-backed ``agent.planner.Planner``.
    """

    def plan_parsed(self, brief: str, state_summary=None, diagnostics=None) -> ParsedOps:
        return ParsedOps([], error="no LLM planner configured for this session")


class BridgeTracer:
    """A ``Tracer`` that surfaces harness progress to the client as chunks.

    Harness trace events (harness.py: harness_start / plan / checkpoint /
    harness_end) become ``agent_message_chunk`` status lines so the client sees
    live progress. Per the Tracer contract, ``event`` never raises into the loop.
    """

    def __init__(self, bridge: ACPBridge) -> None:
        self.bridge = bridge

    def event(self, kind: str, run_id: str, data: dict) -> None:
        try:
            from surfaces.ui.events import UIEvent
            if kind == "harness_start":
                self.bridge.emit(UIEvent.status("Starting text-to-CAD run."))
            elif kind == "plan":
                if data.get("ok"):
                    self.bridge.emit(UIEvent.status(
                        f"Planned {data.get('op_count', 0)} operation(s)."))
            elif kind == "checkpoint":
                self.bridge.emit(UIEvent.status("Checkpointed a verified result."))
            elif kind == "harness_end":
                self.bridge.emit(UIEvent.status(
                    f"Run finished ({data.get('stop_reason', '')})."))
        except Exception:
            return None


class BridgingExecutor:
    """Harness ``executor`` seam that drives each op through a ``ToolExecutor``
    while emitting ACP ``session/update`` notifications in real time.

    Exposes ``apply_ops(ops) -> ApplyOpsResult`` so the harness dispatches to it
    (harness.py ``_dispatch`` tries ``apply_ops`` first). For each op it: sends
    the plan, emits a ``tool_call``, runs the op through the ToolExecutor (whose
    approval callback runs the blocking permission round-trip for Tier-3 ops),
    then emits a ``tool_call_update`` (completed / failed). A denied Tier-3 op is
    surfaced as ``action_rejected`` and stops the batch (block-and-correct).

    Cancellation: if the bridge's cancel flag is set it raises ``PromptCancelled``,
    which propagates cleanly out of ``AgentHarness.run`` to the prompt handler.
    """

    def __init__(self, session: HarnessSession, bridge: ACPBridge,
                 executor: ToolExecutor) -> None:
        self.session = session
        self.bridge = bridge
        self.executor = executor

    def apply_ops(self, ops: List[Any]) -> ApplyOpsResult:
        if self.bridge.cancelled:
            raise PromptCancelled()
        self.bridge.start_plan(ops)
        applied = 0
        diags: List[Any] = []
        for i, op in enumerate(ops):
            if self.bridge.cancelled:
                raise PromptCancelled()
            call_id = self.bridge.plan_call_id(i)
            self.bridge._active_call_id = call_id
            self.bridge.plan_mark(i, "in_progress")
            self.bridge.emit_tool_call(op, call_id)

            res = self.executor.execute(op, self.session)
            diags += list(res.diagnostics)

            if res.ok:
                applied += res.result.applied if res.result is not None else 1
                self.bridge.plan_mark(i, "completed")
                self.bridge.emit_tool_result(op, call_id, ok=True)
                continue

            # Failure: either a denied Tier-3 op (approved is False) or a
            # backend/verifier rejection. Both stop the batch.
            if not res.approved:
                self.bridge.emit_action_rejected(
                    op, call_id, "permission-denied", res.diagnostics)
            else:
                self.bridge.emit_tool_result(op, call_id, ok=False)
            rejected = op.to_dict() if hasattr(op, "to_dict") else {"op": op_name(op)}
            return ApplyOpsResult(False, applied, self.session.digest(),
                                  diags, rejected=rejected)

        return ApplyOpsResult(True, applied, self.session.digest(), diags)


class ACPAgent:
    """The ACP agent. One instance handles many sessions over one Connection."""

    def __init__(self, connection: Connection, backend: str = "stub",
                 planner_factory: Optional[Callable[[Any], Any]] = None,
                 max_iterations: int = 8) -> None:
        self.connection = connection
        self.backend_name = backend
        # planner_factory(backend) -> planner. Default: the offline null planner.
        self.planner_factory = planner_factory or (lambda backend: _NullPlanner())
        self.max_iterations = max_iterations
        self.client_can_write_fs = False
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._session_seq = 0

    # --- id derivation (deterministic, no uuid/clock) ---------------------
    def _make_session_id(self, cwd: str) -> str:
        self._session_seq += 1
        h = hashlib.sha256(f"{self._session_seq}|{cwd}".encode("utf-8")).hexdigest()
        return f"sess-{self._session_seq}-{h[:8]}"

    # --- ACP methods ------------------------------------------------------
    def initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        caps = params.get("clientCapabilities") or {}
        fs = caps.get("fs") or {}
        # ACP advertises fs.writeTextFile / fs.readTextFile as capability flags.
        self.client_can_write_fs = bool(fs.get("writeTextFile"))
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "agentCapabilities": {
                "loadSession": False,
                "promptCapabilities": {},
            },
            "agentInfo": {"name": AGENT_NAME, "version": AGENT_VERSION},
        }

    def session_new(self, params: Dict[str, Any]) -> Dict[str, Any]:
        cwd = params.get("cwd") or "."
        # mcpServers is accepted (ACP passes it) but the stub agent runs no MCP.
        backend, backend_name, _note = _make_backend(self.backend_name)
        session = HarnessSession(backend)
        session_id = self._make_session_id(cwd)

        bridge = ACPBridge(self.connection, session_id, cwd,
                           client_can_write_fs=self.client_can_write_fs)
        # The ToolExecutor approval callback is the blocking permission round-trip.
        approval = lambda op: bridge.request_permission(op)  # noqa: E731
        tool_executor = ToolExecutor(approval=approval)
        executor = BridgingExecutor(session, bridge, tool_executor)
        planner = self.planner_factory(backend)
        harness = AgentHarness(
            session, planner, executor=executor,
            tracer=BridgeTracer(bridge), max_iterations=self.max_iterations)

        self._sessions[session_id] = {
            "session": session,
            "backend": backend,
            "bridge": bridge,
            "harness": harness,
        }
        return {"sessionId": session_id}

    def session_prompt(self, params: Dict[str, Any]) -> Dict[str, Any]:
        session_id = params.get("sessionId")
        entry = self._sessions.get(session_id)
        if entry is None:
            raise KeyError(f"unknown sessionId '{session_id}'")
        bridge: ACPBridge = entry["bridge"]
        harness: AgentHarness = entry["harness"]
        bridge.cancelled = False

        brief = self._brief_of(params.get("prompt") or [])
        try:
            run = harness.run(brief)
        except PromptCancelled:
            return {"stopReason": "cancelled"}

        if bridge.cancelled:
            return {"stopReason": "cancelled"}

        # STEP delivery on success.
        if run.ok:
            try:
                content = entry["backend"].export("step")
                bridge.deliver_step(content)
            except Exception:
                pass

        return {"stopReason": self._stop_reason(run)}

    def session_cancel(self, params: Dict[str, Any]) -> None:
        entry = self._sessions.get(params.get("sessionId"))
        if entry is not None:
            entry["bridge"].cancelled = True

    # --- helpers ----------------------------------------------------------
    @staticmethod
    def _brief_of(prompt: List[Any]) -> str:
        """Extract the brief from ``prompt[0].text`` (the first ContentBlock)."""
        if prompt and isinstance(prompt[0], dict):
            return prompt[0].get("text", "")
        return ""

    @staticmethod
    def _stop_reason(run) -> str:
        if run.ok:
            return "end_turn"
        if run.stop_reason == "max_iterations":
            return "max_turn_requests"
        return "refusal"

    # --- dispatch (used by the stdio reader in __main__) ------------------
    _REQUEST_METHODS = {
        "initialize": "initialize",
        "session/new": "session_new",
        "session/prompt": "session_prompt",
    }

    def handle_request(self, method: str, params: Dict[str, Any]) -> Any:
        """Dispatch a request method to its handler (synchronous)."""
        name = self._REQUEST_METHODS.get(method)
        if name is None:
            raise _MethodNotFound(method)
        return getattr(self, name)(params)

    def dispatch(self, msg: Dict[str, Any]) -> None:
        """Route one inbound request/notification and reply if it is a request.

        Notifications (session/cancel) are handled without a reply. Long-running
        ``session/prompt`` should be run on a worker thread by the caller so the
        reader stays free to deliver the permission / fs responses it initiates;
        see ``__main__``.
        """
        method = msg.get("method")
        rid = msg.get("id")
        params = msg.get("params") or {}

        if method == "session/cancel":
            self.session_cancel(params)
            return
        if rid is None:
            return  # unknown notification: ignore

        try:
            result = self.handle_request(method, params)
            self.connection.respond(rid, result=result)
        except _MethodNotFound as exc:
            self.connection.respond(rid, error={
                "code": METHOD_NOT_FOUND, "message": f"unknown method '{exc}'"})
        except KeyError as exc:
            self.connection.respond(rid, error={
                "code": INVALID_REQUEST, "message": str(exc)})
        except Exception as exc:  # noqa: BLE001 - surface as JSON-RPC error
            self.connection.respond(rid, error={
                "code": INTERNAL_ERROR, "message": str(exc)})


class _MethodNotFound(Exception):
    pass
