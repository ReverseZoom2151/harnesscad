"""The A2A JSON-RPC 2.0 method dispatcher, backed by an ``a2a.TaskStore``.

``A2AHandler`` is the protocol brain, transport-free: it takes JSON-RPC request
dicts and returns JSON-RPC response dicts (``dispatch``) or, for the streaming
methods, yields SSE frame strings (``stream``). The HTTP ``app`` is a thin skin
over it, and tests drive it directly in-process (no socket).

Method map (A2A protocol v0.3.0):
  - ``message/send``   -> create+run a task synchronously, return the final Task.
  - ``message/stream`` -> create+run a task, stream status/artifact updates.
  - ``tasks/get``      -> the task's ``to_a2a`` snapshot (-32001 if unknown).
  - ``tasks/cancel``   -> cancel the task (-32002 if not cancelable).
  - ``tasks/resubscribe`` -> re-attach an SSE stream to a live/terminal task.
  - ``tasks/pushNotificationConfig/*`` -> -32003 (not supported yet).
  - anything else      -> -32601 method not found.

Each run drives the task through its guarded lifecycle
(submit -> start -> [artifact] -> complete/fail) inside a worker thread, exactly
as a real async solve would; ``message/send`` simply joins that thread.
"""

from __future__ import annotations

import base64
import queue
import threading
import uuid
from typing import Any, Dict, Iterator, Optional

from harnesscad.agents.a2a.messages import Artifact, Part, agent_message
from harnesscad.agents.a2a.task import IllegalTransition, TaskStore
from harnesscad.io.surfaces.a2a_server import wire

# Sentinel pushed onto the event queue when a worker finishes.
_SENTINEL = object()


class _RpcError(Exception):
    """Carries a JSON-RPC error code/message/data for the dispatcher to emit."""

    def __init__(self, code: int, message: str, data: Optional[Any] = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


class A2AHandler:
    """Stateful A2A endpoint: one ``TaskStore`` shared across requests.

    ``harness_factory`` is a zero-arg callable returning a fresh ``AgentHarness``
    (with its own ``HarnessSession``/backend) per run, so concurrent tasks never
    share mutable geometry state.
    """

    def __init__(
        self,
        harness_factory,
        store: Optional[TaskStore] = None,
    ) -> None:
        self.harness_factory = harness_factory
        self.store = store if store is not None else TaskStore()

    # --- non-streaming dispatch ------------------------------------------
    def dispatch(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle one JSON-RPC request, returning a single JSON-RPC response."""
        rpc_id = request.get("id") if isinstance(request, dict) else None
        try:
            if not isinstance(request, dict):
                raise _RpcError(wire.ERROR_INVALID_REQUEST, "request must be an object")
            method = request.get("method")
            params = request.get("params") or {}
            if not isinstance(params, dict):
                raise _RpcError(wire.ERROR_INVALID_PARAMS, "'params' must be an object")

            if method == "message/send":
                result = self._message_send(params)
            elif method == "tasks/get":
                result = self._tasks_get(params)
            elif method == "tasks/cancel":
                result = self._tasks_cancel(params)
            elif method in ("message/stream", "tasks/resubscribe"):
                raise _RpcError(
                    wire.ERROR_UNSUPPORTED_OPERATION,
                    f"method '{method}' requires the streaming transport",
                )
            elif isinstance(method, str) and method.startswith(
                "tasks/pushNotificationConfig"
            ):
                raise _RpcError(
                    wire.ERROR_PUSH_NOTIFICATION_NOT_SUPPORTED,
                    "push notifications are not supported",
                )
            else:
                raise _RpcError(
                    wire.ERROR_METHOD_NOT_FOUND, f"method not found: {method}"
                )
            return wire.rpc_result(rpc_id, result)
        except _RpcError as exc:
            return wire.rpc_error(rpc_id, exc.code, exc.message, exc.data)
        except Exception as exc:  # noqa: BLE001 - map any surprise to a JSON-RPC error
            return wire.rpc_error(rpc_id, wire.ERROR_INTERNAL, f"internal error: {exc}")

    # --- streaming dispatch ----------------------------------------------
    def stream(self, request: Dict[str, Any]) -> Iterator[str]:
        """Handle a streaming JSON-RPC request, yielding SSE frame strings."""
        rpc_id = request.get("id") if isinstance(request, dict) else None
        try:
            method = request.get("method")
            params = request.get("params") or {}
            if method == "message/stream":
                yield from self._message_stream(rpc_id, params)
            elif method == "tasks/resubscribe":
                yield from self._resubscribe(rpc_id, params)
            else:
                yield wire.sse_frame(
                    wire.rpc_error(
                        rpc_id, wire.ERROR_METHOD_NOT_FOUND, f"method not found: {method}"
                    )
                )
        except _RpcError as exc:
            yield wire.sse_frame(
                wire.rpc_error(rpc_id, exc.code, exc.message, exc.data)
            )
        except Exception as exc:  # noqa: BLE001
            yield wire.sse_frame(
                wire.rpc_error(rpc_id, wire.ERROR_INTERNAL, f"internal error: {exc}")
            )

    # --- methods ----------------------------------------------------------
    def _message_send(self, params: Dict[str, Any]) -> Dict[str, Any]:
        msg = wire.a2a_message_from_params(params)
        task = self.store.create(contextId=msg.contextId)
        brief = msg.text()
        errbox: list = []

        def work() -> None:
            try:
                self._execute(task, brief, msg)
            except Exception as exc:  # noqa: BLE001 - surface via errbox
                errbox.append(exc)

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        thread.join()
        if errbox:
            raise _RpcError(wire.ERROR_INTERNAL, f"run failed: {errbox[0]}")
        return task.to_a2a()

    def _message_stream(
        self, rpc_id: Any, params: Dict[str, Any]
    ) -> Iterator[str]:
        msg = wire.a2a_message_from_params(params)
        task = self.store.create(contextId=msg.contextId)
        brief = msg.text()
        events: "queue.Queue" = queue.Queue()
        unsubscribe = task.subscribe(events.put)

        def work() -> None:
            try:
                self._execute(task, brief, msg)
            finally:
                events.put(_SENTINEL)

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        try:
            while True:
                event = events.get()
                if event is _SENTINEL:
                    break
                yield wire.sse_frame(
                    wire.rpc_result(rpc_id, wire.event_to_wire(event))
                )
        finally:
            unsubscribe()
            thread.join()

    def _resubscribe(self, rpc_id: Any, params: Dict[str, Any]) -> Iterator[str]:
        task = self.store.get(params.get("id"))
        if task is None:
            yield wire.sse_frame(
                wire.rpc_error(
                    rpc_id,
                    wire.ERROR_TASK_NOT_FOUND,
                    "task not found",
                    {"id": params.get("id")},
                )
            )
            return
        if task.is_terminal:
            # Nothing more will be emitted; replay the current terminal status.
            snapshot = {
                "kind": "status_update",
                "taskId": task.taskId,
                "contextId": task.contextId,
                "ts": None,
                "data": {
                    "state": task.state.value if task.state is not None else "unknown",
                    "message": None,
                    "final": True,
                },
            }
            yield wire.sse_frame(
                wire.rpc_result(rpc_id, wire.event_to_wire(snapshot))
            )
            return
        events: "queue.Queue" = queue.Queue()
        unsubscribe = task.subscribe(events.put)
        try:
            while True:
                event = events.get()
                wire_event = wire.event_to_wire(event)
                yield wire.sse_frame(wire.rpc_result(rpc_id, wire_event))
                if wire_event.get("final"):
                    break
        finally:
            unsubscribe()

    def _tasks_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        task = self.store.get(params.get("id"))
        if task is None:
            raise _RpcError(
                wire.ERROR_TASK_NOT_FOUND, "task not found", {"id": params.get("id")}
            )
        return task.to_a2a()

    def _tasks_cancel(self, params: Dict[str, Any]) -> Dict[str, Any]:
        task = self.store.get(params.get("id"))
        if task is None:
            raise _RpcError(
                wire.ERROR_TASK_NOT_FOUND, "task not found", {"id": params.get("id")}
            )
        try:
            task.cancel(agent_message(Part.from_text("canceled by client")))
        except IllegalTransition:
            raise _RpcError(
                wire.ERROR_TASK_NOT_CANCELABLE,
                f"task '{task.taskId}' is not in a cancelable state",
                {"id": task.taskId},
            )
        return task.to_a2a()

    # --- the run ----------------------------------------------------------
    def _execute(self, task, brief: str, msg) -> None:
        """Drive one task through its lifecycle around a single harness run."""
        task.submit(msg)
        task.start(agent_message(Part.from_text("planning the model")))
        harness = self.harness_factory()
        run = harness.run(brief)
        if run.ok:
            backend = harness.session.backend
            content = backend.export("step")
            raw = content if isinstance(content, (bytes, bytearray)) else str(content).encode("utf-8")
            b64 = base64.b64encode(raw).decode("ascii")
            file_part = Part.from_file(
                name="out.step", mime_type="model/step", bytes_b64=b64
            )
            artifact = Artifact(
                artifactId=f"{task.taskId}-step",
                name="out.step",
                description="Generated STEP model",
                parts=(file_part,),
                metadata={"digest": run.digest, "applied": run.applied},
            )
            task.add_artifact(artifact)
            task.complete(
                agent_message(
                    Part.from_text(
                        f"Built STEP model ({run.applied} ops applied)."
                    )
                )
            )
        else:
            task.fail(
                agent_message(
                    Part.from_text(
                        f"Build failed (stop_reason={run.stop_reason})."
                    ),
                    Part.from_data({"diagnostics": run.diagnostics}),
                )
            )
