"""``python -m surfaces.acp`` — the ACP agent entrypoint Zed (or any ACP client)
spawns.

Wires the newline-delimited JSON-RPC 2.0 stdio transport to an ``ACPAgent``:

  * a single writer lock serialises outbound frames onto stdout (one message per
    line, no embedded newlines);
  * a reader loop consumes stdin line-by-line; responses to agent-initiated
    requests are routed to their pending futures via ``Connection.deliver``,
    while inbound requests / notifications are dispatched to the agent;
  * ``session/prompt`` is dispatched on a worker thread so the reader stays free
    to deliver the ``session/request_permission`` / ``fs/write_text_file``
    responses the agent blocks on mid-turn.

stderr is left entirely for logs.
"""

from __future__ import annotations

import argparse
import sys
import threading
from typing import Any, Dict, Optional, TextIO

from harnesscad.io.surfaces.acp.agent import ACPAgent
from harnesscad.io.surfaces.acp.jsonrpc import Connection, decode, encode, is_response


def _default_planner(backend):
    """Best-effort real planner if an LLM is importable/configured, else null.

    Kept lazy and defensive so a bare offline invocation never fails to start.
    """
    from harnesscad.io.surfaces.acp.agent import _NullPlanner
    return _NullPlanner()


def serve(stdin: Optional[TextIO] = None, stdout: Optional[TextIO] = None,
          backend: str = "stub", max_iterations: int = 8) -> int:
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    write_lock = threading.Lock()

    def send(msg: Dict[str, Any]) -> None:
        line = encode(msg)
        with write_lock:
            stdout.write(line + "\n")
            stdout.flush()

    connection = Connection(send)
    agent = ACPAgent(connection, backend=backend,
                     planner_factory=_default_planner,
                     max_iterations=max_iterations)

    workers = []
    for raw in stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            msg = decode(line)
        except ValueError:
            continue  # malformed line: skip (stderr is free for logs)
        if is_response(msg):
            connection.deliver(msg)
            continue
        # Inbound request / notification.
        if msg.get("method") == "session/prompt":
            # Long-running: worker thread keeps the reader free for the
            # permission / fs round-trips the prompt initiates.
            t = threading.Thread(target=agent.dispatch, args=(msg,))
            t.start()
            workers.append(t)
            workers = [w for w in workers if w.is_alive()]
        else:
            agent.dispatch(msg)
    # stdin closed: let any in-flight prompt finish before exiting so its
    # response is not dropped.
    for w in workers:
        w.join()
    return 0


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="surfaces.acp", description="HarnessCAD ACP agent (stdio)")
    parser.add_argument("--backend", default="stub", choices=["stub", "cadquery"])
    parser.add_argument("--max-iterations", type=int, default=8)
    args = parser.parse_args(argv)
    return serve(backend=args.backend, max_iterations=args.max_iterations)


if __name__ == "__main__":
    raise SystemExit(main())
