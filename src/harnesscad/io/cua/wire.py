"""wire — the Computer/Agent JSON wire-envelope + reflective handler dispatch.

Ported from cua-main (trycua): its ``computer-server`` speaks a tiny, uniform
JSON protocol between the agent and the machine under control. Every request is
``{"command": <name>, "params": {...}}``; every reply is
``{"success": <bool>, ...result}`` or ``{"success": false, "error": <str>}``.
Commands are dispatched REFLECTIVELY: a name->callable table, the params filtered
by ``inspect.signature`` to exactly what the handler accepts, aliases resolved
first, and an unknown command answered with "did you mean ...". No per-command
plumbing; the handler's own signature IS its schema (the same table drives the
``/commands`` introspection endpoint).

Why HarnessCAD wants this (and how it differs from what we have)
---------------------------------------------------------------
:mod:`harnesscad.io.cua.uia` is an in-process driver — it calls UIA directly.
This module is the ENVELOPE for driving a CAD GUI OUT OF PROCESS: a FreeCAD or
Blender instance in a container/VM, or across the Python-console channel of
:mod:`harnesscad.io.cua.console`. It is transport-agnostic (no websocket, no
FastAPI here — those are the caller's) and deterministic: given a handler table
and an envelope, dispatch is a pure function of the two. That makes the protocol
itself unit-testable with a table of fakes, exactly what the live server cannot be.

This is a DISTINCT extension of the CUA surface: nothing in ``io/cua`` currently
defines the request/response envelope or a reflective command router. The
primitive VOCABULARY it routes is the one in :mod:`harnesscad.io.cua.primitives`;
the envelope here is how a remote peer invokes it.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


class WireError(RuntimeError):
    """A malformed envelope (not a handler failure — those are reported in-band)."""


# The alias table cua-main ships: ergonomic short names -> canonical command.
DEFAULT_ALIASES: Dict[str, str] = {
    "type": "type_text",
    "shell": "run_command",
    "exec": "run_command",
    "move": "move_cursor",
    "click": "left_click",
}


@dataclass(frozen=True)
class Request:
    """One inbound envelope: a command name and its params. Pure value."""

    command: str
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Request":
        if not isinstance(data, dict):
            raise WireError("envelope must be a JSON object, got %r" % type(data))
        command = data.get("command")
        if not isinstance(command, str) or not command:
            raise WireError("envelope missing a non-empty 'command' string")
        params = data.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise WireError("'params' must be an object, got %r" % type(params))
        return cls(command=command, params=dict(params))

    def to_dict(self) -> dict:
        return {"command": self.command, "params": dict(self.params)}


@dataclass(frozen=True)
class Response:
    """One outbound envelope. ``success`` plus either result fields or an error.

    The invariant cua-main relies on: a reply is ``{"success": true, **result}``
    or ``{"success": false, "error": ...}`` — never both, never neither.
    """

    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    suggestions: Tuple[str, ...] = ()

    @classmethod
    def ok(cls, **data: Any) -> "Response":
        return cls(success=True, data=dict(data))

    @classmethod
    def fail(cls, error: str, suggestions: Tuple[str, ...] = ()) -> "Response":
        return cls(success=False, error=error, suggestions=tuple(suggestions))

    def to_dict(self) -> dict:
        if self.success:
            out: Dict[str, Any] = {"success": True}
            out.update(self.data)
            return out
        out = {"success": False, "error": self.error}
        if self.suggestions:
            out["suggestions"] = list(self.suggestions)
        return out


def command_params(handler: Callable[..., Any]) -> List[Dict[str, Any]]:
    """The handler's parameters, as cua-main's ``/commands`` reports them:
    ``[{"name", "required", "default"}]``. This is the reflective schema — the
    signature is the contract, no hand-maintained param list."""
    try:
        sig = inspect.signature(handler)
    except (ValueError, TypeError):
        return []
    out: List[Dict[str, Any]] = []
    for p in sig.parameters.values():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL,
                      inspect.Parameter.VAR_KEYWORD):
            continue
        required = p.default is inspect.Parameter.empty
        out.append({"name": p.name, "required": required,
                    "default": None if required else p.default})
    return out


def _accepts_kwargs(handler: Callable[..., Any]) -> bool:
    try:
        sig = inspect.signature(handler)
    except (ValueError, TypeError):
        return False
    return any(p.kind is inspect.Parameter.VAR_KEYWORD
               for p in sig.parameters.values())


def _filter_params(handler: Callable[..., Any],
                   params: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only params the handler declares — cua-main's exact rule, which lets a
    caller send a superset (e.g. a shared ``element_index``) without every handler
    having to accept it. A ``**kwargs`` handler takes everything."""
    if _accepts_kwargs(handler):
        return dict(params)
    try:
        sig = inspect.signature(handler)
    except (ValueError, TypeError):
        return {}
    return {k: v for k, v in params.items() if k in sig.parameters}


class Dispatcher:
    """A reflective command router: name -> callable, with aliases and filtering.

    Build it with a handler table (any callables — real drivers, or fakes in a
    test). :meth:`dispatch` resolves aliases, rejects unknown commands with a
    suggestion, filters params to the handler's signature, calls it, and wraps the
    result in a :class:`Response`. A handler returning a dict has that dict spread
    into the success envelope; a handler that raises becomes an in-band failure
    (never propagates), which is the property that keeps a driving loop alive.
    """

    def __init__(self, handlers: Dict[str, Callable[..., Any]],
                 aliases: Optional[Dict[str, str]] = None) -> None:
        self.handlers = dict(handlers)
        self.aliases = dict(DEFAULT_ALIASES if aliases is None else aliases)

    # -- introspection (cua-main's /commands) ------------------------------
    def resolve(self, command: str) -> str:
        return self.aliases.get(command, command)

    def catalog(self) -> Dict[str, Any]:
        """The self-describing command list: every command, its params, and the
        aliases pointing at it. The agent needs no out-of-band schema."""
        by_command: Dict[str, List[str]] = {}
        for alias, canonical in self.aliases.items():
            by_command.setdefault(canonical, []).append(alias)
        commands: Dict[str, Any] = {}
        for name, handler in self.handlers.items():
            entry: Dict[str, Any] = {"params": command_params(handler)}
            if name in by_command:
                entry["aliases"] = sorted(by_command[name])
            commands[name] = entry
        return {"commands": commands, "aliases": dict(self.aliases)}

    def suggest(self, command: str) -> List[str]:
        """cua-main's typo helper: known names sharing a 2-char prefix with, or
        containing, the unknown command."""
        pool = list(self.handlers) + list(self.aliases)
        low = command.lower()
        out: List[str] = []
        for cand in pool:
            cl = cand.lower()
            if (low[:2] and cl.startswith(low[:2])) or (low and low in cl):
                if cand not in out:
                    out.append(cand)
        return sorted(out)

    # -- the dispatch ------------------------------------------------------
    def dispatch(self, envelope: Any) -> Response:
        """Route one envelope (a :class:`Request` or a raw dict) to a handler.

        Never raises for a handler-level problem: an unknown command, bad params,
        or a handler exception all come back as ``success=False`` with a message,
        because the protocol's job is to keep answering. A malformed ENVELOPE
        (not a dict, no command) is the one thing that fails loudly, via
        :class:`WireError` from :meth:`Request.from_dict`.
        """
        req = envelope if isinstance(envelope, Request) else Request.from_dict(envelope)
        name = self.resolve(req.command)
        handler = self.handlers.get(name)
        if handler is None:
            sugg = self.suggest(req.command)
            msg = "Unknown command: %s" % req.command
            if sugg:
                msg += ". Did you mean: %s?" % ", ".join(sugg)
            return Response.fail(msg, suggestions=tuple(sugg))
        kwargs = _filter_params(handler, req.params)
        try:
            result = handler(**kwargs)
        except TypeError as exc:
            # A missing REQUIRED param survives filtering and shows up here.
            return Response.fail("bad params for %s: %s" % (name, exc))
        except Exception as exc:  # noqa: BLE001 - handler failures are in-band
            return Response.fail("%s: %s" % (name, exc))
        if isinstance(result, Response):
            return result
        if result is None:
            return Response.ok()
        if isinstance(result, dict):
            # cua-main spreads a handler's dict into {"success": True, **result};
            # honour an explicit success=False the handler put there.
            if result.get("success") is False:
                return Response.fail(str(result.get("error", "handler reported failure")))
            data = {k: v for k, v in result.items() if k != "success"}
            return Response.ok(**data)
        return Response.ok(result=result)
