"""console_agent — the Tier-0 text-verified GUI agent, WIRED to a real console.

``audit/cua_synthesis.md`` calls the app's Python console Tier 0, the highest tier of
the action stack and "THE STEP 1": FreeCAD and Blender both ship a Python console, and
typing into it is a full GUI-app agent with TOTAL text observability -- no pixels, no
coordinate grounding, no accessibility tree. The console echoes an exception as a
traceback the instant a step is wrong, which makes the success signal the console's own
text rather than a model's opinion of a screenshot.

:mod:`harnesscad.agents.cua.console_iterate` already ports BabyCommandAGI's controller
(the stall-adjudicator :func:`adjudicate`, the failure-slicer :func:`focus_failure`, and
the :class:`ConsoleController` loop) against an ABSTRACT :class:`ConsoleChannel` whose
base ``write``/``read_new`` raise. This module supplies the missing half: concrete
channels that target a live app's interpreter, and the wiring that assembles a runnable
Tier-0 agent -- WITHOUT editing the environment it drives.

How the console is reached, without touching ``environment_freecad``
--------------------------------------------------------------------
``environment_freecad.FreeCADGuiEnvironment`` deliberately exposes only a measurement /
export macro channel, not arbitrary ``eval``; that is correct and this module does not
change it. Instead the interpreter channel is INJECTED. Anything that can run a line of
Python in the app and hand back a result fits:

* :class:`~harnesscad.io.cua.viewport.FreeCADGuiBridge` already runs source in the live
  FreeCAD GUI's interpreter and returns whatever the snippet set ``RESULT`` to (raising
  on a traceback). :func:`call_console_eval` adapts any such ``.call(source)`` object.
* Blender's console is ``bpy``-driven; inject a callable that ``exec``s in Blender's
  interpreter. :func:`callable_console_eval` wraps any ``Callable[[str], str]``.

Both are passed in from OUTSIDE. The environment/driver is composed, never modified.

.. warning:: REWARD-HACKING GUARD -- the console is a KNIFE.

   A Python console is the cheapest CORRECT path precisely because it can express the
   whole answer in one line. That is exactly why it must NEVER be in a TRAINING
   environment's action space: if a policy under training can type into this channel,
   the optimal policy is "paste the answer script, announce done" -- full reward, zero
   GUI-driving learned. This agent is legitimate as a Tier-0 *evaluation* / bootstrap
   agent and as an oracle for compiling expert data (see
   :mod:`harnesscad.agents.cua.trajectory_compiler`). Keep it on the eval / data side
   of the wall. Same warning as ``viewport.py`` and ``grounding/corpus.py``.

Import-safe and deterministic: nothing launches an app here. The channels take an
already-started bridge / callable; the tests drive a scripted fake through the same
``ConsoleChannel`` contract. No live app, subprocess, or model is run in this module.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

from harnesscad.agents.cua.console_iterate import (
    DEFAULT_DONE_MARKERS,
    DEFAULT_ERROR_MARKERS,
    DEFAULT_PROMPT_MARKERS,
    ConsoleChannel,
    ConsoleController,
    Step,
    Transcript,
)

__all__ = [
    "EvalFn",
    "CallableConsoleChannel",
    "call_console_eval",
    "callable_console_eval",
    "capture_wrapped",
    "FREECAD_DONE_MARKERS",
    "FREECAD_ERROR_MARKERS",
    "BLENDER_DONE_MARKERS",
    "BLENDER_ERROR_MARKERS",
    "ConsoleAgent",
    "build_freecad_console_agent",
    "build_blender_console_agent",
]


#: The narrow thing a console channel needs: run a line of source, return its TEXT.
#: A traceback (as text) IS a valid return -- the adjudicator reads it as an error.
EvalFn = Callable[[str], str]


# --------------------------------------------------------------------------- #
# App marker presets. A console reports done/failure in-band; these are the
# tails that carry the signal for each app. Callers may override.
# --------------------------------------------------------------------------- #
#: FreeCAD: the harness convention is to print ``__DONE__`` from a finished script;
#: Python tracebacks and the FreeCAD console's own error prefixes carry failures.
FREECAD_DONE_MARKERS: Tuple[str, ...] = DEFAULT_DONE_MARKERS
FREECAD_ERROR_MARKERS: Tuple[str, ...] = DEFAULT_ERROR_MARKERS + (
    "<Exception>", "Part.OCCError", "FreeCAD exception",
)

#: Blender: same done convention; ``bpy`` surfaces failures as ``RuntimeError`` and
#: the operator-poll message ``RuntimeError: Operator ... poll() failed``.
BLENDER_DONE_MARKERS: Tuple[str, ...] = DEFAULT_DONE_MARKERS
BLENDER_ERROR_MARKERS: Tuple[str, ...] = DEFAULT_ERROR_MARKERS + (
    "poll() failed", "RuntimeError: Operator",
)


# --------------------------------------------------------------------------- #
# Turning a line of source into captured console text.
# --------------------------------------------------------------------------- #
#: Wrap arbitrary source so its stdout/stderr are captured into ``RESULT`` as text,
#: the way a REPL echoes what a statement printed. A bridge that returns ``RESULT``
#: then hands back exactly the console's visible output. An exception inside the body
#: is NOT swallowed: it propagates to the bridge, which reports the traceback, which
#: the adjudicator classifies as an error -- so a failing line reads as INTERRUPT, as
#: it would in a real console.
_CAPTURE_WRAPPER = (
    "import io as _io, contextlib as _ctx\n"
    "_buf = _io.StringIO()\n"
    "with _ctx.redirect_stdout(_buf), _ctx.redirect_stderr(_buf):\n"
    "%(body)s\n"
    "RESULT = _buf.getvalue()\n"
)


def capture_wrapped(source: str) -> str:
    """Return ``source`` wrapped to capture its printed output into ``RESULT``.

    Deterministic string transform; runs nothing. Used by :func:`call_console_eval`
    so a ``.call`` bridge that returns ``RESULT`` yields the console's visible text.
    """
    body = textwrap.indent(source if source.strip() else "pass", "    ")
    return _CAPTURE_WRAPPER % {"body": body}


def call_console_eval(bridge: Any, *, timeout: Optional[float] = None,
                      wrap: bool = True) -> EvalFn:
    """Adapt a ``.call(source[, timeout])`` bridge into an :data:`EvalFn`.

    Fits :class:`~harnesscad.io.cua.viewport.FreeCADGuiBridge` directly: it runs source
    in the live interpreter and returns ``RESULT`` (raising on a traceback). This
    adapter wraps the source to capture printed output (unless ``wrap=False``), and
    converts a raised error into its text so the controller sees a normal error result
    rather than an exception -- exactly what a console would show. Nothing runs until
    the returned callable is invoked.
    """
    def _eval(source: str) -> str:
        payload = capture_wrapped(source) if wrap else source
        try:
            result = (bridge.call(payload, timeout=timeout)
                      if timeout is not None else bridge.call(payload))
        except Exception as exc:  # noqa: BLE001 - a traceback IS the console output
            return str(exc)
        return "" if result is None else str(result)
    return _eval


def callable_console_eval(fn: EvalFn) -> EvalFn:
    """Adapt a bare ``Callable[[str], str]`` (e.g. a Blender ``exec`` shim).

    The callable is expected to run the source in the app's interpreter and return the
    resulting console text (a traceback string on failure). Errors it raises instead of
    returning are converted to text, so the channel contract holds either way.
    """
    def _eval(source: str) -> str:
        try:
            out = fn(source)
        except Exception as exc:  # noqa: BLE001
            return str(exc)
        return "" if out is None else str(out)
    return _eval


class CallableConsoleChannel(ConsoleChannel):
    """A :class:`ConsoleChannel` backed by an injected :data:`EvalFn`.

    ``write`` runs the source through the eval function and buffers the resulting text;
    ``read_new`` drains and returns the buffered text since the last read. That is the
    whole contract the :class:`ConsoleController` needs. The eval function is the ONLY
    coupling to a live app, and it is injected -- so this class never imports FreeCAD,
    Blender, or the environment, and a test supplies a scripted ``EvalFn``.
    """

    def __init__(self, eval_fn: EvalFn) -> None:
        self._eval = eval_fn
        self._pending: List[str] = []

    def write(self, source: str) -> None:
        out = self._eval(source)
        self._pending.append(out if isinstance(out, str) else str(out))

    def read_new(self) -> str:
        text = "\n".join(self._pending)
        self._pending = []
        return text


# --------------------------------------------------------------------------- #
# The agent: a ConsoleController bound to a live channel, with app markers.
# --------------------------------------------------------------------------- #
@dataclass
class ConsoleAgent:
    """A runnable Tier-0 agent: a :class:`ConsoleController` over a live channel.

    Assemble it with a :class:`CallableConsoleChannel` (or any ``ConsoleChannel``) and
    the app's marker presets, then hand it a list of :class:`Step`s to drive. The whole
    correctness signal is the console's own text, adjudicated by
    :func:`~harnesscad.agents.cua.console_iterate.adjudicate` after every line -- the
    same block-and-verify discipline the a11y tier uses, applied to a REPL.

    Construct directly with an injected channel, or via
    :func:`build_freecad_console_agent` / :func:`build_blender_console_agent`, which
    inject the channel and the right markers for you. Nothing runs until :meth:`run`.
    """

    channel: ConsoleChannel
    done_markers: Tuple[str, ...] = DEFAULT_DONE_MARKERS
    error_markers: Tuple[str, ...] = DEFAULT_ERROR_MARKERS
    prompt_markers: Tuple[str, ...] = DEFAULT_PROMPT_MARKERS
    inputs: Sequence[str] = field(default_factory=tuple)
    max_steps: int = 1000
    app: str = "generic"

    def controller(self) -> ConsoleController:
        """Build the underlying BabyCommandAGI-style controller for this agent."""
        return ConsoleController(
            self.channel,
            done_markers=self.done_markers,
            error_markers=self.error_markers,
            prompt_markers=self.prompt_markers,
            inputs=self.inputs,
            max_steps=self.max_steps,
        )

    def run(self, steps: Sequence[Step]) -> Transcript:
        """Drive ``steps`` through the console, adjudicating after each line.

        Returns the :class:`Transcript` (every step's source/output/verdict, the
        terminal verdict, and a failure-focused log). Runnable; this is where the
        injected channel actually touches the app.
        """
        return self.controller().run(steps)

    def run_sources(self, sources: Sequence[str], *,
                    expect: Optional[Sequence[Optional[str]]] = None) -> Transcript:
        """Convenience: drive plain source lines, with optional per-line ``expect``.

        ``expect[i]`` (when given) is a substring that must appear in line ``i``'s
        output for the step to count as satisfied -- the text-verification hook that
        turns "I typed it" into "I confirmed it".
        """
        exp = list(expect) if expect is not None else [None] * len(sources)
        if len(exp) != len(sources):
            raise ValueError("expect must match sources in length")
        steps = [Step(source=s, expect=e) for s, e in zip(sources, exp)]
        return self.run(steps)


def build_freecad_console_agent(bridge: Any, *, timeout: Optional[float] = None,
                                inputs: Sequence[str] = (),
                                max_steps: int = 1000) -> ConsoleAgent:
    """Wire a Tier-0 FreeCAD console agent over an INJECTED GUI bridge.

    ``bridge`` is anything exposing ``.call(source[, timeout])`` that runs Python in the
    live FreeCAD interpreter -- in practice a started
    :class:`~harnesscad.io.cua.viewport.FreeCADGuiBridge`. The bridge is composed, not
    created here, and ``environment_freecad`` is untouched. Nothing runs until the
    returned agent's :meth:`ConsoleAgent.run` is called.
    """
    channel = CallableConsoleChannel(call_console_eval(bridge, timeout=timeout))
    return ConsoleAgent(channel=channel, done_markers=FREECAD_DONE_MARKERS,
                        error_markers=FREECAD_ERROR_MARKERS, inputs=inputs,
                        max_steps=max_steps, app="freecad")


def build_blender_console_agent(eval_fn: EvalFn, *, inputs: Sequence[str] = (),
                                max_steps: int = 1000) -> ConsoleAgent:
    """Wire a Tier-0 Blender console agent over an INJECTED ``bpy`` eval callable.

    ``eval_fn`` runs a line of Python in Blender's interpreter and returns the console
    text (Blender is invisible to UIA and is driven via ``bpy``, per the synthesis).
    Injected from outside; this module builds no Blender coupling of its own.
    """
    channel = CallableConsoleChannel(callable_console_eval(eval_fn))
    return ConsoleAgent(channel=channel, done_markers=BLENDER_DONE_MARKERS,
                        error_markers=BLENDER_ERROR_MARKERS, inputs=inputs,
                        max_steps=max_steps, app="blender")
