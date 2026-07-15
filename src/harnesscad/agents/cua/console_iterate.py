"""console_iterate — a deterministic controller for driving an app's Python console.

Ported from BabyCommandAGI's iterate-on-a-CLI loop. BabyCommandAGI does not click:
it types a command, reads the result, and ADJUDICATES the result into one of four
verdicts — the task is Complete, it must Interrupt (a failure to repair), it should
Continue (more steps remain), or the program is blocked waiting for <input> and it
must supply some. That single adjudication step, run after every command, is the
whole controller; everything else is bookkeeping.

Why this is the cheapest correct CAD agent
------------------------------------------
FreeCAD and Blender both ship a Python console. So a CAD task can be driven ENTIRELY
by typing Python into that console and reading the result back — ``Part.makeBox(10,
10,10)`` typed and its repr/error read — with no vision, no coordinate grounding, no
a11y tree. The console echoes an exception as a traceback the moment a step is
wrong, which makes it a TEXT-VERIFIED GUI agent: the success signal is the console's
own text, not a model's opinion of a screenshot. This is the same insight as
:mod:`harnesscad.io.cua.uia` (never trust a return, verify the outcome) applied to a
REPL instead of the accessibility tree.

Two ideas ported precisely
---------------------------
* **The stall-adjudicator** (:func:`adjudicate`). A pure function from the console's
  latest output (plus the previous output, to detect a stall) to a
  :class:`Verdict`. Done markers -> COMPLETE; an input prompt at the tail -> INPUT;
  an error/traceback -> INTERRUPT; identical-to-previous output -> INTERRUPT
  (stalled, making no progress); otherwise CONTINUE.
* **Failure-focused log slicing** (:func:`focus_failure`). A console can emit
  thousands of lines; the part that matters is the region AROUND the failure. This
  keeps a window of lines centred on the first error line (with a head/tail
  fallback) so a long log is trimmed to its diagnostic core, exactly as
  BabyCommandAGI trims before reasoning over a result.

Pure stdlib, deterministic, import-safe. The console itself is injected as a
:class:`ConsoleChannel`; the tests drive a scripted fake, and no live app,
subprocess, or model is ever run here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Sequence, Tuple


class Verdict(str, Enum):
    """The four states BabyCommandAGI adjudicates a console result into."""

    COMPLETE = "complete"     # a done-marker seen; the task finished
    CONTINUE = "continue"     # progress made, more steps remain
    INTERRUPT = "interrupt"   # a failure (error, or a stall making no progress)
    INPUT = "input"           # the program is blocked awaiting input


#: Default text markers. A console reports success/failure/blocking in-band, and
#: these are the tails that carry that signal. Callers override per app.
DEFAULT_DONE_MARKERS: Tuple[str, ...] = ("__DONE__", "TASK COMPLETE", "OK: done")
DEFAULT_ERROR_MARKERS: Tuple[str, ...] = (
    "Traceback (most recent call last)", "Error:", "Exception:",
    "SyntaxError", "NameError", "TypeError", "ValueError", "AttributeError",
    "RuntimeError", "KeyError", "IndexError",
)
#: Prompt tails that mean the REPL is WAITING for the user to type something.
#: ``... `` is Python's continuation prompt (an unclosed block); ``: `` / ``? ``
#: are input()-style prompts.
DEFAULT_PROMPT_MARKERS: Tuple[str, ...] = ("... ", ">>> ", ": ", "? ")


@dataclass(frozen=True)
class Adjudication:
    """The verdict for one console result, with the evidence that produced it.

    ``prompt`` is the trailing prompt line when the verdict is INPUT; ``marker`` is
    the specific done/error string that fired; ``reason`` is a short human note.
    """

    verdict: Verdict
    reason: str = ""
    marker: str = ""
    prompt: str = ""

    def to_dict(self) -> dict:
        return {"verdict": self.verdict.value, "reason": self.reason,
                "marker": self.marker, "prompt": self.prompt}


def _last_nonempty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        if line.strip():
            return line
    return ""


def _first_hit(text: str, markers: Sequence[str]) -> str:
    """The earliest-occurring marker in ``text`` (by index), or ""; deterministic."""
    best_pos = -1
    best = ""
    for m in markers:
        pos = text.find(m)
        if pos != -1 and (best_pos == -1 or pos < best_pos):
            best_pos, best = pos, m
    return best


def adjudicate(output: str, *,
               previous: Optional[str] = None,
               done_markers: Sequence[str] = DEFAULT_DONE_MARKERS,
               error_markers: Sequence[str] = DEFAULT_ERROR_MARKERS,
               prompt_markers: Sequence[str] = DEFAULT_PROMPT_MARKERS) -> Adjudication:
    """Classify one console result into a :class:`Verdict`. Pure and deterministic.

    Priority is deliberate and fixed, because two signals can co-occur:

    1. **COMPLETE** — a done-marker anywhere in the output wins over everything: a
       task that announced completion is done even if a warning also printed.
    2. **INTERRUPT (error)** — an error/traceback marker: a failure to repair, not
       a step to continue past.
    3. **INPUT** — no error, but the tail is an input/continuation prompt: the
       program is blocked and needs a value before it will make progress.
    4. **INTERRUPT (stall)** — output byte-identical to ``previous``: the last
       command changed nothing, so continuing would loop forever.
    5. **CONTINUE** — progress was made and none of the above; take the next step.
    """
    done = _first_hit(output, done_markers)
    if done:
        return Adjudication(Verdict.COMPLETE, reason="done marker seen", marker=done)
    err = _first_hit(output, error_markers)
    if err:
        return Adjudication(Verdict.INTERRUPT, reason="error in output", marker=err)
    tail = _last_nonempty_line(output)
    for p in prompt_markers:
        if tail.endswith(p):
            return Adjudication(Verdict.INPUT, reason="console awaiting input",
                                prompt=tail)
    if previous is not None and output == previous:
        return Adjudication(Verdict.INTERRUPT,
                            reason="stalled: output unchanged from the previous step")
    return Adjudication(Verdict.CONTINUE, reason="progress; more steps remain")


def focus_failure(log: str, *, radius: int = 6, max_lines: int = 40,
                  error_markers: Sequence[str] = DEFAULT_ERROR_MARKERS) -> str:
    """Trim a long console ``log`` to the region AROUND its first failure.

    Keeps ``radius`` lines on each side of the first error line, capped at
    ``max_lines`` total. With no error found, falls back to a head+tail window (the
    start sets context, the end carries the latest state) so the result is always
    bounded. An elision marker records how many lines were dropped, so the slice is
    honest about being a slice. Deterministic: same log in, same slice out.
    """
    lines = log.splitlines()
    if len(lines) <= max_lines:
        return log
    err_idx = _first_error_line(lines, error_markers)
    if err_idx is not None:
        lo = max(0, err_idx - radius)
        hi = min(len(lines), err_idx + radius + 1)
        # Grow the window symmetrically toward max_lines if there is room.
        while (hi - lo) < max_lines and (lo > 0 or hi < len(lines)):
            if lo > 0:
                lo -= 1
            if (hi - lo) < max_lines and hi < len(lines):
                hi += 1
        return _join_with_elision(lines, lo, hi)
    # No error: head + tail.
    head = max_lines // 2
    tail = max_lines - head
    kept = lines[:head] + ["... [%d lines elided] ..." % (len(lines) - max_lines)] \
        + lines[len(lines) - tail:]
    return "\n".join(kept)


def _first_error_line(lines: Sequence[str], markers: Sequence[str]) -> Optional[int]:
    for i, line in enumerate(lines):
        for m in markers:
            if m in line:
                return i
    return None


def _join_with_elision(lines: Sequence[str], lo: int, hi: int) -> str:
    out: List[str] = []
    if lo > 0:
        out.append("... [%d lines above elided] ..." % lo)
    out.extend(lines[lo:hi])
    if hi < len(lines):
        out.append("... [%d lines below elided] ..." % (len(lines) - hi))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# The console channel + the controller loop.
# ---------------------------------------------------------------------------


class ConsoleChannel:
    """The narrow interface a Python console must present to be driven.

    Two operations: ``write`` a line of source, and ``read_new`` the output that
    has accrued since the last read. That is the whole contract with a live console
    (FreeCAD's ``PythonConsole``, Blender's interactive console, a subprocess
    REPL); a test supplies a scripted fake. This class is abstract on purpose —
    instantiating the base raises, so a real driver must be provided.
    """

    def write(self, source: str) -> None:
        raise NotImplementedError

    def read_new(self) -> str:
        raise NotImplementedError


@dataclass
class Step:
    """One command to type, and an optional success substring to require.

    ``expect``, when set, must appear in the step's output for the step to be
    considered satisfied even if the adjudicator would otherwise say CONTINUE — the
    text-verification hook that turns "I typed it" into "I confirmed it".
    """

    source: str
    expect: Optional[str] = None


@dataclass
class StepResult:
    """What happened for one driven step: the source, the output, the verdict."""

    source: str
    output: str
    adjudication: Adjudication
    satisfied: bool = False

    def to_dict(self) -> dict:
        return {"source": self.source, "output": self.output,
                "adjudication": self.adjudication.to_dict(),
                "satisfied": self.satisfied}


@dataclass
class Transcript:
    """The whole driven run: every step, the terminal verdict, and a focused log."""

    results: List[StepResult] = field(default_factory=list)
    verdict: Verdict = Verdict.CONTINUE
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.verdict is Verdict.COMPLETE

    @property
    def full_log(self) -> str:
        return "\n".join(r.output for r in self.results)

    def focused_log(self, **kw) -> str:
        """The run's output, trimmed to its failure region (see :func:`focus_failure`)."""
        return focus_failure(self.full_log, **kw)

    def to_dict(self) -> dict:
        return {"verdict": self.verdict.value, "reason": self.reason, "ok": self.ok,
                "results": [r.to_dict() for r in self.results]}


class ConsoleController:
    """Drive a Python console step by step, adjudicating after each command.

    Deterministic given the channel: the controller sends a step's source, reads
    the new output, adjudicates, and acts on the verdict — COMPLETE stops with
    success, INTERRUPT stops with the failure recorded, INPUT pulls the next queued
    input and re-reads (blocking if none is queued), CONTINUE advances. A run also
    stops if it exhausts the steps or hits ``max_steps`` (a runaway guard).

    ``inputs`` is a queue of strings the controller feeds whenever the console
    blocks on input — the deterministic stand-in for a human answering a prompt.
    """

    def __init__(self, channel: ConsoleChannel, *,
                 done_markers: Sequence[str] = DEFAULT_DONE_MARKERS,
                 error_markers: Sequence[str] = DEFAULT_ERROR_MARKERS,
                 prompt_markers: Sequence[str] = DEFAULT_PROMPT_MARKERS,
                 inputs: Optional[Sequence[str]] = None,
                 max_steps: int = 1000) -> None:
        self.channel = channel
        self.done_markers = tuple(done_markers)
        self.error_markers = tuple(error_markers)
        self.prompt_markers = tuple(prompt_markers)
        self._inputs: List[str] = list(inputs or ())
        self.max_steps = int(max_steps)

    def _adjudicate(self, output: str, previous: Optional[str]) -> Adjudication:
        return adjudicate(output, previous=previous,
                          done_markers=self.done_markers,
                          error_markers=self.error_markers,
                          prompt_markers=self.prompt_markers)

    def run(self, steps: Sequence[Step]) -> Transcript:
        """Type each step, adjudicate, and stop at the first terminal verdict."""
        transcript = Transcript()
        previous: Optional[str] = None
        budget = self.max_steps
        for step in steps:
            if budget <= 0:
                transcript.verdict = Verdict.INTERRUPT
                transcript.reason = "step budget exhausted (%d)" % self.max_steps
                return transcript
            budget -= 1
            self.channel.write(step.source)
            output = self.channel.read_new()
            adj = self._adjudicate(output, previous)

            # If blocked on input, feed queued input and re-read, once, so a prompt
            # that a value resolves does not end the run prematurely.
            while adj.verdict is Verdict.INPUT and self._inputs and budget > 0:
                budget -= 1
                feed = self._inputs.pop(0)
                self.channel.write(feed)
                extra = self.channel.read_new()
                output = output + extra
                adj = self._adjudicate(output, previous)

            satisfied = step.expect is None or (step.expect in output)
            transcript.results.append(
                StepResult(step.source, output, adj, satisfied=satisfied))
            previous = output

            if adj.verdict is Verdict.COMPLETE:
                transcript.verdict = Verdict.COMPLETE
                transcript.reason = adj.reason
                return transcript
            if adj.verdict is Verdict.INTERRUPT:
                transcript.verdict = Verdict.INTERRUPT
                transcript.reason = adj.reason
                return transcript
            if adj.verdict is Verdict.INPUT:
                # Still blocked and no input left to give: a real interruption.
                transcript.verdict = Verdict.INTERRUPT
                transcript.reason = "blocked on input with no queued value: %r" % adj.prompt
                return transcript
            if step.expect is not None and not satisfied:
                transcript.verdict = Verdict.INTERRUPT
                transcript.reason = ("step did not produce expected text %r"
                                     % step.expect)
                return transcript
            # CONTINUE: next step.

        # Ran out of steps without a done-marker. That is not a failure by itself,
        # but it is not a proven completion either — report the last verdict.
        transcript.verdict = (transcript.results[-1].adjudication.verdict
                              if transcript.results else Verdict.CONTINUE)
        transcript.reason = "steps exhausted without a completion marker"
        return transcript
