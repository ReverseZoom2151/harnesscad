"""Tests for the BabyCommandAGI-derived console-iterate controller."""

import unittest

from harnesscad.agents.cua.console_iterate import (
    Adjudication, ConsoleChannel, ConsoleController, Step, Transcript, Verdict,
    adjudicate, focus_failure,
)


class AdjudicateTest(unittest.TestCase):
    def test_done_marker_completes(self):
        a = adjudicate("built the box\n__DONE__")
        self.assertEqual(a.verdict, Verdict.COMPLETE)
        self.assertEqual(a.marker, "__DONE__")

    def test_traceback_interrupts(self):
        out = "Traceback (most recent call last):\n  ...\nNameError: name 'Prt'"
        a = adjudicate(out)
        self.assertEqual(a.verdict, Verdict.INTERRUPT)

    def test_input_prompt_detected(self):
        a = adjudicate(">>> x = 1\n... ")
        self.assertEqual(a.verdict, Verdict.INPUT)
        self.assertTrue(a.prompt.endswith("... "))

    def test_stall_is_interrupt(self):
        prev = "same output\n"
        a = adjudicate("same output\n", previous=prev)
        self.assertEqual(a.verdict, Verdict.INTERRUPT)
        self.assertIn("stall", a.reason)

    def test_progress_continues(self):
        a = adjudicate("created Body001", previous="created Body")
        self.assertEqual(a.verdict, Verdict.CONTINUE)

    def test_done_beats_error(self):
        # A done marker present alongside a warning-that-looks-like-error still wins.
        a = adjudicate("ValueError: ignored\n__DONE__")
        self.assertEqual(a.verdict, Verdict.COMPLETE)

    def test_first_marker_by_position(self):
        a = adjudicate("NameError first then TypeError")
        self.assertEqual(a.verdict, Verdict.INTERRUPT)
        self.assertEqual(a.marker, "NameError")


class FocusFailureTest(unittest.TestCase):
    def test_short_log_unchanged(self):
        log = "\n".join("line %d" % i for i in range(5))
        self.assertEqual(focus_failure(log, max_lines=40), log)

    def test_keeps_region_around_error(self):
        lines = ["noise %d" % i for i in range(100)]
        lines[60] = "RuntimeError: boom"
        log = "\n".join(lines)
        out = focus_failure(log, radius=3, max_lines=12)
        self.assertIn("RuntimeError: boom", out)
        self.assertLessEqual(len(out.splitlines()), 12 + 2)  # + elision markers
        self.assertIn("elided", out)

    def test_head_tail_fallback_when_no_error(self):
        lines = ["quiet %d" % i for i in range(100)]
        out = focus_failure("\n".join(lines), max_lines=10)
        self.assertIn("quiet 0", out)
        self.assertIn("quiet 99", out)
        self.assertIn("elided", out)


class _ScriptedConsole(ConsoleChannel):
    """A fake console: each written source maps to a scripted output chunk.

    ``script`` is a list of output strings returned in order on each read; the last
    write's read pops the next chunk. Records everything written for assertions.
    """

    def __init__(self, script):
        self._script = list(script)
        self.written = []

    def write(self, source):
        self.written.append(source)

    def read_new(self):
        return self._script.pop(0) if self._script else ""


class ControllerTest(unittest.TestCase):
    def test_happy_path_completes(self):
        console = _ScriptedConsole([
            "Body created",
            "Pad created",
            "volume = 1000.0\n__DONE__",
        ])
        ctrl = ConsoleController(console)
        t = ctrl.run([Step("makeBody()"), Step("makePad()"), Step("report()")])
        self.assertTrue(t.ok)
        self.assertEqual(t.verdict, Verdict.COMPLETE)
        self.assertEqual(len(t.results), 3)
        self.assertEqual(console.written, ["makeBody()", "makePad()", "report()"])

    def test_error_interrupts_and_stops_early(self):
        console = _ScriptedConsole([
            "Body created",
            "Traceback (most recent call last):\nNameError: Pad",
            "should never run",
        ])
        ctrl = ConsoleController(console)
        t = ctrl.run([Step("makeBody()"), Step("oops()"), Step("more()")])
        self.assertEqual(t.verdict, Verdict.INTERRUPT)
        self.assertEqual(len(t.results), 2)   # stopped after the failing step
        self.assertFalse(t.ok)

    def test_input_prompt_fed_from_queue(self):
        # First step blocks on input; the queued value unblocks and completes.
        console = _ScriptedConsole([
            "enter width: ",     # blocks (ends with ': ')
            "width set\n__DONE__",   # after feeding the input
        ])
        ctrl = ConsoleController(console, inputs=["10"])
        t = ctrl.run([Step("ask_width()")])
        self.assertEqual(t.verdict, Verdict.COMPLETE)
        self.assertIn("10", console.written)

    def test_blocked_without_input_interrupts(self):
        console = _ScriptedConsole(["enter width: "])
        ctrl = ConsoleController(console)   # no inputs queued
        t = ctrl.run([Step("ask_width()")])
        self.assertEqual(t.verdict, Verdict.INTERRUPT)
        self.assertIn("blocked on input", t.reason)

    def test_expect_mismatch_interrupts(self):
        console = _ScriptedConsole(["volume = 999"])
        ctrl = ConsoleController(console)
        t = ctrl.run([Step("report()", expect="volume = 1000")])
        self.assertEqual(t.verdict, Verdict.INTERRUPT)
        self.assertIn("expected text", t.reason)

    def test_max_steps_guard(self):
        console = _ScriptedConsole(["a", "b", "c"])
        ctrl = ConsoleController(console, max_steps=1)
        t = ctrl.run([Step("s1"), Step("s2"), Step("s3")])
        # first step consumes the budget; second trips the guard.
        self.assertEqual(t.verdict, Verdict.INTERRUPT)
        self.assertIn("budget", t.reason)

    def test_transcript_focused_log(self):
        big = ["line %d" % i for i in range(80)]
        big[40] = "TypeError: bad"
        console = _ScriptedConsole(["\n".join(big)])
        ctrl = ConsoleController(console)
        t = ctrl.run([Step("go()")])
        self.assertEqual(t.verdict, Verdict.INTERRUPT)
        self.assertIn("TypeError: bad", t.focused_log(radius=2, max_lines=10))


class ChannelContractTest(unittest.TestCase):
    def test_base_channel_is_abstract(self):
        with self.assertRaises(NotImplementedError):
            ConsoleChannel().write("x")
        with self.assertRaises(NotImplementedError):
            ConsoleChannel().read_new()


if __name__ == "__main__":
    unittest.main()
