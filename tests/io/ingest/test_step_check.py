"""Tests for io.ingest.step_check -- subprocess isolation, the timeout kill, and
the empty-vs-malformed distinction.

The reader is pinned to ``part21`` wherever the result is asserted, so these
tests assert the same thing whether or not OCCT is installed.
"""

import os
import subprocess
import sys
import tempfile
import time
import unittest

from harnesscad.io.ingest.step_check import (
    BBOX_EPS,
    DEFAULT_TIMEOUT_S,
    STATUSES,
    StepCheckResult,
    bbox_overlap,
    check_step,
    interpenetration_candidates,
)

_GOOD = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('t'),'2;1');
FILE_NAME('t.step','2026-07-16',(''),(''),'','','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=CLOSED_SHELL('',(#3));
#3=ADVANCED_FACE('',(),#1,.T.);
#4=MANIFOLD_SOLID_BREP('body',#2);
ENDSEC;
END-ISO-10303-21;
"""

#: Well-formed part-21 carrying no shape entity: parses, contains nothing.
_EMPTY = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('t'),'2;1');
FILE_NAME('e.step','2026-07-16',(''),(''),'','','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=DIRECTION('',(0.,0.,1.));
ENDSEC;
END-ISO-10303-21;
"""

#: Unterminated string literal + truncated entity: the reader must FAIL.
_BROKEN = """ISO-10303-21;
HEADER;
ENDSEC;
DATA;
#1=CARTESIAN_POINT('unterminated,(0.,0.,0.;
#2=MANIFOLD_SOLID_BREP('body',#
ENDSEC;
"""


class _TmpStepMixin:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def write(self, name, text):
        path = os.path.join(self.tmp.name, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path


class WellFormedArmTest(_TmpStepMixin, unittest.TestCase):
    """Arm 1: a well-formed STEP parses."""

    def test_well_formed_step_parses(self):
        r = check_step(self.write("g.step", _GOOD), reader="part21")
        self.assertEqual(r.status, "ok")
        self.assertTrue(r.ok)
        self.assertTrue(r.parsed)
        self.assertGreaterEqual(r.roots, 1)

    def test_reports_which_reader_ran(self):
        r = check_step(self.write("g.step", _GOOD), reader="part21")
        self.assertEqual(r.reader, "part21")

    def test_runs_out_of_process(self):
        # The whole premise: the read happened somewhere this process cannot be
        # killed from. The worker reports its own pid, so the isolation is
        # proven rather than assumed.
        r = check_step(self.write("g.step", _GOOD), reader="part21")
        self.assertTrue(r.ok)
        self.assertIn("pid", r.details)
        self.assertNotEqual(r.details["pid"], os.getpid())

    def test_result_is_serialisable(self):
        r = check_step(self.write("g.step", _GOOD), reader="part21")
        d = r.to_dict()
        self.assertEqual(d["status"], "ok")
        self.assertTrue(d["ok"] and d["parsed"])

    def test_deterministic_across_runs(self):
        path = self.write("g.step", _GOOD)
        first = check_step(path, reader="part21")
        second = check_step(path, reader="part21")
        self.assertEqual(first.status, second.status)
        self.assertEqual(first.roots, second.roots)


class TimeoutArmTest(_TmpStepMixin, unittest.TestCase):
    """Arm 2: a hostile/hanging input is killed, not waited on."""

    def _hanging_worker(self, seconds=60):
        return [sys.executable, "-c", f"import time; time.sleep({seconds})"]

    def test_hanging_worker_is_killed_by_the_timeout(self):
        path = self.write("g.step", _GOOD)
        start = time.monotonic()
        r = check_step(path, timeout_s=1.0, worker_cmd=self._hanging_worker())
        elapsed = time.monotonic() - start
        self.assertEqual(r.status, "timeout")
        self.assertTrue(r.timed_out)
        self.assertTrue(r.killed)
        # The worker wanted 60s. The parent must not have given it 60s.
        self.assertLess(elapsed, 30.0)

    def test_killed_worker_is_reaped_not_leaked(self):
        # A returncode exists only if the parent actually waited on the corpse.
        r = check_step(self.write("g.step", _GOOD), timeout_s=1.0,
                       worker_cmd=self._hanging_worker())
        self.assertIsNotNone(r.returncode)

    def test_timeout_is_not_reported_as_parsed(self):
        r = check_step(self.write("g.step", _GOOD), timeout_s=1.0,
                       worker_cmd=self._hanging_worker())
        self.assertFalse(r.parsed)
        self.assertFalse(r.ok)
        self.assertFalse(r.empty)

    def test_timeout_note_states_the_budget(self):
        r = check_step(self.write("g.step", _GOOD), timeout_s=1.0,
                       worker_cmd=self._hanging_worker())
        self.assertIn("1", r.note)

    def test_fast_worker_does_not_trip_the_timeout(self):
        r = check_step(self.write("g.step", _GOOD), timeout_s=60.0,
                       reader="part21")
        self.assertFalse(r.timed_out)
        self.assertFalse(r.killed)

    def test_worker_dying_silently_is_crashed_not_empty(self):
        # A segfaulting kernel produces no report. Calling that "empty" would be
        # the silence-is-success bug wearing a different hat.
        die = [sys.executable, "-c", "import os; os._exit(139)"]
        r = check_step(self.write("g.step", _GOOD), worker_cmd=die)
        self.assertEqual(r.status, "crashed")
        self.assertFalse(r.parsed)
        self.assertFalse(r.empty)

    def test_worker_emitting_garbage_is_an_error(self):
        liar = [sys.executable, "-c", "print('not json at all')"]
        r = check_step(self.write("g.step", _GOOD), worker_cmd=liar)
        self.assertEqual(r.status, "error")
        self.assertFalse(r.parsed)

    def test_worker_emitting_unknown_status_is_an_error(self):
        liar = [sys.executable, "-c",
                "import json; print(json.dumps({'status': 'brilliant'}))"]
        r = check_step(self.write("g.step", _GOOD), worker_cmd=liar)
        self.assertEqual(r.status, "error")

    def test_unstartable_worker_is_an_error_not_a_crash(self):
        r = check_step(self.write("g.step", _GOOD),
                       worker_cmd=["definitely-not-a-real-binary-xyz"])
        self.assertEqual(r.status, "error")
        self.assertIn("cannot start worker", r.note)


class EmptyVersusMalformedTest(_TmpStepMixin, unittest.TestCase):
    """The distinction: a parse that yields nothing != a parse that failed."""

    def test_empty_file_is_empty(self):
        r = check_step(self.write("e.step", _EMPTY), reader="part21")
        self.assertEqual(r.status, "empty")
        self.assertEqual(r.roots, 0)

    def test_empty_file_counts_as_parsed(self):
        r = check_step(self.write("e.step", _EMPTY), reader="part21")
        self.assertTrue(r.parsed)   # the reader SUCCEEDED
        self.assertTrue(r.empty)
        self.assertFalse(r.ok)      # ... and found no geometry

    def test_malformed_file_is_malformed(self):
        r = check_step(self.write("b.step", _BROKEN), reader="part21")
        self.assertEqual(r.status, "malformed")

    def test_malformed_file_does_not_count_as_parsed(self):
        r = check_step(self.write("b.step", _BROKEN), reader="part21")
        self.assertFalse(r.parsed)
        self.assertFalse(r.empty)
        self.assertFalse(r.ok)

    def test_empty_and_malformed_are_never_the_same_status(self):
        empty = check_step(self.write("e.step", _EMPTY), reader="part21")
        broken = check_step(self.write("b.step", _BROKEN), reader="part21")
        self.assertNotEqual(empty.status, broken.status)
        self.assertNotEqual(empty.parsed, broken.parsed)

    def test_malformed_note_carries_the_reason(self):
        r = check_step(self.write("b.step", _BROKEN), reader="part21")
        self.assertTrue(r.note)

    def test_neither_empty_nor_malformed_is_ok(self):
        for text, name in ((_EMPTY, "e.step"), (_BROKEN, "b.step")):
            with self.subTest(name=name):
                self.assertFalse(check_step(self.write(name, text),
                                            reader="part21").ok)


class GuardTest(_TmpStepMixin, unittest.TestCase):
    def test_missing_file_is_missing(self):
        r = check_step(os.path.join(self.tmp.name, "nope.step"))
        self.assertEqual(r.status, "missing")
        self.assertFalse(r.parsed)

    def test_directory_is_missing_not_malformed(self):
        r = check_step(self.tmp.name)
        self.assertEqual(r.status, "missing")

    def test_empty_path_never_spawns_a_worker(self):
        self.assertEqual(check_step("").status, "missing")

    def test_unknown_reader_is_rejected(self):
        r = check_step(self.write("g.step", _GOOD), reader="haruspicy")
        self.assertEqual(r.status, "error")

    def test_check_step_never_raises(self):
        for path in ("", "/nonexistent/x.step", self.tmp.name):
            with self.subTest(path=path):
                self.assertIsInstance(check_step(path), StepCheckResult)

    def test_every_reported_status_is_declared(self):
        results = [
            check_step(self.write("g.step", _GOOD), reader="part21"),
            check_step(self.write("e.step", _EMPTY), reader="part21"),
            check_step(self.write("b.step", _BROKEN), reader="part21"),
            check_step(""),
        ]
        for r in results:
            with self.subTest(status=r.status):
                self.assertIn(r.status, STATUSES)

    def test_default_timeout_is_finite(self):
        self.assertGreater(DEFAULT_TIMEOUT_S, 0)
        self.assertLess(DEFAULT_TIMEOUT_S, 3600)


class WorkerTest(_TmpStepMixin, unittest.TestCase):
    """The worker's own CLI contract, exercised directly."""

    def _run(self, args):
        return subprocess.run(
            [sys.executable, "-m", "harnesscad.io.ingest._step_check_worker"]
            + args, capture_output=True, text=True, timeout=60)

    def test_worker_exits_zero_on_a_good_file(self):
        p = self._run(["--reader", "part21", self.write("g.step", _GOOD)])
        self.assertEqual(p.returncode, 0)
        self.assertIn('"status": "ok"', p.stdout)

    def test_worker_exits_zero_on_an_empty_file(self):
        # Empty is a successful read: exit 0, status empty.
        p = self._run(["--reader", "part21", self.write("e.step", _EMPTY)])
        self.assertEqual(p.returncode, 0)
        self.assertIn('"status": "empty"', p.stdout)

    def test_worker_exits_nonzero_on_a_malformed_file(self):
        p = self._run(["--reader", "part21", self.write("b.step", _BROKEN)])
        self.assertEqual(p.returncode, 1)
        self.assertIn('"status": "malformed"', p.stdout)

    def test_worker_rejects_bad_usage(self):
        self.assertEqual(self._run([]).returncode, 2)
        self.assertEqual(self._run(["--reader", "tarot", "x.step"]).returncode, 2)


class BboxPrefilterTest(unittest.TestCase):
    """The cheap half of the interpenetration metric."""

    def test_overlapping_boxes_overlap(self):
        self.assertTrue(bbox_overlap((0, 0, 0, 1, 1, 1),
                                     (0.5, 0.5, 0.5, 1.5, 1.5, 1.5)))

    def test_disjoint_boxes_do_not(self):
        self.assertFalse(bbox_overlap((0, 0, 0, 1, 1, 1), (5, 5, 5, 6, 6, 6)))

    def test_face_touching_boxes_do_not_interpenetrate(self):
        # Two parts sharing a mating face are assembled, not interpenetrating.
        self.assertFalse(bbox_overlap((0, 0, 0, 1, 1, 1), (1, 0, 0, 2, 1, 1)))

    def test_separation_on_a_single_axis_is_enough(self):
        # Overlapping in x and y but not z.
        self.assertFalse(bbox_overlap((0, 0, 0, 1, 1, 1), (0, 0, 5, 1, 1, 6)))

    def test_sub_epsilon_overlap_is_numerical_noise(self):
        a = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        b = (1.0 - BBOX_EPS / 2, 0.0, 0.0, 2.0, 1.0, 1.0)
        self.assertFalse(bbox_overlap(a, b))

    def test_candidates_are_the_overlapping_pairs_only(self):
        boxes = [
            (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            (0.5, 0.5, 0.5, 1.5, 1.5, 1.5),
            (5.0, 5.0, 5.0, 6.0, 6.0, 6.0),
        ]
        self.assertEqual(interpenetration_candidates(boxes), [(0, 1)])

    def test_candidates_are_deterministic_and_ordered(self):
        boxes = [(0, 0, 0, 2, 2, 2)] * 3
        self.assertEqual(interpenetration_candidates(boxes),
                         [(0, 1), (0, 2), (1, 2)])

    def test_no_self_pairs(self):
        pairs = interpenetration_candidates([(0, 0, 0, 1, 1, 1)] * 4)
        self.assertTrue(all(i != j for i, j in pairs))

    def test_trivial_inputs(self):
        self.assertEqual(interpenetration_candidates([]), [])
        self.assertEqual(interpenetration_candidates([(0, 0, 0, 1, 1, 1)]), [])


class SelfcheckTest(unittest.TestCase):
    def test_selfcheck_exits_zero(self):
        from harnesscad.io.ingest.step_check import main
        self.assertEqual(main(["--selfcheck"]), 0)


if __name__ == "__main__":
    unittest.main()
