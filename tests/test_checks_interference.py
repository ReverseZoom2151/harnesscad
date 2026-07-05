"""Tests for the standalone interference / collision gate (checks_interference).

The exact OCCT boolean-common path needs cadquery/OCCT, which the dependency-free
test environment may not have; these tests therefore exercise the *pure-python
bounding-box fallback* (always available) plus graceful degradation:

  * two overlapping AABB parts -> approximate WARNING clash;
  * disjoint parts -> no clash (INFO clear);
  * the broad-phase sweep still finds a clash among many parts;
  * INFO-skip on a StubBackend with no 'assembly' query, and never a crash.

When cadquery *is* installed, an extra test confirms the exact solid path flags
two overlapping boxes as a hard ERROR.
"""

import unittest

from backends.stub import StubBackend
from verifiers.verify import Severity
from verifiers.interference import (
    InterferenceCheck, Clash, with_interference,
    _sweep_and_prune, _aabb_overlap, _overlap_dims,
)


def _codes(report):
    return {d.code for d in report.diagnostics}


def _by_severity(report, sev):
    return [d for d in report.diagnostics if d.severity is sev]


def _box(id_, xmin, ymin, zmin, xmax, ymax, zmax):
    return {"id": id_, "bbox": [xmin, ymin, zmin, xmax, ymax, zmax]}


class _AssemblyBackend:
    def __init__(self, parts):
        self._parts = parts

    def query(self, q: str) -> dict:
        if q == "assembly":
            return {"parts": self._parts, "mates": []}
        return {}


class TestBroadPhase(unittest.TestCase):
    def test_aabb_overlap_predicate(self):
        a = (0, 0, 0, 10, 10, 10)
        b = (5, 5, 5, 15, 15, 15)
        c = (20, 20, 20, 30, 30, 30)
        self.assertTrue(_aabb_overlap(a, b))
        self.assertFalse(_aabb_overlap(a, c))

    def test_sweep_and_prune_finds_only_overlapping_pairs(self):
        boxes = [
            (0, 0, 0, 10, 10, 10),     # 0
            (5, 5, 5, 15, 15, 15),     # 1 overlaps 0
            (100, 0, 0, 110, 10, 10),  # 2 far away
        ]
        pairs = _sweep_and_prune(boxes)
        self.assertIn((0, 1), pairs)
        self.assertNotIn((0, 2), pairs)
        self.assertNotIn((1, 2), pairs)

    def test_sweep_skips_boxless_parts(self):
        boxes = [(0, 0, 0, 1, 1, 1), None, (0, 0, 0, 1, 1, 1)]
        pairs = _sweep_and_prune(boxes)
        self.assertEqual(pairs, [(0, 2)])

    def test_overlap_dims(self):
        a = (0, 0, 0, 10, 10, 10)
        b = (6, 0, 0, 16, 4, 10)
        self.assertEqual(_overlap_dims(a, b), (4.0, 4.0, 10.0))


class TestInterferenceFallback(unittest.TestCase):
    """Pure-python bbox fallback (no cadquery required)."""

    def test_overlapping_parts_flagged_approx(self):
        parts = [_box("p1", 0, 0, 0, 10, 10, 10),
                 _box("p2", 5, 5, 5, 15, 15, 15)]
        report = InterferenceCheck().check(_AssemblyBackend(parts), None)
        codes = _codes(report)
        # No OCCT here -> approximate WARNING, not a hard ERROR.
        self.assertIn("interference-approx", codes)
        warns = _by_severity(report, Severity.WARNING)
        self.assertTrue(any(c.code == "interference-approx" for c in warns))

    def test_disjoint_parts_pass(self):
        parts = [_box("p1", 0, 0, 0, 10, 10, 10),
                 _box("p2", 50, 50, 50, 60, 60, 60)]
        report = InterferenceCheck().check(_AssemblyBackend(parts), None)
        codes = _codes(report)
        self.assertIn("interference-clear", codes)
        self.assertNotIn("interference-approx", codes)
        self.assertNotIn("interference", codes)
        self.assertEqual(_by_severity(report, Severity.ERROR), [])
        self.assertTrue(report.ok)

    def test_touching_faces_are_not_a_clash(self):
        # Share a face at x=10: overlap volume is zero -> below min_volume.
        parts = [_box("p1", 0, 0, 0, 10, 10, 10),
                 _box("p2", 10, 0, 0, 20, 10, 10)]
        report = InterferenceCheck().check(_AssemblyBackend(parts), None)
        self.assertNotIn("interference-approx", _codes(report))
        self.assertTrue(report.ok)

    def test_ranked_worst_first(self):
        # p1 overlaps both p2 (big) and p3 (small); big clash reported first.
        parts = [
            _box("p1", 0, 0, 0, 20, 20, 20),
            _box("p2", 1, 1, 1, 19, 19, 19),   # large overlap
            _box("p3", 19, 19, 19, 21, 21, 21),  # tiny overlap corner
        ]
        report = InterferenceCheck().check(_AssemblyBackend(parts), None)
        approx = [d for d in report.diagnostics
                  if d.code == "interference-approx"]
        self.assertGreaterEqual(len(approx), 2)
        # Worst-first: the p1/p2 clash message precedes the p1/p3 one.
        self.assertIn("p2", approx[0].message)

    def test_not_measurable_when_no_bbox_and_no_occt(self):
        # Parts without bbox can never enter the x-sweep, so they cannot clash;
        # a single measurable box plus a boxless one just yields "clear".
        parts = [{"id": "p1"}, {"id": "p2"}]
        report = InterferenceCheck().check(_AssemblyBackend(parts), None)
        self.assertTrue(report.ok)
        self.assertEqual(_by_severity(report, Severity.ERROR), [])


class TestDegradation(unittest.TestCase):
    def test_stub_info_skips(self):
        report = InterferenceCheck().check(StubBackend(), None)
        self.assertIn("interference-skipped", _codes(report))
        self.assertTrue(report.ok)

    def test_single_part_trivial(self):
        report = InterferenceCheck().check(
            _AssemblyBackend([_box("solo", 0, 0, 0, 1, 1, 1)]), None)
        self.assertIn("interference-trivial", _codes(report))
        self.assertTrue(report.ok)

    def test_never_crashes_on_broken_backend(self):
        class _Boom:
            def query(self, q):
                raise RuntimeError("nope")

        report = InterferenceCheck().check(_Boom(), None)
        self.assertIn("interference-skipped", _codes(report))
        self.assertTrue(report.ok)

    def test_check_parts_direct(self):
        parts = [_box("p1", 0, 0, 0, 10, 10, 10),
                 _box("p2", 5, 5, 5, 15, 15, 15)]
        report = InterferenceCheck().check_parts(parts)
        self.assertIn("interference-approx", _codes(report))


class TestWithInterference(unittest.TestCase):
    def test_appends_check(self):
        base = ["x"]
        result = with_interference(base)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[-1], InterferenceCheck)
        self.assertEqual(result[-1].name, "interference")
        self.assertEqual(base, ["x"])


def _has_cadquery():
    try:
        import cadquery  # noqa: F401
        return True
    except Exception:
        return False


@unittest.skipUnless(_has_cadquery(), "cadquery/OCCT not installed")
class TestExactOCCT(unittest.TestCase):
    def test_overlapping_solids_hard_error(self):
        import cadquery as cq
        a = cq.Workplane("XY").box(10, 10, 10)
        b = cq.Workplane("XY").box(10, 10, 10).translate((5, 0, 0))
        parts = [
            {"id": "a", "shape": a.val()},
            {"id": "b", "shape": b.val()},
        ]
        report = InterferenceCheck().check_parts(parts)
        self.assertIn("interference", _codes(report))
        self.assertFalse(report.ok)  # exact solid overlap is an ERROR

    def test_disjoint_solids_clear(self):
        import cadquery as cq
        a = cq.Workplane("XY").box(10, 10, 10)
        b = cq.Workplane("XY").box(10, 10, 10).translate((50, 0, 0))
        parts = [
            {"id": "a", "shape": a.val()},
            {"id": "b", "shape": b.val()},
        ]
        report = InterferenceCheck().check_parts(parts)
        self.assertEqual(_by_severity(report, Severity.ERROR), [])
        self.assertTrue(report.ok)


if __name__ == "__main__":
    unittest.main()
