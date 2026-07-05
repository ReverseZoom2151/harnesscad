"""Tests for the pre-execution plan-feasibility precheck (verifiers.precheck).

The precheck is purely symbolic: it walks a CISP op plan and rejects infeasible
plans BEFORE any geometry is built. These tests build raw op lists and confirm:

  * a hole bigger than the plate wall is flagged;
  * a fillet before any solid exists is flagged;
  * an extrude of an empty sketch is flagged;
  * a valid plate plan (extrude -> hole -> fillet) passes;
  * check_ops works on a raw op list, and check() drives it from an OpDAG.
"""

import unittest

from cisp.ops import (
    NewSketch, AddRectangle, AddCircle, AddLine,
    Extrude, Fillet, Hole, Shell, Boolean,
    LinearPattern, Mate,
)
from state.opdag import OpDAG
from verifiers.verify import Severity
from verifiers.precheck import PrecheckCheck, PrecheckRules, with_precheck


def _codes(report):
    return {d.code for d in report.diagnostics}


def _errors(report):
    return [d for d in report.diagnostics if d.severity is Severity.ERROR]


def _valid_plate_ops():
    """A structurally sound plan: rectangle -> extrude (20 thick) -> small hole -> fillet."""
    return [
        NewSketch(plane="XY"),               # sk1
        AddRectangle(sketch="sk1", x=0, y=0, w=50, h=50),
        Extrude(sketch="sk1", distance=20.0),
        Hole(face_or_sketch="solid", x=10, y=10, diameter=5.0, through=True),
        Fillet(edges=(), radius=2.0),
    ]


class TestRules(unittest.TestCase):
    def test_roundtrip(self):
        r = PrecheckRules(min_wall=1.0, wall_thickness=3.0, min_pattern_count=3)
        self.assertEqual(PrecheckRules.from_dict(r.to_dict()), r)

    def test_from_dict_defaults(self):
        r = PrecheckRules.from_dict(None)
        self.assertEqual(r, PrecheckRules())
        self.assertIsNone(r.wall_thickness)


class TestInfeasiblePlans(unittest.TestCase):
    def test_hole_bigger_than_wall(self):
        # Plate only 2 mm thick; a 10 mm hole is larger than the wall.
        ops = [
            NewSketch(plane="XY"),
            AddRectangle(sketch="sk1", x=0, y=0, w=50, h=50),
            Extrude(sketch="sk1", distance=2.0),
            Hole(face_or_sketch="solid", x=10, y=10, diameter=10.0, through=True),
        ]
        report = PrecheckCheck().check_ops(ops)
        errs = _errors(report)
        self.assertIn("infeasible-plan", _codes(report))
        self.assertTrue(any("wall" in e.message for e in errs))
        self.assertFalse(report.ok)

    def test_fillet_before_solid(self):
        ops = [Fillet(edges=(), radius=2.0)]
        report = PrecheckCheck().check_ops(ops)
        errs = _errors(report)
        self.assertTrue(any("requires an existing solid" in e.message for e in errs))
        self.assertFalse(report.ok)

    def test_empty_extrude(self):
        # A sketch with no profile entities cannot become a solid.
        ops = [NewSketch(plane="XY"), Extrude(sketch="sk1", distance=10.0)]
        report = PrecheckCheck().check_ops(ops)
        errs = _errors(report)
        self.assertTrue(any("empty sketch" in e.message for e in errs))
        self.assertFalse(report.ok)

    def test_boolean_before_two_solids(self):
        ops = [
            NewSketch(plane="XY"),
            AddRectangle(sketch="sk1", x=0, y=0, w=10, h=10),
            Extrude(sketch="sk1", distance=5.0),
            Boolean(kind="union", target="", tool=""),
        ]
        report = PrecheckCheck().check_ops(ops)
        self.assertTrue(any("two solids" in e.message for e in _errors(report)))
        self.assertFalse(report.ok)

    def test_zero_extrude_distance(self):
        ops = [
            NewSketch(plane="XY"),
            AddRectangle(sketch="sk1", x=0, y=0, w=10, h=10),
            Extrude(sketch="sk1", distance=0.0),
        ]
        report = PrecheckCheck().check_ops(ops)
        self.assertTrue(any("non-zero" in e.message for e in _errors(report)))

    def test_negative_dimension_circle(self):
        ops = [NewSketch(plane="XY"), AddCircle(sketch="sk1", cx=0, cy=0, r=-3.0)]
        report = PrecheckCheck().check_ops(ops)
        self.assertTrue(any("radius must be > 0" in e.message for e in _errors(report)))

    def test_pattern_count_below_two(self):
        ops = [
            NewSketch(plane="XY"),
            AddRectangle(sketch="sk1", x=0, y=0, w=10, h=10),
            Extrude(sketch="sk1", distance=5.0),
            LinearPattern(feature="", count=1, spacing=5.0),
        ]
        report = PrecheckCheck().check_ops(ops)
        self.assertTrue(any("pattern count" in e.message for e in _errors(report)))

    def test_shell_thicker_than_stock(self):
        ops = [
            NewSketch(plane="XY"),
            AddRectangle(sketch="sk1", x=0, y=0, w=50, h=50),
            Extrude(sketch="sk1", distance=4.0),
            Shell(faces=(), thickness=5.0),  # >= 4 mm stock
        ]
        report = PrecheckCheck().check_ops(ops)
        self.assertTrue(any("available stock" in e.message for e in _errors(report)))

    def test_dangling_sketch_reference(self):
        ops = [Extrude(sketch="sk9", distance=10.0)]
        report = PrecheckCheck().check_ops(ops)
        self.assertTrue(any("unknown sketch" in e.message for e in _errors(report)))

    def test_duplicate_mate_pair(self):
        ops = [
            Mate(kind="rigid", a="i1", b="i2"),
            Mate(kind="revolute", a="i2", b="i1"),  # same unordered pair
        ]
        report = PrecheckCheck().check_ops(ops)
        self.assertTrue(any("mutually exclusive" in e.message for e in _errors(report)))


class TestFeasiblePlans(unittest.TestCase):
    def test_valid_plate_passes(self):
        report = PrecheckCheck().check_ops(_valid_plate_ops())
        self.assertEqual(_errors(report), [])
        self.assertTrue(report.ok)

    def test_wall_thickness_rule_override(self):
        # Explicit wall makes a 5 mm hole infeasible even on a thick plate.
        rules = PrecheckRules(wall_thickness=4.0)
        report = PrecheckCheck(rules).check_ops(_valid_plate_ops())
        self.assertFalse(report.ok)
        self.assertTrue(any("wall" in e.message for e in _errors(report)))


class TestBackendEntrypoint(unittest.TestCase):
    def test_check_drives_from_opdag(self):
        dag = OpDAG()
        for op in _valid_plate_ops():
            dag.append(op)
        report = PrecheckCheck().check(None, dag)
        self.assertTrue(report.ok)

    def test_check_flags_bad_opdag(self):
        dag = OpDAG()
        dag.append(Fillet(edges=(), radius=1.0))
        report = PrecheckCheck().check(None, dag)
        self.assertFalse(report.ok)

    def test_check_ops_accepts_raw_list(self):
        report = PrecheckCheck().check_ops([Fillet(edges=(), radius=1.0)])
        # check() also accepts a raw op list passed as the opdag argument.
        report2 = PrecheckCheck().check(None, [Fillet(edges=(), radius=1.0)])
        self.assertFalse(report.ok)
        self.assertFalse(report2.ok)

    def test_no_plan_info_skips(self):
        report = PrecheckCheck().check(None, None)
        self.assertIn("precheck-skipped", _codes(report))
        self.assertTrue(report.ok)


class TestWithPrecheck(unittest.TestCase):
    def test_appends_check(self):
        base = ["x"]
        result = with_precheck(base)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[-1], PrecheckCheck)
        self.assertEqual(result[-1].name, "precheck")
        self.assertEqual(base, ["x"])


if __name__ == "__main__":
    unittest.main()
