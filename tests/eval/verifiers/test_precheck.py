"""Tests for the pre-execution plan-feasibility precheck (verifiers.precheck).

The precheck is purely symbolic: it walks a CISP op plan and rejects infeasible
plans BEFORE any geometry is built. These tests build raw op lists and confirm:

  * a hole that eats the material AROUND it in-plane is flagged (and a hole that
    is merely wider than the plate is thick is NOT: those axes are orthogonal);
  * an ambiguous sketch (two profiles / two extrudes) is warned about;
  * a fillet before any solid exists is flagged;
  * an extrude of an empty sketch is flagged;
  * a valid plate plan (extrude -> hole -> fillet) passes;
  * check_ops works on a raw op list, and check() drives it from an OpDAG.
"""

import unittest

from harnesscad.core.cisp.ops import (
    NewSketch, AddRectangle, AddCircle, AddLine,
    Extrude, Fillet, Hole, Shell, Boolean,
    LinearPattern, Mate,
)
from harnesscad.core.state.opdag import OpDAG
from harnesscad.eval.verifiers.verify import Severity
from harnesscad.eval.verifiers.precheck import PrecheckCheck, PrecheckRules, with_precheck


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


class TestHoleFeasibility(unittest.TestCase):
    """The hole rule compares the diameter with the IN-PLANE material only.

    Regression guard for the bug that caused every harness regression in the
    pressure run: the diameter (in-plane) was compared against the extrude
    distance (Z). Orthogonal quantities.
    """

    def test_washer_is_not_infeasible(self):
        # 80 mm disc, 8 mm thick, 30 mm bore -> a washer. It builds correctly
        # (volume 34217, bbox 80x80x8). The old rule raised infeasible-plan
        # because 30 >= 8. Nothing may fire.
        ops = [
            NewSketch(plane="XY"),
            AddCircle(sketch="sk1", cx=0, cy=0, r=40.0),
            Extrude(sketch="sk1", distance=8.0),
            Hole(face_or_sketch="sk1", x=0, y=0, diameter=30.0, through=True),
        ]
        report = PrecheckCheck().check_ops(ops)
        self.assertEqual(_errors(report), [])
        self.assertEqual(report.diagnostics, [])
        self.assertTrue(report.ok)

    def test_through_hole_wider_than_the_plate_is_thickness_is_not_infeasible(self):
        # 40x40x10 plate, 12 mm through hole: routine machining (the
        # trap_hole_oversize brief). The diameter exceeds the 10 mm thickness
        # and that is irrelevant.
        ops = [
            NewSketch(plane="XY"),
            AddRectangle(sketch="sk1", x=0, y=0, w=40, h=40),
            Extrude(sketch="sk1", distance=10.0),
            Hole(face_or_sketch="solid", x=20, y=20, diameter=12.0, through=True),
        ]
        report = PrecheckCheck().check_ops(ops)
        self.assertEqual(_errors(report), [])
        self.assertTrue(report.ok)

    def test_hole_spanning_the_in_plane_extent_is_infeasible(self):
        # 50x20 plate; a 25 mm hole on the centre line spans the whole 20 mm
        # in-plane Y extent: no wall survives on either side.
        ops = [
            NewSketch(plane="XY"),
            AddRectangle(sketch="sk1", x=0, y=0, w=50, h=20),
            Extrude(sketch="sk1", distance=6.0),
            Hole(face_or_sketch="solid", x=25, y=10, diameter=25.0, through=True),
        ]
        report = PrecheckCheck().check_ops(ops)
        errs = _errors(report)
        self.assertTrue(any("in-plane" in e.message for e in errs))
        self.assertFalse(report.ok)

    def test_hole_breaking_out_of_the_edge_warns(self):
        # Centre inside the stock, disc crossing the boundary: an open notch.
        ops = [
            NewSketch(plane="XY"),
            AddRectangle(sketch="sk1", x=0, y=0, w=50, h=50),
            Extrude(sketch="sk1", distance=6.0),
            Hole(face_or_sketch="solid", x=2, y=25, diameter=10.0, through=True),
        ]
        report = PrecheckCheck().check_ops(ops)
        self.assertIn("thin-wall", _codes(report))
        self.assertEqual(_errors(report), [])

    def test_hole_stays_silent_when_the_in_plane_extent_is_unknown(self):
        # No sketch bounds are knowable (hole on a bare solid-less datum): the
        # rule must not fire. A false positive is more costly than a miss.
        ops = [Hole(face_or_sketch="sk1", x=0, y=0, diameter=99.0, through=True)]
        report = PrecheckCheck().check_ops(ops)
        self.assertFalse(any("in-plane" in d.message for d in report.diagnostics))


class TestAmbiguousSketchConsumption(unittest.TestCase):
    def test_sketch_with_two_profiles_warns(self):
        ops = [
            NewSketch(plane="XY"),
            AddRectangle(sketch="sk1", x=0, y=0, w=50, h=40),
            AddCircle(sketch="sk1", cx=25, cy=20, r=15.0),
            Extrude(sketch="sk1", distance=6.0),
        ]
        report = PrecheckCheck().check_ops(ops)
        self.assertIn("ambiguous-sketch", _codes(report))
        self.assertEqual(_errors(report), [])

    def test_sketch_extruded_twice_warns(self):
        ops = [
            NewSketch(plane="XY"),
            AddRectangle(sketch="sk1", x=0, y=0, w=50, h=40),
            Extrude(sketch="sk1", distance=6.0),
            Extrude(sketch="sk1", distance=30.0),
        ]
        report = PrecheckCheck().check_ops(ops)
        self.assertIn("sketch-reused", _codes(report))

    def test_one_profile_one_extrude_is_silent(self):
        ops = [
            NewSketch(plane="XY"),
            AddRectangle(sketch="sk1", x=0, y=0, w=50, h=40),
            Extrude(sketch="sk1", distance=6.0),
        ]
        report = PrecheckCheck().check_ops(ops)
        self.assertEqual(report.diagnostics, [])


class TestInfeasiblePlans(unittest.TestCase):
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

    def test_wall_thickness_rule_governs_the_shell_not_the_hole(self):
        # `wall_thickness` is the stock available to a SHELL. It must not touch
        # holes (the deleted diameter-vs-thickness rule): a 5 mm hole in a plate
        # declared 4 mm thick is still a perfectly good hole.
        rules = PrecheckRules(wall_thickness=4.0)
        report = PrecheckCheck(rules).check_ops(_valid_plate_ops())
        self.assertEqual(_errors(report), [])
        self.assertTrue(report.ok)
        shelled = list(_valid_plate_ops()) + [Shell(faces=(), thickness=4.0)]
        report = PrecheckCheck(rules).check_ops(shelled)
        self.assertTrue(any("available stock" in e.message for e in _errors(report)))


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
