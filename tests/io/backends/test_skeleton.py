"""Tests for the top-down master-sketch / layout front-of-pipeline (skeleton)."""

import unittest

from harnesscad.io.backends.stub import StubBackend
from harnesscad.core.cisp.ops import AddPoint, NewSketch
from harnesscad.core.loop import HarnessSession
from harnesscad.domain.sizing.calc import SizingCalc
from harnesscad.domain.skeleton.layout import Skeleton, build_skeleton


def _fresh_session():
    return HarnessSession(StubBackend())


class TestBuildSkeleton(unittest.TestCase):
    def test_brief_produces_datums_and_reference_points(self):
        sk = build_skeleton("mounting plate 100 x 50 x 10 mm with 4 holes")
        self.assertEqual(sk.envelope.width, 100.0)
        self.assertEqual(sk.envelope.height, 50.0)
        self.assertEqual(sk.envelope.depth, 10.0)
        # Named datum reference frame present.
        for name in ("origin", "center", "x_axis", "y_axis", "z_axis", "XY"):
            self.assertIn(name, sk.datum_names())
        # One reference point per hole.
        self.assertEqual(len(sk.reference_points), 4)
        self.assertIn("hole_1_center", sk.reference_points)

    def test_parameter_table_populated(self):
        sk = build_skeleton("bracket 80x40 with 2 bolts")
        self.assertGreater(len(sk.parameters), 0)
        self.assertEqual(sk.parameters["envelope_width"], 80.0)
        self.assertEqual(sk.parameters["envelope_height"], 40.0)
        self.assertEqual(sk.parameters["hole_count"], 2.0)
        # The table is editable.
        sk.parameters["envelope_width"] = 90.0
        self.assertEqual(sk.parameters["envelope_width"], 90.0)

    def test_defaults_when_brief_is_bare(self):
        sk = build_skeleton("a part")
        self.assertEqual(sk.envelope.width, 100.0)
        self.assertEqual(sk.envelope.height, 100.0)
        self.assertEqual(len(sk.reference_points), 0)

    def test_structured_spec_dict(self):
        sk = build_skeleton({"width": 120.0, "height": 60.0, "depth": 8.0,
                             "hole_count": 1, "name": "cover"})
        self.assertEqual(sk.name, "cover")
        self.assertEqual(sk.envelope.width, 120.0)
        # A single hole lands at the centre.
        self.assertEqual(sk.reference_points["hole_1_center"], (60.0, 30.0))


class TestToOpsApplies(unittest.TestCase):
    def test_ops_apply_ok_on_stub_session(self):
        sk = build_skeleton("plate 100 x 50 x 10 with 4 holes")
        ops = sk.to_ops()
        self.assertIsInstance(ops[0], NewSketch)
        res = _fresh_session().apply_ops(ops)
        self.assertTrue(res.ok, msg=[d.to_dict() for d in res.diagnostics])
        self.assertEqual(res.applied, len(ops))
        self.assertIsNone(res.rejected)

    def test_reference_points_emitted_as_points(self):
        sk = build_skeleton("plate 100 x 50 with 4 holes")
        ops = sk.to_ops()
        point_ops = [o for o in ops if isinstance(o, AddPoint)]
        # 4 envelope corners + 1 centre + 4 hole reference points.
        self.assertEqual(len(point_ops), 9)

    def test_bare_part_ops_also_apply(self):
        sk = build_skeleton("a part")
        res = _fresh_session().apply_ops(sk.to_ops())
        self.assertTrue(res.ok)


class TestSizingFeedsSkeleton(unittest.TestCase):
    def test_sizing_results_merge_into_parameters(self):
        calc = SizingCalc()
        shaft = calc.size({"formula": "shaft_diameter_torsion",
                           "torque": 1.0e6, "allowable_shear": 40.0})
        sk = build_skeleton("shaft housing 100 x 100", sizing=[shaft])
        key = "sized_shaft_diameter_torsion"
        self.assertIn(key, sk.parameters)
        self.assertAlmostEqual(sk.parameters[key], shaft["value"], places=6)


class TestDeterminism(unittest.TestCase):
    def test_same_brief_same_ops_and_digest(self):
        brief = "plate 100 x 50 x 10 with 4 holes"
        a = build_skeleton(brief)
        b = build_skeleton(brief)
        self.assertEqual([o.to_dict() for o in a.to_ops()],
                         [o.to_dict() for o in b.to_ops()])
        s1, s2 = _fresh_session(), _fresh_session()
        r1 = s1.apply_ops(a.to_ops())
        r2 = s2.apply_ops(b.to_ops())
        self.assertTrue(r1.ok and r2.ok)
        self.assertEqual(s1.digest(), s2.digest())


if __name__ == "__main__":
    unittest.main()
