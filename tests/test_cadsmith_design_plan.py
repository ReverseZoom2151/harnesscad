import unittest

from generation.cadsmith_design_plan import (
    Component, GeometricConstraints, DesignPlan,
    validate, is_valid, check_prompt_conventions,
)


def _good_plan():
    return DesignPlan(
        components=(
            Component("flange", "bottom disc", (0.0, 10.0)),
            Component("hub", "central cylinder", (10.0, 50.0)),
        ),
        target_bbox_mm=(50.0, 50.0, 60.0),
        constraints=GeometricConstraints(
            hole_count=6, hole_diameters_mm=(6.5,), symmetry=("axial-z",),
        ),
        notes="flanged shaft coupling",
    )


class TestSerialization(unittest.TestCase):
    def test_json_round_trip(self):
        plan = _good_plan()
        again = DesignPlan.from_json(plan.to_json())
        self.assertEqual(again, plan)

    def test_json_is_stable(self):
        plan = _good_plan()
        self.assertEqual(plan.to_json(), plan.to_json())

    def test_component_z_range_preserved(self):
        again = DesignPlan.from_json(_good_plan().to_json())
        self.assertEqual(again.components[0].z_range, (0.0, 10.0))

    def test_none_z_range(self):
        c = Component("x")
        self.assertNotIn("z_range", c.to_dict())
        self.assertIsNone(Component.from_dict(c.to_dict()).z_range)


class TestValidation(unittest.TestCase):
    def test_good_plan_valid(self):
        self.assertTrue(is_valid(_good_plan()))
        self.assertEqual(validate(_good_plan()), ())

    def test_no_components(self):
        p = DesignPlan((), (1.0, 1.0, 1.0))
        self.assertIn("no-components", validate(p))

    def test_duplicate_component(self):
        p = DesignPlan((Component("a"), Component("a")), (1.0, 1.0, 1.0))
        self.assertIn("duplicate-component:a", validate(p))

    def test_inverted_z_range(self):
        p = DesignPlan((Component("a", z_range=(5.0, 1.0)),), (1.0, 1.0, 1.0))
        self.assertIn("inverted-z-range:a", validate(p))

    def test_non_positive_bbox(self):
        p = DesignPlan((Component("a"),), (0.0, 1.0, 1.0))
        self.assertIn("bbox-non-positive", validate(p))

    def test_negative_hole_count(self):
        p = DesignPlan((Component("a"),), (1.0, 1.0, 1.0),
                       GeometricConstraints(hole_count=-1))
        self.assertIn("negative-hole-count", validate(p))

    def test_more_diameters_than_holes(self):
        p = DesignPlan((Component("a"),), (1.0, 1.0, 1.0),
                       GeometricConstraints(hole_count=1,
                                            hole_diameters_mm=(1.0, 2.0)))
        self.assertIn("more-diameters-than-holes", validate(p))


class TestPromptConventions(unittest.TestCase):
    def test_compliant_prompt(self):
        prompt = ("A cylinder 20mm in diameter, centered at the origin, "
                  "standing along the Z axis, Z-up.")
        self.assertEqual(check_prompt_conventions(prompt), ())

    def test_missing_mm(self):
        prompt = "A cylinder centered at the origin along the Z axis."
        self.assertIn("no-explicit-mm-dimensions", check_prompt_conventions(prompt))

    def test_missing_axis(self):
        prompt = "A 20mm cube centered at the origin."
        self.assertIn("no-axis-annotation", check_prompt_conventions(prompt))

    def test_missing_origin(self):
        prompt = "A 20mm cube on the XY plane, Z-up."
        self.assertIn("no-origin-centering", check_prompt_conventions(prompt))


if __name__ == "__main__":
    unittest.main()
