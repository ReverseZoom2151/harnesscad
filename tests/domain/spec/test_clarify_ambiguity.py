import unittest

from harnesscad.domain.spec.clarify_ambiguity import (
    CADSpec, Feature, AmbiguityDetector, audit,
    UNDER_SPECIFIED, CONFLICTING, IMPOSSIBLE,
    under_specification_score, missing_slots, vague_phrases, question_for,
)


def full_spec():
    return CADSpec(
        general_shape="a rectangular mounting plate with a hole",
        workplane="XY", origin=(0.0, 0.0, 0.0),
        extrude_direction="positive_normal", extrude_distance=20.0,
        features=[
            Feature("rectangle", "plate", {"width": 200.0, "height": 160.0}),
            Feature("hole", "hole", {"radius": 8.0}),
        ],
    )


class TestUnderSpecified(unittest.TestCase):
    def test_full_spec_has_no_issues(self):
        self.assertEqual(AmbiguityDetector().detect(full_spec()), [])
        self.assertFalse(audit(full_spec()).is_misleading)

    def test_missing_workplane_and_origin(self):
        spec = full_spec()
        spec.workplane = None
        spec.origin = None
        issues = AmbiguityDetector().detect(spec)
        keys = {i.key for i in issues}
        self.assertIn("setup.workplane", keys)
        self.assertIn("setup.origin", keys)
        self.assertTrue(all(i.type == UNDER_SPECIFIED for i in issues))

    def test_missing_extrusion_distance_and_direction(self):
        spec = full_spec()
        spec.extrude_distance = None
        spec.extrude_direction = None
        keys = {i.key for i in AmbiguityDetector().detect(spec)}
        self.assertIn("build.extrude_distance", keys)
        self.assertIn("build.extrude_direction", keys)

    def test_missing_feature_param(self):
        spec = full_spec()
        spec.features[1].params["radius"] = None
        issues = AmbiguityDetector().detect(spec)
        self.assertTrue(any(i.key == "hole.radius"
                            and i.type == UNDER_SPECIFIED for i in issues))

    def test_vague_shape_language(self):
        self.assertIn("large", vague_phrases("a large bracket"))
        self.assertEqual(vague_phrases("a 200 mm bracket"), [])
        spec = full_spec()
        spec.general_shape = "a large plate"
        self.assertTrue(any(i.key.startswith("shape.vague")
                            for i in AmbiguityDetector().detect(spec)))


class TestConflicting(unittest.TestCase):
    def test_feature_value_conflict(self):
        spec = full_spec()
        spec.features[1].params["radius"] = [8.0, 10.0]
        issues = AmbiguityDetector().detect(spec)
        conf = [i for i in issues if i.type == CONFLICTING]
        self.assertEqual(len(conf), 1)
        self.assertEqual(conf[0].key, "hole.radius")
        self.assertEqual(set(conf[0].values), {8.0, 10.0})

    def test_extrude_distance_conflict(self):
        spec = full_spec()
        spec.extrude_distance = [20.0, 25.0]
        conf = [i for i in AmbiguityDetector().detect(spec)
                if i.type == CONFLICTING]
        self.assertEqual(conf[0].key, "build.extrude_distance")

    def test_single_value_not_conflict(self):
        spec = full_spec()
        spec.features[1].params["radius"] = [8.0, 8.0]
        conf = [i for i in AmbiguityDetector().detect(spec)
                if i.type == CONFLICTING]
        self.assertEqual(conf, [])


class TestImpossible(unittest.TestCase):
    def test_non_positive_dimension(self):
        spec = full_spec()
        spec.features[0].params["width"] = -5.0
        imp = [i for i in AmbiguityDetector().detect(spec)
               if i.type == IMPOSSIBLE]
        self.assertTrue(imp)

    def test_inner_ge_outer_radius(self):
        spec = full_spec()
        spec.features.append(
            Feature("circle", "ring", {"outer_radius": 5.0, "inner_radius": 6.0,
                                       "radius": 5.0}))
        imp = [i for i in AmbiguityDetector().detect(spec)
               if i.type == IMPOSSIBLE]
        self.assertTrue(any("inner_radius" in i.key for i in imp))

    def test_zero_extrusion(self):
        spec = full_spec()
        spec.extrude_distance = 0.0
        imp = [i for i in AmbiguityDetector().detect(spec)
               if i.type == IMPOSSIBLE]
        self.assertTrue(imp)


class TestAuditAndScore(unittest.TestCase):
    def test_minimal_questions_one_per_key(self):
        spec = full_spec()
        spec.features[1].params["radius"] = None
        spec.workplane = None
        rep = audit(spec)
        self.assertTrue(rep.is_misleading)
        keys = [q.key for q in rep.questions]
        self.assertEqual(len(keys), len(set(keys)))

    def test_envelope_shapes(self):
        spec = full_spec()
        env = audit(spec).envelope("standardized")
        self.assertEqual(env, {"is_misleading": False,
                               "standardized_prompt": "standardized"})
        spec.workplane = None
        env2 = audit(spec).envelope()
        self.assertTrue(env2["is_misleading"])
        self.assertIn("questions", env2)

    def test_under_specification_score(self):
        self.assertEqual(under_specification_score(full_spec()), 1.0)
        spec = full_spec()
        spec.workplane = None
        spec.origin = None
        self.assertLess(under_specification_score(spec), 1.0)
        self.assertIn("setup.workplane", missing_slots(spec))

    def test_conflict_question_offers_both_values(self):
        spec = full_spec()
        spec.features[1].params["radius"] = [8.0, 10.0]
        q = audit(spec).questions[0]
        self.assertIn("8", q.text)
        self.assertIn("10", q.text)

    def test_deterministic(self):
        spec = full_spec()
        spec.features[1].params["radius"] = None
        self.assertEqual([i.key for i in AmbiguityDetector().detect(spec)],
                         [i.key for i in AmbiguityDetector().detect(spec)])


if __name__ == "__main__":
    unittest.main()
