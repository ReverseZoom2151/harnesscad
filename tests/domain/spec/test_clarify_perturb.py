import unittest

from harnesscad.domain.spec.clarify_ambiguity import CADSpec, Feature, AmbiguityDetector, CONFLICTING
from harnesscad.domain.spec.clarify_perturb import AmbiguityGenerator, keep_sample, UNDER, CONFLICT
from harnesscad.domain.spec.clarify_dialogue import run_dialogue


def spec():
    return CADSpec(
        general_shape="a plate with a hole",
        workplane="XY", origin=(0.0, 0.0, 0.0),
        extrude_direction="positive_normal", extrude_distance=20.0,
        features=[
            Feature("rectangle", "plate", {"width": 200.0, "height": 160.0}),
            Feature("hole", "hole", {"radius": 8.0}),
        ],
    )


class TestUnderSpecified(unittest.TestCase):
    def test_omits_exactly_k(self):
        gen = AmbiguityGenerator(seed=1)
        traj = gen.perturb(spec(), UNDER, 2)
        self.assertEqual(traj.num_issues, 2)
        self.assertEqual(len(traj.keys), 2)
        # perturbed spec is detectably under-specified.
        issues = AmbiguityDetector().detect(traj.ambiguous)
        self.assertTrue(issues)
        # answers recover ground truth.
        self.assertEqual(len(traj.answers), 2)

    def test_trajectory_resolvable(self):
        gen = AmbiguityGenerator(seed=3)
        traj = gen.perturb(spec(), UNDER, 1)
        result = run_dialogue(traj.ambiguous, traj.original)
        self.assertTrue(result.is_misleading)
        self.assertFalse(AmbiguityDetector().detect(result.corrected))


class TestConflicting(unittest.TestCase):
    def test_injects_conflict(self):
        gen = AmbiguityGenerator(seed=2)
        traj = gen.perturb(spec(), CONFLICT, 1)
        conf = [i for i in AmbiguityDetector().detect(traj.ambiguous)
                if i.type == CONFLICTING]
        self.assertEqual(len(conf), 1)
        self.assertEqual(conf[0].key, traj.keys[0])

    def test_conflict_only_numeric_slots(self):
        gen = AmbiguityGenerator(seed=0)
        traj = gen.perturb(spec(), CONFLICT, 3)
        for key in traj.keys:
            self.assertNotIn("workplane", key)
            self.assertNotIn("origin", key)
            self.assertNotIn("direction", key)


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_output(self):
        a = AmbiguityGenerator(seed=7).perturb(spec(), UNDER, 2)
        b = AmbiguityGenerator(seed=7).perturb(spec(), UNDER, 2)
        self.assertEqual(a.keys, b.keys)
        self.assertEqual(a.questions, b.questions)

    def test_invalid_k_raises(self):
        gen = AmbiguityGenerator(seed=0)
        with self.assertRaises(ValueError):
            gen.perturb(spec(), UNDER, 999)


class TestKeepSample(unittest.TestCase):
    def test_selection_rules(self):
        # high-quality original, harmful perturbation, ratio >= 10.
        self.assertTrue(keep_sample(1e-5, 1e-3))
        # original not high quality.
        self.assertFalse(keep_sample(1e-3, 1e-2))
        # perturbation not harmful.
        self.assertFalse(keep_sample(1e-5, 1e-5))
        # ratio too small.
        self.assertFalse(keep_sample(1e-4, 5e-4))


if __name__ == "__main__":
    unittest.main()
