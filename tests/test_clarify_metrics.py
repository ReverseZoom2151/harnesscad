import unittest

from clarify_ambiguity import CADSpec, Feature
from clarify_metrics import (
    efficiency, resolution, score_interaction, aggregate, ClarifierScore,
)


def target():
    return CADSpec(
        general_shape="a rod", workplane="XY", origin=(0.0, 0.0, 0.0),
        extrude_direction="positive_normal", extrude_distance=200.0,
        features=[Feature("circle", "rod", {"radius": 19.0})],
    )


class TestEfficiency(unittest.TestCase):
    def test_perfect_match(self):
        e = efficiency(["rod.radius", "setup.origin"],
                       ["setup.origin", "rod.radius"])
        self.assertEqual(e.f1, 1.0)
        self.assertEqual(e.redundant, ())
        self.assertEqual(e.missed, ())

    def test_redundant_question_lowers_precision(self):
        e = efficiency(["rod.radius", "hole.depth"], ["rod.radius"])
        self.assertEqual(e.precision, 0.5)
        self.assertEqual(e.recall, 1.0)
        self.assertIn("hole.depth", e.redundant)
        self.assertAlmostEqual(e.f1, 2 / 3)

    def test_missed_question_lowers_recall(self):
        e = efficiency(["rod.radius"], ["rod.radius", "setup.origin"])
        self.assertEqual(e.recall, 0.5)
        self.assertIn("setup.origin", e.missed)


class TestResolution(unittest.TestCase):
    def test_fully_resolved(self):
        clarified = target()
        self.assertEqual(resolution(clarified, target(),
                                    ["rod.radius"]), 1.0)

    def test_partial_resolution(self):
        clarified = target()
        clarified.features[0].params["radius"] = 99.0  # wrong value, present
        score = resolution(clarified, target(),
                           ["rod.radius", "setup.origin"])
        self.assertEqual(score, 0.5)

    def test_unresolved_still_missing(self):
        clarified = target()
        clarified.origin = None
        self.assertEqual(resolution(clarified, target(),
                                    ["setup.origin"]), 0.0)


class TestSpecialCases(unittest.TestCase):
    def test_unambiguous_correctly_accepted(self):
        s = score_interaction(prompt_is_ambiguous=False, agent_flagged=False)
        self.assertEqual(s, ClarifierScore(1.0, 1.0))

    def test_unambiguous_false_flag(self):
        s = score_interaction(prompt_is_ambiguous=False, agent_flagged=True)
        self.assertEqual(s, ClarifierScore(0.0, 0.0))

    def test_ambiguous_missed(self):
        s = score_interaction(prompt_is_ambiguous=True, agent_flagged=False)
        self.assertEqual(s, ClarifierScore(0.0, 0.0))

    def test_ambiguous_handled(self):
        clarified = target()
        s = score_interaction(
            prompt_is_ambiguous=True, agent_flagged=True,
            generated_keys=["rod.radius"], ground_truth_keys=["rod.radius"],
            clarified=clarified, target=target(), issue_keys=["rod.radius"])
        self.assertEqual(s.efficiency, 1.0)
        self.assertEqual(s.resolution, 1.0)


class TestAggregate(unittest.TestCase):
    def test_means(self):
        rep = aggregate([ClarifierScore(1.0, 1.0), ClarifierScore(0.0, 0.5)])
        self.assertEqual(rep.n, 2)
        self.assertEqual(rep.mean_efficiency, 0.5)
        self.assertEqual(rep.mean_resolution, 0.75)

    def test_empty(self):
        self.assertEqual(aggregate([]).n, 0)


if __name__ == "__main__":
    unittest.main()
