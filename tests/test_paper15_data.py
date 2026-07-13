import unittest

from harnesscad.data.datagen.complexity import measure_complexity, voxel_entropy
from harnesscad.data.datagen.instruction_taxonomy import (
    CATEGORIES, LENGTH_BUCKETS, STYLES, InstructionSample, quota_matrix,
    seeded_slots, deduplicate,
)
from harnesscad.data.dataengine.curation.cascade_filter import FilterDecision, cascade_filter
from harnesscad.data.dataengine.selftrain.self_improvement import self_improve


class TaxonomyTests(unittest.TestCase):
    def test_full_taxonomy_and_seed_replay(self):
        self.assertEqual(len(quota_matrix()), 16 * 8 * 5)
        self.assertEqual(seeded_slots(1, 7), seeded_slots(1, 7))
        self.assertEqual(len(CATEGORIES), 16)
        self.assertEqual(len(STYLES), 8)
        self.assertEqual(len(LENGTH_BUCKETS), 5)

    def test_similarity_and_name_dedup(self):
        slot = quota_matrix()[0]
        samples = [
            InstructionSample("Make a bolt", slot),
            InstructionSample("make a bolt!", slot),
        ]
        kept = deduplicate(
            samples,
            similarity=lambda left, right: 1.0 if left.casefold() == right.casefold() else 0.0,
        )
        self.assertEqual(len(kept), 1)


class ComplexityTests(unittest.TestCase):
    def test_metrics(self):
        result = measure_complexity(
            [{"parameters": {"x": 1, "y": 2}}, {"parameters": {"z": 3}}],
            [0, 0, 1, 1],
        )
        self.assertEqual(result.unit_count, 2)
        self.assertEqual(result.parameter_density, 1.5)
        self.assertEqual(result.occupancy_entropy, 1.0)

    def test_empty_entropy(self):
        self.assertEqual(voxel_entropy([]), 0.0)


class CascadeTests(unittest.TestCase):
    def test_fine_only_receives_coarse_passes(self):
        seen = []
        report = cascade_filter(
            [1, 2, 3],
            lambda item: FilterDecision(item > 1, "coarse", "small"),
            lambda item: (seen.append(item) or FilterDecision(item == 2, "fine")),
        )
        self.assertEqual(seen, [2, 3])
        self.assertEqual(report.fine_calls, 2)
        self.assertEqual(report.accepted_count, 1)


class SelfImprovementTests(unittest.TestCase):
    def test_retains_best_and_stops_before_degradation(self):
        scores = {0: 0.1, 1: 0.5, 2: 0.4}
        run = self_improve(
            0,
            lambda model, round_index: range(5),
            lambda sample: sample % 2 == 0,
            lambda model, samples: model + 1,
            lambda model: scores[model],
            maximum_rounds=4,
            sample_cap=2,
        )
        self.assertEqual(run.best_model, 1)
        self.assertEqual(run.stop_reason, "validation_degraded")
        self.assertEqual([round_.accepted for round_ in run.rounds], [2, 2])

    def test_stops_on_too_few_samples(self):
        run = self_improve(
            "m", lambda model, round_index: [], lambda sample: True,
            lambda model, samples: "new", lambda model: 1.0,
        )
        self.assertEqual(run.stop_reason, "insufficient_accepted_samples")


if __name__ == "__main__":
    unittest.main()
