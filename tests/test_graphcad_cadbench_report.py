import unittest

from harnesscad.eval.bench.protocols.graphcad_cadbench_report import (
    DIMENSIONS,
    SampleConfig,
    report,
    score_dimension,
    score_parameter,
    score_sample,
)

ATTR, SPAT, INST = DIMENSIONS


def make_config(sample_id="s1", split="Simulative"):
    return SampleConfig(
        sample_id=sample_id,
        criteria={
            ATTR: {
                "Shape accuracy": ["is a cylinder", "is uniform"],
                "Size": ["diameter is 8 in"],
            },
            SPAT: {"Object distance and contact": ["sits on a flat surface"]},
            INST: {"Instruction execution accuracy": ["follows the instruction"]},
        },
        split=split,
    )


ALL_PASS = {
    ATTR: {"Shape accuracy": [1, 1], "Size": [1]},
    SPAT: {"Object distance and contact": [1]},
    INST: {"Instruction execution accuracy": [1]},
}


class ConfigTests(unittest.TestCase):
    def test_requirement_count(self):
        self.assertEqual(make_config().requirement_count(), 5)

    def test_unknown_dimension_rejected(self):
        with self.assertRaises(ValueError):
            SampleConfig("s", {"Vibes": {"p": ["r"]}})

    def test_empty_requirement_list_rejected(self):
        with self.assertRaises(ValueError):
            SampleConfig("s", {ATTR: {"Shape accuracy": []}})


class ParameterTests(unittest.TestCase):
    def test_mean_of_requirements(self):
        self.assertEqual(score_parameter([1, 0, 1, 0]), 0.5)

    def test_non_binary_rejected(self):
        with self.assertRaises(ValueError):
            score_parameter([2])

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            score_parameter([])


class DimensionTests(unittest.TestCase):
    def test_parameters_are_weighted_equally(self):
        parameters = {"Shape accuracy": ["a", "b"], "Size": ["c"]}
        # Shape accuracy fully fails (0.0), Size fully passes (1.0) -> 0.5,
        # not 1/3 as a flat requirement mean would give.
        self.assertEqual(
            score_dimension(parameters, {"Shape accuracy": [0, 0], "Size": [1]}), 0.5
        )

    def test_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            score_dimension({"Size": ["a", "b"]}, {"Size": [1]})

    def test_missing_parameter_rejected(self):
        with self.assertRaises(KeyError):
            score_dimension({"Size": ["a"]}, {})


class SampleTests(unittest.TestCase):
    def test_all_pass(self):
        score = score_sample(make_config(), ALL_PASS)
        self.assertEqual(score.dimension_scores, {"Attr": 1.0, "Spat": 1.0, "Inst": 1.0})
        self.assertEqual(score.average, 1.0)
        self.assertTrue(score.judged)

    def test_partial(self):
        judgements = {
            ATTR: {"Shape accuracy": [1, 0], "Size": [0]},
            SPAT: {"Object distance and contact": [1]},
            INST: {"Instruction execution accuracy": [0]},
        }
        score = score_sample(make_config(), judgements)
        self.assertAlmostEqual(score.dimension_scores["Attr"], 0.25)  # mean(0.5, 0.0)
        self.assertEqual(score.dimension_scores["Spat"], 1.0)
        self.assertEqual(score.dimension_scores["Inst"], 0.0)
        self.assertAlmostEqual(score.average, (0.25 + 1.0 + 0.0) / 3.0)

    def test_unjudged_sample_scores_zero(self):
        score = score_sample(make_config(), None, has_geometry=False)
        self.assertEqual(score.average, 0.0)
        self.assertFalse(score.judged)
        self.assertFalse(score.has_geometry)


class ReportTests(unittest.TestCase):
    def test_failed_sample_stays_in_the_denominator(self):
        scores = [
            score_sample(make_config("a"), ALL_PASS),
            score_sample(make_config("b"), None, has_geometry=False),
        ]
        rows = report(scores)
        overall = rows["overall"]
        self.assertEqual(overall.total, 2)
        self.assertEqual(overall.judged, 1)
        self.assertEqual(overall.average, 0.5)  # not 1.0
        self.assertEqual(overall.syntax_error_rate, 50.0)

    def test_split_rows(self):
        scores = [
            score_sample(make_config("a", "Simulative"), ALL_PASS),
            score_sample(make_config("b", "Wild"), None, has_geometry=False),
        ]
        rows = report(scores)
        self.assertEqual(set(rows), {"overall", "Simulative", "Wild"})
        self.assertEqual(rows["Simulative"].average, 1.0)
        self.assertEqual(rows["Simulative"].syntax_error_rate, 0.0)
        self.assertEqual(rows["Wild"].average, 0.0)
        self.assertEqual(rows["Wild"].syntax_error_rate, 100.0)

    def test_as_row_keys(self):
        row = report([score_sample(make_config(), ALL_PASS)])["overall"].as_row()
        self.assertEqual(set(row), {"Attr", "Spat", "Inst", "Avg", "Esyntax"})
        self.assertEqual(row["Avg"], 1.0)

    def test_empty_report(self):
        rows = report([])
        self.assertEqual(rows["overall"].total, 0)
        self.assertEqual(rows["overall"].average, 0.0)

    def test_geometry_present_but_low_scores(self):
        judgements = {
            ATTR: {"Shape accuracy": [0, 0], "Size": [0]},
            SPAT: {"Object distance and contact": [0]},
            INST: {"Instruction execution accuracy": [0]},
        }
        rows = report([score_sample(make_config(), judgements, has_geometry=True)])
        self.assertEqual(rows["overall"].average, 0.0)
        self.assertEqual(rows["overall"].syntax_error_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
