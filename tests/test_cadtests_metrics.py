"""Tests for bench.cadtests_metrics (benchmark metrics + mutation analysis)."""

import unittest

from harnesscad.eval.bench.protocols.test_assertions import (
    assert_aspect_ratio,
    assert_face_count,
    assert_typed_face_count,
    assert_valid_solid,
    assert_volume,
)
from harnesscad.eval.bench.data.cad_model_schema import CADModel, Edge, Face
from harnesscad.eval.bench.protocols.test_suite_quality import (
    analyze_test_suite,
    benchmark_scores,
    evaluate_method,
    refinement_gain,
)
from harnesscad.eval.bench.protocols.test_suite_runner import run_suite


def _box(volume=6.0, size=(2.0, 3.0, 1.0), faces=6, solids=1):
    fs = tuple(Face("plane", 1.0) for _ in range(faces))
    es = tuple(Edge("line", 1.0) for _ in range(12))
    com = tuple(s / 2.0 for s in size)
    return CADModel(fs, es, 8, (0.0, 0.0, 0.0), size, volume, com, solids=solids)


def _suite():
    return [
        assert_valid_solid(requirement="validity"),
        assert_face_count(6, requirement="topology"),
        assert_volume(6.0, requirement="volume"),
    ]


class TestBenchmarkScores(unittest.TestCase):
    def test_all_pass_sample(self):
        rows = [run_suite(_box(), _suite())]
        s = benchmark_scores(rows)
        self.assertEqual(s["pass_rate"], 1.0)
        self.assertEqual(s["requirement_score"], 1.0)
        self.assertEqual(s["invalid_ratio"], 0.0)
        self.assertEqual(s["n"], 1)

    def test_mixed_samples(self):
        good = run_suite(_box(), _suite())
        bad = run_suite(_box(volume=99.0), _suite())      # volume test fails
        invalid = run_suite(None, _suite())               # invalid generation
        s = benchmark_scores([good, bad, invalid])
        # only 1 of 3 samples passes all tests
        self.assertAlmostEqual(s["pass_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(s["invalid_ratio"], 1.0 / 3.0)
        # RS: good=1.0, bad=2/3 (volume group fails), invalid=0 -> mean
        self.assertAlmostEqual(s["requirement_score"],
                               (1.0 + 2.0 / 3.0 + 0.0) / 3.0)

    def test_percent_scaling(self):
        rows = [run_suite(_box(), _suite())]
        s = benchmark_scores(rows, as_percent=True)
        self.assertEqual(s["pass_rate"], 100.0)

    def test_category_accuracy(self):
        good = run_suite(_box(), _suite())
        bad = run_suite(_box(volume=99.0), _suite())
        s = benchmark_scores([good, bad])
        acc = s["accuracy"]
        # validity + topology pass on both samples -> 100%
        self.assertEqual(acc["solid_shell_validity"], 1.0)
        self.assertEqual(acc["topology"], 1.0)
        # volumetric passes on 1 of 2 samples
        self.assertAlmostEqual(acc["volumetric"], 0.5)
        # no geometric-type tests -> None
        self.assertIsNone(acc["geometric_types"])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            benchmark_scores([])

    def test_evaluate_method_helper(self):
        samples = [
            (_box(), _suite()),
            (None, _suite()),
        ]
        s = evaluate_method(samples)
        self.assertAlmostEqual(s["pass_rate"], 0.5)
        self.assertAlmostEqual(s["invalid_ratio"], 0.5)
        self.assertEqual(len(s["rows"]), 2)


class TestMutationAnalysis(unittest.TestCase):
    def _reference(self):
        return _box(volume=6.0, size=(2.0, 3.0, 1.0), faces=6)

    def _mutants(self):
        return [
            _box(volume=99.0),                 # wrong volume
            _box(faces=8),                     # wrong face count
            _box(solids=2),                    # not a valid solid
        ]

    def test_all_valid_sound(self):
        q = analyze_test_suite(_suite(), self._reference(), [])
        self.assertEqual(q.validity, 1.0)
        self.assertEqual(q.soundness, 1.0)
        self.assertEqual(q.n_sound, 3)

    def test_kills_all_mutants(self):
        q = analyze_test_suite(_suite(), self._reference(), self._mutants())
        self.assertEqual(q.n_mutants, 3)
        self.assertEqual(q.n_killed, 3)
        self.assertEqual(q.mutation_score, 1.0)
        self.assertEqual(q.killed, (True, True, True))

    def test_weak_suite_misses_mutant(self):
        # a suite that only checks validity cannot catch the volume mutant
        weak = [assert_valid_solid()]
        q = analyze_test_suite(weak, self._reference(),
                               [_box(volume=99.0), _box(solids=2)])
        # volume mutant survives, solids mutant is killed
        self.assertEqual(q.killed, (False, True))
        self.assertAlmostEqual(q.mutation_score, 0.5)

    def test_unsound_test_excluded(self):
        # a test that fails on the reference is not sound and cannot kill
        unsound = [assert_volume(999.0)]     # fails on reference
        q = analyze_test_suite(unsound, self._reference(), [_box(volume=1.0)])
        self.assertEqual(q.n_valid, 1)       # it executes (valid)
        self.assertEqual(q.n_sound, 0)       # but does not pass reference
        self.assertEqual(q.mutation_score, 0.0)

    def test_invalid_test_not_valid(self):
        # aspect ratio dividing by a zero extent raises on the reference
        flat_ref = _box(size=(2.0, 3.0, 0.0), volume=0.0)
        suite = [assert_aspect_ratio("x", "z", 1.0)]
        q = analyze_test_suite(suite, flat_ref, [])
        self.assertEqual(q.n_valid, 0)
        self.assertEqual(q.validity, 0.0)

    def test_refinement_gain(self):
        ref = self._reference()
        muts = self._mutants()
        before = analyze_test_suite([assert_valid_solid()], ref, muts)
        after = analyze_test_suite(_suite(), ref, muts)
        g = refinement_gain(before, after)
        self.assertGreater(g["absolute_gain"], 0.0)
        self.assertEqual(g["mutation_score_after"], 1.0)
        self.assertGreater(g["sound_tests_delta"], 0)


if __name__ == "__main__":
    unittest.main()
