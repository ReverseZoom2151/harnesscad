import unittest

from harnesscad.eval.bench.geometry.img2cadrev_metrics import (
    chamfer_distance, mirror_points, symmetry_chamfer, num_scc,
    structure_accuracy, attribute_error, factorization_fidelity,
    factorization_report,
)
from harnesscad.domain.reconstruction.sequences.img2cadrev_factorization import factorize


def sample_model():
    return [
        {"label": "seat", "commands": [
            {"type": "L", "attrs": [1.0, 0.0]},
            {"type": "Ej", "attrs": [0, 0, 0, 0, 0, 0, 0.2]},
        ]},
        {"label": "leg", "commands": [
            {"type": "R", "attrs": [0.0, 0.0, 0.1]},
        ]},
    ]


class ChamferTest(unittest.TestCase):
    def test_identical_zero(self):
        pts = [(0, 0, 0), (1, 1, 1)]
        self.assertAlmostEqual(chamfer_distance(pts, pts), 0.0)

    def test_shifted(self):
        a = [(0.0, 0.0)]
        b = [(3.0, 4.0)]
        # 5 each direction -> 10
        self.assertAlmostEqual(chamfer_distance(a, b), 10.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            chamfer_distance([], [(0, 0)])


class SymmetryTest(unittest.TestCase):
    def test_mirror_points(self):
        self.assertEqual(mirror_points([(1.0, 2.0)], "x"), [(-1.0, 2.0)])
        self.assertEqual(mirror_points([(1.0, 2.0, 3.0)], "z"), [(1.0, 2.0, -3.0)])

    def test_symmetric_set_zero(self):
        # Symmetric about x=0.
        pts = [(-1.0, 0.0), (1.0, 0.0), (0.0, 2.0)]
        self.assertAlmostEqual(symmetry_chamfer(pts, "x"), 0.0)

    def test_asymmetric_positive(self):
        pts = [(1.0, 0.0), (2.0, 0.0)]
        self.assertGreater(symmetry_chamfer(pts, "x"), 0.0)


class SccTest(unittest.TestCase):
    def test_single_cluster(self):
        pts = [(0.0, 0.0), (0.01, 0.0), (0.02, 0.0)]
        self.assertEqual(num_scc(pts, threshold=0.05), 1)

    def test_two_clusters(self):
        pts = [(0.0, 0.0), (0.01, 0.0), (10.0, 0.0), (10.01, 0.0)]
        self.assertEqual(num_scc(pts, threshold=0.05), 2)

    def test_all_isolated(self):
        pts = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        self.assertEqual(num_scc(pts, threshold=0.05), 3)

    def test_empty(self):
        self.assertEqual(num_scc([]), 0)


class FactorizationMetricTest(unittest.TestCase):
    def test_structure_accuracy_perfect(self):
        s, _ = factorize(sample_model())
        self.assertEqual(structure_accuracy(s, s), 1.0)

    def test_structure_accuracy_label_mismatch(self):
        s, _ = factorize(sample_model())
        pred = [dict(p) for p in s]
        pred[0] = {"label": "backrest", "command_types": s[0]["command_types"]}
        # Part 0 has 2 of 3 commands wrong-labelled; leg part (1 cmd) correct.
        acc = structure_accuracy(pred, s)
        self.assertAlmostEqual(acc, 1.0 / 3.0)

    def test_structure_accuracy_extra_command_penalized(self):
        s, _ = factorize(sample_model())
        pred = [dict(p) for p in s]
        pred[1] = {"label": "leg", "command_types": ["R", "Ej"]}  # extra cmd
        acc = structure_accuracy(pred, s)
        # denom = max(3,4)=4, hits=3 -> 0.75
        self.assertAlmostEqual(acc, 0.75)

    def test_attribute_error(self):
        _, a = factorize(sample_model())
        pred = [[v + 1.0 for v in vec] for vec in a]
        self.assertAlmostEqual(attribute_error(pred, a), 1.0)

    def test_attribute_error_mismatch_raises(self):
        _, a = factorize(sample_model())
        with self.assertRaises(ValueError):
            attribute_error(a[:-1], a)

    def test_factorization_fidelity(self):
        self.assertEqual(factorization_fidelity(sample_model()), 1.0)

    def test_report(self):
        s, a = factorize(sample_model())
        rep = factorization_report(s, a, s, a)
        self.assertEqual(rep["structure_accuracy"], 1.0)
        self.assertAlmostEqual(rep["attribute_error"], 0.0)
        self.assertTrue(rep["command_count_match"])


if __name__ == "__main__":
    unittest.main()
