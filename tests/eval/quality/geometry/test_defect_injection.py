"""Tests for the Roshera injected-defect benchmark."""

import unittest

from harnesscad.eval.quality.geometry import defect_injection as di


class BaseMeshTest(unittest.TestCase):
    def test_tetrahedron_is_sound(self):
        self.assertTrue(di.topology_verifier(di.unit_tetrahedron()))

    def test_cube_is_sound(self):
        self.assertTrue(di.topology_verifier(di.unit_cube_mesh()))


class InjectorTest(unittest.TestCase):
    def setUp(self):
        self.mesh = di.unit_tetrahedron()

    def test_flip_makes_unsound(self):
        self.assertFalse(di.topology_verifier(di.inject_flipped_normal(self.mesh)))

    def test_torn_seam_makes_unsound(self):
        self.assertFalse(di.topology_verifier(di.inject_torn_seam(self.mesh)))

    def test_non_manifold_makes_unsound(self):
        self.assertFalse(di.topology_verifier(di.inject_non_manifold(self.mesh)))

    def test_degenerate_makes_unsound(self):
        self.assertFalse(di.topology_verifier(di.inject_degenerate_facet(self.mesh)))

    def test_injectors_do_not_mutate_original(self):
        before = self.mesh
        di.inject_flipped_normal(self.mesh)
        self.assertEqual(before, self.mesh)


class BenchmarkTest(unittest.TestCase):
    def test_topology_verifier_catches_all_four(self):
        result = di.run_benchmark(di.unit_tetrahedron())
        self.assertTrue(result.base_sound)
        self.assertEqual(result.catch_count, 4)
        self.assertEqual(result.total, 4)
        self.assertAlmostEqual(result.catch_rate, 1.0)

    def test_cube_base_catches_all(self):
        result = di.run_benchmark(di.unit_cube_mesh())
        self.assertEqual(result.catch_count, 4)

    def test_blind_verifier_catches_none(self):
        # A verifier that always says "sound" catches nothing.
        result = di.run_benchmark(di.unit_tetrahedron(), verifier=lambda m: True)
        self.assertEqual(result.catch_count, 0)

    def test_unknown_class_raises(self):
        with self.assertRaises(di.DefectError):
            di.run_benchmark(di.unit_tetrahedron(), classes=["not_a_class"])

    def test_summary_string(self):
        result = di.run_benchmark(di.unit_tetrahedron())
        text = result.summary()
        self.assertIn("caught 4/4", text)
        self.assertIn("base_sound=True", text)

    def test_only_watertight_heuristic_partial(self):
        # A heuristic that only checks edge incidence (ignores orientation and
        # degeneracy) should miss the flipped-normal lie.
        def watertight_only(mesh):
            from collections import Counter
            edges = Counter()
            for f in mesh.faces:
                n = len(f)
                for k in range(n):
                    i, j = f[k], f[(k + 1) % n]
                    edges[tuple(sorted((i, j)))] += 1
            return all(c == 2 for c in edges.values())

        result = di.run_benchmark(di.unit_tetrahedron(), verifier=watertight_only)
        self.assertFalse(result.caught["flipped_normal"])
        self.assertTrue(result.caught["torn_seam"])


if __name__ == "__main__":
    unittest.main()
