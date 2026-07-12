"""Tests for bench.voxhammer_coherence."""
import unittest

from bench.voxhammer_coherence import (
    boundary_discontinuity,
    boundary_pairs,
    coherence_report,
    preservation_max_error,
    preservation_mse,
    preservation_psnr,
)


class TestBoundaryPairs(unittest.TestCase):
    def setUp(self):
        # line of 4 voxels; edit = first two
        self.coords = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]
        self.edit = {(0, 0, 0), (1, 0, 0)}

    def test_single_boundary(self):
        p = boundary_pairs(self.coords, self.edit, 6)
        self.assertEqual(p, (((1, 0, 0), (2, 0, 0)),))

    def test_no_boundary_when_all_edit(self):
        p = boundary_pairs(self.coords, set(self.coords), 6)
        self.assertEqual(p, ())

    def test_bad_connectivity(self):
        with self.assertRaises(ValueError):
            boundary_pairs(self.coords, self.edit, 7)


class TestBoundaryDiscontinuity(unittest.TestCase):
    def test_smooth_is_low(self):
        coords = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        edit = {(0, 0, 0)}
        latents = {(0, 0, 0): (1.0,), (1, 0, 0): (1.0,), (2, 0, 0): (1.0,)}
        self.assertEqual(boundary_discontinuity(latents, coords, edit), 0.0)

    def test_jump_measured(self):
        coords = [(0, 0, 0), (1, 0, 0)]
        edit = {(0, 0, 0)}
        latents = {(0, 0, 0): (0.0,), (1, 0, 0): (3.0,)}
        # boundary pair (0,0,0)-(1,0,0), L2 = 3
        self.assertEqual(boundary_discontinuity(latents, coords, edit), 3.0)

    def test_none_without_boundary(self):
        coords = [(0, 0, 0)]
        edit = {(0, 0, 0)}
        latents = {(0, 0, 0): (1.0,)}
        self.assertIsNone(boundary_discontinuity(latents, coords, edit))


class TestPreservation(unittest.TestCase):
    def setUp(self):
        self.keep = {(1, 0, 0), (2, 0, 0)}
        self.source = {(1, 0, 0): (1.0, 1.0), (2, 0, 0): (2.0, 2.0)}

    def test_perfect_preservation(self):
        edited = dict(self.source)
        self.assertEqual(preservation_mse(edited, self.source, self.keep), 0.0)
        self.assertEqual(preservation_max_error(edited, self.source, self.keep), 0.0)
        self.assertEqual(preservation_psnr(edited, self.source, self.keep), float("inf"))

    def test_mse_value(self):
        edited = {(1, 0, 0): (2.0, 1.0), (2, 0, 0): (2.0, 2.0)}
        # only one component differs by 1 over 4 components -> mse 0.25
        self.assertAlmostEqual(preservation_mse(edited, self.source, self.keep), 0.25)

    def test_max_error(self):
        edited = {(1, 0, 0): (4.0, 1.0), (2, 0, 0): (2.0, 2.0)}
        self.assertAlmostEqual(preservation_max_error(edited, self.source, self.keep), 3.0)

    def test_psnr_finite(self):
        edited = {(1, 0, 0): (2.0, 1.0), (2, 0, 0): (2.0, 2.0)}
        val = preservation_psnr(edited, self.source, self.keep, peak=1.0)
        self.assertTrue(val < float("inf"))
        self.assertAlmostEqual(val, 10.0 * (0 - 0) - 0, delta=100)  # finite sanity

    def test_empty_keep(self):
        self.assertIsNone(preservation_mse(self.source, self.source, set()))


class TestCoherenceReport(unittest.TestCase):
    def test_keys_and_perfect(self):
        coords = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        edit = {(0, 0, 0)}
        keep = {(1, 0, 0), (2, 0, 0)}
        latents = {(0, 0, 0): (1.0,), (1, 0, 0): (1.0,), (2, 0, 0): (1.0,)}
        rep = coherence_report(latents, latents, coords, edit, keep)
        self.assertEqual(rep["boundary_discontinuity"], 0.0)
        self.assertEqual(rep["preservation_mse"], 0.0)
        self.assertEqual(rep["preservation_psnr"], float("inf"))
        self.assertEqual(rep["n_boundary_pairs"], 1)

    def test_deterministic(self):
        coords = [(0, 0, 0), (1, 0, 0)]
        edit = {(0, 0, 0)}
        keep = {(1, 0, 0)}
        src = {(0, 0, 0): (0.0,), (1, 0, 0): (1.0,)}
        ed = {(0, 0, 0): (0.5,), (1, 0, 0): (1.2,)}
        self.assertEqual(
            coherence_report(ed, src, coords, edit, keep),
            coherence_report(ed, src, coords, edit, keep),
        )


if __name__ == "__main__":
    unittest.main()
