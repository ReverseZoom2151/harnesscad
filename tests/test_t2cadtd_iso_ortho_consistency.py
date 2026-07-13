import unittest

from harnesscad.domain.drawings.orthographic_projection import Box, project_three_views
from harnesscad.domain.drawings.iso_ortho_consistency import (
    recover_extents_from_isometric, isometric_edge_lengths,
    check_iso_ortho_consistency, mirror_extents, IsoOrthoResult,
)


class TestRoundTrip(unittest.TestCase):
    def test_forward_then_inverse_recovers_extents(self):
        edges = isometric_edge_lengths(2.0, 3.0, 5.0)
        rec = recover_extents_from_isometric(edges)
        self.assertAlmostEqual(rec["x"], 2.0)
        self.assertAlmostEqual(rec["y"], 3.0)
        self.assertAlmostEqual(rec["z"], 5.0)

    def test_missing_axis_is_zero(self):
        rec = recover_extents_from_isometric({"x": 1.0})
        self.assertAlmostEqual(rec["y"], 0.0)
        self.assertAlmostEqual(rec["z"], 0.0)

    def test_negative_length_raises(self):
        with self.assertRaises(ValueError):
            recover_extents_from_isometric({"x": -1.0})


class TestConsistency(unittest.TestCase):
    def setUp(self):
        self.box = Box(0, 0, 0, 2.0, 3.0, 5.0)
        self.views = project_three_views([self.box])
        self.edges = isometric_edge_lengths(2.0, 3.0, 5.0)

    def test_consistent_when_iso_matches_ortho(self):
        res = check_iso_ortho_consistency(self.edges, self.views)
        self.assertIsInstance(res, IsoOrthoResult)
        self.assertTrue(res.consistent)
        self.assertEqual(res.mismatches, ())

    def test_dimensions_reported(self):
        res = check_iso_ortho_consistency(self.edges, self.views)
        self.assertAlmostEqual(res.iso_dimensions["width"], 2.0)
        self.assertAlmostEqual(res.iso_dimensions["height"], 5.0)
        self.assertAlmostEqual(res.iso_dimensions["depth"], 3.0)
        self.assertAlmostEqual(res.ortho_dimensions["width"], 2.0)

    def test_inconsistent_when_iso_edge_perturbed(self):
        bad = dict(self.edges)
        bad["z"] = bad["z"] * 1.5  # height drifts in the isometric
        res = check_iso_ortho_consistency(bad, self.views)
        self.assertFalse(res.consistent)
        dims = {m["dimension"] for m in res.mismatches}
        self.assertIn("height", dims)

    def test_to_dict(self):
        d = check_iso_ortho_consistency(self.edges, self.views).to_dict()
        self.assertIn("iso_dimensions", d)
        self.assertIn("ortho_dimensions", d)
        self.assertTrue(d["consistent"])


class TestMirror(unittest.TestCase):
    def test_mirror_swaps_width_and_depth(self):
        dims = {"width": 2.0, "height": 5.0, "depth": 3.0}
        m = mirror_extents(dims)
        self.assertAlmostEqual(m["width"], 3.0)
        self.assertAlmostEqual(m["depth"], 2.0)
        self.assertAlmostEqual(m["height"], 5.0)

    def test_double_mirror_is_identity(self):
        dims = {"width": 2.0, "height": 5.0, "depth": 3.0}
        self.assertEqual(mirror_extents(mirror_extents(dims)), dims)


if __name__ == "__main__":
    unittest.main()
