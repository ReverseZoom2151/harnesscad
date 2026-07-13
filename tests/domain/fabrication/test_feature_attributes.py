"""Tests for fabrication/mfgfeat_attributes.py."""

from __future__ import annotations

import unittest

from harnesscad.domain.fabrication.feature_attributes import extract_attributes


class TestHole(unittest.TestCase):
    def test_blind_hole_derives_subtype_and_aspect(self):
        out = extract_attributes("hole", {"diameter": 5.0, "depth": 15.0})
        self.assertEqual(out["subtype"], "blind")
        self.assertFalse(out["through"])
        self.assertAlmostEqual(out["aspect_ratio"], 3.0)

    def test_through_hole(self):
        out = extract_attributes("through hole",
                                 {"diameter": 4.0, "through": True})
        self.assertEqual(out["subtype"], "through")
        self.assertTrue(out["through"])

    def test_explicit_subtype(self):
        out = extract_attributes("hole",
                                 {"diameter": 4.0, "subtype": "countersink"})
        self.assertEqual(out["subtype"], "countersink")

    def test_threaded_flag(self):
        out = extract_attributes("hole", {"diameter": 6.0, "threaded": True})
        self.assertEqual(out["subtype"], "threaded")

    def test_counterbore_derived(self):
        out = extract_attributes("hole",
                                 {"diameter": 4.0, "counterbore_diameter": 8.0})
        self.assertEqual(out["subtype"], "counterbore")

    def test_bad_subtype(self):
        with self.assertRaises(ValueError):
            extract_attributes("hole", {"diameter": 4.0, "subtype": "banana"})

    def test_nonpositive_diameter(self):
        with self.assertRaises(ValueError):
            extract_attributes("hole", {"diameter": 0.0})


class TestPrismatic(unittest.TestCase):
    def test_slot(self):
        out = extract_attributes("slot",
                                 {"width": 4.0, "length": 20.0, "depth": 8.0})
        self.assertEqual(out["width"], 4.0)
        self.assertAlmostEqual(out["aspect_ratio"], 2.0)

    def test_pocket_corner_radius(self):
        out = extract_attributes("pocket",
                                 {"width": 10.0, "length": 10.0, "depth": 5.0,
                                  "corner_radius": 2.0})
        self.assertEqual(out["corner_radius"], 2.0)

    def test_step(self):
        out = extract_attributes("step", {"width": 6.0, "depth": 3.0})
        self.assertAlmostEqual(out["aspect_ratio"], 0.5)


class TestEdgeBlends(unittest.TestCase):
    def test_chamfer(self):
        out = extract_attributes("chamfer", {"width": 1.0, "angle": 45.0})
        self.assertEqual(out["angle"], 45.0)

    def test_chamfer_bad_angle(self):
        with self.assertRaises(ValueError):
            extract_attributes("chamfer", {"angle": 120.0})

    def test_fillet(self):
        out = extract_attributes("fillet", {"radius": 2.5})
        self.assertEqual(out["radius"], 2.5)


class TestPipeTube(unittest.TestCase):
    def test_tube_hollow(self):
        out = extract_attributes("pipe/tube",
                                 {"outer_diameter": 10.0, "inner_diameter": 6.0,
                                  "length": 30.0})
        self.assertEqual(out["class"], "tube")
        self.assertAlmostEqual(out["wall_thickness"], 2.0)

    def test_pipe_solid(self):
        out = extract_attributes("pipe_tube",
                                 {"outer_diameter": 10.0, "inner_diameter": 0.0})
        self.assertEqual(out["class"], "pipe")

    def test_inner_ge_outer_rejected(self):
        with self.assertRaises(ValueError):
            extract_attributes("pipe_tube",
                               {"outer_diameter": 5.0, "inner_diameter": 5.0})


class TestDraftAndGeneric(unittest.TestCase):
    def test_draft_sufficiency(self):
        out = extract_attributes("draft", {"angle": 2.0, "min_draft": 1.0})
        self.assertTrue(out["sufficient"])
        bad = extract_attributes("draft", {"angle": 0.5, "min_draft": 1.0})
        self.assertFalse(bad["sufficient"])

    def test_generic_gear_teeth_count(self):
        out = extract_attributes("gear teeth", {"count": 24, "module": 1.5})
        self.assertEqual(out["count"], 24)
        self.assertEqual(out["module"], 1.5)

    def test_generic_drops_unknown_keys(self):
        out = extract_attributes("rib",
                                 {"thickness": 2.0, "height": 10.0,
                                  "bogus": 99})
        self.assertNotIn("bogus", out)

    def test_unknown_feature(self):
        with self.assertRaises(KeyError):
            extract_attributes("wormhole", {})


if __name__ == "__main__":
    unittest.main()
