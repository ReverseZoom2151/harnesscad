"""Tests for domain.reconstruction.sequences.spcc_components."""

import unittest

from harnesscad.domain.reconstruction.sequences.spcc_components import (
    classify_extrusion_direction,
    collapse_equivalent_components,
)


class ExtrusionDirectionTest(unittest.TestCase):
    def test_canonical_axes(self):
        self.assertEqual(classify_extrusion_direction((0, 0, 5))["word"], "up")
        self.assertEqual(classify_extrusion_direction((0, 0, -5))["word"], "down")
        self.assertEqual(classify_extrusion_direction((3, 0, 0))["word"], "right")
        self.assertEqual(classify_extrusion_direction((-3, 0, 0))["word"], "left")
        self.assertEqual(classify_extrusion_direction((0, 2, 0))["word"], "back")
        self.assertEqual(classify_extrusion_direction((0, -2, 0))["word"], "front")

    def test_axis_tuple(self):
        self.assertEqual(classify_extrusion_direction((0, 0, 7))["axis"], (0, 0, 1))

    def test_near_axis_within_tolerance(self):
        # small off-axis component within 10% is still on-axis.
        self.assertEqual(classify_extrusion_direction((10, 0.5, 0))["word"], "right")

    def test_skew_returns_none(self):
        self.assertIsNone(classify_extrusion_direction((10, 8, 0)))

    def test_zero_vector(self):
        self.assertIsNone(classify_extrusion_direction((0, 0, 0)))


class ComponentCollapseTest(unittest.TestCase):
    def _pair(self, r, origin):
        return {"cmd": "circle", "radius": r, "px": origin[0], "py": origin[1],
                "pz": origin[2]}

    def test_collapses_long_run(self):
        pairs = [self._pair(48, (i, 0, 0)) for i in range(5)]  # 5 equal > 3
        comps = collapse_equivalent_components(pairs, threshold=3)
        self.assertEqual(len(comps), 1)
        self.assertTrue(comps[0]["collapsed"])
        self.assertEqual(comps[0]["count"], 5)
        self.assertEqual(comps[0]["indices"], (0, 1, 2, 3, 4))

    def test_short_run_kept_individual(self):
        pairs = [self._pair(48, (i, 0, 0)) for i in range(3)]  # 3 == threshold
        comps = collapse_equivalent_components(pairs, threshold=3)
        self.assertEqual(len(comps), 3)
        self.assertTrue(all(not c["collapsed"] for c in comps))

    def test_origin_ignored_for_equivalence(self):
        # differ only by origin -> equivalent.
        a = self._pair(48, (0, 0, 0))
        b = self._pair(48, (60, 0, 0))
        comps = collapse_equivalent_components([a, b, a, b], threshold=3)
        self.assertEqual(comps[0]["count"], 4)
        self.assertTrue(comps[0]["collapsed"])

    def test_differing_params_not_equivalent(self):
        pairs = [self._pair(48, (0, 0, 0)), self._pair(30, (0, 0, 0)),
                 self._pair(48, (0, 0, 0))]
        comps = collapse_equivalent_components(pairs, threshold=1)
        self.assertEqual(len(comps), 3)

    def test_mixed_runs(self):
        pairs = ([self._pair(48, (i, 0, 0)) for i in range(4)]
                 + [self._pair(10, (0, 0, 0))])
        comps = collapse_equivalent_components(pairs, threshold=3)
        self.assertEqual(len(comps), 2)
        self.assertTrue(comps[0]["collapsed"])
        self.assertEqual(comps[0]["count"], 4)
        self.assertFalse(comps[1]["collapsed"])

    def test_bad_threshold(self):
        with self.assertRaises(ValueError):
            collapse_equivalent_components([], threshold=0)


if __name__ == "__main__":
    unittest.main()
