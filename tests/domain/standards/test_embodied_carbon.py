"""Tests for domain.standards.embodied_carbon."""

import unittest

from harnesscad.domain.standards.embodied_carbon import (
    DEFAULT_CO2E,
    MaterialUse,
    aggregate,
    carbon_intensity,
    embodied_carbon,
    top_contributors,
)


class IntensityTest(unittest.TestCase):
    def test_lookup_case_insensitive(self):
        self.assertEqual(carbon_intensity("Steel"), DEFAULT_CO2E["steel"])

    def test_unknown_raises(self):
        with self.assertRaises(KeyError):
            carbon_intensity("unobtanium")


class EmbodiedTest(unittest.TestCase):
    def test_single(self):
        self.assertAlmostEqual(
            embodied_carbon(MaterialUse("steel", 10.0)), DEFAULT_CO2E["steel"] * 10.0
        )

    def test_negative_mass_rejected(self):
        with self.assertRaises(ValueError):
            MaterialUse("steel", -1.0)

    def test_aggregate_sums(self):
        uses = [MaterialUse("steel", 2.0), MaterialUse("concrete", 100.0)]
        expected = DEFAULT_CO2E["steel"] * 2 + DEFAULT_CO2E["concrete"] * 100
        self.assertAlmostEqual(aggregate(uses), expected)


class TopContributorsTest(unittest.TestCase):
    def test_ranking_descending(self):
        uses = [
            MaterialUse("concrete", 1000.0),   # 110
            MaterialUse("aluminium", 50.0),    # 412
            MaterialUse("timber", 10.0),       # 4.5
        ]
        top = top_contributors(uses, n=2)
        self.assertEqual(top[0][0], "aluminium")
        self.assertEqual(top[1][0], "concrete")
        self.assertEqual(len(top), 2)

    def test_repeated_material_combined(self):
        uses = [MaterialUse("steel", 1.0), MaterialUse("steel", 3.0)]
        top = top_contributors(uses)
        self.assertEqual(len(top), 1)
        self.assertAlmostEqual(top[0][1], DEFAULT_CO2E["steel"] * 4.0)

    def test_deterministic(self):
        uses = [MaterialUse("brick", 5.0), MaterialUse("glass", 5.0)]
        self.assertEqual(top_contributors(uses), top_contributors(uses))


if __name__ == "__main__":
    unittest.main()
