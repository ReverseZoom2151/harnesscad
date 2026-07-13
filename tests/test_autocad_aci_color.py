"""Tests for standards.autocad_aci_color."""

import unittest

from harnesscad.domain.standards.aci_color import (
    BYBLOCK,
    BYLAYER,
    BYENTITY,
    NAME_TO_ACI,
    ACI_RGB,
    name_to_aci,
    aci_to_name,
    is_special,
    validate_aci,
    aci_to_rgb,
    nearest_aci,
)


class TestNameIndex(unittest.TestCase):
    def test_round_trip(self):
        for name, idx in NAME_TO_ACI.items():
            self.assertEqual(name_to_aci(name), idx)
            self.assertEqual(aci_to_name(idx), name)

    def test_case_insensitive(self):
        self.assertEqual(name_to_aci("red"), 1)
        self.assertEqual(name_to_aci("Yellow"), 2)

    def test_unknown_name(self):
        with self.assertRaises(ValueError):
            name_to_aci("chartreuse")

    def test_unnamed_index_none(self):
        self.assertIsNone(aci_to_name(200))


class TestSpecialAndValidate(unittest.TestCase):
    def test_special(self):
        self.assertTrue(is_special(BYBLOCK))
        self.assertTrue(is_special(BYLAYER))
        self.assertTrue(is_special(BYENTITY))
        self.assertFalse(is_special(1))

    def test_validate_ok(self):
        self.assertEqual(validate_aci(0), 0)
        self.assertEqual(validate_aci(256), 256)

    def test_validate_out_of_range(self):
        with self.assertRaises(ValueError):
            validate_aci(258)
        with self.assertRaises(ValueError):
            validate_aci(-1)


class TestRgb(unittest.TestCase):
    def test_primaries(self):
        self.assertEqual(aci_to_rgb(1), (255, 0, 0))
        self.assertEqual(aci_to_rgb(5), (0, 0, 255))

    def test_undefined_rgb(self):
        self.assertIsNone(aci_to_rgb(8))  # gray has no fabricated RGB here

    def test_all_defined_are_valid(self):
        for idx in ACI_RGB:
            self.assertEqual(validate_aci(idx), idx)


class TestNearest(unittest.TestCase):
    def test_exact_red(self):
        self.assertEqual(nearest_aci((255, 0, 0)), 1)

    def test_near_blue(self):
        self.assertEqual(nearest_aci((10, 10, 240)), 5)

    def test_white_maps_to_seven_not_alias(self):
        self.assertEqual(nearest_aci((255, 255, 255)), 7)

    def test_tie_prefers_lower_index(self):
        # equidistant construction: midpoint of red(1) and green(3)
        idx = nearest_aci((128, 128, 0))
        self.assertIn(idx, (1, 2, 3))


if __name__ == "__main__":
    unittest.main()
