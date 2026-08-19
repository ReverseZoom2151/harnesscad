"""Tests for the Onshape ConstraintType integer-id fact table."""

import unittest

from harnesscad.domain.geometry.sketch import constraint_type_ids as cti
from harnesscad.domain.geometry.sketch.constraint_satisfaction import CONSTRAINT_TYPES


class ConstraintTypeIdsTest(unittest.TestCase):
    def test_unique_ids(self):
        ids = [int(m.value) for m in cti.ConstraintType]
        self.assertEqual(len(ids), len(set(ids)))

    def test_core_contiguous_plus_sentinel(self):
        ids = {int(m.value) for m in cti.ConstraintType}
        self.assertTrue(set(range(0, 30)) <= ids)
        self.assertIn(101, ids)

    def test_name_id_round_trip_total(self):
        for member in cti.ConstraintType:
            self.assertEqual(cti.id_for_name(member.name), int(member.value))
            self.assertEqual(cti.name_for_id(int(member.value)), member.name)

    def test_case_insensitive_lookup(self):
        self.assertEqual(cti.id_for_name("COINCIDENT"), 0)
        self.assertEqual(cti.id_for_name("coincident"), 0)

    def test_histcad_names_are_subset(self):
        # Every HistCAD evaluation-side name has an authoritative integer id.
        for name in CONSTRAINT_TYPES:
            self.assertIn(name, cti.HISTCAD_TO_ID)
            self.assertIsInstance(int(cti.HISTCAD_TO_ID[name]), int)

    def test_specific_ids_match_source(self):
        self.assertEqual(cti.ConstraintType.Coincident, 0)
        self.assertEqual(cti.ConstraintType.Angle, 17)
        self.assertEqual(cti.ConstraintType.Rho, 28)
        self.assertEqual(cti.ConstraintType.Unknown, 29)
        self.assertEqual(cti.ConstraintType.Subnode, 101)

    def test_has_parameters(self):
        self.assertTrue(cti.has_parameters(cti.ConstraintType.Distance))
        self.assertTrue(cti.has_parameters(cti.ConstraintType.Radius))
        self.assertFalse(cti.has_parameters(cti.ConstraintType.Coincident))

    def test_selfcheck_exits_zero(self):
        self.assertEqual(cti._selfcheck(), 0)


if __name__ == "__main__":
    unittest.main()
