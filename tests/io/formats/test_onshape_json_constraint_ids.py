"""Tests that onshape_json resolves constraint names via the ConstraintType table.

These prove the wiring: onshape_json's new resolvers delegate to the authoritative
``constraint_type_ids`` fact table (not a second mapping), so a parsed Onshape
sketch constraint gets a canonical SketchGraphs integer id.
"""

import unittest

from harnesscad.domain.geometry.sketch import constraint_type_ids
from harnesscad.io.formats.onshape_json import (
    ConstraintType,
    constraint_type_id,
    constraint_type_name,
    inspect_constraint_type,
)


class ConstraintTypeResolverTest(unittest.TestCase):
    def test_reexports_enum(self):
        # onshape_json re-exports the authoritative enum, not a copy.
        self.assertIs(ConstraintType, constraint_type_ids.ConstraintType)

    def test_known_name_resolves_to_canonical_id(self):
        # A known Onshape constraint name -> its integer id, through onshape_json.
        self.assertEqual(constraint_type_id("Coincident"), 0)
        self.assertEqual(constraint_type_id("Tangent"), 7)
        self.assertEqual(constraint_type_id("Circular_Pattern"), 18)

    def test_case_insensitive(self):
        self.assertEqual(constraint_type_id("COINCIDENT"), 0)
        self.assertEqual(constraint_type_id("  tangent  "), 7)

    def test_histcad_radius_aliases(self):
        # HistCAD labels these by radius; SketchGraphs by diameter -- same id.
        self.assertEqual(
            constraint_type_id("minor_radius"),
            int(ConstraintType.Minor_Diameter),
        )
        self.assertEqual(
            constraint_type_id("major_radius"),
            int(ConstraintType.Major_Diameter),
        )

    def test_resolver_agrees_with_bridge_table(self):
        # Every HistCAD name resolves exactly to its HISTCAD_TO_ID entry.
        for name, cid in constraint_type_ids.HISTCAD_TO_ID.items():
            self.assertEqual(constraint_type_id(name), int(cid))

    def test_name_round_trip(self):
        self.assertEqual(constraint_type_name(0), "Coincident")
        self.assertEqual(constraint_type_name(constraint_type_id("Tangent")), "Tangent")

    def test_unknown_name_raises(self):
        with self.assertRaises(KeyError):
            constraint_type_id("not_a_real_constraint")

    def test_inspect_raw_constraint_blob(self):
        # Onshape BTMSketchConstraint layout: name lives in message.constraintType.
        blob = {
            "type": 2,
            "typeName": "BTMSketchConstraint",
            "message": {"constraintType": "COINCIDENT", "entityId": "c0"},
        }
        self.assertEqual(inspect_constraint_type(blob), 0)

    def test_inspect_top_level_fallback(self):
        self.assertEqual(inspect_constraint_type({"constraintType": "Tangent"}), 7)

    def test_inspect_missing_type_raises(self):
        with self.assertRaises(KeyError):
            inspect_constraint_type({"message": {"entityId": "c0"}})


if __name__ == "__main__":
    unittest.main()
