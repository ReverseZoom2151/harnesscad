"""Tests for persistent topological naming and query resolution (WHUCAD/CADFS)."""

import unittest

from harnesscad.domain.programs import topological_naming as tn


class PersistentNameTest(unittest.TestCase):
    def test_same_provenance_same_digest(self):
        a = tn.make_persistent_name("edge", "F5", "SWEPT_EDGE", ["F1", "F5"], "c=1,2,3")
        b = tn.make_persistent_name("edge", "F5", "SWEPT_EDGE", ["F1", "F5"], "c=1,2,3")
        self.assertEqual(a.digest, b.digest)

    def test_history_change_changes_digest(self):
        a = tn.make_persistent_name("edge", "F5", "SWEPT_EDGE", ["F1", "F5"])
        b = tn.make_persistent_name("edge", "F5", "SWEPT_EDGE", ["F2", "F5"])
        self.assertNotEqual(a.digest, b.digest)

    def test_bad_entity_type(self):
        with self.assertRaises(ValueError):
            tn.make_persistent_name("blob", "F1", "role", [])


class QueryTest(unittest.TestCase):
    def setUp(self):
        self.records = [
            tn.EntityRecord("e1", "edge", "F5", "SWEPT_EDGE", ancestors=("F1",)),
            tn.EntityRecord("e2", "edge", "F5", "SWEPT_EDGE", ancestors=("F2",)),
            tn.EntityRecord("f1", "face", "F5", "SIDE_FACE"),
        ]

    def test_resolves_unique_with_disambiguation(self):
        q = tn.EntityQuery("F5", "SWEPT_EDGE", "edge", require_ancestors=("F1",))
        self.assertEqual(tn.resolve_query(q, self.records), ("e1",))
        self.assertTrue(tn.reference_survives(q, self.records))

    def test_ambiguous_without_disambiguation(self):
        q = tn.EntityQuery("F5", "SWEPT_EDGE", "edge")
        self.assertEqual(set(tn.resolve_query(q, self.records)), {"e1", "e2"})
        self.assertFalse(tn.reference_survives(q, self.records))

    def test_dangling_reference(self):
        # after an edit the creating op F5 no longer produces this role
        q = tn.EntityQuery("F9", "SWEPT_EDGE", "edge")
        self.assertEqual(tn.resolve_query(q, self.records), ())
        self.assertFalse(tn.reference_survives(q, self.records))

    def test_entity_type_filter(self):
        q = tn.EntityQuery("F5", "SIDE_FACE", "face")
        self.assertEqual(tn.resolve_query(q, self.records), ("f1",))

    def test_geometry_key_disambiguation(self):
        recs = [
            tn.EntityRecord("a", "vertex", "F1", "CORNER", geometry_key="0,0,0"),
            tn.EntityRecord("b", "vertex", "F1", "CORNER", geometry_key="1,0,0"),
        ]
        q = tn.EntityQuery("F1", "CORNER", "vertex", require_geometry_key="1,0,0")
        self.assertEqual(tn.resolve_query(q, recs), ("b",))


if __name__ == "__main__":
    unittest.main()
