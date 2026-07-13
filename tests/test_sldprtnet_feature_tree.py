"""Tests for reconstruction.sldprtnet_feature_tree."""

import unittest

from harnesscad.domain.reconstruction.sequences.feature_tree import (
    FEATURE_TYPES,
    FeatureNode,
    FeatureTree,
    is_supported_feature,
)


def _sample_tree() -> FeatureTree:
    nodes = [
        FeatureNode("Top View", "RefPlane", parent=None,
                    params=(("Vertex1", "-80.9, 0.0, -50.0"),
                            ("Vertex2", "80.9, 0.0, -50.0"))),
        FeatureNode("Sketch1", "ProfileFeature", parent="Top View",
                    params=(("SketchPlane", "Top View"),
                            ("Radius", "994.0"))),
        FeatureNode("Extrude1", "Extrusion", parent="Sketch1",
                    params=(("d1", "150.0"), ("merge", "True")),
                    depends_on=("Sketch1",)),
    ]
    return FeatureTree(nodes=nodes)


class TestFeatureNode(unittest.TestCase):
    def test_rejects_unknown_type(self):
        with self.assertRaises(ValueError):
            FeatureNode("X", "NotAType")

    def test_rejects_empty_name(self):
        with self.assertRaises(ValueError):
            FeatureNode("  ", "Extrusion")

    def test_param_lookup(self):
        n = FeatureNode("E", "Extrusion", params=(("d1", "5"),))
        self.assertEqual(n.param("d1"), "5")
        self.assertIsNone(n.param("missing"))

    def test_thirteen_feature_types(self):
        self.assertEqual(len(FEATURE_TYPES), 13)
        self.assertEqual(len(set(FEATURE_TYPES)), 13)


class TestFeatureTree(unittest.TestCase):
    def test_validate_ok(self):
        _sample_tree().validate()

    def test_duplicate_names_rejected(self):
        t = FeatureTree([FeatureNode("A", "Extrusion"),
                         FeatureNode("A", "Fillet")])
        with self.assertRaises(ValueError):
            t.validate()

    def test_parent_must_precede(self):
        t = FeatureTree([FeatureNode("Child", "Extrusion", parent="Parent"),
                         FeatureNode("Parent", "RefPlane")])
        with self.assertRaises(ValueError):
            t.validate()

    def test_missing_dependency_rejected(self):
        t = FeatureTree([FeatureNode("E", "Extrusion", depends_on=("Ghost",))])
        with self.assertRaises(ValueError):
            t.validate()

    def test_depth(self):
        t = _sample_tree()
        self.assertEqual(t.depth_of("Top View"), 0)
        self.assertEqual(t.depth_of("Sketch1"), 1)
        self.assertEqual(t.depth_of("Extrude1"), 2)

    def test_feature_counts(self):
        t = _sample_tree()
        self.assertEqual(
            t.feature_counts(),
            {"Extrusion": 1, "ProfileFeature": 1, "RefPlane": 1},
        )
        self.assertEqual(t.num_features, 3)

    def test_by_name(self):
        t = _sample_tree()
        self.assertEqual(t.by_name("Sketch1").ftype, "ProfileFeature")
        with self.assertRaises(KeyError):
            t.by_name("nope")


class TestRoundTrip(unittest.TestCase):
    def test_text_round_trip(self):
        t = _sample_tree()
        text = t.to_text()
        back = FeatureTree.from_text(text)
        self.assertEqual(back.to_text(), text)

    def test_round_trip_preserves_structure(self):
        t = _sample_tree()
        back = FeatureTree.from_text(t.to_text())
        self.assertEqual([n.name for n in back.nodes],
                         [n.name for n in t.nodes])
        self.assertEqual(back.by_name("Extrude1").parent, "Sketch1")
        self.assertEqual(back.by_name("Extrude1").depends_on, ("Sketch1",))
        self.assertEqual(back.by_name("Sketch1").params,
                         (("SketchPlane", "Top View"), ("Radius", "994.0")))

    def test_header_present(self):
        text = _sample_tree().to_text()
        self.assertTrue(text.startswith("Feature Tree:\n"))
        self.assertIn("Extrude1 (Extrusion)", text)

    def test_from_text_rejects_missing_header(self):
        with self.assertRaises(ValueError):
            FeatureTree.from_text("no header here\n")

    def test_deterministic_serialization(self):
        t = _sample_tree()
        self.assertEqual(t.to_text(), t.to_text())


class TestHelpers(unittest.TestCase):
    def test_is_supported(self):
        self.assertTrue(is_supported_feature("Fillet"))
        self.assertFalse(is_supported_feature("Weldment"))


if __name__ == "__main__":
    unittest.main()
