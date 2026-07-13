"""Tests for generation.worldcraft_customization."""

import math
import unittest

from harnesscad.domain.reconstruction.scene.layout_spec import ObjectPlacement, Pose
from harnesscad.agents.generation.scene_customization import (
    Color,
    CustomizationSchema,
    MaterialSpec,
    ObjectCustomization,
    apply_customization,
    merge_customizations,
)


class TestColor(unittest.TestCase):
    def test_channel_range(self):
        with self.assertRaises(ValueError):
            Color(1.2, 0.0, 0.0)

    def test_from_hex_rgb(self):
        c = Color.from_hex("#ff8000")
        self.assertAlmostEqual(c.r, 1.0)
        self.assertAlmostEqual(c.g, 128 / 255.0)
        self.assertAlmostEqual(c.b, 0.0)
        self.assertEqual(c.a, 1.0)

    def test_from_hex_rgba_and_no_hash(self):
        c = Color.from_hex("00ff0080")
        self.assertAlmostEqual(c.g, 1.0)
        self.assertAlmostEqual(c.a, 128 / 255.0)

    def test_from_hex_bad_length(self):
        with self.assertRaises(ValueError):
            Color.from_hex("#fff")


class TestMaterial(unittest.TestCase):
    def test_range_validation(self):
        with self.assertRaises(ValueError):
            MaterialSpec(metallic=1.5)
        with self.assertRaises(ValueError):
            MaterialSpec(emission=-1.0)

    def test_to_dict(self):
        m = MaterialSpec(base_color=Color(0.1, 0.2, 0.3), metallic=0.4, roughness=0.6)
        d = m.to_dict()
        self.assertEqual(d["metallic"], 0.4)
        self.assertEqual(d["base_color"], [0.1, 0.2, 0.3, 1.0])


class TestSchema(unittest.TestCase):
    def test_unknown_attribute_rejected(self):
        s = CustomizationSchema()
        self.assertIn("unknown attribute 'bogus'", s.validate({"bogus": 1}))

    def test_scale_range(self):
        s = CustomizationSchema(min_scale=0.5, max_scale=2.0)
        self.assertTrue(s.is_valid({"scale": 1.0}))
        self.assertFalse(s.is_valid({"scale": 5.0}))
        self.assertFalse(s.is_valid({"scale": "big"}))

    def test_bool_not_number(self):
        s = CustomizationSchema()
        self.assertFalse(s.is_valid({"scale": True}))

    def test_type_checks(self):
        s = CustomizationSchema()
        self.assertFalse(s.is_valid({"material": "shiny"}))
        self.assertFalse(s.is_valid({"color": "red"}))
        self.assertFalse(s.is_valid({"tags": [1, 2]}))
        self.assertTrue(s.is_valid({"tags": ["a", "b"], "label": "chair"}))

    def test_bad_schema_bounds(self):
        with self.assertRaises(ValueError):
            CustomizationSchema(min_scale=2.0, max_scale=1.0)


class TestApply(unittest.TestCase):
    def _obj(self):
        return ObjectPlacement("a", "chair", (0.5, 0.5, 1.0),
                               Pose(position=(1.0, 2.0, 3.0), scale=(2.0, 2.0, 2.0)),
                               attributes={"tags": ["old"]})

    def test_scale_multiplies_existing(self):
        out = apply_customization(self._obj(), ObjectCustomization(scale=1.5))
        self.assertEqual(out.pose.scale, (3.0, 3.0, 3.0))

    def test_yaw_absolute(self):
        out = apply_customization(self._obj(), ObjectCustomization(yaw=math.pi / 2.0))
        self.assertAlmostEqual(out.pose.yaw, math.pi / 2.0)

    def test_material_and_color_written(self):
        m = MaterialSpec(metallic=0.9)
        out = apply_customization(self._obj(),
                                  ObjectCustomization(material=m, color=Color(1.0, 0.0, 0.0),
                                                      label="red chair", tags=("new",)))
        self.assertIs(out.attributes["material"], m)
        self.assertEqual(out.attributes["label"], "red chair")
        self.assertEqual(out.attributes["tags"], ["old", "new"])

    def test_input_not_mutated(self):
        obj = self._obj()
        before = obj.to_dict()
        apply_customization(obj, ObjectCustomization(scale=2.0, color=Color(0.0, 0.0, 1.0)))
        self.assertEqual(obj.to_dict(), before)

    def test_invalid_scale_raises(self):
        with self.assertRaises(ValueError):
            apply_customization(self._obj(), ObjectCustomization(scale=1000.0),
                                schema=CustomizationSchema(max_scale=10.0))


class TestMerge(unittest.TestCase):
    def test_later_overrides_earlier(self):
        merged = merge_customizations([
            ObjectCustomization(scale=1.0, label="a"),
            ObjectCustomization(scale=2.0),
            ObjectCustomization(label="b"),
        ])
        self.assertEqual(merged.scale, 2.0)
        self.assertEqual(merged.label, "b")

    def test_tags_accumulate_dedup(self):
        merged = merge_customizations([
            ObjectCustomization(tags=("x", "y")),
            ObjectCustomization(tags=("y", "z")),
        ])
        self.assertEqual(merged.tags, ("x", "y", "z"))

    def test_empty(self):
        merged = merge_customizations([])
        self.assertIsNone(merged.scale)
        self.assertEqual(merged.tags, ())


if __name__ == "__main__":
    unittest.main()
