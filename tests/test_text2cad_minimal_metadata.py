import unittest

from harnesscad.data.dataengine.annotation.minimal_metadata import (
    MinimalMetadataError,
    generate_minimal_metadata,
    is_random_key,
)


class RandomKeyTests(unittest.TestCase):
    def test_deepcad_uuid_is_random(self):
        self.assertTrue(is_random_key("FI4bCL9y0XvsF52"))

    def test_meaningful_snake_case_not_random(self):
        self.assertFalse(is_random_key("part_1"))
        self.assertFalse(is_random_key("loop_12"))

    def test_short_word_not_random(self):
        self.assertFalse(is_random_key("type"))
        self.assertFalse(is_random_key("Sketch"))

    def test_whitelist_overrides(self):
        self.assertFalse(is_random_key("ABC123DEF456", whitelist=frozenset({"ABC123DEF456"})))


class RenamingTests(unittest.TestCase):
    def setUp(self):
        self.raw = {
            "FI4bCL9y0XvsF52": {"type": "Part"},
            "Zx99QqPlmn01AB": {"type": "Part"},
            "Loop77TokenXyz": {"type": "Loop"},
        }

    def test_random_keys_renamed_positionally(self):
        md = generate_minimal_metadata(self.raw)
        self.assertIn("part_1", md.entities)
        self.assertIn("part_2", md.entities)
        self.assertIn("loop_1", md.entities)
        self.assertNotIn("FI4bCL9y0XvsF52", md.entities)

    def test_key_map_records_original(self):
        md = generate_minimal_metadata(self.raw)
        self.assertEqual(md.key_map["FI4bCL9y0XvsF52"], "part_1")
        self.assertEqual(md.key_map["Loop77TokenXyz"], "loop_1")

    def test_deterministic(self):
        a = generate_minimal_metadata(self.raw)
        b = generate_minimal_metadata(self.raw)
        self.assertEqual(a.entities, b.entities)
        self.assertEqual(a.key_map, b.key_map)


class RedundancyTests(unittest.TestCase):
    def test_redundant_key_dropped(self):
        raw = {"Sketch01TokenAB": {"type": "Sketch", "role": "AgainstDistance"}}
        md = generate_minimal_metadata(raw)
        entity = md.entities["sketch_1"]
        self.assertNotIn("role", entity)
        self.assertIn("role", md.dropped_keys)

    def test_redundant_typed_entity_dropped(self):
        raw = {
            "GoodSketch01Ab": {"type": "Sketch"},
            "ParamXY99token": {"type": "ModelParameter", "role": "AgainstDistance"},
        }
        md = generate_minimal_metadata(raw)
        self.assertIn("sketch_1", md.entities)
        self.assertEqual(len(md.entities), 1)
        self.assertIn("ParamXY99token", md.dropped_entities)

    def test_nested_cleaning(self):
        raw = {
            "PartToken001Ab": {
                "type": "Part",
                "children": {
                    "LoopToken009Zz": {"type": "Loop", "uuid": "xxxx"},
                },
            },
        }
        md = generate_minimal_metadata(raw)
        part = md.entities["part_1"]
        self.assertIn("loop_1", part["children"])
        self.assertNotIn("uuid", part["children"]["loop_1"])

    def test_meaningful_keys_preserved(self):
        raw = {"PartToken001Ab": {"type": "Part", "name": "base", "sketch_scale": 0.7}}
        md = generate_minimal_metadata(raw)
        self.assertEqual(md.entities["part_1"]["name"], "base")
        self.assertEqual(md.entities["part_1"]["sketch_scale"], 0.7)


class MiscTests(unittest.TestCase):
    def test_shape_descriptions_attached(self):
        raw = {"PartToken001Ab": {"type": "Part"}}
        md = generate_minimal_metadata(raw, shape_descriptions={"final": "a ring"})
        self.assertEqual(md.shape_descriptions["final"], "a ring")
        self.assertIn("shape_descriptions", md.as_dict())

    def test_non_mapping_raises(self):
        with self.assertRaises(MinimalMetadataError):
            generate_minimal_metadata([1, 2, 3])


if __name__ == "__main__":
    unittest.main()
