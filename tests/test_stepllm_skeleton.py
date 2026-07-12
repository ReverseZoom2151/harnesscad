import unittest

from formats.stepllm_parser import Real, Ref, parse, serialize
from formats.stepllm_schema import check_attributes
from formats.stepllm_skeleton import (
    StepBuilder, default_header, detect_primitives, skeleton_from_keywords,
)


class TestBuilder(unittest.TestCase):
    def test_auto_numbering(self):
        b = StepBuilder()
        r1 = b.point(0, 0, 0)
        r2 = b.direction(0, 0, 1)
        self.assertEqual((r1, r2), (Ref(1), Ref(2)))

    def test_origin_placement_wires_refs(self):
        b = StepBuilder()
        p = b.origin_placement()
        step = b.build()
        placement = step.entities[p.id]
        self.assertEqual(placement.keyword, "AXIS2_PLACEMENT_3D")
        self.assertEqual(placement.params[1:], [Ref(1), Ref(2), Ref(3)])

    def test_add_validates_arity(self):
        b = StepBuilder()
        with self.assertRaises(ValueError):
            b.add("PLANE", "")  # missing position ref

    def test_to_text_roundtrips(self):
        b = StepBuilder()
        b.origin_placement()
        text = b.to_text()
        reparsed = parse(text)
        self.assertEqual(reparsed.order, [1, 2, 3, 4])

    def test_header_present(self):
        b = StepBuilder(name="demo")
        step = b.build()
        self.assertEqual([r.keyword for r in step.header],
                         ["FILE_DESCRIPTION", "FILE_NAME", "FILE_SCHEMA"])


class TestDefaultHeader(unittest.TestCase):
    def test_schema_recorded(self):
        header = default_header(schema="CONFIG_CONTROL_DESIGN")
        schema_rec = header[2]
        self.assertEqual(schema_rec.params[0], ["CONFIG_CONTROL_DESIGN"])


class TestDetectPrimitives(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(
            detect_primitives("A flat circular lid"), ["PLANE", "CIRCLE"])

    def test_dedup_and_order(self):
        self.assertEqual(
            detect_primitives("circle and another circle then a plane"),
            ["CIRCLE", "PLANE"])

    def test_punctuation_ignored(self):
        self.assertEqual(detect_primitives("cylinder, with holes."),
                         ["CYLINDRICAL_SURFACE"])

    def test_no_keywords(self):
        self.assertEqual(detect_primitives("an abstract widget"), [])


class TestSkeleton(unittest.TestCase):
    def test_plane_skeleton_valid(self):
        step = skeleton_from_keywords("a flat plate")
        problems = []
        for e in step.entities.values():
            problems.extend(check_attributes(e))
        self.assertEqual(problems, [])

    def test_circle_radius_applied(self):
        step = skeleton_from_keywords("a round disc", radius=2.5)
        circle = next(e for e in step.entities.values()
                      if e.keyword == "CIRCLE")
        self.assertEqual(circle.params[2], Real("2.5"))

    def test_multiple_primitives(self):
        step = skeleton_from_keywords("a cylinder capped with a plane")
        kinds = sorted(e.keyword for e in step.entities.values()
                       if e.keyword in ("CYLINDRICAL_SURFACE", "PLANE"))
        self.assertEqual(kinds, ["CYLINDRICAL_SURFACE", "PLANE"])

    def test_empty_caption_still_valid_file(self):
        step = skeleton_from_keywords("nondescript object")
        text = serialize(step)
        self.assertTrue(text.startswith("ISO-10303-21;"))
        self.assertTrue(text.rstrip().endswith("END-ISO-10303-21;"))
        self.assertEqual(len(step.entities), 0)

    def test_deterministic(self):
        a = serialize(skeleton_from_keywords("circular flat lid"))
        b = serialize(skeleton_from_keywords("circular flat lid"))
        self.assertEqual(a, b)

    def test_skeleton_parses_back(self):
        step = skeleton_from_keywords("a sphere")
        reparsed = parse(serialize(step))
        self.assertTrue(any(e.keyword == "SPHERICAL_SURFACE"
                            for e in reparsed.entities.values()))


if __name__ == "__main__":
    unittest.main()
