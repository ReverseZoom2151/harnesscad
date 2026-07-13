import unittest

from harnesscad.domain.programs.emit.cadtalk_primitive_program import (
    Cuboid,
    Ellipsoid,
    synthesize,
    ProgramEntry,
)
from harnesscad.domain.programs.annotate import cadtalk_parser as parser


class TestPrimitives(unittest.TestCase):
    def test_cuboid_scad(self):
        c = Cuboid(label="body", size=(2, 3, 4), center=(1, 0, 0))
        s = c.to_scad()
        self.assertIn("cube([2, 3, 4], center=true)", s)
        self.assertIn("translate([1, 0, 0])", s)

    def test_cuboid_no_transform_when_origin(self):
        c = Cuboid(label="body", size=(1, 1, 1))
        s = c.to_scad()
        self.assertNotIn("translate", s)
        self.assertNotIn("rotate", s)

    def test_ellipsoid_scad(self):
        e = Ellipsoid(label="head", semi_axes=(1, 2, 1), center=(0, 0, 5))
        s = e.to_scad()
        self.assertIn("scale([1, 2, 1])", s)
        self.assertIn("sphere(1", s)
        self.assertIn("translate([0, 0, 5])", s)


class TestSynthesize(unittest.TestCase):
    def test_one_block_per_primitive(self):
        prims = [
            Cuboid(label="body", size=(1, 1, 1), center=(0, 0, 0)),
            Cuboid(label="wing", size=(2, 1, 1), center=(2, 0, 0)),
        ]
        entry = synthesize(prims, group_consecutive=False)
        self.assertEqual(entry.num_blocks, 2)
        self.assertEqual(entry.block_labels, {0: "body", 1: "wing"})

    def test_group_consecutive_same_label(self):
        prims = [
            Cuboid(label="leg", size=(1, 1, 1), center=(0, 0, 0)),
            Cuboid(label="leg", size=(1, 1, 1), center=(2, 0, 0)),
            Cuboid(label="seat", size=(3, 3, 1), center=(0, 0, 2)),
        ]
        entry = synthesize(prims, group_consecutive=True)
        # two leg prims collapse into one block
        self.assertEqual(entry.num_blocks, 2)
        self.assertEqual(entry.block_labels, {0: "leg", 1: "seat"})
        self.assertEqual(entry.block_primitives[0], [0, 1])
        self.assertIn("union()", entry.source)

    def test_non_consecutive_same_label_not_grouped(self):
        prims = [
            Cuboid(label="leg", size=(1, 1, 1), center=(0, 0, 0)),
            Cuboid(label="seat", size=(3, 3, 1), center=(0, 0, 2)),
            Cuboid(label="leg", size=(1, 1, 1), center=(2, 0, 0)),
        ]
        entry = synthesize(prims, group_consecutive=True)
        self.assertEqual(entry.num_blocks, 3)

    def test_source_parses_to_matching_blocks(self):
        prims = [
            Ellipsoid(label="body", semi_axes=(2, 1, 1)),
            Ellipsoid(label="wing", semi_axes=(3, 1, 0.2), center=(0, 2, 0)),
        ]
        entry = synthesize(prims, group_consecutive=False)
        blocks = parser.commentable_blocks(entry.source)
        # each ellipsoid is a single-solid irreducible leaf
        irr = [b for b in blocks if b.irreducible]
        self.assertEqual(len(irr), 2)

    def test_category_comment(self):
        prims = [Cuboid(label="body", size=(1, 1, 1))]
        entry = synthesize(prims, category="Airplane")
        self.assertIn("category: Airplane", entry.source)

    def test_deterministic(self):
        prims = [Cuboid(label="body", size=(1, 2, 3), center=(1, 1, 1))]
        self.assertEqual(synthesize(prims).source, synthesize(prims).source)


if __name__ == "__main__":
    unittest.main()
