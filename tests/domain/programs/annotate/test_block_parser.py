import unittest

from harnesscad.domain.programs.annotate.block_parser import (
    parse,
    identify_blocks,
    commentable_blocks,
    is_single_solid,
    annotate,
    Node,
)


class TestParse(unittest.TestCase):
    def test_flat_primitives(self):
        src = "cube([1,2,3]); sphere(2);"
        nodes = parse(src)
        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0].head, "cube")
        self.assertEqual(nodes[0].kind, "primitive")
        self.assertEqual(nodes[1].head, "sphere")

    def test_transform_chain(self):
        src = "translate([1,0,0]) rotate([0,0,90]) cube(1);"
        nodes = parse(src)
        self.assertEqual(len(nodes), 1)
        t = nodes[0]
        self.assertEqual(t.head, "translate")
        self.assertEqual(t.children[0].head, "rotate")
        self.assertEqual(t.children[0].children[0].head, "cube")

    def test_braced_body(self):
        src = "union() { cube(1); sphere(1); }"
        nodes = parse(src)
        self.assertEqual(nodes[0].head, "union")
        self.assertEqual(len(nodes[0].children), 2)

    def test_assignment_and_module(self):
        src = "w = 10; module leg() { cube([1,1,w]); } leg();"
        nodes = parse(src)
        kinds = [n.kind for n in nodes]
        self.assertIn("assignment", kinds)
        self.assertIn("module", kinds)

    def test_comments_stripped(self):
        src = "// a comment\ncube(1); /* block */ sphere(1);"
        nodes = parse(src)
        self.assertEqual(len(nodes), 2)


class TestSingleSolid(unittest.TestCase):
    def test_primitive_is_single_solid(self):
        n = parse("cube(1);")[0]
        self.assertTrue(is_single_solid(n))

    def test_difference_of_primitives_is_single_solid(self):
        n = parse("difference() { cube(2); sphere(1); }")[0]
        self.assertTrue(is_single_solid(n))

    def test_union_is_not_single_solid(self):
        n = parse("union() { cube(1); sphere(1); }")[0]
        self.assertFalse(is_single_solid(n))

    def test_transform_over_union_not_single_solid(self):
        n = parse("translate([1,0,0]) union() { cube(1); sphere(1); }")[0]
        self.assertFalse(is_single_solid(n))


class TestIdentifyBlocks(unittest.TestCase):
    def test_irreducible_leaves(self):
        # A union of two primitives -> two irreducible leaves + the union.
        src = "union() { cube(1); sphere(1); }"
        blocks = commentable_blocks(src)
        leaves = [b for b in blocks if b.irreducible]
        self.assertEqual(len(leaves), 2)
        # union is commentable but not irreducible
        union = [b for b in blocks if b.head == "union"][0]
        self.assertTrue(union.commentable)
        self.assertFalse(union.irreducible)

    def test_difference_is_single_irreducible_block(self):
        src = "difference() { cube(2); translate([0,0,1]) cylinder(1); }"
        blocks = commentable_blocks(src)
        # The whole difference is one irreducible block; its primitive children
        # are inside a single-solid subtree so they are not separate leaves.
        irr = [b for b in blocks if b.irreducible]
        self.assertEqual(len(irr), 1)
        self.assertEqual(irr[0].head, "difference")

    def test_block_ids_sequential(self):
        src = "union() { cube(1); sphere(1); }"
        blocks = commentable_blocks(src)
        ids = [b.block_id for b in blocks]
        self.assertEqual(ids, list(range(len(blocks))))

    def test_multi_level(self):
        # nested unions -> commentable blocks at several levels
        src = "union() { union() { cube(1); sphere(1); } cylinder(1); }"
        blocks = commentable_blocks(src)
        levels = [b.head for b in blocks]
        self.assertEqual(levels.count("union"), 2)
        self.assertEqual(sum(1 for b in blocks if b.irreducible), 3)

    def test_assignments_not_commentable(self):
        src = "w = 5; cube([w,w,w]);"
        blocks = commentable_blocks(src)
        self.assertTrue(all(b.kind != "assignment" for b in blocks))
        self.assertEqual(len(blocks), 1)


class TestAnnotate(unittest.TestCase):
    def test_tbc_default(self):
        out = annotate("union() { cube(1); sphere(1); }")
        self.assertIn("TBC", out)

    def test_labels_applied(self):
        src = "union() { cube(1); sphere(1); }"
        blocks = commentable_blocks(src)
        labels = {b.block_id: "seat" for b in blocks}
        out = annotate(src, labels=labels)
        self.assertIn("seat", out)
        self.assertNotIn("TBC", out)

    def test_deterministic(self):
        src = "difference() { cube(2); sphere(1); } cylinder(3);"
        self.assertEqual(annotate(src), annotate(src))


if __name__ == "__main__":
    unittest.main()
