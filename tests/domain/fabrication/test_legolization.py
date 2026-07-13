import unittest

from harnesscad.domain.fabrication.brick_library import is_library_part
from harnesscad.domain.fabrication.legolization import (
    covers_exactly,
    legolize,
    legolize_variants,
)


def _rect_layer(nx, ny, z=0):
    return {(x, y, z) for x in range(nx) for y in range(ny)}


class TestLegolize(unittest.TestCase):
    def test_single_cell(self):
        bricks = legolize({(0, 0, 0)})
        self.assertEqual(len(bricks), 1)
        self.assertEqual(bricks[0].footprint(), (1, 1))

    def test_covers_exactly_no_overlap(self):
        vox = _rect_layer(4, 4)
        bricks = legolize(vox)
        self.assertTrue(covers_exactly(bricks, vox))

    def test_all_parts_in_library(self):
        vox = _rect_layer(8, 6)
        bricks = legolize(vox)
        for b in bricks:
            self.assertTrue(is_library_part(b.h, b.w), (b.h, b.w))
        self.assertTrue(covers_exactly(bricks, vox))

    def test_prefers_larger_bricks(self):
        # A 2x4 strip should be covered by a single 2x4 (or 4x2), not 8 units.
        vox = {(x, y, 0) for x in range(2) for y in range(4)}
        bricks = legolize(vox)
        self.assertEqual(len(bricks), 1)
        self.assertEqual(bricks[0].footprint(), (2, 4))

    def test_multi_layer(self):
        vox = _rect_layer(4, 4, 0) | _rect_layer(4, 4, 1)
        bricks = legolize(vox)
        self.assertTrue(covers_exactly(bricks, vox))
        self.assertEqual({b.z for b in bricks}, {0, 1})

    def test_deterministic_default(self):
        vox = _rect_layer(6, 6)
        self.assertEqual(legolize(vox), legolize(vox))

    def test_seeded_variant_reproducible(self):
        vox = _rect_layer(6, 6)
        a = legolize(vox, seed=7)
        b = legolize(vox, seed=7)
        self.assertEqual(a, b)
        self.assertTrue(covers_exactly(a, vox))

    def test_variants_all_cover(self):
        vox = _rect_layer(5, 5)
        for layout in legolize_variants(vox, [1, 2, 3]):
            self.assertTrue(covers_exactly(layout, vox))

    def test_irregular_shape(self):
        # L-shape
        vox = {(0, 0, 0), (1, 0, 0), (0, 1, 0)}
        bricks = legolize(vox)
        self.assertTrue(covers_exactly(bricks, vox))


if __name__ == "__main__":
    unittest.main()
