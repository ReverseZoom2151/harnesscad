import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harnesscad.domain.geometry.views.lasdiff_patch_stitch import (
    provenance,
    rect_patches,
    region_patches,
    stitch,
    stitch_rect,
)


def make_grid(n, tag):
    return {(r, c): f"{tag}_{r}_{c}" for r in range(n) for c in range(n)}


class TestPatchStitch(unittest.TestCase):
    def test_region_top_bottom(self):
        top = region_patches(4, "top")
        bottom = region_patches(4, "bottom")
        self.assertEqual(top, {(r, c) for r in (0, 1) for c in range(4)})
        self.assertEqual(bottom, {(r, c) for r in (2, 3) for c in range(4)})
        self.assertEqual(top | bottom, {(r, c) for r in range(4) for c in range(4)})
        self.assertEqual(top & bottom, set())

    def test_region_left_right(self):
        left = region_patches(4, "left")
        right = region_patches(4, "right")
        self.assertTrue(all(c < 2 for _, c in left))
        self.assertTrue(all(c >= 2 for _, c in right))

    def test_region_bad(self):
        with self.assertRaises(ValueError):
            region_patches(4, "middle")
        with self.assertRaises(ValueError):
            region_patches(0, "top")

    def test_stitch_top_from_other(self):
        base = make_grid(4, "A")
        other = make_grid(4, "B")
        out = stitch(base, other, 4, "bottom")
        # bottom half from B, top half from A (mimics Fig. 16 top+bottom stitch)
        self.assertEqual(out[(0, 0)], "A_0_0")
        self.assertEqual(out[(3, 0)], "B_3_0")

    def test_stitch_left_right(self):
        base = make_grid(4, "car")
        other = make_grid(4, "plane")
        out = stitch(base, other, 4, "right")
        self.assertEqual(out[(0, 0)], "car_0_0")
        self.assertEqual(out[(0, 3)], "plane_0_3")

    def test_stitch_incomplete_grid(self):
        base = make_grid(4, "A")
        del base[(0, 0)]
        with self.assertRaises(ValueError):
            stitch(base, make_grid(4, "B"), 4, "top")

    def test_rect_patches(self):
        self.assertEqual(rect_patches((0, 0, 2, 2)),
                         {(0, 0), (0, 1), (1, 0), (1, 1)})
        with self.assertRaises(ValueError):
            rect_patches((2, 2, 2, 2))

    def test_stitch_rect(self):
        base = make_grid(4, "A")
        other = make_grid(4, "B")
        out = stitch_rect(base, other, 4, (1, 1, 3, 3))
        self.assertEqual(out[(1, 1)], "B_1_1")
        self.assertEqual(out[(0, 0)], "A_0_0")

    def test_stitch_rect_out_of_bounds(self):
        base = make_grid(4, "A")
        other = make_grid(4, "B")
        with self.assertRaises(ValueError):
            stitch_rect(base, other, 4, (0, 0, 5, 5))

    def test_provenance(self):
        base = make_grid(4, "A")
        other = make_grid(4, "B")
        out = stitch(base, other, 4, "bottom")
        prov = provenance(base, out)
        self.assertEqual(prov[(0, 0)], "base")
        self.assertEqual(prov[(3, 3)], "other")
        self.assertEqual(sum(1 for v in prov.values() if v == "other"), 8)

    def test_odd_grid_split(self):
        # n=3: half=1, top is row 0 only, bottom rows 1-2
        top = region_patches(3, "top")
        self.assertEqual(top, {(0, 0), (0, 1), (0, 2)})


if __name__ == "__main__":
    unittest.main()
