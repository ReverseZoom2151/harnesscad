"""Tests for the ShowUI-derived union-find region grouper."""

import unittest

from harnesscad.io.cua.region_group import (
    RegionMap, UnionFind, build_regions, largest_regions, patch_distance,
    patchify,
)

BLACK = (0.0, 0.0, 0.0)
WHITE = (255.0, 255.0, 255.0)
GREY = (128.0, 128.0, 128.0)


class TestUnionFind(unittest.TestCase):
    def test_union_and_find_connect(self):
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        self.assertEqual(uf.find(0), uf.find(2))
        self.assertNotEqual(uf.find(0), uf.find(3))

    def test_deterministic_root_regardless_of_order(self):
        a = UnionFind(4)
        a.union(0, 1)
        a.union(2, 3)
        a.union(1, 2)
        b = UnionFind(4)
        b.union(2, 3)
        b.union(1, 2)
        b.union(0, 1)
        self.assertEqual({a.find(i) for i in range(4)}, {a.find(0)})
        self.assertEqual({b.find(i) for i in range(4)}, {b.find(0)})


class TestPatchDistance(unittest.TestCase):
    def test_identical_is_zero(self):
        self.assertEqual(patch_distance(GREY, GREY), 0.0)

    def test_black_white_is_large(self):
        self.assertAlmostEqual(patch_distance(BLACK, WHITE), (3 * 255 ** 2) ** 0.5)


class TestBuildRegions(unittest.TestCase):
    def test_uniform_grid_is_one_region(self):
        grid = [[GREY] * 4 for _ in range(4)]
        rmap = build_regions(grid, threshold=12.0)
        self.assertEqual(rmap.count, 1)
        self.assertAlmostEqual(rmap.compression(), 1 / 16)

    def test_two_blocks_split_at_a_hard_edge(self):
        # Left half black, right half white: two regions, no matter the threshold
        # below the black/white distance.
        grid = [[BLACK, BLACK, WHITE, WHITE] for _ in range(3)]
        rmap = build_regions(grid, threshold=50.0)
        self.assertEqual(rmap.count, 2)
        # Every patch on the left shares a label distinct from the right.
        self.assertEqual(rmap.label_at(0, 0), rmap.label_at(2, 1))
        self.assertNotEqual(rmap.label_at(0, 0), rmap.label_at(0, 3))

    def test_threshold_controls_granularity(self):
        # A smooth gradient UNIQUE PER PATCH (varies in both row and col, so no two
        # neighbours are identical): a big threshold merges it all into one region,
        # a tiny one keeps every patch separate. This is ShowUI's uigraph_diff knob.
        # NB: the gradient must vary along BOTH axes -- a column-only gradient leaves
        # vertically-adjacent patches identical, which correctly merge (that is not
        # a grouping bug, it is the union-find working).
        grid = [[(float((r * 6 + c) * 20), 0.0, 0.0) for c in range(6)]
                for r in range(2)]
        coarse = build_regions(grid, threshold=1000.0)
        fine = build_regions(grid, threshold=1.0)
        self.assertEqual(coarse.count, 1)
        self.assertEqual(fine.count, 12)

    def test_labels_are_compact(self):
        grid = [[BLACK, WHITE], [WHITE, BLACK]]
        rmap = build_regions(grid, threshold=10.0)
        labels = {rmap.label_at(r, c) for r in range(2) for c in range(2)}
        self.assertEqual(labels, set(range(rmap.count)))

    def test_region_mean_and_bbox(self):
        grid = [[BLACK, BLACK, WHITE, WHITE] for _ in range(2)]
        rmap = build_regions(grid, threshold=50.0)
        black = rmap.region_at(0, 0)
        self.assertEqual(black.mean, BLACK)
        self.assertEqual(black.bbox, (0, 0, 1, 1))
        self.assertEqual(black.area, 4)

    def test_ragged_grid_rejected(self):
        with self.assertRaises(ValueError):
            build_regions([[BLACK, WHITE], [BLACK]])

    def test_empty_grid(self):
        rmap = build_regions([])
        self.assertEqual(rmap.count, 0)

    def test_roundtrip_dict(self):
        grid = [[BLACK, WHITE], [WHITE, BLACK]]
        d = build_regions(grid, threshold=10.0).to_dict()
        self.assertIn("regions", d)
        self.assertEqual(d["count"], len(d["regions"]))


class TestPatchify(unittest.TestCase):
    def test_block_mean(self):
        # 4x4 of a single colour, patch 2 -> 2x2 grid of that colour.
        pixels = [[GREY] * 4 for _ in range(4)]
        grid = patchify(pixels, patch=2)
        self.assertEqual(len(grid), 2)
        self.assertEqual(len(grid[0]), 2)
        self.assertEqual(grid[0][0], GREY)

    def test_partial_block_averaged(self):
        pixels = [[BLACK, BLACK, WHITE]]  # 1x3, patch 2 -> block0=black, block1=white
        grid = patchify(pixels, patch=2)
        self.assertEqual(grid[0][0], BLACK)
        self.assertEqual(grid[0][1], WHITE)

    def test_patchify_then_group(self):
        # A viewport-like frame: white background with a grey square blob.
        pixels = [[WHITE] * 8 for _ in range(8)]
        for r in range(2, 6):
            for c in range(2, 6):
                pixels[r][c] = GREY
        grid = patchify(pixels, patch=2)
        rmap = build_regions(grid, threshold=20.0)
        self.assertGreaterEqual(rmap.count, 2)  # background + blob at least


class TestLargestRegions(unittest.TestCase):
    def test_biggest_first(self):
        grid = [[BLACK, BLACK, BLACK, WHITE] for _ in range(3)]
        rmap = build_regions(grid, threshold=50.0)
        top = largest_regions(rmap, k=2)
        self.assertEqual(top[0].area, 9)   # black dominates
        self.assertEqual(top[1].area, 3)


if __name__ == "__main__":
    unittest.main()
