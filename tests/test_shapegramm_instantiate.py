"""Tests for procedural.shapegramm_instantiate (seeded reproducible model)."""

import unittest

from procedural.shapegramm_instantiate import (
    MassiveModel, Instance, generate_view, full_object_count,
)
from procedural.shapegramm_scope import make_aabb_frustum


class CellInstancingTest(unittest.TestCase):
    def test_reproducible_same_cell(self):
        m = MassiveModel(dims=(3, 3, 3), base_seed=42)
        a = m.cell_instances((1, 2, 0))
        b = m.cell_instances((1, 2, 0))
        self.assertEqual(a, b)

    def test_different_cells_differ(self):
        m = MassiveModel(dims=(3, 3, 3), base_seed=42)
        self.assertNotEqual(m.cell_instances((0, 0, 0)), m.cell_instances((1, 0, 0)))

    def test_seed_changes_output(self):
        a = MassiveModel(dims=(2, 2, 2), base_seed=1).cell_instances((0, 0, 0))
        b = MassiveModel(dims=(2, 2, 2), base_seed=2).cell_instances((0, 0, 0))
        self.assertNotEqual(a, b)

    def test_instances_lie_in_cell_box(self):
        m = MassiveModel(dims=(2, 2, 2), cell_size=10.0, base_seed=7)
        lo, hi = m.cell_box((1, 0, 1))
        for inst in m.cell_instances((1, 0, 1)):
            for d in range(3):
                self.assertGreaterEqual(inst.translation[d], lo[d])
                self.assertLessEqual(inst.translation[d], hi[d])

    def test_lod_reduces_object_count_and_swaps_geometry(self):
        m = MassiveModel(dims=(2, 2, 2), max_objects=8, base_seed=3)
        full = m.cell_instances((0, 0, 0), lod=0)
        coarse = m.cell_instances((0, 0, 0), lod=2)
        self.assertGreater(len(full), len(coarse))
        self.assertEqual(full[0].geometry_id, "mesh_full")
        self.assertEqual(coarse[0].geometry_id, "mesh_coarsest")

    def test_coarse_lod_never_empty(self):
        m = MassiveModel(dims=(1, 1, 1), max_objects=4, base_seed=0)
        self.assertGreaterEqual(len(m.cell_instances((0, 0, 0), lod=10)), 1)


class GenerateViewTest(unittest.TestCase):
    def setUp(self):
        # 4x4x1 grid of 10-unit cells spanning [0,40] x [0,40] x [0,10]
        self.model = MassiveModel(dims=(4, 4, 1), cell_size=10.0, base_seed=5,
                                  max_objects=6)
        self.thresholds = [200, 40]

    def test_culls_cells_outside_frustum(self):
        # frustum only covers the first cell region
        planes = make_aabb_frustum((0, 0, 0), (10, 10, 10))
        batches, stats = generate_view(self.model, planes, (5, 5, 100), 500.0,
                                       self.thresholds)
        self.assertLess(stats["cells_visible"], stats["cells_total"])
        self.assertEqual(stats["cells_visible"] + stats["cells_culled"],
                         stats["cells_total"])
        self.assertGreater(stats["cells_culled"], 0)

    def test_wide_frustum_sees_more_than_narrow(self):
        narrow = make_aabb_frustum((0, 0, 0), (10, 10, 10))
        wide = make_aabb_frustum((0, 0, 0), (40, 40, 10))
        _, s_narrow = generate_view(self.model, narrow, (5, 5, 100), 500.0,
                                    self.thresholds)
        _, s_wide = generate_view(self.model, wide, (20, 20, 100), 500.0,
                                  self.thresholds)
        self.assertGreater(s_wide["cells_visible"], s_narrow["cells_visible"])

    def test_deterministic_view(self):
        planes = make_aabb_frustum((0, 0, 0), (40, 40, 10))
        a = generate_view(self.model, planes, (20, 20, 80), 500.0, self.thresholds)
        b = generate_view(self.model, planes, (20, 20, 80), 500.0, self.thresholds)
        self.assertEqual(a, b)

    def test_batches_keyed_by_geometry_id(self):
        planes = make_aabb_frustum((0, 0, 0), (40, 40, 10))
        batches, stats = generate_view(self.model, planes, (20, 20, 80), 500.0,
                                       self.thresholds)
        total = sum(len(v) for v in batches.values())
        self.assertEqual(total, stats["objects"])
        for gid, insts in batches.items():
            for inst in insts:
                self.assertIsInstance(inst, Instance)
                self.assertEqual(inst.geometry_id, gid)

    def test_far_camera_uses_coarser_lod(self):
        planes = make_aabb_frustum((0, 0, 0), (40, 40, 1000))
        _, near = generate_view(self.model, planes, (20, 20, 60), 500.0,
                                 self.thresholds)
        _, far = generate_view(self.model, planes, (20, 20, 900), 500.0,
                                self.thresholds)
        # farther camera pushes cells into higher LOD levels -> fewer objects
        self.assertLessEqual(far["objects"], near["objects"])

    def test_full_object_count(self):
        self.assertEqual(full_object_count(self.model), 16 * 6)


if __name__ == "__main__":
    unittest.main()
