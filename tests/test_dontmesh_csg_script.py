"""Tests for programs.dontmesh_csg_script."""

import unittest

from geometry.dontmesh_halfspace_csg import (
    CSGModel,
    Cylinder,
    HalfSpace,
    Plane,
    axis_box_cell,
    axis_cylinder_cell,
)
from programs.dontmesh_csg_script import (
    ScriptSyntaxError,
    all_splits,
    build_training_pair,
    build_training_pairs,
    cells_of,
    collect_surfaces,
    correct_syntax,
    input_surfaces,
    parameter_signature,
    parse,
    same_cell_count,
    same_structure,
    same_structure_and_params,
    serialize,
    split_sequence,
    structural_signature,
)


def _two_box_model():
    return CSGModel((
        axis_box_cell((0, 0, 0), (1, 1, 1)),
        axis_box_cell((1, 0, 0), (2, 1, 1)),
    ))


class TestSerializeParse(unittest.TestCase):
    def test_roundtrip_structure(self):
        m = _two_box_model()
        script = serialize(m)
        parsed = parse(script)
        self.assertTrue(same_structure(m, parsed.model))
        self.assertTrue(same_structure_and_params(m, parsed.model))

    def test_cylinder_roundtrip(self):
        m = CSGModel((axis_cylinder_cell("z", 0.5, 0.5, 1.0, 0.0, 3.0),))
        parsed = parse(serialize(m))
        self.assertTrue(same_structure_and_params(m, parsed.model))
        kinds = {s.kind() for s in collect_surfaces(parsed.model)}
        self.assertIn("cylinder", kinds)
        self.assertIn("plane", kinds)

    def test_input_surface_line(self):
        m = _two_box_model()
        script = serialize(m, ["s0", "s1"])
        self.assertIn("# input_surfaces: s0 s1", script)
        parsed = parse(script)
        self.assertEqual(parsed.input_surfaces, ["s0", "s1"])

    def test_surface_reuse_dedup(self):
        # Two boxes sharing the plane x=1 -> shared surface collected once.
        m = _two_box_model()
        surfaces = collect_surfaces(m)
        # box A has planes x=0,x=1,y=0,y=1,z=0,z=1 ; box B x=1,x=2,... ; x=1 shared
        names_x1 = [s for s in surfaces if isinstance(s, Plane) and s.a == 1 and s.d == 1]
        self.assertEqual(len(names_x1), 1)

    def test_membership_preserved(self):
        m = _two_box_model()
        parsed = parse(serialize(m))
        for p in [(0.5, 0.5, 0.5), (1.5, 0.5, 0.5), (3, 3, 3)]:
            self.assertEqual(m.contains(p), parsed.model.contains(p))


class TestSyntaxMetric(unittest.TestCase):
    def test_correct_syntax_true(self):
        self.assertTrue(correct_syntax(serialize(_two_box_model())))

    def test_correct_syntax_false(self):
        self.assertFalse(correct_syntax("cell0 = [~s0]\n"))
        self.assertFalse(correct_syntax("s0 = plane(1, 2, 3)\n"))

    def test_undefined_surface_raises(self):
        with self.assertRaises(ScriptSyntaxError):
            parse("cell0 = [+s9]\n")

    def test_bad_cylinder_axis(self):
        self.assertFalse(correct_syntax("s0 = cylinder(w, 0, 0, 1)\ncell0 = [-s0]\n"))


class TestSplitPipeline(unittest.TestCase):
    def test_split_sequence(self):
        cells = _two_box_model().cells
        inp, out = split_sequence(cells, 1)
        self.assertEqual(len(inp), 1)
        self.assertEqual(len(out), 1)

    def test_split_bounds(self):
        cells = _two_box_model().cells
        with self.assertRaises(ValueError):
            split_sequence(cells, 0)
        with self.assertRaises(ValueError):
            split_sequence(cells, 2)

    def test_all_splits_count(self):
        cells = CSGModel((
            axis_box_cell((0, 0, 0), (1, 1, 1)),
            axis_box_cell((1, 0, 0), (2, 1, 1)),
            axis_box_cell((2, 0, 0), (3, 1, 1)),
        )).cells
        # 3 cells -> 2 valid cut points.
        self.assertEqual(len(all_splits(cells)), 2)

    def test_input_surfaces_reused(self):
        # Two boxes share the plane x=1; splitting after box 0 makes it reused.
        cells = _two_box_model().cells
        inp, out = split_sequence(cells, 1)
        reused = input_surfaces(inp, out)
        # The shared plane x=1 appears in both input and output.
        self.assertTrue(any(isinstance(s, Plane) and s.a == 1 and s.d == 1 for s in reused))

    def test_input_surfaces_none_when_disjoint(self):
        a = axis_box_cell((0, 0, 0), (1, 1, 1))
        b = axis_box_cell((5, 5, 5), (6, 6, 6))  # no shared plane on any axis
        reused = input_surfaces([a], [b])
        self.assertEqual(reused, [])


class TestTrainingPairs(unittest.TestCase):
    def test_build_training_pair(self):
        cells = _two_box_model().cells
        pair = build_training_pair(cells, 1)
        self.assertTrue(correct_syntax(pair.input_script))
        self.assertTrue(correct_syntax(pair.output_script))
        self.assertEqual(len(pair.input_cells), 1)
        self.assertEqual(len(pair.output_cells), 1)

    def test_build_training_pairs_augmentation(self):
        m = CSGModel((
            axis_box_cell((0, 0, 0), (1, 1, 1)),
            axis_box_cell((1, 0, 0), (2, 1, 1)),
            axis_box_cell((2, 0, 0), (3, 1, 1)),
        ))
        orderings = [
            cells_of(m, (0, 1, 2)),
            cells_of(m, (2, 1, 0)),
        ]
        pairs = build_training_pairs(orderings)
        # 2 orderings x 2 cuts each = 4 pairs.
        self.assertEqual(len(pairs), 4)
        for pr in pairs:
            self.assertTrue(correct_syntax(pr.input_script))

    def test_determinism(self):
        cells = _two_box_model().cells
        self.assertEqual(
            build_training_pair(cells, 1).input_script,
            build_training_pair(cells, 1).input_script,
        )


class TestStructuralMetrics(unittest.TestCase):
    def test_same_structure_diff_params(self):
        a = CSGModel((axis_box_cell((0, 0, 0), (1, 1, 1)),))
        b = CSGModel((axis_box_cell((0, 0, 0), (5, 5, 5)),))  # same structure, diff params
        self.assertTrue(same_structure(a, b))
        self.assertFalse(same_structure_and_params(a, b))

    def test_same_structure_and_params(self):
        a = CSGModel((axis_box_cell((0, 0, 0), (2, 2, 2)),))
        b = CSGModel((axis_box_cell((0, 0, 0), (2, 2, 2)),))
        self.assertTrue(same_structure_and_params(a, b))

    def test_different_structure(self):
        box = CSGModel((axis_box_cell((0, 0, 0), (1, 1, 1)),))
        cyl = CSGModel((axis_cylinder_cell("z", 0, 0, 1, 0, 1),))
        self.assertFalse(same_structure(box, cyl))

    def test_same_cell_count(self):
        a = CSGModel((axis_box_cell((0, 0, 0), (1, 1, 1)),))
        b = CSGModel((axis_box_cell((0, 0, 0), (2, 2, 2)),))
        self.assertTrue(same_cell_count(a, b))
        c = _two_box_model()
        self.assertFalse(same_cell_count(a, c))

    def test_signatures_hashable(self):
        m = _two_box_model()
        self.assertIsInstance(hash(structural_signature(m)), int)
        self.assertIsInstance(hash(parameter_signature(m)), int)


if __name__ == "__main__":
    unittest.main()
