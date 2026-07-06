import unittest

from reconstruction.histcad_sequence import (
    Line, Circle, Arc, SketchPlane, Sketch, Extrusion, Feature, ModelingSequence,
)
from reconstruction.histcad_replay import (
    reconstruct_loops, hierarchical_loops, replay_validity,
)


def _square(dx=0.0, dy=0.0, s=1.0):
    return [
        Line(dx, dy, dx + s, dy),
        Line(dx + s, dy, dx + s, dy + s),
        Line(dx + s, dy + s, dx, dy + s),
        Line(dx, dy + s, dx, dy),
    ]


class TestReconstructLoops(unittest.TestCase):
    def test_closed_square(self):
        loops = reconstruct_loops(_square())
        closed = [lp for lp in loops if lp.closed]
        self.assertEqual(len(closed), 1)
        self.assertAlmostEqual(closed[0].area, 1.0)

    def test_open_chain_not_closed(self):
        prims = [Line(0, 0, 1, 0), Line(1, 0, 2, 0)]  # open
        loops = reconstruct_loops(prims)
        self.assertFalse(any(lp.closed for lp in loops))

    def test_circle_is_loop(self):
        loops = reconstruct_loops([Circle(0, 0, 2)])
        self.assertEqual(len(loops), 1)
        self.assertTrue(loops[0].closed)
        self.assertTrue(loops[0].is_circle)
        self.assertAlmostEqual(loops[0].area, 3.141592653589793 * 4)


class TestHierarchical(unittest.TestCase):
    def test_hole_containment(self):
        outer = _square(0, 0, 10)
        inner = _square(3, 3, 2)  # inside outer
        loops = reconstruct_loops(outer + inner)
        loop_dict, obb = hierarchical_loops(loops)
        # exactly one outer with one hole
        self.assertEqual(len(loop_dict), 1)
        node = list(loop_dict.values())[0]
        self.assertEqual(len(node.holes), 1)
        self.assertEqual(obb, (0.0, 0.0, 10.0, 10.0))

    def test_two_disjoint_outers(self):
        loops = reconstruct_loops(_square(0, 0, 2) + _square(10, 10, 2))
        loop_dict, _ = hierarchical_loops(loops)
        self.assertEqual(len(loop_dict), 2)
        for node in loop_dict.values():
            self.assertEqual(node.holes, ())


class TestReplayValidity(unittest.TestCase):
    def test_valid_sequence(self):
        f1 = Feature(Sketch(SketchPlane(), tuple(_square())), Extrusion(0, 0, 1, 5.0), "create")
        f2 = Feature(Sketch(SketchPlane(), tuple(_square(0, 0, 0.5))), Extrusion(0, 0, 1, 2.0), "join")
        rep = replay_validity(ModelingSequence((f1, f2)))
        self.assertTrue(rep.valid)
        self.assertEqual(rep.n_invalid, 0)

    def test_no_closed_loop(self):
        f = Feature(Sketch(SketchPlane(), (Line(0, 0, 1, 0),)), Extrusion(0, 0, 1, 1.0), "create")
        rep = replay_validity(ModelingSequence((f,)))
        self.assertFalse(rep.valid)
        self.assertIn("no-closed-loop", rep.features[0].errors)

    def test_zero_length(self):
        f = Feature(Sketch(SketchPlane(), tuple(_square())), Extrusion(0, 0, 1, 0.0), "create")
        rep = replay_validity(ModelingSequence((f,)))
        self.assertIn("zero-length-extrusion", rep.features[0].errors)

    def test_first_must_create(self):
        f = Feature(Sketch(SketchPlane(), tuple(_square())), Extrusion(0, 0, 1, 1.0), "join")
        rep = replay_validity(ModelingSequence((f,)))
        self.assertIn("first-feature-not-create", rep.features[0].errors)

    def test_redundant_create(self):
        f1 = Feature(Sketch(SketchPlane(), tuple(_square())), Extrusion(0, 0, 1, 1.0), "create")
        f2 = Feature(Sketch(SketchPlane(), tuple(_square())), Extrusion(0, 0, 1, 1.0), "create")
        rep = replay_validity(ModelingSequence((f1, f2)))
        self.assertIn("redundant-create", rep.features[1].errors)


if __name__ == "__main__":
    unittest.main()
