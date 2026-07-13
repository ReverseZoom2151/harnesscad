import unittest

from harnesscad.domain.reconstruction.sequences.histcad_sequence import (
    Line, Circle, Arc, SketchPlane, Constraint, Sketch, Extrusion, Feature,
    ModelingSequence, symmetric_difference, flatten_faces, token_estimate,
    primitive_from_dict, CONSTRAINT_TYPES, BOOLEAN_OPS,
)


def _square(dx=0.0, dy=0.0):
    return [
        Line(0 + dx, 0 + dy, 1 + dx, 0 + dy),
        Line(1 + dx, 0 + dy, 1 + dx, 1 + dy),
        Line(1 + dx, 1 + dy, 0 + dx, 1 + dy),
        Line(0 + dx, 1 + dy, 0 + dx, 0 + dy),
    ]


def _seq():
    sk = Sketch(SketchPlane(), tuple(_square()),
                (Constraint("horizontal", (0,)), Constraint("vertical", (1,))))
    ex = Extrusion(0, 0, 1, 5.0)
    return ModelingSequence((Feature(sk, ex, "create"),))


class TestSchema(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(len(CONSTRAINT_TYPES), 10)
        self.assertEqual(BOOLEAN_OPS, ("create", "join", "subtract", "intersect"))

    def test_roundtrip(self):
        seq = _seq()
        d = seq.to_dict()
        back = ModelingSequence.from_dict(d)
        self.assertEqual(back.to_dict(), d)

    def test_primitive_roundtrip(self):
        for p in (Line(0, 0, 1, 2), Circle(1, 1, 3), Arc(0, 0, 1, 1, 2, 0)):
            self.assertEqual(primitive_from_dict(p.to_dict()).to_dict(), p.to_dict())

    def test_bad_constraint(self):
        with self.assertRaises(ValueError):
            Constraint("bogus", (0,))

    def test_bad_boolean(self):
        with self.assertRaises(ValueError):
            Feature(Sketch(SketchPlane()), Extrusion(0, 0, 1, 1.0), "melt")


class TestSymmetricDifference(unittest.TestCase):
    def test_shared_edge_removed(self):
        # two unit squares sharing the edge x=1..? Build two squares sharing an edge
        left = [Line(0, 0, 1, 0), Line(1, 0, 1, 1), Line(1, 1, 0, 1), Line(0, 1, 0, 0)]
        right = [Line(1, 0, 2, 0), Line(2, 0, 2, 1), Line(2, 1, 1, 1), Line(1, 1, 1, 0)]
        # shared edge (1,0)-(1,1) appears in both loops -> removed
        result = symmetric_difference([left, right])
        keys = {p.canonical_key() for p in result}
        shared = Line(1, 0, 1, 1).canonical_key()
        self.assertNotIn(shared, keys)
        self.assertEqual(len(result), 6)  # 8 edges - 2 shared

    def test_no_duplicates_keeps_all(self):
        loops = [_square()]
        self.assertEqual(len(symmetric_difference(loops)), 4)

    def test_flatten_faces(self):
        faces = [[_square()], [[Circle(0.5, 0.5, 0.2)]]]
        flat = flatten_faces(faces)
        self.assertEqual(len(flat), 5)

    def test_deterministic_order(self):
        loops = [_square()]
        r1 = symmetric_difference(loops)
        r2 = symmetric_difference(loops)
        self.assertEqual([p.canonical_key() for p in r1],
                         [p.canonical_key() for p in r2])


class TestTokenEstimate(unittest.TestCase):
    def test_constraints_add_overhead(self):
        seq = _seq()
        self.assertGreater(token_estimate(seq, True), token_estimate(seq, False))

    def test_rotated_extrusion_costs_more(self):
        sk = Sketch(SketchPlane(), tuple(_square()))
        up = ModelingSequence((Feature(sk, Extrusion(0, 0, 1, 5.0)),))
        rot = ModelingSequence((Feature(sk, Extrusion(0, 0, 1, 5.0, rotated=True)),))
        self.assertGreater(token_estimate(rot), token_estimate(up))


if __name__ == "__main__":
    unittest.main()
