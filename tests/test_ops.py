"""Round-trip and canonical-serialisation tests for the CISP op set."""

import unittest

from harnesscad.core.cisp.ops import (
    NewSketch, AddPoint, AddLine, AddCircle, AddRectangle,
    Constrain, Extrude, Fillet, Boolean,
    parse_op, canonical_json,
    CONSTRAINT_DOF, PRIMITIVE_DOF,
)


class TestOpRoundTrip(unittest.TestCase):
    def _sample_ops(self):
        return [
            NewSketch(plane="XZ"),
            AddPoint(sketch="sk1", x=1.0, y=2.0),
            AddLine(sketch="sk1", x1=0.0, y1=0.0, x2=3.0, y2=4.0),
            AddCircle(sketch="sk1", cx=1.0, cy=1.0, r=2.5),
            AddRectangle(sketch="sk1", x=0.0, y=0.0, w=10.0, h=5.0),
            Constrain(kind="distance", a="e1", b="e2", value=10.0),
            Constrain(kind="coincident", a="e1"),
            Extrude(sketch="sk1", distance=5.0),
            Fillet(edges=("e1", "e2", "e3"), radius=0.5),
            Boolean(kind="cut", target="f1", tool="f2"),
        ]

    def test_round_trip_preserves_equality(self):
        for op in self._sample_ops():
            with self.subTest(op=type(op).__name__):
                restored = parse_op(op.to_dict())
                self.assertEqual(op, restored)
                self.assertIs(type(op), type(restored))

    def test_fillet_edges_tuple_preserved(self):
        op = Fillet(edges=("a", "b", "c"), radius=1.5)
        d = op.to_dict()
        # to_dict lowers the tuple to a list for JSON.
        self.assertIsInstance(d["edges"], list)
        self.assertEqual(d["edges"], ["a", "b", "c"])
        restored = parse_op(d)
        self.assertIsInstance(restored.edges, tuple)
        self.assertEqual(restored.edges, ("a", "b", "c"))
        self.assertEqual(op, restored)

    def test_parse_op_on_fillet_dict_restores_tuple(self):
        # A raw dict (as it would arrive over the wire) with edges as a list.
        d = {"op": "fillet", "edges": ["e1", "e2"], "radius": 2.0}
        op = parse_op(d)
        self.assertIsInstance(op, Fillet)
        self.assertIsInstance(op.edges, tuple)
        self.assertEqual(op.edges, ("e1", "e2"))


class TestCanonicalJson(unittest.TestCase):
    def test_canonical_json_is_sorted(self):
        op = Constrain(kind="distance", a="e1", b="e2", value=10.0)
        cj = canonical_json(op)
        d = op.to_dict()
        keys = sorted(d.keys())
        # Keys must appear in sorted order in the canonical form.
        positions = [cj.index('"%s"' % k) for k in keys]
        self.assertEqual(positions, sorted(positions))

    def test_canonical_json_is_stable(self):
        op = AddRectangle(sketch="sk1", x=1.0, y=2.0, w=3.0, h=4.0)
        self.assertEqual(canonical_json(op), canonical_json(op))
        # Two equal ops produce identical canonical JSON.
        op2 = AddRectangle(sketch="sk1", x=1.0, y=2.0, w=3.0, h=4.0)
        self.assertEqual(canonical_json(op), canonical_json(op2))

    def test_canonical_json_distinguishes_different_ops(self):
        a = Extrude(sketch="sk1", distance=5.0)
        b = Extrude(sketch="sk1", distance=6.0)
        self.assertNotEqual(canonical_json(a), canonical_json(b))


class TestDofTables(unittest.TestCase):
    def test_constraint_dof_keys(self):
        self.assertEqual(CONSTRAINT_DOF["distance"], 1)
        self.assertEqual(CONSTRAINT_DOF["coincident"], 2)

    def test_primitive_dof_keys(self):
        self.assertEqual(PRIMITIVE_DOF["rectangle"], 4)
        self.assertEqual(PRIMITIVE_DOF["point"], 2)


if __name__ == "__main__":
    unittest.main()
