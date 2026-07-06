import unittest

from reconstruction.geofusion_hierarchy import (
    Curve, Loop, Face, Sketch, Extrusion, SePair, Solid, Token,
    serialize, EC, ELOOP,
)
from bench.geofusion_structure_consistency import (
    closure_valid, valid_ratio, structure_signature, structure_match,
    structure_f1,
)


def _solid(n_curves=3, n_loops=1, n_faces=1):
    loops = tuple(
        Loop(tuple(Curve("line", (11, 11, 100, 100)) for _ in range(n_curves)))
        for _ in range(n_loops)
    )
    faces = tuple(Face(loops) for _ in range(n_faces))
    return Solid((SePair(Sketch(faces), Extrusion((0,) * 10)),))


class TestClosureValid(unittest.TestCase):
    def test_valid(self):
        ok, reason = closure_valid(serialize(_solid()))
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_empty(self):
        ok, reason = closure_valid(())
        self.assertFalse(ok)

    def test_missing_cls(self):
        ok, reason = closure_valid(serialize(_solid())[1:])
        self.assertFalse(ok)
        self.assertIn("cls", reason)

    def test_missing_esolid(self):
        toks = serialize(_solid())[:-1]
        ok, reason = closure_valid(toks)
        self.assertFalse(ok)
        self.assertIn("esolid", reason)

    def test_curve_without_ec(self):
        toks = list(serialize(_solid()))
        # find first ec and remove it -> curve no longer closed
        for i, t in enumerate(toks):
            if t.kind == "ctl" and t.payload == EC:
                del toks[i]
                break
        ok, reason = closure_valid(tuple(toks))
        self.assertFalse(ok)

    def test_never_raises_on_garbage(self):
        garbage = (Token("ctl", 99), Token("line", (1,)))
        ok, reason = closure_valid(garbage)
        self.assertFalse(ok)


class TestValidRatio(unittest.TestCase):
    def test_ratio(self):
        good = serialize(_solid())
        bad = good[1:]
        self.assertEqual(valid_ratio((good, good, bad, bad)), 0.5)

    def test_empty_batch(self):
        self.assertEqual(valid_ratio(()), 0.0)

    def test_all_valid(self):
        good = serialize(_solid())
        self.assertEqual(valid_ratio((good, good)), 1.0)


class TestStructureMetrics(unittest.TestCase):
    def test_signature_ignores_params(self):
        a = _solid()
        # same topology, different coordinates
        b = Solid((SePair(
            Sketch((Face((Loop(tuple(Curve("line", (55, 55, 66, 66)) for _ in range(3))),)),)),
            Extrusion((9,) * 10)),))
        self.assertEqual(structure_signature(a), structure_signature(b))
        self.assertTrue(structure_match(a, b))

    def test_signature_distinguishes_topology(self):
        self.assertNotEqual(structure_signature(_solid(n_curves=3)),
                            structure_signature(_solid(n_curves=4)))
        self.assertFalse(structure_match(_solid(n_loops=1), _solid(n_loops=2)))

    def test_f1_identical(self):
        a = _solid()
        m = structure_f1(a, a)
        self.assertEqual(m["f1"], 1.0)
        self.assertEqual(m["precision"], 1.0)
        self.assertEqual(m["recall"], 1.0)

    def test_f1_partial(self):
        a = _solid(n_curves=4)
        b = _solid(n_curves=2)
        m = structure_f1(a, b)
        self.assertGreater(m["f1"], 0.0)
        self.assertLess(m["f1"], 1.0)

    def test_f1_curve_kind_matters(self):
        a = _solid(n_curves=1)
        b = Solid((SePair(
            Sketch((Face((Loop((Curve("arc", (1, 2, 3, 4, 5, 6)),)),)),)),
            Extrusion((0,) * 10)),))
        m = structure_f1(a, b)
        # line vs arc leaf differs, but solid/pair/sketch/face/loop/extrusion align
        self.assertLess(m["f1"], 1.0)
        self.assertGreater(m["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
