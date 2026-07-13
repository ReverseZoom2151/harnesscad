import unittest

from harnesscad.domain.reconstruction.geofusion_hierarchy import (
    Curve, Loop, Face, Sketch, Extrusion, SePair, Solid, Token,
    serialize, deserialize, quantize, dequantize,
    count_nodes, tree_depth, type_paths,
    CLS, ESOLID, ESKETCH, EFACE, ELOOP, EC, EE,
    QUANT_LO, QUANT_HI,
)


def _sample_solid():
    square = Loop((
        Curve("line", (11, 11, 100, 11)),
        Curve("line", (100, 11, 100, 100)),
        Curve("line", (100, 100, 11, 100)),
        Curve("line", (11, 100, 11, 11)),
    ))
    hole = Loop((Curve("circle", (55, 55, 20)),))
    sketch = Sketch((Face((square, hole)),))
    ext = Extrusion((128, 128, 128, 50, 50, 50, 200, 100, 11, 7))
    return Solid((SePair(sketch, ext),))


class TestQuantization(unittest.TestCase):
    def test_range(self):
        self.assertEqual(quantize(0.0), QUANT_LO)
        self.assertEqual(quantize(1.0), QUANT_HI)
        self.assertEqual(quantize(-5.0), QUANT_LO)
        self.assertEqual(quantize(9.0), QUANT_HI)

    def test_roundtrip_monotone(self):
        prev = -1.0
        for code in range(QUANT_LO, QUANT_HI + 1):
            v = dequantize(code)
            self.assertGreater(v, prev)
            prev = v
        self.assertAlmostEqual(dequantize(QUANT_LO), 0.0)
        self.assertAlmostEqual(dequantize(QUANT_HI), 1.0)

    def test_dequantize_out_of_range(self):
        with self.assertRaises(ValueError):
            dequantize(QUANT_LO - 1)
        with self.assertRaises(ValueError):
            dequantize(QUANT_HI + 1)


class TestSerialization(unittest.TestCase):
    def test_roundtrip(self):
        solid = _sample_solid()
        tokens = serialize(solid)
        rebuilt = deserialize(tokens)
        self.assertEqual(rebuilt, solid)

    def test_starts_cls_ends_esolid(self):
        tokens = serialize(_sample_solid())
        self.assertEqual(tokens[0], Token("ctl", CLS))
        self.assertEqual(tokens[-1], Token("ctl", ESOLID))

    def test_end_token_counts(self):
        # one sketch (1 face, 2 loops, 5 curves), one extrusion.
        tokens = serialize(_sample_solid())
        ctl = [t.payload for t in tokens if t.kind == "ctl"]
        self.assertEqual(ctl.count(EC), 5)     # 4 lines + 1 circle
        self.assertEqual(ctl.count(ELOOP), 2)
        self.assertEqual(ctl.count(EFACE), 1)
        self.assertEqual(ctl.count(ESKETCH), 1)
        self.assertEqual(ctl.count(EE), 1)
        self.assertEqual(ctl.count(ESOLID), 1)

    def test_multi_pair_roundtrip(self):
        s = _sample_solid()
        two = Solid(s.pairs + s.pairs)
        self.assertEqual(deserialize(serialize(two)), two)

    def test_empty_solid_roundtrip(self):
        empty = Solid(())
        self.assertEqual(deserialize(serialize(empty)), empty)

    def test_malformed_missing_cls(self):
        tokens = serialize(_sample_solid())[1:]
        with self.assertRaises(ValueError):
            deserialize(tokens)

    def test_malformed_missing_ec(self):
        tokens = list(serialize(_sample_solid()))
        # drop the first ec (index 2: cls, line, ec, ...)
        del tokens[2]
        with self.assertRaises(ValueError):
            deserialize(tuple(tokens))

    def test_unknown_curve_kind(self):
        bad = Solid((SePair(Sketch((Face((Loop((Curve("spline", (1,)),)),)),)),
                            Extrusion((0,) * 10)),))
        with self.assertRaises(ValueError):
            serialize(bad)


class TestStructuralHelpers(unittest.TestCase):
    def test_count_nodes(self):
        counts = count_nodes(_sample_solid())
        self.assertEqual(counts["solid"], 1)
        self.assertEqual(counts["sketch"], 1)
        self.assertEqual(counts["face"], 1)
        self.assertEqual(counts["loop"], 2)
        self.assertEqual(counts["curve"], 5)
        self.assertEqual(counts["extrusion"], 1)

    def test_tree_depth(self):
        self.assertEqual(tree_depth(_sample_solid()), 5)
        self.assertEqual(tree_depth(Solid(())), 1)

    def test_type_paths_stable(self):
        s = _sample_solid()
        self.assertEqual(type_paths(s), type_paths(_sample_solid()))
        # 5 curve paths + 2 loop + 1 face + 1 extrusion = 9
        self.assertEqual(len(type_paths(s)), 9)


if __name__ == "__main__":
    unittest.main()
