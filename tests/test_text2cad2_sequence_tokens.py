import unittest

from harnesscad.domain.reconstruction.tokens import text2cad_tokens as t2


class TestQuantization(unittest.TestCase):
    def test_quantize_endpoints(self):
        self.assertEqual(t2.quantize(0.0), 0)
        self.assertEqual(t2.quantize(1.0), 255)

    def test_quantize_clamps(self):
        self.assertEqual(t2.quantize(-5.0), 0)
        self.assertEqual(t2.quantize(2.0), 255)

    def test_quantize_dequantize_roundtrip_levels(self):
        for level in (0, 1, 127, 200, 255):
            self.assertEqual(t2.quantize(t2.dequantize(level)), level)

    def test_dequantize_range_error(self):
        with self.assertRaises(ValueError):
            t2.dequantize(256)
        with self.assertRaises(ValueError):
            t2.dequantize(-1)

    def test_value_token_range(self):
        self.assertEqual(t2.value_to_token_id(0.0), t2.COORD_MIN_TOKEN)
        self.assertEqual(t2.value_to_token_id(1.0), t2.COORD_MAX_TOKEN)
        self.assertEqual(t2.COORD_MIN_TOKEN, 11)
        self.assertEqual(t2.COORD_MAX_TOKEN, 266)

    def test_value_token_never_collides_with_specials(self):
        for v in (0.0, 0.25, 0.5, 0.75, 1.0):
            self.assertGreater(t2.value_to_token_id(v), 10)

    def test_value_token_roundtrip(self):
        for tid in (11, 100, 266):
            v = t2.token_id_to_value(tid)
            self.assertEqual(t2.value_to_token_id(v), tid)

    def test_token_id_to_value_rejects_specials(self):
        with self.assertRaises(ValueError):
            t2.token_id_to_value(6)


class TestTokens(unittest.TestCase):
    def test_special_token_shape(self):
        self.assertEqual(t2.special_token(t2.END_SKETCH), (2, 0))
        self.assertEqual(t2.special_token(t2.PAD), (0, 0))

    def test_special_token_rejects_bad_id(self):
        with self.assertRaises(ValueError):
            t2.special_token(11)

    def test_coord_token_both_slots(self):
        tok = t2.coord_token(0.0, 1.0)
        self.assertEqual(tok, (11, 266))
        self.assertTrue(t2.is_coordinate_token(tok))

    def test_special_not_coordinate(self):
        self.assertFalse(t2.is_coordinate_token(t2.special_token(t2.END_LOOP)))

    def test_coord_token_roundtrip(self):
        x, y = t2.decode_coord_token(t2.coord_token(0.5, 0.25))
        self.assertAlmostEqual(x, 128 / 255)
        self.assertAlmostEqual(y, 64 / 255)

    def test_vocabulary_size(self):
        self.assertEqual(t2.vocabulary_size(), 267)

    def test_boolean_ids(self):
        self.assertEqual(
            sorted(t2.BOOLEAN_IDS.values()), [7, 8, 9, 10]
        )
        self.assertEqual(t2.BOOLEAN_NAMES[7], "new")


class TestCurves(unittest.TestCase):
    def test_line_tokens_end_with_curve_marker(self):
        toks = t2.Line((0.0, 0.0), (1.0, 1.0)).tokens()
        self.assertEqual(toks[-1], (t2.END_CURVE, 0))
        self.assertEqual(len(toks), 3)

    def test_arc_has_three_coords(self):
        toks = t2.Arc((0.0, 0.0), (0.5, 0.5), (1.0, 0.0)).tokens()
        coords = [x for x in toks if t2.is_coordinate_token(x)]
        self.assertEqual(len(coords), 3)

    def test_circle_topmost(self):
        self.assertEqual(t2.circle_topmost(0.5, 0.5, 0.25), (0.5, 0.75))

    def test_circle_tokens_center_and_topmost(self):
        c = t2.Circle((0.5, 0.5), 0.25)
        toks = c.tokens()
        self.assertEqual(t2.decode_coord_token(toks[0]), (128 / 255, 128 / 255))
        # top-most y = 0.75
        self.assertAlmostEqual(t2.decode_coord_token(toks[1])[1], t2.quantize(0.75) / 255)


class TestSerialize(unittest.TestCase):
    def _model(self):
        loop = [t2.Line((0.0, 0.0), (1.0, 0.0)), t2.Line((1.0, 0.0), (0.0, 0.0))]
        face = [loop]
        extr = t2.Extrusion(
            d_plus=0.5, d_minus=0.0, tx=0.1, ty=0.2, tz=0.3,
            theta=0.4, phi=0.5, gamma=0.6, sigma=0.7, boolean="new",
        )
        return t2.CadModel(faces=[face], extrusion=extr)

    def test_serialize_model_boundaries(self):
        toks = t2.serialize_model(self._model())
        self.assertEqual(toks[0], (t2.START, 0))
        self.assertEqual(toks[-1], (t2.END_SEQUENCE, 0))

    def test_structural_markers_present(self):
        toks = t2.serialize_model(self._model())
        ids = [x[0] for x in toks]
        for marker in (t2.END_CURVE, t2.END_LOOP, t2.END_FACE, t2.END_SKETCH, t2.END_EXTRUDE):
            self.assertIn(marker, ids)

    def test_extrusion_block_has_11_tokens(self):
        extr = self._model().extrusion
        toks = extr.tokens()
        self.assertEqual(len(toks), 11)
        self.assertEqual(toks[-1], (t2.END_EXTRUDE, 0))
        # boolean token before ee
        self.assertEqual(toks[-2], (t2.BOOL_NEW, 0))

    def test_extrusion_param_order(self):
        extr = t2.Extrusion(
            d_plus=1.0, d_minus=0.0, tx=0.0, ty=0.0, tz=0.0,
            theta=0.0, phi=0.0, gamma=0.0, sigma=0.0, boolean="cut",
        )
        toks = extr.tokens()
        # first param is d_plus == 1.0 -> max token
        self.assertEqual(toks[0], (t2.COORD_MAX_TOKEN, 0))
        self.assertEqual(toks[-2], (t2.BOOL_CUT, 0))


if __name__ == "__main__":
    unittest.main()
