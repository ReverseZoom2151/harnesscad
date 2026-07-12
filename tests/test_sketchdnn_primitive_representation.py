import unittest

from reconstruction.sketchdnn_primitive_representation import (
    CLASS_NAMES,
    FEATURE_DIM,
    PARAM_DIMS,
    PARAM_TYPES,
    class_logits,
    decode_primitive,
    encode_primitive,
    mask_irrelevant_params,
    param_block,
    reflect_arc,
    rescale_probs,
    weight_params_by_type,
)


class TestLayout(unittest.TestCase):
    def test_feature_dim(self):
        # 2 (construction) + 5 (class incl NONE) + 4+3+5+2 params
        self.assertEqual(FEATURE_DIM, 2 + 5 + 4 + 3 + 5 + 2)

    def test_class_names(self):
        self.assertIn("NONE", CLASS_NAMES)
        for t in PARAM_TYPES:
            self.assertIn(t, CLASS_NAMES)


class TestEncodeDecode(unittest.TestCase):
    def test_roundtrip_line(self):
        vec = encode_primitive("LINE", [0.1, 0.2, 0.3, 0.4])
        cons, cls, params = decode_primitive(vec)
        self.assertFalse(cons)
        self.assertEqual(cls, "LINE")
        self.assertEqual(params, [0.1, 0.2, 0.3, 0.4])

    def test_roundtrip_circle(self):
        vec = encode_primitive("CIRCLE", [1.0, 2.0, 0.5])
        _, cls, params = decode_primitive(vec)
        self.assertEqual(cls, "CIRCLE")
        self.assertEqual(params, [1.0, 2.0, 0.5])

    def test_roundtrip_arc(self):
        vec = encode_primitive("ARC", [0.0, 0.0, 1.0, 1.0, 0.25])
        _, cls, params = decode_primitive(vec)
        self.assertEqual(cls, "ARC")
        self.assertEqual(params, [0.0, 0.0, 1.0, 1.0, 0.25])

    def test_roundtrip_point(self):
        vec = encode_primitive("POINT", [3.0, 4.0])
        _, cls, params = decode_primitive(vec)
        self.assertEqual(cls, "POINT")
        self.assertEqual(params, [3.0, 4.0])

    def test_construction_flag(self):
        vec = encode_primitive("LINE", [0, 0, 1, 1], construction=True)
        cons, _, _ = decode_primitive(vec)
        self.assertTrue(cons)

    def test_only_active_block_filled(self):
        vec = encode_primitive("POINT", [5.0, 6.0])
        # LINE block should remain zeros
        self.assertEqual(param_block(vec, "LINE"), [0.0, 0.0, 0.0, 0.0])

    def test_wrong_param_count(self):
        with self.assertRaises(ValueError):
            encode_primitive("LINE", [1.0, 2.0])

    def test_unknown_class(self):
        with self.assertRaises(ValueError):
            encode_primitive("SPLINE", [1.0])

    def test_decode_none_class(self):
        vec = [0.0] * FEATURE_DIM
        # class block all-zero -> argmax picks first (LINE); force NONE win
        none_idx = CLASS_NAMES.index("NONE")
        vec[2 + none_idx] = 1.0
        _, cls, params = decode_primitive(vec)
        self.assertEqual(cls, "NONE")
        self.assertEqual(params, [])


class TestMasking(unittest.TestCase):
    def test_mask_keeps_true_type(self):
        vec = encode_primitive("CIRCLE", [1.0, 2.0, 3.0])
        # inject noise into a foreign block
        line_off = 2 + 5
        vec[line_off] = 9.9
        masked = mask_irrelevant_params(vec, "CIRCLE")
        self.assertEqual(param_block(masked, "CIRCLE"), [1.0, 2.0, 3.0])
        self.assertEqual(param_block(masked, "LINE"), [0.0, 0.0, 0.0, 0.0])

    def test_mask_leaves_class_block(self):
        vec = encode_primitive("ARC", [0, 0, 1, 1, 0.5])
        masked = mask_irrelevant_params(vec, "ARC")
        self.assertEqual(class_logits(masked), class_logits(vec))


class TestWeighting(unittest.TestCase):
    def test_rescale_max_is_one(self):
        r = rescale_probs([0.2, 0.5, 0.3])
        self.assertAlmostEqual(max(r), 1.0)
        self.assertAlmostEqual(r[1], 1.0)

    def test_rescale_proportions(self):
        r = rescale_probs([0.1, 0.4])
        self.assertAlmostEqual(r[0], 0.25)
        self.assertAlmostEqual(r[1], 1.0)

    def test_weight_attenuates_unlikely(self):
        vec = [0.0] * FEATURE_DIM
        # fill every param block with 1.0
        for t in PARAM_TYPES:
            b = encode_primitive(t, [1.0] * PARAM_DIMS[t])
            for i, v in enumerate(b):
                if v == 1.0 and i >= 2 + 5:
                    vec[i] = 1.0
        # probs favouring CIRCLE (index 1 in PARAM_TYPES)
        probs = [0.1, 0.6, 0.2, 0.1]
        weighted = weight_params_by_type(vec, probs)
        # CIRCLE block untouched (weight 1.0), LINE attenuated
        self.assertEqual(param_block(weighted, "CIRCLE"), [1.0, 1.0, 1.0])
        self.assertTrue(all(v < 1.0 for v in param_block(weighted, "LINE")))

    def test_weight_bad_length(self):
        with self.assertRaises(ValueError):
            weight_params_by_type([0.0] * FEATURE_DIM, [0.5, 0.5])


class TestReflectArc(unittest.TestCase):
    def test_reflect_negates_kappa(self):
        vec = encode_primitive("ARC", [0, 0, 1, 1, 0.3])
        r = reflect_arc(vec)
        _, _, params = decode_primitive(r)
        self.assertAlmostEqual(params[-1], -0.3)

    def test_reflect_preserves_coords(self):
        vec = encode_primitive("ARC", [0.1, 0.2, 0.3, 0.4, 0.5])
        r = reflect_arc(vec)
        self.assertEqual(param_block(r, "ARC")[:4], [0.1, 0.2, 0.3, 0.4])


if __name__ == "__main__":
    unittest.main()
