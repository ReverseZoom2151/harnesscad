import unittest

from harnesscad.domain.reconstruction import pointercad_tokens as t


class TokenRangesTest(unittest.TestCase):
    def test_families_are_disjoint_and_cover_ids(self):
        q = 8
        seen = {}
        for tid in range(1, t.vocab_size(q)):
            fam = t.token_family(tid, q)
            seen.setdefault(fam, 0)
            seen[fam] += 1
        self.assertEqual(set(seen), {t.LABEL, t.VALUE, t.POINTER})
        self.assertEqual(seen[t.POINTER], 2)      # <pe>, <pd>
        self.assertEqual(seen[t.VALUE], 1 << q)   # 2**q value tokens
        # labels: em..sx (9) + or(2) + dr(6) + bo(4) = 21
        self.assertEqual(seen[t.LABEL], 21)

    def test_value_base_boundary(self):
        lo, hi = t.value_range(8)
        self.assertEqual(lo, 24)
        self.assertEqual(hi, 24 + 256)
        self.assertEqual(t.token_family(23, 8), t.LABEL)   # last boolean
        self.assertEqual(t.token_family(24, 8), t.VALUE)   # first value

    def test_out_of_range_raises(self):
        with self.assertRaises(t.PointerTokenError):
            t.token_family(0, 8)
        with self.assertRaises(t.PointerTokenError):
            t.token_family(t.vocab_size(8), 8)


class PointerTokenTest(unittest.TestCase):
    def test_pe_enabled_pd_disabled(self):
        self.assertTrue(t.is_pointer(t.TOK_PE))
        self.assertTrue(t.is_pointer(t.TOK_PD))
        self.assertTrue(t.pointer_enabled(t.TOK_PE))
        self.assertFalse(t.pointer_enabled(t.TOK_PD))

    def test_pointer_enabled_rejects_non_pointer(self):
        with self.assertRaises(t.PointerTokenError):
            t.pointer_enabled(t.TOK_SS)


class LabelNameTest(unittest.TestCase):
    def test_direction_roundtrip(self):
        for name in t.DIRECTIONS:
            tid = t.direction_id(name)
            self.assertTrue(t.DR_BASE <= tid < t.BO_BASE)
            self.assertEqual(t.direction_name(tid), name)

    def test_boolean_roundtrip(self):
        for name in t.BOOLEANS:
            self.assertEqual(t.boolean_name(t.boolean_id(name)), name)

    def test_orientation_roundtrip(self):
        for name in t.ORIENTATIONS:
            self.assertEqual(t.orientation_name(t.orientation_id(name)), name)

    def test_label_name_dispatch(self):
        self.assertEqual(t.label_name(t.TOK_SS), "<ss>")
        self.assertEqual(t.label_name(t.direction_id("Z+")), "<dr:Z+>")
        self.assertEqual(t.label_name(t.boolean_id("Cut")), "<bo:Cut>")
        self.assertEqual(t.label_name(t.orientation_id("CCW")), "<or:CCW>")


class QuantizeTest(unittest.TestCase):
    def test_nv_endpoints(self):
        self.assertEqual(t.quantize_nv(0.0, 8), t.VALUE_BASE)
        self.assertEqual(t.quantize_nv(1.0, 8), t.VALUE_BASE + 255)

    def test_nv_clamps(self):
        self.assertEqual(t.quantize_nv(-5.0, 8), t.VALUE_BASE)
        self.assertEqual(t.quantize_nv(5.0, 8), t.VALUE_BASE + 255)

    def test_nv_roundtrip_within_halfbin(self):
        rep = t.quantization_report(8)
        for i in range(0, 101):
            v = i / 100.0
            back = t.dequantize_nv(t.quantize_nv(v, 8), 8)
            self.assertLessEqual(abs(back - v), rep.max_abs_error + 1e-9)

    def test_ag_wraps(self):
        self.assertEqual(t.quantize_ag(0.0, 8), t.quantize_ag(360.0, 8))
        self.assertAlmostEqual(t.dequantize_ag(t.quantize_ag(90.0, 8), 8), 90.0, delta=1.0)

    def test_higher_q_smaller_error(self):
        self.assertLess(
            t.quantization_report(10).max_abs_error,
            t.quantization_report(6).max_abs_error,
        )

    def test_dequantize_rejects_non_value(self):
        with self.assertRaises(t.PointerTokenError):
            t.dequantize_nv(t.TOK_SS, 8)


if __name__ == "__main__":
    unittest.main()
