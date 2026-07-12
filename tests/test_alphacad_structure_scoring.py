import unittest

from procedural.alphacad_brick_templates import generate_tower, generate_table
from quality.alphacad_structure_scoring import (
    aesthetics_score,
    base_top_counts,
    confidence_score,
    materials_score,
    score_all,
    support_and_violations,
)


def _model(bricks, w, d, h, features=None):
    m = {"width": w, "depth": d, "height": h, "bricks": bricks}
    if features is not None:
        m["features"] = features
    return m


def _b(bid, x, y, z, part="default"):
    return {"id": bid, "x": x, "y": y, "z": z, "part_type": part}


class TestStructureScoring(unittest.TestCase):
    def test_no_violations_when_grounded(self):
        bricks = [_b(0, 0, 0, 0), _b(1, 0, 0, 1), _b(2, 0, 0, 2)]
        _, viol = support_and_violations(_model(bricks, 1, 1, 3))
        self.assertEqual(viol, [])

    def test_floating_brick_is_violation(self):
        # brick id 1 at z=2 with nothing at z=1 -> floating
        bricks = [_b(0, 0, 0, 0), _b(1, 0, 0, 2)]
        _, viol = support_and_violations(_model(bricks, 1, 1, 3))
        self.assertEqual(viol, [1])

    def test_base_top_counts(self):
        bricks = [_b(0, 0, 0, 0), _b(1, 1, 0, 0), _b(2, 0, 0, 1)]
        base, top = base_top_counts(_model(bricks, 2, 1, 2))
        self.assertEqual((base, top), (2, 1))

    def test_confidence_wide_base_beats_tall(self):
        wide = generate_table(8, 6, 3, seed=1)
        tall = generate_tower(2, 2, 20, seed=1)
        self.assertGreater(confidence_score(wide).score, confidence_score(tall).score)

    def test_confidence_in_range(self):
        c = confidence_score(generate_table(6, 4, 3, seed=2))
        self.assertTrue(0 <= c.score <= 100)
        self.assertTrue(len(c.reasons) >= 2)

    def test_materials_zero_for_single_type(self):
        bricks = [_b(i, i, 0, 0, "leg") for i in range(5)]
        self.assertEqual(materials_score(_model(bricks, 5, 1, 1)), 0)

    def test_materials_higher_for_diverse(self):
        uniform = [_b(i, i, 0, 0, "leg") for i in range(4)]
        diverse = [_b(0, 0, 0, 0, "leg"), _b(1, 1, 0, 0, "seat"),
                   _b(2, 2, 0, 0, "wall"), _b(3, 3, 0, 0, "surface")]
        self.assertGreater(materials_score(_model(diverse, 4, 1, 1)),
                           materials_score(_model(uniform, 4, 1, 1)))

    def test_aesthetics_symmetry(self):
        square = _model([], 4, 4, 2)
        oblong = _model([], 10, 2, 2)
        self.assertGreater(aesthetics_score(square), aesthetics_score(oblong))

    def test_aesthetics_feature_bonus(self):
        plain = _model([], 4, 4, 2, features={})
        fancy = _model([], 4, 4, 2, features={"surface_pattern": "border",
                                              "center_support": True})
        self.assertGreaterEqual(aesthetics_score(fancy), aesthetics_score(plain))

    def test_score_all_keys(self):
        s = score_all(generate_table(6, 4, 3, seed=1))
        for k in ("stability", "materials", "aesthetics", "violations", "reasons"):
            self.assertIn(k, s)

    def test_deterministic(self):
        m = generate_table(6, 4, 3, seed=9)
        self.assertEqual(score_all(m), score_all(m))


if __name__ == "__main__":
    unittest.main()
