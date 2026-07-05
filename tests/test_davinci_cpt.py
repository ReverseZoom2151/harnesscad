import unittest

from ingest.davinci_cpt import (
    ORIENTATION_DEPENDENT, Sketch, apply_permutation, constraints_preserved,
    random_permutation, rotate_sketch,
)


def _sketch():
    prims = (
        {"coords": ((0.0, 0.0), (1.0, 0.0))},
        {"coords": ((1.0, 0.0), (1.0, 1.0))},
        {"coords": ((0.5, 0.5),)},
    )
    cons = (
        ("coincident", 0, 2, 1, 1),
        ("perpendicular", 0, 4, 1, 4),
        ("horizontal", 0, 4, 0, 4),
    )
    return Sketch(primitives=prims, constraints=cons)


class TestPermutation(unittest.TestCase):
    def test_apply_permutation_remaps_indices(self):
        s = _sketch()
        perm = (2, 0, 1)   # old->new
        out = apply_permutation(s, perm)
        self.assertEqual(out.primitives[2], s.primitives[0])
        # coincident 0..1 becomes 2..0
        self.assertIn(("coincident", 2, 2, 0, 1), out.constraints)

    def test_permutation_preserves_constraints(self):
        s = _sketch()
        perm = (2, 0, 1)
        out = apply_permutation(s, perm)
        self.assertTrue(constraints_preserved(s, out, perm))

    def test_random_permutation_deterministic(self):
        s = _sketch()
        a = random_permutation(s, seed=7)
        b = random_permutation(s, seed=7)
        self.assertEqual(a.constraints, b.constraints)
        self.assertEqual(a.primitives, b.primitives)

    def test_random_permutation_is_preserving(self):
        s = _sketch()
        import random
        for seed in range(5):
            rng = random.Random(seed)
            perm = list(range(len(s.primitives)))
            rng.shuffle(perm)
            out = random_permutation(s, seed=seed)
            self.assertTrue(constraints_preserved(s, out, perm))

    def test_bad_perm_rejected(self):
        with self.assertRaises(ValueError):
            apply_permutation(_sketch(), (0, 0, 1))

    def test_accepts_tuple_pair(self):
        s = _sketch()
        out = apply_permutation((s.primitives, s.constraints), (0, 1, 2))
        self.assertEqual(out.constraints, s.constraints)


class TestRotation(unittest.TestCase):
    def test_180_keeps_orientation_constraints(self):
        s = _sketch()
        out = rotate_sketch(s, 2)
        kinds = {c[0] for c in out.constraints}
        self.assertIn("horizontal", kinds)

    def test_90_drops_orientation_constraints(self):
        s = _sketch()
        out = rotate_sketch(s, 1)
        kinds = {c[0] for c in out.constraints}
        self.assertFalse(kinds & ORIENTATION_DEPENDENT)
        self.assertIn("coincident", kinds)

    def test_rotation_stays_in_unit_square(self):
        s = _sketch()
        out = rotate_sketch(s, 1)
        for p in out.primitives:
            for (x, y) in p["coords"]:
                self.assertTrue(-1e-9 <= x <= 1 + 1e-9)
                self.assertTrue(-1e-9 <= y <= 1 + 1e-9)

    def test_four_quarter_turns_identity(self):
        s = _sketch()
        out = rotate_sketch(s, 4)
        for a, b in zip(out.primitives, s.primitives):
            for (x0, y0), (x1, y1) in zip(a["coords"], b["coords"]):
                self.assertAlmostEqual(x0, x1)
                self.assertAlmostEqual(y0, y1)


if __name__ == "__main__":
    unittest.main()
