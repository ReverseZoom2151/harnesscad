import unittest

from harnesscad.data.datagen.symmetry_balance import (
    OrderingInstance,
    apply_permutation,
    augment_full,
    balance_by_symmetry,
    class_distribution,
    imbalance_ratio,
    inverse_permutation,
    orbit,
    random_relabel,
)


def _inst(labels_first_second_third):
    # blocks encode the item index so we can track relabelling
    blocks = ((10.0, 0.0), (20.0, 0.0), (30.0, 0.0))
    return OrderingInstance(blocks, tuple(labels_first_second_third))


class TestInversePermutation(unittest.TestCase):
    def test_inverse(self):
        self.assertEqual(inverse_permutation((2, 0, 1)), (1, 2, 0))

    def test_inverse_roundtrip(self):
        p = (2, 0, 1)
        q = inverse_permutation(p)
        self.assertEqual(inverse_permutation(q), p)


class TestApplyPermutation(unittest.TestCase):
    def test_identity(self):
        inst = _inst((0, 1, 2))
        out = apply_permutation(inst, (0, 1, 2))
        self.assertEqual(out, inst)

    def test_swap_relabels_blocks_and_label(self):
        # swap item names 0 and 1: new j0 <- old 1, new j1 <- old 0
        inst = OrderingInstance(((1.0,), (2.0,), (3.0,)), (0, 1, 2))
        out = apply_permutation(inst, (1, 0, 2))
        self.assertEqual(out.blocks, ((2.0,), (1.0,), (3.0,)))
        # old label 0>1>2 with old0->new1, old1->new0 => 1>0>2
        self.assertEqual(out.label, (1, 0, 2))

    def test_label_permutation_matches_paper_example(self):
        # paper: swapping x1,x2 turns optimal x2>x1>x3 into x1>x2>x3
        # items: x1=0,x2=1,x3=2; label x2>x1>x3 == (1,0,2)
        inst = OrderingInstance(((1.0,), (2.0,), (3.0,)), (1, 0, 2))
        out = apply_permutation(inst, (1, 0, 2))
        self.assertEqual(out.label, (0, 1, 2))

    def test_bad_permutation_raises(self):
        with self.assertRaises(ValueError):
            apply_permutation(_inst((0, 1, 2)), (0, 0, 1))


class TestOrbit(unittest.TestCase):
    def test_orbit_size_and_balance(self):
        orb = orbit(_inst((0, 1, 2)))
        self.assertEqual(len(orb), 6)
        # every ordering class appears exactly once
        self.assertEqual(len(set(o.label for o in orb)), 6)

    def test_orbit_deterministic(self):
        a = [o.label for o in orbit(_inst((2, 0, 1)))]
        b = [o.label for o in orbit(_inst((2, 0, 1)))]
        self.assertEqual(a, b)

    def test_blocks_preserved_as_multiset(self):
        orb = orbit(OrderingInstance(((1.0,), (2.0,), (3.0,)), (0, 1, 2)))
        for o in orb:
            self.assertEqual(sorted(o.blocks), [(1.0,), (2.0,), (3.0,)])


class TestAugmentAndBalance(unittest.TestCase):
    def test_augment_full_perfectly_balanced(self):
        data = [_inst((0, 1, 2)), _inst((0, 1, 2)), _inst((2, 1, 0))]
        aug = augment_full(data)
        self.assertEqual(len(aug), 18)
        dist = class_distribution(aug)
        self.assertEqual(len(dist), 6)
        self.assertEqual(set(dist.values()), {3})

    def test_imbalance_ratio_flags_skew(self):
        data = [_inst((0, 1, 2))] * 5 + [_inst((2, 1, 0))]
        self.assertEqual(imbalance_ratio(data), 5.0)

    def test_balance_keeps_size_and_reduces_imbalance(self):
        data = [_inst((0, 1, 2))] * 12
        balanced = balance_by_symmetry(data, seed=1)
        self.assertEqual(len(balanced), 12)
        dist = class_distribution(balanced)
        self.assertEqual(len(dist), 6)
        self.assertEqual(set(dist.values()), {2})  # 12/6 exactly

    def test_balance_deterministic(self):
        data = [_inst((0, 1, 2))] * 7 + [_inst((1, 0, 2))] * 3
        a = [i.label for i in balance_by_symmetry(data, seed=5)]
        b = [i.label for i in balance_by_symmetry(data, seed=5)]
        self.assertEqual(a, b)

    def test_balance_preserves_block_multiset_per_instance(self):
        inst = OrderingInstance(((1.0,), (2.0,), (3.0,)), (0, 1, 2))
        out = balance_by_symmetry([inst] * 6, seed=0)
        for o in out:
            self.assertEqual(sorted(o.blocks), [(1.0,), (2.0,), (3.0,)])

    def test_random_relabel_deterministic_and_size(self):
        data = [_inst((0, 1, 2))] * 10
        a = [i.label for i in random_relabel(data, seed=3)]
        b = [i.label for i in random_relabel(data, seed=3)]
        self.assertEqual(a, b)
        self.assertEqual(len(a), 10)


class TestTupleInput(unittest.TestCase):
    def test_accepts_plain_tuples(self):
        out = orbit((((1.0,), (2.0,), (3.0,)), (0, 1, 2)))
        self.assertEqual(len(out), 6)


if __name__ == "__main__":
    unittest.main()
