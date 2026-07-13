"""Tests for dataengine.sldprtnet_complexity."""

import unittest

from harnesscad.data.dataengine.curation.complexity_levels import (
    COMPLEXITY_LEVELS,
    REFERENCE_COUNTS,
    ComplexityItem,
    classify_complexity,
    curriculum_order,
    distribution_l1,
    level_histogram,
    level_label,
    level_proportions,
    reference_proportions,
    stratify,
)


class TestClassify(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(classify_complexity(1), 1)
        self.assertEqual(classify_complexity(5), 1)
        self.assertEqual(classify_complexity(6), 2)
        self.assertEqual(classify_complexity(10), 2)
        self.assertEqual(classify_complexity(11), 3)
        self.assertEqual(classify_complexity(100), 3)
        self.assertEqual(classify_complexity(101), 4)
        self.assertEqual(classify_complexity(5000), 4)

    def test_zero_rejected(self):
        with self.assertRaises(ValueError):
            classify_complexity(0)

    def test_four_levels(self):
        self.assertEqual(len(COMPLEXITY_LEVELS), 4)

    def test_labels(self):
        self.assertEqual(level_label(1), "Simple")
        self.assertEqual(level_label(4), "Expert")
        with self.assertRaises(ValueError):
            level_label(9)


class TestItem(unittest.TestCase):
    def test_level_property(self):
        self.assertEqual(ComplexityItem("a", 7).level, 2)


class TestStratify(unittest.TestCase):
    def _items(self):
        return [
            ComplexityItem("a", 3),
            ComplexityItem("b", 8),
            ComplexityItem("c", 50),
            ComplexityItem("d", 200),
            ComplexityItem("e", 4),
        ]

    def test_stratify(self):
        b = stratify(self._items())
        self.assertEqual([i.id for i in b[1]], ["a", "e"])
        self.assertEqual([i.id for i in b[2]], ["b"])
        self.assertEqual([i.id for i in b[3]], ["c"])
        self.assertEqual([i.id for i in b[4]], ["d"])

    def test_histogram(self):
        self.assertEqual(level_histogram(self._items()), {1: 2, 2: 1, 3: 1, 4: 1})

    def test_proportions(self):
        p = level_proportions(self._items())
        self.assertAlmostEqual(p[1], 2 / 5)
        self.assertAlmostEqual(sum(p.values()), 1.0)

    def test_proportions_empty(self):
        self.assertEqual(level_proportions([]), {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0})


class TestCurriculum(unittest.TestCase):
    def test_order_simple_first(self):
        items = [
            ComplexityItem("z", 150),
            ComplexityItem("a", 3),
            ComplexityItem("m", 3),
            ComplexityItem("k", 8),
        ]
        ordered = curriculum_order(items)
        self.assertEqual([i.id for i in ordered], ["a", "m", "k", "z"])

    def test_deterministic(self):
        items = [ComplexityItem("b", 2), ComplexityItem("a", 2)]
        self.assertEqual(
            [i.id for i in curriculum_order(items)],
            [i.id for i in curriculum_order(items)],
        )


class TestReference(unittest.TestCase):
    def test_reference_counts(self):
        self.assertEqual(REFERENCE_COUNTS[1], 93188)
        self.assertEqual(set(REFERENCE_COUNTS), {1, 2, 3, 4})

    def test_reference_proportions_sum(self):
        self.assertAlmostEqual(sum(reference_proportions().values()), 1.0)
        # Level 1 is the largest bucket in the paper.
        rp = reference_proportions()
        self.assertGreater(rp[1], rp[4])

    def test_distribution_l1_zero_when_matching(self):
        # Build items matching reference proportions in miniature.
        items = (
            [ComplexityItem(f"s{i}", 3) for i in range(93188 // 1000)]
            + [ComplexityItem(f"m{i}", 8) for i in range(78926 // 1000)]
            + [ComplexityItem(f"a{i}", 50) for i in range(69259 // 1000)]
            + [ComplexityItem(f"e{i}", 200) for i in range(1234 // 1000)]
        )
        # Not exactly zero due to rounding, but small.
        self.assertLess(distribution_l1(items), 0.05)

    def test_distribution_l1_range(self):
        items = [ComplexityItem("a", 3)]  # all level 1
        val = distribution_l1(items)
        self.assertGreater(val, 0.0)
        self.assertLessEqual(val, 2.0)


if __name__ == "__main__":
    unittest.main()
