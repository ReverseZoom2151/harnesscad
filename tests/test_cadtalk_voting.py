import unittest

from harnesscad.domain.programs.annotate.voting import (
    Evidence,
    accumulate,
    assign_labels,
    vote,
    confidence_matrix_dense,
    DEFAULT_THRESHOLDS,
)


class TestEvidence(unittest.TestCase):
    def test_confidence_product(self):
        e = Evidence(view=0, image=0, block=0, label="a", cdino=0.5, iou=0.4)
        self.assertAlmostEqual(e.confidence(), 0.2)

    def test_ci_override(self):
        e = Evidence(view=0, image=0, block=0, label="a", ci=0.9)
        self.assertEqual(e.confidence(), 0.9)


class TestVote(unittest.TestCase):
    def _ev(self, view, image, block, label, ci):
        return Evidence(view=view, image=image, block=block, label=label, ci=ci)

    def test_single_block_single_label(self):
        ev = [self._ev(0, 0, 0, "body", 0.5)]
        result = vote(ev)
        self.assertEqual(result, {0: "body"})

    def test_argmax_over_labels(self):
        ev = [
            self._ev(0, 0, 0, "body", 0.9),
            self._ev(0, 0, 0, "wing", 0.1),
        ]
        self.assertEqual(vote(ev), {0: "body"})

    def test_accumulation_over_views(self):
        # 'wing' wins because it accumulates across two views though weaker each
        ev = [
            self._ev(0, 0, 0, "body", 0.6),
            self._ev(0, 0, 0, "wing", 0.5),
            self._ev(1, 0, 0, "wing", 0.5),
        ]
        # body: 0.6 ; wing: 0.5+0.5 = 1.0 -> wing
        self.assertEqual(vote(ev), {0: "wing"})

    def test_threshold_filters_weak(self):
        # ci below t1 (0.001) dropped entirely
        ev = [self._ev(0, 0, 0, "body", 0.0005)]
        self.assertEqual(vote(ev), {})

    def test_progressive_threshold_step3(self):
        # sum over one view = 0.015, below t3=0.02 -> dropped
        ev = [self._ev(0, 0, 0, "body", 0.015)]
        # passes t1(0.001) and t2(0.01) but fails t3(0.02)
        self.assertEqual(vote(ev), {})

    def test_tie_broken_by_sort(self):
        ev = [
            self._ev(0, 0, 0, "b", 0.5),
            self._ev(0, 0, 0, "a", 0.5),
        ]
        self.assertEqual(vote(ev), {0: "a"})

    def test_multiple_blocks(self):
        ev = [
            self._ev(0, 0, 0, "body", 0.9),
            self._ev(0, 0, 1, "wing", 0.9),
        ]
        self.assertEqual(vote(ev), {0: "body", 1: "wing"})

    def test_four_images_sum_per_view(self):
        # four images each 0.3 -> view sum 1.2
        ev = [self._ev(0, i, 0, "body", 0.3) for i in range(4)]
        m = accumulate(ev)
        self.assertAlmostEqual(m[(0, "body")], 1.2)


class TestDense(unittest.TestCase):
    def test_dense_matrix(self):
        ev = [
            Evidence(0, 0, 0, "a", ci=0.5),
            Evidence(0, 0, 1, "b", ci=0.7),
        ]
        m = accumulate(ev)
        blocks, labels, rows = confidence_matrix_dense(m)
        self.assertEqual(blocks, [0, 1])
        self.assertEqual(labels, ["a", "b"])
        self.assertAlmostEqual(rows[0][0], 0.5)
        self.assertAlmostEqual(rows[1][1], 0.7)

    def test_default_thresholds(self):
        self.assertEqual(DEFAULT_THRESHOLDS, (0.001, 0.01, 0.02))


if __name__ == "__main__":
    unittest.main()
