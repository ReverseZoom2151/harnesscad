import unittest

from harnesscad.domain.programs.annotate.label_transfer import (
    majority_label,
    multi_labels,
    iou_label,
)


class TestMajorityLabel(unittest.TestCase):
    def test_simple_majority(self):
        block_points = {0: ["p1", "p2", "p3"]}
        point_labels = {"p1": "body", "p2": "body", "p3": "wing"}
        self.assertEqual(majority_label(block_points, point_labels), {0: "body"})

    def test_unlabelled_points_ignored(self):
        block_points = {0: ["p1", "p2", "pX"]}
        point_labels = {"p1": "wing", "p2": "wing"}
        self.assertEqual(majority_label(block_points, point_labels), {0: "wing"})

    def test_empty_block_omitted(self):
        block_points = {0: ["pX"]}
        point_labels = {"p1": "body"}
        self.assertEqual(majority_label(block_points, point_labels), {})

    def test_tie_sort_order(self):
        block_points = {0: ["p1", "p2"]}
        point_labels = {"p1": "wing", "p2": "body"}
        # tie -> alphabetical 'body'
        self.assertEqual(majority_label(block_points, point_labels), {0: "body"})


class TestMultiLabels(unittest.TestCase):
    def test_keeps_spanning_labels(self):
        # block spans wing + engine roughly equally
        block_points = {0: ["a", "b", "c", "d"]}
        point_labels = {"a": "wing", "b": "wing", "c": "engine", "d": "engine"}
        result = multi_labels(block_points, point_labels, share=0.25)
        self.assertEqual(result[0], {"wing", "engine"})

    def test_share_threshold_drops_minor(self):
        block_points = {0: ["a", "b", "c", "d"]}
        point_labels = {"a": "body", "b": "body", "c": "body", "d": "tip"}
        # tip share = 0.25, body = 0.75
        result = multi_labels(block_points, point_labels, share=0.5)
        self.assertEqual(result[0], {"body"})

    def test_never_empty_when_points(self):
        block_points = {0: ["a", "b", "c", "d"]}
        point_labels = {"a": "w", "b": "x", "c": "y", "d": "z"}
        # all share 0.25 < 0.5, so keep majority (tie -> 'w')
        result = multi_labels(block_points, point_labels, share=0.5)
        self.assertEqual(result[0], {"w"})


class TestIoULabel(unittest.TestCase):
    def test_iou_assignment(self):
        # block 0 overlaps entirely with 'body' point set
        block_points = {0: ["a", "b"]}
        point_labels = {"a": "body", "b": "body", "c": "wing"}
        self.assertEqual(iou_label(block_points, point_labels), {0: "body"})

    def test_iou_picks_higher_overlap(self):
        block_points = {0: ["a", "b", "c"]}
        point_labels = {
            "a": "body", "b": "body", "c": "wing",
            "d": "wing", "e": "wing",
        }
        # body: |{a,b}| / |{a,b,c}| = 2/3 ; wing: |{c}|/|{a,b,c,d,e}| = 1/5
        self.assertEqual(iou_label(block_points, point_labels), {0: "body"})

    def test_no_overlap_omitted(self):
        block_points = {0: ["z"]}
        point_labels = {"a": "body"}
        self.assertEqual(iou_label(block_points, point_labels), {})


if __name__ == "__main__":
    unittest.main()
