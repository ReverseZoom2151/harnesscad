import unittest

from harnesscad.domain.assembly.interference import AABB, check_interference


class TestAABB(unittest.TestCase):
    def test_center(self):
        self.assertEqual(AABB(0, 0, 0, 2, 4, 6).center, (1.0, 2.0, 3.0))

    def test_invalid_box_rejected(self):
        with self.assertRaises(ValueError):
            AABB(5, 0, 0, 1, 1, 1)


class TestInterference(unittest.TestCase):
    def test_disjoint_no_clip(self):
        boxes = {"a": AABB(0, 0, 0, 10, 10, 10), "b": AABB(20, 0, 0, 30, 10, 10)}
        res = check_interference(boxes)
        self.assertTrue(res.passed)
        self.assertEqual(res.clips, [])

    def test_shared_face_not_a_clip(self):
        boxes = {"a": AABB(0, 0, 0, 10, 10, 10), "b": AABB(10, 0, 0, 20, 10, 10)}
        res = check_interference(boxes)
        self.assertTrue(res.passed)

    def test_overlap_detected_with_volume(self):
        boxes = {"a": AABB(0, 0, 0, 10, 10, 10), "b": AABB(8, 0, 0, 20, 10, 10)}
        res = check_interference(boxes)
        self.assertFalse(res.passed)
        self.assertEqual(len(res.clips), 1)
        clip = res.clips[0]
        self.assertEqual(clip.label_a, "a")
        self.assertEqual(clip.label_b, "b")
        # overlap is 2 x 10 x 10 = 200
        self.assertAlmostEqual(clip.volume, 200.0)
        self.assertEqual(clip.suggest_axis, "x")

    def test_suggested_shift_clears_overlap(self):
        boxes = {"a": AABB(0, 0, 0, 10, 10, 10), "b": AABB(8, 0, 0, 20, 10, 10)}
        clip = check_interference(boxes, min_clearance=1.0).clips[0]
        # Move A negative so its max ends 1mm below b.min=8 -> a.max should be 7.
        # shift = b.min - clearance - a.max = 8 - 1 - 10 = -3
        self.assertAlmostEqual(clip.suggest_shift, -3.0)
        # Apply the shift and confirm clearance.
        a = boxes["a"]
        new_max = a.xmax + clip.suggest_shift
        self.assertLessEqual(new_max, boxes["b"].xmin - clip.clearance + 1e-9)

    def test_cheapest_axis_chosen(self):
        # Large overlap in x/y, tiny in z -> z is cheapest to clear.
        boxes = {"a": AABB(0, 0, 0, 10, 10, 10), "b": AABB(0, 0, 9, 10, 10, 20)}
        clip = check_interference(boxes).clips[0]
        self.assertEqual(clip.suggest_axis, "z")

    def test_skip_labels(self):
        boxes = {"belt": AABB(0, 0, 0, 10, 10, 10), "plate": AABB(8, 0, 0, 20, 10, 10)}
        res = check_interference(boxes, skip_labels={"belt"})
        self.assertTrue(res.passed)

    def test_min_volume_threshold(self):
        # Overlap volume 1 x 10 x 10 = 100; threshold above it -> no clip.
        boxes = {"a": AABB(0, 0, 0, 10, 10, 10), "b": AABB(9, 0, 0, 20, 10, 10)}
        res = check_interference(boxes, min_volume=200.0)
        self.assertTrue(res.passed)

    def test_deterministic_ordering(self):
        boxes = {
            "z": AABB(0, 0, 0, 10, 10, 10),
            "a": AABB(8, 0, 0, 20, 10, 10),
            "m": AABB(5, 0, 0, 15, 10, 10),
        }
        clips = check_interference(boxes).clips
        pairs = [(c.label_a, c.label_b) for c in clips]
        self.assertEqual(pairs, sorted(pairs))


if __name__ == "__main__":
    unittest.main()
