import unittest

from harnesscad.domain.vision.img2cadsvg_edge_extract import (
    gaussian_kernel,
    gaussian_blur,
    sobel_gradients,
    non_max_suppression,
    hysteresis,
    extract_edges,
    edge_pixels,
)


def _step_image(h=9, w=9, left=0.0, right=1.0, split=4):
    return [[left if x <= split else right for x in range(w)] for _ in range(h)]


class KernelTest(unittest.TestCase):
    def test_normalised(self):
        k = gaussian_kernel(1.0)
        self.assertAlmostEqual(sum(k), 1.0)
        # symmetric
        self.assertEqual(k, k[::-1])

    def test_bad_sigma(self):
        with self.assertRaises(ValueError):
            gaussian_kernel(0.0)


class BlurTest(unittest.TestCase):
    def test_preserves_constant(self):
        img = [[5.0] * 4 for _ in range(4)]
        out = gaussian_blur(img, 1.0)
        for row in out:
            for v in row:
                self.assertAlmostEqual(v, 5.0)

    def test_rejects_ragged(self):
        with self.assertRaises(ValueError):
            gaussian_blur([[1.0, 2.0], [3.0]], 1.0)


class GradientTest(unittest.TestCase):
    def test_vertical_edge_horizontal_gradient(self):
        img = _step_image()
        mag, ang = sobel_gradients(img)
        # strongest response near the vertical step column (x=4/5)
        row = mag[4]
        peak = max(range(len(row)), key=lambda x: row[x])
        self.assertIn(peak, (4, 5))


class NMSAndHysteresisTest(unittest.TestCase):
    def test_nms_thins(self):
        img = _step_image()
        mag, ang = sobel_gradients(img)
        thin = non_max_suppression(mag, ang)
        # thinned map has no more nonzero pixels than the magnitude map
        n_mag = sum(1 for r in mag for v in r if v > 0)
        n_thin = sum(1 for r in thin for v in r if v > 0)
        self.assertLessEqual(n_thin, n_mag)

    def test_hysteresis_links(self):
        # strong at (1,1), weak neighbour at (1,2), isolated weak at (3,3)
        nms = [[0.0] * 5 for _ in range(5)]
        nms[1][1] = 1.0
        nms[1][2] = 0.2
        nms[3][3] = 0.2
        out = hysteresis(nms, low=0.1, high=0.5)
        self.assertEqual(out[1][1], 1)
        self.assertEqual(out[1][2], 1)  # linked to strong
        self.assertEqual(out[3][3], 0)  # isolated weak dropped

    def test_bad_thresholds(self):
        with self.assertRaises(ValueError):
            hysteresis([[0.0]], low=0.5, high=0.1)


class PipelineTest(unittest.TestCase):
    def test_extract_edges_finds_step(self):
        img = _step_image(h=11, w=11, split=5)
        edges = extract_edges(img, sigma=1.0, low=0.05, high=0.15)
        px = edge_pixels(edges)
        self.assertTrue(len(px) > 0)
        # edges concentrate near the vertical step (x in {4,5,6})
        near = sum(1 for (_, x) in px if 4 <= x <= 6)
        self.assertEqual(near, len(px))

    def test_deterministic(self):
        img = _step_image()
        a = extract_edges(img)
        b = extract_edges(img)
        self.assertEqual(a, b)

    def test_edge_pixels_empty(self):
        self.assertEqual(edge_pixels([[0, 0], [0, 0]]), [])


if __name__ == "__main__":
    unittest.main()
