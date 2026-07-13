"""Tests for backends.cadhub_render_request."""

import unittest

from harnesscad.io.backends.cadhub_render_request import (
    Camera,
    Vec3,
    Viewport,
    cache_key,
    camera_args,
    canonical_request,
    device_size,
    quantize_camera,
    request_key,
    round_to,
    same_render,
)


class TestRounding(unittest.TestCase):
    def test_half_up_not_bankers(self):
        self.assertEqual(round_to(0.25, 1), 0.3)
        self.assertEqual(round_to(0.35, 1), 0.4)
        self.assertEqual(round_to(1.04, 1), 1)
        self.assertEqual(round_to(-0.25, 1), -0.3)

    def test_integral_result_is_int(self):
        self.assertEqual(round_to(40.02, 1), 40)
        self.assertIsInstance(round_to(40.02, 1), int)

    def test_camera_quantisation(self):
        cam = Camera(position=Vec3(40.04, 40.06, 39.999), rotation=Vec3(55.01, 0, 25.0), dist=200.049)
        snapped = quantize_camera(cam)
        self.assertEqual(snapped.position.as_list(), [40, 40.1, 40])
        self.assertEqual(snapped.rotation.as_list(), [55, 0, 25])
        self.assertEqual(snapped.dist, 200)


class TestViewport(unittest.TestCase):
    def test_pixel_ratio(self):
        self.assertEqual(device_size(Viewport(500, 300), 2.0), (1000, 600))
        self.assertEqual(device_size(Viewport(500, 300)), (500, 300))
        self.assertEqual(device_size(Viewport(333, 111), 1.5), (500, 167))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            device_size(Viewport(0, 10))
        with self.assertRaises(ValueError):
            device_size(Viewport(10, 10), 0)


class TestCameraArgs(unittest.TestCase):
    def test_openscad_argument(self):
        cam = Camera(position=Vec3(40, 40, 40), rotation=Vec3(55, 0, 25), dist=200)
        self.assertEqual(camera_args(cam), "40,40,40,55,0,25,200")

    def test_rounded_argument(self):
        cam = Camera(position=Vec3(40.049, 0, 0), rotation=Vec3(0, 0, 0), dist=12.34)
        self.assertEqual(camera_args(cam), "40,0,0,0,0,0,12.3")


class TestCanonicalRequest(unittest.TestCase):
    def setUp(self):
        self.cam = Camera(position=Vec3(40, 40, 40), rotation=Vec3(55, 0, 25), dist=200)
        self.vp = Viewport(500, 400)

    def test_openscad_full(self):
        req = canonical_request(
            "openscad",
            "cube(1);",
            camera=self.cam,
            viewport=self.vp,
            pixel_ratio=2.0,
            parameters={"b": 2, "a": 1},
            view_all=True,
        )
        self.assertEqual(req["size"], {"x": 1000, "y": 800})
        self.assertEqual(req["camera"]["dist"], 200)
        self.assertTrue(req["viewAll"])
        self.assertEqual(list(req["parameters"]), ["a", "b"])  # key-sorted

    def test_curv_drops_camera_and_params(self):
        req = canonical_request(
            "curv", "cube 1", camera=self.cam, viewport=self.vp, parameters={"a": 1}
        )
        self.assertNotIn("camera", req)
        self.assertNotIn("parameters", req)  # curv has no parameter support
        self.assertEqual(req["size"], {"x": 500, "y": 400})

    def test_jscad_drops_size(self):
        req = canonical_request("jscad", "// x", viewport=self.vp, parameters={"a": 1})
        self.assertNotIn("size", req)
        self.assertEqual(req["parameters"], {"a": 1})


class TestCacheKey(unittest.TestCase):
    def setUp(self):
        self.vp = Viewport(500, 400)

    def _key(self, cam):
        return request_key("openscad", "cube(1);", camera=cam, viewport=self.vp)

    def test_jitter_within_grid_hits_same_key(self):
        a = Camera(position=Vec3(40.01, 40.02, 40.0), rotation=Vec3(55, 0, 25), dist=200.03)
        b = Camera(position=Vec3(40.04, 39.98, 40.01), rotation=Vec3(55.02, 0, 25), dist=199.99)
        self.assertEqual(self._key(a), self._key(b))

    def test_real_move_changes_key(self):
        a = Camera(position=Vec3(40, 40, 40), rotation=Vec3(55, 0, 25), dist=200)
        b = Camera(position=Vec3(45, 40, 40), rotation=Vec3(55, 0, 25), dist=200)
        self.assertNotEqual(self._key(a), self._key(b))

    def test_source_and_language_matter(self):
        self.assertNotEqual(
            request_key("openscad", "cube(1);"), request_key("openscad", "cube(2);")
        )
        self.assertNotEqual(request_key("curv", "x"), request_key("openscad", "x"))

    def test_same_render_helper_and_stability(self):
        r1 = canonical_request("openscad", "cube(1);", parameters={"a": 1, "b": 2})
        r2 = canonical_request("openscad", "cube(1);", parameters={"b": 2, "a": 1})
        self.assertTrue(same_render(r1, r2))
        self.assertEqual(cache_key(r1), cache_key(r1))
        self.assertEqual(len(cache_key(r1)), 64)


if __name__ == "__main__":
    unittest.main()
