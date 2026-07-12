import unittest

from reconstruction.s2cadsig_op_router import spec_for
from reconstruction.s2cadsig_param_decode import (
    DecodeError,
    OrthoCamera,
    PinholeCamera,
    connected_components,
    decode_offset,
    decode_operation,
    decode_stitching_face,
    extract_curve_pixels,
    largest_component,
    lift_curve_to_plane,
    normalize,
    ray_plane_intersect,
    threshold_pixels,
)

H = W = 4


def _flat(rows):
    out = []
    for r in rows:
        out.extend(r)
    return out


class TestVectorsAndCameras(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize((0.0, 0.0, 3.0)), (0.0, 0.0, 1.0))
        with self.assertRaises(DecodeError):
            normalize((0.0, 0.0, 0.0))

    def test_pinhole_unproject_and_ray(self):
        cam = PinholeCamera(fx=2.0, fy=2.0, cx=1.0, cy=1.0)
        self.assertEqual(cam.unproject(1.0, 1.0, 5.0), (0.0, 0.0, 5.0))
        p = cam.unproject(3.0, 1.0, 2.0)
        self.assertAlmostEqual(p[0], 2.0)
        r = cam.ray(1.0, 1.0)
        self.assertEqual(r, (0.0, 0.0, 1.0))
        self.assertEqual(cam.origin(3.0, 3.0), (0.0, 0.0, 0.0))

    def test_ortho(self):
        cam = OrthoCamera(scale=2.0, cx=1.0, cy=1.0)
        self.assertEqual(cam.ray(3.0, 3.0), (0.0, 0.0, 1.0))
        self.assertEqual(cam.origin(3.0, 1.0), (4.0, 0.0, 0.0))
        self.assertEqual(cam.unproject(3.0, 1.0, 7.0), (4.0, 0.0, 7.0))

    def test_ray_plane_intersect(self):
        p = ray_plane_intersect((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 5.0), (0.0, 0.0, 1.0))
        self.assertEqual(p, (0.0, 0.0, 5.0))
        with self.assertRaises(DecodeError):
            ray_plane_intersect((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 5.0), (0.0, 0.0, 1.0))


class TestComponents(unittest.TestCase):
    def test_threshold_pixels(self):
        m = _flat([[0, 1, 0, 0]] * 4)
        px = threshold_pixels(m, H, W, 0.5)
        self.assertEqual(px, [(0, 1), (1, 1), (2, 1), (3, 1)])

    def test_threshold_size_check(self):
        with self.assertRaises(DecodeError):
            threshold_pixels([0.0], H, W, 0.5)

    def test_components_sorted_by_size(self):
        px = [(0, 0), (0, 1), (1, 0), (3, 3)]
        comps = connected_components(px)
        self.assertEqual(len(comps), 2)
        self.assertEqual(len(comps[0]), 3)
        self.assertEqual(comps[1], [(3, 3)])

    def test_largest_component_ignores_stray(self):
        m = _flat([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
        ])
        comp = largest_component(m, H, W, 0.5)
        self.assertEqual(comp, [(0, 0), (0, 1), (1, 0), (1, 1)])

    def test_largest_component_empty(self):
        with self.assertRaises(DecodeError):
            largest_component([0.0] * 16, H, W, 0.5)


def _face_setup():
    heat = _flat([
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],  # stray pixel, must be ignored
    ])
    normals = [(0.0, 0.0, 1.0)] * (H * W)
    # the stray pixel carries a bogus normal that must not leak into the plane
    normals[15] = (1.0, 0.0, 0.0)
    depth = [10.0] * (H * W)
    return heat, normals, depth


class TestStitchingFace(unittest.TestCase):
    def test_plane_from_heatmap(self):
        heat, normals, depth = _face_setup()
        cam = OrthoCamera()
        face = decode_stitching_face(heat, normals, depth, H, W, cam, threshold=0.5)
        self.assertEqual(face.area_px, 4)
        self.assertEqual(face.normal, (0.0, 0.0, 1.0))
        self.assertAlmostEqual(face.centroid_uv[0], 0.5)
        self.assertAlmostEqual(face.centroid_uv[1], 0.5)
        self.assertAlmostEqual(face.mean_depth, 10.0)
        self.assertEqual(face.point, (0.5, 0.5, 10.0))
        self.assertAlmostEqual(face.peak_value, 1.0)
        self.assertAlmostEqual(face.signed_distance((0.0, 0.0, 12.0)), 2.0)

    def test_size_mismatch(self):
        heat, normals, depth = _face_setup()
        with self.assertRaises(DecodeError):
            decode_stitching_face(heat, normals[:3], depth, H, W, OrthoCamera())

    def test_zero_normal_is_degenerate(self):
        heat = [1.0] * (H * W)
        normals = [(0.0, 0.0, 0.0)] * (H * W)
        with self.assertRaises(DecodeError):
            decode_stitching_face(heat, normals, [1.0] * (H * W), H, W, OrthoCamera())


class TestCurve(unittest.TestCase):
    def test_extract_masks_by_stroke(self):
        curve = _flat([[0, 1, 1, 0]] + [[0, 0, 0, 0]] * 3)
        stroke = [0.0] * (H * W)
        stroke[2] = 1.0  # pixel (0,2) is a user-stroke pixel -> excluded
        px = extract_curve_pixels(curve, H, W, 0.5, user_stroke=stroke)
        self.assertEqual(px, [(0, 1)])
        self.assertEqual(len(extract_curve_pixels(curve, H, W, 0.5)), 2)

    def test_lift_to_plane(self):
        heat, normals, depth = _face_setup()
        cam = OrthoCamera()
        face = decode_stitching_face(heat, normals, depth, H, W, cam)
        pts = lift_curve_to_plane([(0, 0), (1, 2)], face, cam)
        self.assertEqual(pts[0], (0.0, 0.0, 10.0))
        self.assertEqual(pts[1], (2.0, 1.0, 10.0))

    def test_lift_with_pinhole_stays_on_plane(self):
        cam = PinholeCamera(fx=4.0, fy=4.0, cx=1.5, cy=1.5)
        heat, normals, depth = _face_setup()
        face = decode_stitching_face(heat, normals, depth, H, W, cam)
        for p in lift_curve_to_plane([(0, 0), (3, 3), (2, 1)], face, cam):
            self.assertAlmostEqual(face.signed_distance(p), 0.0, places=9)


class TestOffset(unittest.TestCase):
    def test_aggregate(self):
        n = H * W
        dist = [0.0] * n
        direction = [(0.0, 0.0, 0.0)] * n
        sign = [0.0] * n
        for i in (1, 2):
            dist[i] = 2.0
            direction[i] = (0.0, 0.0, 3.0)
            sign[i] = -1.0
        off = decode_offset(dist, direction, sign, [(0, 1), (0, 2)], W)
        self.assertAlmostEqual(off.distance, 2.0)
        self.assertEqual(off.direction, (0.0, 0.0, 1.0))
        self.assertEqual(off.sign, -1)
        self.assertEqual(off.vector, (0.0, 0.0, -2.0))

    def test_majority_sign_ties_positive(self):
        n = H * W
        dist = [1.0] * n
        direction = [(1.0, 0.0, 0.0)] * n
        sign = [0.0] * n
        sign[0] = 1.0
        sign[1] = -1.0
        off = decode_offset(dist, direction, sign, [(0, 0), (0, 1)], W)
        self.assertEqual(off.sign, 1)

    def test_empty_pixels(self):
        with self.assertRaises(DecodeError):
            decode_offset([], [], [], [], W)


class TestDecodeOperation(unittest.TestCase):
    def _maps(self):
        heat, normals, depth = _face_setup()
        n = H * W
        curve = [0.0] * n
        curve[4] = 1.0
        curve[5] = 1.0
        return {
            "face_heatmap": heat,
            "context_normal": normals,
            "context_depth": depth,
            "offset_curve": curve,
            "base_curve": curve,
            "offset_distance": [3.0] * n,
            "offset_direction": [(0.0, 0.0, 1.0)] * n,
            "offset_sign": [1.0] * n,
        }

    def test_extrusion(self):
        params = decode_operation(
            spec_for("extrusion"), self._maps(), H, W, OrthoCamera()
        )
        self.assertEqual(params.op_name, "extrusion")
        self.assertEqual(params.curve_name, "offset_curve")
        self.assertEqual(len(params.curve_points), 2)
        self.assertEqual(params.offset_vector, (0.0, 0.0, 3.0))
        self.assertEqual(params.summary()["offset_sign"], 1)

    def test_bevel_has_no_offset(self):
        maps = self._maps()
        del maps["offset_distance"]
        params = decode_operation(spec_for("bevel"), maps, H, W, OrthoCamera())
        self.assertIsNone(params.offset)
        self.assertIsNone(params.offset_vector)

    def test_missing_map(self):
        maps = self._maps()
        del maps["context_depth"]
        with self.assertRaises(DecodeError):
            decode_operation(spec_for("bevel"), maps, H, W, OrthoCamera())

    def test_missing_offset_map_for_offset_op(self):
        maps = self._maps()
        del maps["offset_sign"]
        with self.assertRaises(DecodeError):
            decode_operation(spec_for("extrusion"), maps, H, W, OrthoCamera())

    def test_empty_curve(self):
        maps = self._maps()
        maps["offset_curve"] = [0.0] * (H * W)
        with self.assertRaises(DecodeError):
            decode_operation(spec_for("extrusion"), maps, H, W, OrthoCamera())

    def test_deterministic(self):
        a = decode_operation(spec_for("sweep"), dict(self._maps(), profile_curve=self._maps()["base_curve"]), H, W, OrthoCamera())
        b = decode_operation(spec_for("sweep"), dict(self._maps(), profile_curve=self._maps()["base_curve"]), H, W, OrthoCamera())
        self.assertEqual(a.curve_points, b.curve_points)
        self.assertEqual(a.face.point, b.face.point)


if __name__ == "__main__":
    unittest.main()
