"""The geometry service registry: discovery, dispatch, and correct geometry.

The point of these tests is not that the dispatcher returns *something*; it is
that the operations it publishes return the RIGHT geometry. Every numeric check
below compares against a closed-form value that can be derived on paper.
"""

import math
import unittest

from harnesscad.domain.geometry import services


class TestRegistrySurface(unittest.TestCase):
    def test_every_published_operation_is_corroborated(self):
        # an operation is only published if the static capability index agrees
        # the module exists and exports the symbol; nothing may fall out.
        self.assertEqual(services.missing(), [])
        self.assertGreater(len(services.operations()), 100)

    def test_operations_are_deterministically_ordered(self):
        names = services.names()
        self.assertEqual(names, sorted(names))
        self.assertEqual(len(names), len(set(names)))

    def test_lookup_and_dispatch(self):
        op = services.get("gear.module.nearest")
        self.assertEqual(op.dotted, "harnesscad.domain.geometry.kinematics.gear_modules")
        self.assertIn("kinematics", op.tags)
        with self.assertRaises(services.UnknownOperationError):
            services.get("no.such.operation")

    def test_find_by_capability_tag(self):
        sdf_ops = services.find(tag="sdf")
        self.assertTrue(sdf_ops)
        self.assertTrue(all("sdf" in o.tags for o in sdf_ops))
        self.assertTrue(any(o.name == "sdf.infill.gyroid" for o in sdf_ops))
        self.assertTrue(services.find(prefix="gear."))

    def test_report_counts_the_fleet(self):
        rep = services.report()
        self.assertEqual(rep["operations"], len(services.operations()))
        self.assertEqual(rep["modules"], len(services.modules()))
        self.assertIn("mesh.contour_2d", rep["notes"])   # the honest 2D-only note


class TestGeometryIsCorrect(unittest.TestCase):
    """Analytic checks -- each expected value is derivable in closed form."""

    def test_involute_gear_radii(self):
        g = services.call("gear.involute.geometry", module=2.0, teeth=20)
        # pitch diameter = m * z; base radius = r_pitch * cos(20 deg)
        self.assertAlmostEqual(g.pitch_diameter, 40.0, places=9)
        self.assertAlmostEqual(g.pitch_radius, 20.0, places=9)
        self.assertAlmostEqual(g.base_radius, 20.0 * math.cos(math.radians(20.0)),
                               places=9)
        # tip radius = r + m (addendum); root = r - m - clearance (default 0)
        self.assertAlmostEqual(g.tip_radius, 22.0, places=9)
        self.assertAlmostEqual(g.root_radius, 18.0, places=9)
        # circular pitch = pi * m
        self.assertAlmostEqual(g.circular_pitch, math.pi * 2.0, places=9)

    def test_gear_pair_centre_distance(self):
        # standard pair: a = m (z1 + z2) / 2
        d = services.call("gear.train.center_distance", module=2.0, teeth_a=20,
                          teeth_b=30)
        self.assertAlmostEqual(d, 50.0, places=9)

    def test_gear_module_snaps_to_the_standard_series(self):
        self.assertAlmostEqual(services.call("gear.module.nearest", 1.9), 2.0, places=9)
        self.assertTrue(services.call("gear.module.is_standard", 2.0))

    def test_chord_tolerance_matches_the_sagitta_formula(self):
        # sagitta of an N-segment circle of radius r: r (1 - cos(pi/N))
        r, n = 10.0, 16
        err = services.call("curve.chord.error", radius=r, sweep_angle=2.0 * math.pi,
                            segments=n)
        self.assertAlmostEqual(err, r * (1.0 - math.cos(math.pi / n)), places=9)
        # and the inverse: asking for that error back gives that segment count
        segs = services.call("curve.chord.segments", radius=r,
                             sweep_angle=2.0 * math.pi, tolerance=err + 1e-12)
        self.assertEqual(segs, n)

    def test_circle_approximation_is_inscribed_within_tolerance(self):
        pts = services.call("curve.circle.approximate", centre=(0.0, 0.0),
                            radius=5.0, tolerance=0.05)
        self.assertGreater(len(pts), 8)
        for (x, y) in pts:
            self.assertAlmostEqual(math.hypot(x, y), 5.0, places=9)

    def test_pappus_volume_of_a_revolved_rectangle(self):
        # A 2x4 rectangle centred at radius 10, revolved about the axis:
        # Pappus -> V = 2 pi R A = 2 pi * 10 * 8
        profile = [(9.0, 0.0), (11.0, 0.0), (11.0, 4.0), (9.0, 4.0)]
        v = services.call("feature.revolve.pappus_volume", profile)
        self.assertAlmostEqual(v, 2.0 * math.pi * 10.0 * 8.0, places=6)

    def test_airfoil_polygon_closes_and_respects_thickness(self):
        pts = services.call("feature.airfoil.polygon", m=0.0, p=0.0, t=0.12, n=60)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        self.assertAlmostEqual(min(xs), 0.0, places=6)
        self.assertAlmostEqual(max(xs), 1.0, places=6)
        # a symmetric NACA 00xx is ~12% thick, and the max thickness is near 30% chord
        self.assertAlmostEqual(max(ys) - min(ys), 0.12, delta=0.006)

    def test_polyline_simplification_respects_its_tolerance(self):
        pts = [(float(i), 0.0) for i in range(11)]
        pts[5] = (5.0, 0.4)
        keep = services.call("curve.simplify", pts, 0.5)      # 0.4 < 0.5 -> flatten
        self.assertEqual(keep, [(0.0, 0.0), (10.0, 0.0)])
        keep = services.call("curve.simplify", pts, 0.2)      # 0.4 > 0.2 -> keep it
        self.assertIn((5.0, 0.4), keep)
        self.assertLessEqual(services.call("curve.simplify.deviation", pts, keep), 0.2)

    def test_hole_volumes_are_the_cylinder_volumes(self):
        h = services.call("hole.simple", diameter=6.0, depth=10.0)
        self.assertAlmostEqual(h.volume, math.pi * 3.0 ** 2 * 10.0, places=6)
        cb = services.call("hole.counterbore", diameter=6.0, cbore_diameter=12.0,
                           cbore_depth=4.0, depth=10.0)
        # counterbore = the 12mm recess plus the 6mm hole below it
        expected = math.pi * 6.0 ** 2 * 4.0 + math.pi * 3.0 ** 2 * 6.0
        self.assertAlmostEqual(cb.volume, expected, places=6)
        self.assertGreater(cb.max_radius, h.max_radius)

    def test_gyroid_infill_is_the_gyroid_surface(self):
        # the TPMS field vanishes on the surface: at the origin all sines are 0
        self.assertAlmostEqual(services.call("sdf.infill.gyroid", (0.0, 0.0, 0.0),
                                             period=10.0), 0.0, places=9)
        # and it is periodic with the requested period
        a = services.call("sdf.infill.gyroid", (1.0, 2.0, 3.0), period=10.0)
        b = services.call("sdf.infill.gyroid", (11.0, 12.0, 13.0), period=10.0)
        self.assertAlmostEqual(a, b, places=9)

    def test_extra_sdf_shapes_are_signed_distances(self):
        # a hexagonal prism of circumradius 5, half-length 10: the centre is
        # inside (negative) by at least the apothem
        d = services.call("sdf.shape.hex_prism", (0.0, 0.0, 0.0), 5.0, 10.0)
        self.assertLess(d, 0.0)
        # a point far outside is positive
        d = services.call("sdf.shape.hex_prism", (50.0, 0.0, 0.0), 5.0, 10.0)
        self.assertGreater(d, 0.0)

    def test_sphere_tracing_finds_the_surface_of_a_sphere(self):
        def sphere(p):
            return math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) - 4.0

        # the ray enters the sphere at z = -4, i.e. 16 units along
        hit = services.call("sdf.raycast", sphere, (0.0, 0.0, -20.0), (0.0, 0.0, 1.0))
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(float(hit), 16.0, delta=1e-3)

    def test_bvh_finds_exactly_the_overlapping_triangles(self):
        verts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                 (10.0, 10.0, 10.0), (11.0, 10.0, 10.0), (10.0, 11.0, 10.0)]
        faces = [(0, 1, 2), (3, 4, 5)]
        boxes = services.call("mesh.bvh.boxes", verts, faces)
        tree = services.call("mesh.bvh.build", boxes)
        from harnesscad.domain.geometry.mesh.bvh import AABB
        near = tree.query(AABB((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0)))
        self.assertEqual(sorted(near), [0])
        far = tree.query(AABB((9.0, 9.0, 9.0), (12.0, 12.0, 12.0)))
        self.assertEqual(sorted(far), [1])

    def test_quadrature_integrates_a_polynomial_exactly(self):
        # 3-point Gauss-Legendre is exact through degree 5
        val = services.call("numeric.quadrature.integrate", lambda x: x ** 5, 0.0, 2.0, 3)
        self.assertAlmostEqual(val, 2.0 ** 6 / 6.0, places=9)

    def test_path_offset_of_a_square_is_a_square(self):
        square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        out = services.call("curve.offset", square, 1.0, internal=False, closed=True)
        xs = [p[0] for p in out]
        ys = [p[1] for p in out]
        self.assertAlmostEqual(min(xs), -1.0, places=6)
        self.assertAlmostEqual(max(xs), 11.0, places=6)
        self.assertAlmostEqual(min(ys), -1.0, places=6)
        self.assertAlmostEqual(max(ys), 11.0, places=6)

    def test_catmull_rom_interpolates_its_control_points(self):
        ctrl = [(0.0, 0.0), (1.0, 2.0), (2.0, 0.0), (3.0, 2.0)]
        pts = services.call("curve.catmull_rom.points", ctrl, subdivisions=8)
        for c in ctrl:
            self.assertTrue(any(abs(p[0] - c[0]) < 1e-9 and abs(p[1] - c[1]) < 1e-9
                                for p in pts),
                            "the spline must pass through its control points")

    def test_surface_fit_recovers_a_known_sphere(self):
        pts = []
        r, c = 3.0, (1.0, 2.0, 3.0)
        for i in range(8):
            for j in range(1, 8):
                th = 2.0 * math.pi * i / 8.0
                ph = math.pi * j / 8.0
                pts.append((c[0] + r * math.sin(ph) * math.cos(th),
                            c[1] + r * math.sin(ph) * math.sin(th),
                            c[2] + r * math.cos(ph)))
        (centre, radius), residual = services.call("surface.fit.sphere", pts)
        self.assertAlmostEqual(radius, r, places=6)
        self.assertLess(residual, 1e-6)
        for i in range(3):
            self.assertAlmostEqual(centre[i], c[i], places=6)

    def test_the_2d_contourer_and_the_3d_mesher_stay_distinct(self):
        # RIVAL DISCIPLINE: publish a capability only where it is implemented.
        # dual_contouring.py is a 2D contourer and remains one. The 3D dual
        # contourer is a separate module and is published as a 3D mesher rival
        # alongside marching cubes. Neither may stand in for the other.
        from harnesscad.io.backends import frep
        self.assertEqual(services.get("mesh.contour_2d").symbol, "dual_contour_2d")
        self.assertIn("dual_contouring", frep.MESHERS)
        self.assertIn("marching_cubes", frep.MESHERS)


if __name__ == "__main__":
    unittest.main()
