"""Metric booleans run on manifold3d, never on OCCT.

The enforcing test for the policy in
:mod:`harnesscad.eval.verifiers.metric_booleans`. cadgenbench documented a hang
where its scorer wedged indefinitely on a raw OCCT BREP boolean over
interface-overlay geometry, and fixed it structurally: every metric-side boolean
goes through the manifold3d mesh kernel, pinned by a test so an OCCT boolean
cannot silently return.

This is NOT a hypothetical guard. Both ``interference._common_volume`` and
``access._swept_common_volume`` called ``BRepAlgoAPI_Common`` on exactly the
geometry the hang was reported for (overlapping placed parts; a tool corridor
swept flush against a face). They now call metric_booleans instead, and
:func:`TestNoOcctBooleansOnTheMetricPath` is what keeps them there.

Four things are pinned:
  1. no module on the metric path may reach an OCCT boolean (the policy);
  2. the scanner that enforces (1) actually detects an offence, and is not
     fooled by prose -- a test that cannot fail is not a test;
  3. the manifold3d booleans compute the right volumes on known geometry;
  4. sub-epsilon overlap is classified as numerical noise, not as a defect.
"""

import math
import unittest

from harnesscad.eval.verifiers import access, interference
from harnesscad.eval.verifiers.metric_booleans import (
    METRIC_BOOLEAN_PATH,
    OCCT_BOOLEAN_MARKERS,
    OVERLAP_NOISE_EPSILON,
    classify_overlap,
    common_volume,
    intersection_volume,
    manifold_available,
    mesh_to_manifold,
    occt_boolean_offenders,
    shape_to_manifold,
    shape_to_mesh,
    swept_cylinder_common_volume,
)


def _cadquery_available() -> bool:
    try:
        import cadquery  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


HAVE_CQ = _cadquery_available()
HAVE_M3D = manifold_available()


# ======================================================================
# 1. The policy
# ======================================================================
class TestNoOcctBooleansOnTheMetricPath(unittest.TestCase):
    def test_no_module_on_the_metric_path_reaches_an_occt_boolean(self):
        offenders = occt_boolean_offenders()
        self.assertEqual(
            offenders, {},
            "OCCT booleans have re-entered the metric path. OCCT booleans hang "
            "on the near-tangent overlay geometry these verifiers exist to "
            "measure; route the boolean through metric_booleans (manifold3d) "
            "instead. Offenders: %r" % offenders)

    def test_the_known_boolean_call_sites_are_on_the_governed_path(self):
        # If someone moves the boolean somewhere else, the policy must follow it.
        for module in (interference, access):
            self.assertIn(module.__name__, METRIC_BOOLEAN_PATH)

    def test_interference_and_access_still_expose_their_volume_helpers(self):
        # The migration must not have quietly deleted the exact path; these are
        # what the verifiers call to get a real (non-bbox) overlap.
        self.assertTrue(callable(interference._common_volume))
        self.assertTrue(callable(access._swept_common_volume))


# ======================================================================
# 2. The scanner itself (a test that cannot fail is not a test)
# ======================================================================
class TestScannerActuallyDetects(unittest.TestCase):
    def test_scanner_flags_a_module_that_imports_an_occt_boolean(self):
        # A module that really does import BRepAlgoAPI must be caught. eval/
        # hardcorpus/occt.py is off the metric path (it is allowed to use OCCT),
        # which makes it a safe positive control.
        offenders = occt_boolean_offenders(["harnesscad.eval.hardcorpus.occt"])
        self.assertTrue(
            offenders,
            "the scanner failed to flag a module that does import an OCCT "
            "boolean; the policy test would pass vacuously")

    def test_scanner_is_not_fooled_by_prose(self):
        # metric_booleans' own docstring discusses BRepAlgoAPI at length. A
        # naive substring scan would flag it and the policy would be unusable.
        self.assertNotIn("harnesscad.eval.verifiers.metric_booleans",
                         occt_boolean_offenders())

    def test_scanner_fails_closed_on_an_unreadable_module(self):
        offenders = occt_boolean_offenders(["harnesscad.no.such.module"])
        self.assertIn("harnesscad.no.such.module", offenders)

    def test_markers_cover_the_real_routes_to_an_occt_boolean(self):
        self.assertIn("BRepAlgoAPI", OCCT_BOOLEAN_MARKERS)  # Common/Cut/Fuse
        self.assertIn("build123d", OCCT_BOOLEAN_MARKERS)    # operator booleans


# ======================================================================
# 3. Sub-epsilon overlap is numerical noise
# ======================================================================
class TestOverlapClassification(unittest.TestCase):
    def test_unknown_is_not_zero(self):
        # "not measurable" must never read as "no overlap".
        self.assertEqual(classify_overlap(None), "unknown")

    def test_zero_is_none(self):
        self.assertEqual(classify_overlap(0.0), "none")

    def test_sub_epsilon_is_noise(self):
        self.assertEqual(classify_overlap(1e-9), "noise")
        self.assertEqual(classify_overlap(OVERLAP_NOISE_EPSILON), "noise")

    def test_above_epsilon_is_a_clash(self):
        self.assertEqual(classify_overlap(OVERLAP_NOISE_EPSILON * 1.001), "clash")

    def test_a_stricter_epsilon_is_honoured(self):
        self.assertEqual(classify_overlap(0.5, epsilon=0.1), "clash")
        self.assertEqual(classify_overlap(0.05, epsilon=0.1), "noise")


# ======================================================================
# 4. Degradation: no kernel -> None, never a wrong number
# ======================================================================
class TestDegradesCleanly(unittest.TestCase):
    def test_none_inputs_yield_none(self):
        self.assertIsNone(intersection_volume(None, None))
        self.assertIsNone(common_volume(None, None))
        self.assertIsNone(mesh_to_manifold(None))
        self.assertIsNone(shape_to_mesh(None))
        self.assertIsNone(shape_to_manifold(None))

    def test_degenerate_tool_yields_none(self):
        self.assertIsNone(
            swept_cylinder_common_volume((0, 0, 0), (0, 0, 1), 0.0, 10.0, None))
        self.assertIsNone(
            swept_cylinder_common_volume((0, 0, 0), (0, 0, 1), 1.0, 0.0, None))


# ======================================================================
# 5. Real kernel properties
# ======================================================================
@unittest.skipUnless(HAVE_M3D, "manifold3d not installed")
class TestManifoldBooleanVolumes(unittest.TestCase):
    def _cube(self, size=10.0, origin=(0.0, 0.0, 0.0)):
        import manifold3d as m3d
        return m3d.Manifold.cube([size, size, size], center=False).translate(list(origin))

    def test_overlapping_cubes_intersect_in_the_shared_cube(self):
        # Two 10mm cubes offset 5mm on each axis share a 5mm cube.
        vol = intersection_volume(self._cube(), self._cube(origin=(5, 5, 5)))
        self.assertAlmostEqual(vol, 125.0, places=3)
        self.assertEqual(classify_overlap(vol), "clash")

    def test_disjoint_cubes_intersect_in_nothing(self):
        vol = intersection_volume(self._cube(), self._cube(origin=(100, 0, 0)))
        self.assertEqual(vol, 0.0)
        self.assertEqual(classify_overlap(vol), "none")

    def test_a_grazing_overlap_is_classified_as_noise(self):
        # 0.5mm cube of shared material = 0.125 mm^3, under the epsilon: two
        # parts that merely touch are not a clash.
        vol = intersection_volume(self._cube(), self._cube(origin=(9.5, 9.5, 9.5)))
        self.assertGreater(vol, 0.0)
        self.assertLess(vol, OVERLAP_NOISE_EPSILON)
        self.assertEqual(classify_overlap(vol), "noise")

    def test_intersection_is_deterministic(self):
        a, b = self._cube(), self._cube(origin=(5, 5, 5))
        self.assertEqual(intersection_volume(a, b), intersection_volume(a, b))


@unittest.skipUnless(HAVE_CQ and HAVE_M3D, "cadquery and manifold3d required")
class TestOcctToManifoldBridge(unittest.TestCase):
    """The tessellate+weld must really produce a closed 2-manifold."""

    def _box(self, dx=10, dy=10, dz=10, at=(0, 0, 0)):
        import cadquery as cq
        return cq.Workplane("XY").box(dx, dy, dz).translate(at).val()

    def test_an_occt_box_tessellates_and_welds(self):
        mesh = shape_to_mesh(self._box())
        self.assertIsNotNone(mesh)
        # A welded box is 8 vertices and 12 triangles; an unwelded soup would
        # carry 24+ vertices and manifold3d would reject it.
        self.assertEqual(mesh.n_vertices, 8)
        self.assertEqual(mesh.n_triangles, 12)

    def test_the_bridge_preserves_volume(self):
        manifold = shape_to_manifold(self._box())
        self.assertIsNotNone(manifold)
        self.assertAlmostEqual(float(manifold.volume()), 1000.0, places=3)

    def test_a_curved_solid_crosses_the_bridge(self):
        import cadquery as cq
        cyl = cq.Workplane("XY").circle(5).extrude(10).val()
        manifold = shape_to_manifold(cyl)
        self.assertIsNotNone(manifold)
        # Tessellated, so approximate — but well inside 1% at this deflection.
        self.assertAlmostEqual(float(manifold.volume()), math.pi * 25 * 10,
                               delta=math.pi * 25 * 10 * 0.01)

    def test_common_volume_of_two_occt_solids(self):
        # Boxes overlapping in a 4 x 10 x 10 slab.
        vol = common_volume(self._box(), self._box(at=(6, 0, 0)))
        self.assertAlmostEqual(vol, 400.0, places=2)

    def test_common_volume_of_disjoint_occt_solids(self):
        self.assertEqual(common_volume(self._box(), self._box(at=(50, 0, 0))), 0.0)

    def test_swept_cylinder_passes_through_the_part(self):
        # An r=2 tool driven -Z from above passes through 5mm of the box.
        vol = swept_cylinder_common_volume((0, 0, 10), (0, 0, -1), 2.0, 10.0,
                                           self._box())
        self.assertAlmostEqual(vol, 5.0 * math.pi * 4.0, delta=1.0)

    def test_swept_cylinder_aimed_away_hits_nothing(self):
        vol = swept_cylinder_common_volume((0, 0, 10), (0, 0, 1), 2.0, 10.0,
                                           self._box())
        self.assertEqual(vol, 0.0)

    def test_swept_cylinder_on_an_oblique_axis_is_measurable(self):
        # The +Z -> axis rotation must work for a non-trivial direction.
        vol = swept_cylinder_common_volume((-10, -10, 0), (1, 1, 0), 1.0, 30.0,
                                           self._box())
        self.assertIsNotNone(vol)
        self.assertGreater(vol, 0.0)

    def test_interference_common_volume_goes_through_the_mesh_kernel(self):
        # End-to-end: the migrated verifier helper returns the same volume the
        # policy module does.
        a, b = self._box(), self._box(at=(6, 0, 0))
        self.assertAlmostEqual(interference._common_volume(a, b), 400.0, places=2)

    def test_access_swept_common_volume_goes_through_the_mesh_kernel(self):
        feat = {"pos": (0, 0, 10), "axis": (0, 0, -1)}
        vol = access._swept_common_volume(feat, 2.0, 10.0, self._box())
        self.assertAlmostEqual(vol, 5.0 * math.pi * 4.0, delta=1.0)

    def test_access_swept_common_volume_degrades_on_a_malformed_feature(self):
        self.assertIsNone(access._swept_common_volume({}, 2.0, 10.0, self._box()))


if __name__ == "__main__":
    unittest.main()
