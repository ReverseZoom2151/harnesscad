"""Tests for the CAD validity gate and its advisory diagnostics."""
import math
import unittest

from harnesscad.eval.bench.geometry.betti_graded import MeshSurface
from harnesscad.eval.verifiers.validity_gate import (
    MAX_BREP_TOLERANCE_MM,
    MAX_FACE_ASPECT_RATIO,
    MIN_FACE_AREA_MM2,
    advisory_flags,
    candidate_status,
    mesh_face_diagnostics,
    triangle_area,
    validate_candidate,
)


def _cube(scale=1.0):
    s = scale
    verts = [
        (0, 0, 0), (s, 0, 0), (s, s, 0), (0, s, 0),
        (0, 0, s), (s, 0, s), (s, s, s), (0, s, s),
    ]
    tris = [
        (0, 3, 2), (0, 2, 1),
        (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4),
        (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6),
        (3, 0, 4), (3, 4, 7),
    ]
    return MeshSurface(verts, tris)


class TestGate(unittest.TestCase):
    def test_clean_cube_is_valid(self):
        result = validate_candidate(mesh=_cube(10.0))
        self.assertTrue(result.is_valid)
        self.assertEqual(result.reasons, [])
        self.assertEqual(result.status, "valid")

    def test_brep_errors_fail_the_gate(self):
        result = validate_candidate(brep_errors=["self-intersecting wire"], mesh=_cube())
        self.assertFalse(result.is_valid)
        self.assertTrue(result.reasons[0].startswith("brep:"))

    def test_not_watertight_fails(self):
        result = validate_candidate(is_watertight=False, mesh=_cube())
        self.assertFalse(result.is_valid)
        self.assertTrue(any("watertight" in r for r in result.reasons))

    def test_unmeshable_fails(self):
        result = validate_candidate(mesh=None)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("not meshable" in r for r in result.reasons))

    def test_open_mesh_fails(self):
        cube = _cube()
        open_mesh = MeshSurface(cube.vertices, list(cube.triangles)[:-2])
        result = validate_candidate(mesh=open_mesh)
        self.assertFalse(result.is_valid)
        self.assertTrue(any(r.startswith("mesh:") for r in result.reasons))

    def test_to_dict(self):
        payload = validate_candidate(mesh=_cube()).to_dict()
        self.assertTrue(payload["is_valid"])
        self.assertIn("diagnostics", payload)


class TestAdvisories(unittest.TestCase):
    def test_flags_never_gate(self):
        # A cube 0.01 mm on a side: valid, but its faces are below the area floor.
        result = validate_candidate(mesh=_cube(0.01), max_brep_tolerance=0.5)
        self.assertTrue(result.is_valid)
        self.assertTrue(result.flags)
        self.assertEqual(result.reasons, [])

    def test_min_face_area_flag(self):
        flags = advisory_flags(min_face_area=MIN_FACE_AREA_MM2 / 10)
        self.assertEqual(len(flags), 1)
        self.assertIn("min face area", flags[0])

    def test_healthy_face_area_is_not_flagged(self):
        self.assertEqual(advisory_flags(min_face_area=0.05), [])

    def test_sliver_aspect_flag(self):
        flags = advisory_flags(max_face_aspect_ratio=MAX_FACE_ASPECT_RATIO * 100)
        self.assertIn("sliver face", flags[0])

    def test_loose_tolerance_flag(self):
        flags = advisory_flags(max_brep_tolerance=MAX_BREP_TOLERANCE_MM * 2)
        self.assertIn("loose export", flags[0])

    def test_healthy_tolerance_is_not_flagged(self):
        self.assertEqual(advisory_flags(max_brep_tolerance=0.05), [])

    def test_no_measurements_no_flags(self):
        self.assertEqual(advisory_flags(), [])


class TestMeshDiagnostics(unittest.TestCase):
    def test_triangle_area(self):
        self.assertAlmostEqual(
            triangle_area((0, 0, 0), (2, 0, 0), (0, 2, 0)), 2.0, places=9
        )

    def test_cube_diagnostics(self):
        diag = mesh_face_diagnostics(_cube(10.0))
        self.assertAlmostEqual(diag["min_face_area"], 50.0, places=6)
        # A right isosceles triangle: longest^2 / (2*area) = 200/100 = 2.
        self.assertAlmostEqual(diag["max_face_aspect_ratio"], 2.0, places=6)

    def test_degenerate_face_is_infinite_aspect(self):
        mesh = MeshSurface([(0, 0, 0), (1, 0, 0), (2, 0, 0)], [(0, 1, 2)])
        diag = mesh_face_diagnostics(mesh)
        self.assertEqual(diag["min_face_area"], 0.0)
        self.assertTrue(math.isinf(diag["max_face_aspect_ratio"]))

    def test_empty_mesh_has_no_diagnostics(self):
        self.assertEqual(mesh_face_diagnostics(MeshSurface([], [])), {})


class TestStatus(unittest.TestCase):
    def test_missing_beats_invalid(self):
        self.assertEqual(candidate_status(candidate_exists=False, is_valid=False), "missing")

    def test_invalid(self):
        self.assertEqual(candidate_status(candidate_exists=True, is_valid=False), "invalid")

    def test_valid(self):
        self.assertEqual(candidate_status(candidate_exists=True, is_valid=True), "valid")


if __name__ == "__main__":
    unittest.main()
