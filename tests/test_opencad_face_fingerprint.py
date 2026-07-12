import unittest

from geometry.opencad_face_fingerprint import (
    FaceRecord,
    Provenance,
    fingerprint,
    match_cost,
    match_topology,
    resolve_reference,
)


def _box_faces(prefix):
    defs = [
        ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0), 4.0),
        ((0.0, 0.0, -1.0), (0.0, 0.0, -1.0), 4.0),
        ((0.0, 1.0, 0.0), (0.0, 1.0, 0.0), 4.0),
        ((0.0, -1.0, 0.0), (0.0, -1.0, 0.0), 4.0),
        ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), 4.0),
        ((-1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), 4.0),
    ]
    return [
        FaceRecord(
            id="%s:face:%d" % (prefix, i),
            surface="planar",
            centroid=c,
            normal=n,
            area=a,
        )
        for i, (c, n, a) in enumerate(defs)
    ]


class TestFingerprint(unittest.TestCase):
    def test_stable_and_quantised(self):
        a = FaceRecord("f0", "planar", (0.0, 0.0, 1.0), (0.0, 0.0, 5.0), 4.0)
        b = FaceRecord("other", "planar", (0.0, 0.0, 1.0), (0.0, 0.0, 5.00001), 4.0)
        self.assertEqual(fingerprint(a), fingerprint(a))
        self.assertEqual(fingerprint(a), fingerprint(b))

    def test_differs_on_area(self):
        a = FaceRecord("f0", "planar", (0.0, 0.0, 1.0), (0.0, 0.0, 5.0), 4.0)
        b = FaceRecord("f0", "planar", (0.0, 0.0, 1.0), (0.0, 0.0, 5.0), 9.0)
        self.assertNotEqual(fingerprint(a), fingerprint(b))

    def test_no_normal_faces_hash(self):
        a = FaceRecord("f", "cylindrical", None, (0.0, 0.0, 0.0), 3.0)
        self.assertEqual(len(fingerprint(a)), 32)


class TestMatchCost(unittest.TestCase):
    def test_incompatible_surface_kinds(self):
        a = FaceRecord("a", "planar", (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 1.0)
        b = FaceRecord("b", "cylindrical", None, (0.0, 0.0, 0.0), 1.0)
        self.assertIsNone(match_cost(a, b))

    def test_flipped_normal_rejected(self):
        a = FaceRecord("a", "planar", (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 1.0)
        b = FaceRecord("b", "planar", (0.0, 0.0, -1.0), (0.0, 0.0, 0.0), 1.0)
        self.assertIsNone(match_cost(a, b))

    def test_identical_faces_cost_zero(self):
        a = FaceRecord("a", "planar", (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 1.0)
        b = FaceRecord("b", "planar", (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 1.0)
        self.assertAlmostEqual(match_cost(a, b), 0.0)

    def test_provenance_bonus_lowers_cost(self):
        a = FaceRecord("old", "planar", (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 1.0)
        plain = FaceRecord("n1", "planar", (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 1.0)
        derived = FaceRecord(
            "n2",
            "planar",
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0),
            1.0,
            provenance=Provenance(operation="fillet", parent_face_ids=("old",)),
        )
        self.assertLess(match_cost(a, derived), match_cost(a, plain))


class TestMatchTopology(unittest.TestCase):
    def test_identical_rebuild_matches_all(self):
        old = _box_faces("box")
        new = _box_faces("fillet")  # reindexed but same geometry
        report = match_topology(old, new)
        self.assertEqual(len(report.matched), 6)
        self.assertEqual(report.deleted, [])
        self.assertEqual(report.created, [])
        self.assertEqual(report.mapping()["box:face:0"], "fillet:face:0")

    def test_reordered_faces_still_match_geometrically(self):
        old = _box_faces("box")
        new = list(reversed(_box_faces("v2")))
        report = match_topology(old, new)
        self.assertEqual(report.mapping()["box:face:4"], "v2:face:4")
        self.assertEqual(len(report.matched), 6)

    def test_fillet_split_via_provenance(self):
        old = _box_faces("box")
        new = _box_faces("v2")
        blend = FaceRecord(
            id="v2:face:6",
            surface="blend",
            normal=None,
            centroid=(0.9, 0.0, 0.9),
            area=0.5,
            provenance=Provenance(
                operation="fillet_edges", parent_face_ids=("box:face:0",), local_index=0
            ),
        )
        report = match_topology(old, new + [blend])
        self.assertIn("box:face:0", report.splits)
        self.assertEqual(report.splits["box:face:0"], ["v2:face:0", "v2:face:6"])
        resolution = resolve_reference(
            "box:face:0", report, new_faces=new + [blend]
        )
        self.assertEqual(resolution.status, "split")
        self.assertEqual(resolution.new_id, "v2:face:0")  # largest fragment
        self.assertEqual(resolution.alternatives, ("v2:face:6",))
        self.assertFalse(resolution.is_stale)

    def test_boolean_deletes_face(self):
        old = _box_faces("box")
        new = [f for f in _box_faces("cut") if f.id != "cut:face:0"]
        report = match_topology(old, new)
        self.assertEqual(report.deleted, ["box:face:0"])
        resolution = resolve_reference("box:face:0", report)
        self.assertEqual(resolution.status, "deleted")
        self.assertTrue(resolution.is_stale)

    def test_created_face_reported(self):
        old = _box_faces("box")
        extra = FaceRecord(
            "v2:face:9", "cylindrical", None, (5.0, 5.0, 5.0), 12.0
        )
        report = match_topology(old, _box_faces("v2") + [extra])
        self.assertEqual(report.created, ["v2:face:9"])

    def test_symmetric_geometry_flagged_ambiguous(self):
        old = [FaceRecord("box:face:0", "planar", (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 4.0)]
        new = [
            FaceRecord("v2:face:a", "planar", (0.0, 0.0, 1.0), (0.0, 0.0, 0.5), 4.0),
            FaceRecord("v2:face:b", "planar", (0.0, 0.0, 1.0), (0.0, 0.0, -0.5), 4.0),
        ]
        report = match_topology(old, new)
        self.assertEqual(report.ambiguous, ["box:face:0"])
        resolution = resolve_reference("box:face:0", report)
        self.assertEqual(resolution.status, "ambiguous")
        self.assertIn("symmetric", resolution.reason)

    def test_merge_detected(self):
        old = _box_faces("box")
        old.append(
            FaceRecord("box:face:6", "planar", (0.0, 0.0, 1.0), (0.5, 0.5, 1.0), 1.0)
        )
        new = _box_faces("v2")
        report = match_topology(old, new)
        self.assertIn("v2:face:0", report.merges)
        self.assertIn("box:face:6", report.merges["v2:face:0"])
        resolution = resolve_reference("box:face:6", report)
        self.assertEqual(resolution.status, "merged")
        self.assertEqual(resolution.new_id, "v2:face:0")

    def test_unknown_reference(self):
        report = match_topology(_box_faces("box"), _box_faces("v2"))
        self.assertEqual(resolve_reference("nope", report).status, "unknown")

    def test_matching_is_deterministic_and_order_independent(self):
        old = _box_faces("box")
        new = _box_faces("v2")
        first = match_topology(old, new).mapping()
        second = match_topology(list(reversed(old)), list(reversed(new))).mapping()
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
