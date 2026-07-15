"""Tests for the OnshapeBackend -- a scriptable GeometryBackend that actuates
Onshape geometry over its signed REST API.

Credentials (ONSHAPE_ACCESS_KEY / ONSHAPE_SECRET_KEY) are almost certainly ABSENT
here, so the *live* path cannot run and is not faked. Instead a mock client stands
in for the signed REST transport and RECORDS every feature-definition JSON the
backend POSTs, so these tests prove the op -> Onshape-feature mapping end to end
offline: that a sketch + extrude lower to exactly the documented BTMSketch-151 /
BTMFeature-134 payloads, that units convert mm -> m, that a boolean names the right
operand bodies, and that pick-dependent ops are refused with typed diagnostics
rather than approximated.

One test also asserts the honest credential gate: with no credentials and no
injected client, the constructor raises BackendUnavailable naming the two env vars.
"""

import os
import unittest

from harnesscad.core.cisp.ops import (
    AddCircle, AddLine, AddPoint, AddRectangle, Boolean, Chamfer, Constrain,
    Extrude, Fillet, Hole, Loft, Mirror, NewSketch, Revolve, SetParam, Shell,
    Sweep,
)
from harnesscad.io.backends.base import BackendUnavailable
from harnesscad.io.backends.onshape import (
    BOOLEAN_OP, MM_TO_M, PLANE_TO_ONSHAPE, REFUSED_OPS, SUPPORTED_OPS,
    OnshapeBackend, build_boolean_feature, build_extrude_feature,
)
from harnesscad.io.cua.environment_onshape import (
    BoundingBox, DocumentRef, MassProperties,
)


class MockOnshapeClient:
    """A stand-in for OnshapeFeatureClient that never touches the network.

    It records each POSTed feature JSON in ``self.posted`` and mints deterministic
    featureIds ("FID1", "FID2", ...), so a test can assert the exact op->feature
    mapping. The oracle reads return canned geometry so query() can be exercised.
    """

    def __init__(self):
        self.posted = []
        self.deleted = []
        self.created = []
        self._n = 0

    def create_scratch_document(self, name):
        self.created.append(name)
        return DocumentRef("DID", "WID", "EID")

    def delete_document(self, did):
        self.deleted.append(did)

    def add_feature(self, ref, feature):
        self._n += 1
        self.posted.append(feature)
        return {"feature": {"featureId": "FID%d" % self._n}}

    def features(self, ref):
        return [{"featureId": "FID%d" % (i + 1)} for i in range(len(self.posted))]

    def mass_properties(self, ref):
        return MassProperties(volume_mm3=1000.0, surface_area_mm2=600.0,
                              centroid_mm=(5.0, 5.0, 5.0), mass=1.0, raw={})

    def bounding_box(self, ref):
        return BoundingBox(low_mm=(0.0, 0.0, 0.0), high_mm=(10.0, 10.0, 10.0))

    def export_stl(self, ref):
        return b"solid mock\nendsolid mock\n"


def _backend():
    return OnshapeBackend(client=MockOnshapeClient())


class TestCredentialGate(unittest.TestCase):
    def test_no_credentials_no_client_raises_named_unavailable(self):
        # Ensure the env vars are absent for this check.
        saved = {k: os.environ.pop(k, None)
                 for k in ("ONSHAPE_ACCESS_KEY", "ONSHAPE_SECRET_KEY")}
        try:
            with self.assertRaises(BackendUnavailable) as ctx:
                OnshapeBackend()
            msg = str(ctx.exception)
            self.assertIn("ONSHAPE_ACCESS_KEY", msg)
            self.assertIn("ONSHAPE_SECRET_KEY", msg)
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_available_reflects_credentials(self):
        saved = {k: os.environ.pop(k, None)
                 for k in ("ONSHAPE_ACCESS_KEY", "ONSHAPE_SECRET_KEY")}
        try:
            self.assertFalse(OnshapeBackend.available())
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_injected_client_bypasses_credential_gate(self):
        # A mock client lets the whole mapping run with no credentials at all.
        b = _backend()
        self.assertTrue(b.apply(NewSketch(plane="XY")).ok)


class TestLazyDocument(unittest.TestCase):
    def test_construction_creates_no_document(self):
        client = MockOnshapeClient()
        OnshapeBackend(client=client)
        self.assertEqual(client.created, [])  # lazy: nothing yet

    def test_document_created_on_first_posting_op(self):
        client = MockOnshapeClient()
        b = OnshapeBackend(client=client)
        # Pure sketch ops are buffered -> still no document.
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0, y=0, w=20, h=10))
        self.assertEqual(client.created, [])
        # Extrude flushes the sketch -> the document is created now.
        b.apply(Extrude(sketch="sk1", distance=5))
        self.assertEqual(len(client.created), 1)


class TestSketchExtrudeMapping(unittest.TestCase):
    def test_rectangle_extrude_posts_two_features(self):
        client = MockOnshapeClient()
        b = OnshapeBackend(client=client)
        self.assertTrue(b.apply(NewSketch(plane="XY")).ok)
        self.assertTrue(b.apply(AddRectangle(sketch="sk1", x=0, y=0, w=20, h=10)).ok)
        self.assertTrue(b.apply(Extrude(sketch="sk1", distance=5)).ok)
        # Exactly two POSTs: the flushed sketch, then the extrude.
        self.assertEqual(len(client.posted), 2)
        sketch, extrude = client.posted

        # -- the sketch feature --
        sf = sketch["feature"]
        self.assertEqual(sketch["btType"], "BTFeatureDefinitionCall-1406")
        self.assertEqual(sf["btType"], "BTMSketch-151")
        self.assertEqual(sf["featureType"], "newSketch")
        # plane query names the Onshape default plane for XY (Top)
        plane_q = sf["parameters"][0]
        self.assertEqual(plane_q["parameterId"], "sketchPlane")
        self.assertIn("makeId(\"Top\")", plane_q["queries"][0]["queryString"])
        # a rectangle is four line segments, coords converted mm -> m
        segs = sf["entities"]
        self.assertEqual(len(segs), 4)
        self.assertTrue(all(s["btType"] == "BTMSketchCurveSegment-155" for s in segs))
        self.assertTrue(all(s["geometry"]["btType"] == "BTCurveGeometryLine-117"
                            for s in segs))
        # first segment runs from (0,0) to (20mm,0): length 0.02 m, dir +x
        s0 = segs[0]
        self.assertAlmostEqual(s0["geometry"]["pntX"], 0.0)
        self.assertAlmostEqual(s0["endParam"], 20 * MM_TO_M)
        self.assertAlmostEqual(s0["geometry"]["dirX"], 1.0)
        self.assertAlmostEqual(s0["geometry"]["dirY"], 0.0)

        # -- the extrude feature --
        ef = extrude["feature"]
        self.assertEqual(ef["btType"], "BTMFeature-134")
        self.assertEqual(ef["featureType"], "extrude")
        params = {p["parameterId"]: p for p in ef["parameters"]}
        self.assertEqual(params["bodyType"]["value"], "SOLID")
        self.assertEqual(params["operationType"]["value"], "NEW")
        self.assertEqual(params["endBound"]["value"], "BLIND")
        self.assertEqual(params["depth"]["expression"], "5 mm")
        # entities is a sketch-region query pointing at the flushed sketch id
        region = params["entities"]["queries"][0]
        self.assertEqual(region["btType"], "BTMIndividualSketchRegionQuery-140")
        self.assertEqual(region["featureId"], "FID1")

    def test_circle_maps_to_one_circle_entity(self):
        client = MockOnshapeClient()
        b = OnshapeBackend(client=client)
        b.apply(NewSketch(plane="XZ"))
        b.apply(AddCircle(sketch="sk1", cx=3, cy=4, r=6))
        b.apply(Extrude(sketch="sk1", distance=2))
        sketch = client.posted[0]["feature"]
        self.assertIn("makeId(\"Front\")",
                      sketch["parameters"][0]["queries"][0]["queryString"])
        ents = sketch["entities"]
        self.assertEqual(len(ents), 1)
        geom = ents[0]["geometry"]
        self.assertEqual(geom["btType"], "BTCurveGeometryCircle-115")
        self.assertAlmostEqual(geom["radius"], 6 * MM_TO_M)
        self.assertAlmostEqual(geom["xCenter"], 3 * MM_TO_M)
        self.assertAlmostEqual(geom["yCenter"], 4 * MM_TO_M)

    def test_negative_distance_sets_opposite_direction(self):
        feat = build_extrude_feature("E", "FIDX", -7.5)
        params = {p["parameterId"]: p for p in feat["feature"]["parameters"]}
        self.assertEqual(params["depth"]["expression"], "7.5 mm")
        self.assertTrue(params["oppositeDirection"]["value"])

    def test_sketch_flushed_once_even_if_reused(self):
        client = MockOnshapeClient()
        b = OnshapeBackend(client=client)
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0, y=0, w=10, h=10))
        b.apply(Extrude(sketch="sk1", distance=5))
        # A second entity add after flush is refused (the sketch is locked).
        r = b.apply(AddCircle(sketch="sk1", cx=0, cy=0, r=2))
        self.assertFalse(r.ok)
        self.assertEqual(r.diagnostics[0].code, "locked-sketch")


class TestBooleanMapping(unittest.TestCase):
    def _two_solids(self, client):
        b = OnshapeBackend(client=client)
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0, y=0, w=20, h=20))
        b.apply(Extrude(sketch="sk1", distance=10))     # -> f-feature, FID2 body
        b.apply(NewSketch(plane="XY"))
        b.apply(AddCircle(sketch="sk2", cx=10, cy=10, r=4))
        b.apply(Extrude(sketch="sk2", distance=10))     # -> f-feature, FID4 body
        return b

    def test_union_names_both_bodies_as_tools(self):
        client = MockOnshapeClient()
        b = self._two_solids(client)
        self.assertTrue(b.apply(Boolean(kind="union")).ok)
        boolean = client.posted[-1]["feature"]
        self.assertEqual(boolean["featureType"], "boolean")
        params = {p["parameterId"]: p for p in boolean["parameters"]}
        self.assertEqual(params["operationType"]["value"], "UNION")
        tool_queries = params["tools"]["queries"]
        self.assertEqual(len(tool_queries), 2)
        joined = " ".join(q["queryString"] for q in tool_queries)
        self.assertIn("FID2", joined)
        self.assertIn("FID4", joined)

    def test_cut_splits_targets_and_tools(self):
        client = MockOnshapeClient()
        b = self._two_solids(client)
        self.assertTrue(b.apply(Boolean(kind="cut")).ok)
        params = {p["parameterId"]: p
                  for p in client.posted[-1]["feature"]["parameters"]}
        self.assertEqual(params["operationType"]["value"], "SUBTRACT")
        self.assertIn("FID2", params["targets"]["queries"][0]["queryString"])
        self.assertIn("FID4", params["tools"]["queries"][0]["queryString"])

    def test_boolean_requires_two_solids(self):
        client = MockOnshapeClient()
        b = OnshapeBackend(client=client)
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0, y=0, w=20, h=20))
        b.apply(Extrude(sketch="sk1", distance=10))
        r = b.apply(Boolean(kind="union"))
        self.assertFalse(r.ok)
        self.assertEqual(r.diagnostics[0].code, "no-solid")

    def test_boolean_unknown_target_ref_refused(self):
        client = MockOnshapeClient()
        b = self._two_solids(client)
        r = b.apply(Boolean(kind="union", target="nope", tool="f2"))
        self.assertFalse(r.ok)
        self.assertEqual(r.diagnostics[0].code, "bad-ref")


class TestBlockAndCorrect(unittest.TestCase):
    def test_extrude_unknown_sketch_refused_without_posting(self):
        client = MockOnshapeClient()
        b = OnshapeBackend(client=client)
        r = b.apply(Extrude(sketch="sk9", distance=5))
        self.assertFalse(r.ok)
        self.assertEqual(r.diagnostics[0].code, "bad-ref")
        self.assertEqual(client.posted, [])

    def test_bad_rectangle_dims_refused(self):
        b = _backend()
        b.apply(NewSketch(plane="XY"))
        r = b.apply(AddRectangle(sketch="sk1", x=0, y=0, w=0, h=10))
        self.assertFalse(r.ok)
        self.assertEqual(r.diagnostics[0].code, "bad-value")

    def test_zero_distance_extrude_refused(self):
        b = _backend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0, y=0, w=5, h=5))
        r = b.apply(Extrude(sketch="sk1", distance=0))
        self.assertFalse(r.ok)
        self.assertEqual(r.diagnostics[0].code, "bad-value")

    def test_unknown_plane_refused(self):
        b = _backend()
        r = b.apply(NewSketch(plane="WW"))
        self.assertFalse(r.ok)
        self.assertEqual(r.diagnostics[0].code, "bad-value")


class TestRefusedOps(unittest.TestCase):
    """Every pick-dependent op is refused with a typed reason, never approximated."""

    def _solid(self, client):
        b = OnshapeBackend(client=client)
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0, y=0, w=20, h=20))
        b.apply(Extrude(sketch="sk1", distance=10))
        return b

    def test_pick_dependent_ops_are_refused_typed(self):
        cases = [
            Fillet(edges=("|Z",), radius=1),
            Chamfer(edges=(">Z",), distance=1),
            Hole(face_or_sketch="last", x=0, y=0, diameter=4),
            Shell(faces=(">Z",), thickness=1),
            Revolve(sketch="sk1", angle=360),
            Loft(sketches=("sk1", "sk2")),
            Sweep(sketch="sk1", path="sk2"),
            Mirror(feature_or_body="f1", plane="XZ"),
            AddPoint(sketch="sk1", x=1, y=1),
            AddLine(sketch="sk1", x1=0, y1=0, x2=1, y2=1),
            Constrain(kind="horizontal", a="e1"),
        ]
        for op in cases:
            client = MockOnshapeClient()
            b = self._solid(client)
            before = len(client.posted)
            r = b.apply(op)
            with self.subTest(op=type(op).__name__):
                self.assertFalse(r.ok)
                self.assertEqual(r.diagnostics[0].code, "unsupported-op")
                # a refusal never POSTs a feature
                self.assertEqual(len(client.posted), before)

    def test_supported_and_refused_partition_the_op_set(self):
        overlap = set(SUPPORTED_OPS) & set(REFUSED_OPS)
        self.assertEqual(overlap, set())


class TestOracleReadsAndLifecycle(unittest.TestCase):
    def test_measure_goes_through_oracle(self):
        b = _backend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0, y=0, w=10, h=10))
        b.apply(Extrude(sketch="sk1", distance=10))
        m = b.query("measure")
        self.assertAlmostEqual(m["volume"], 1000.0)     # mm^3 from the oracle
        self.assertEqual(m["bbox"], [10.0, 10.0, 10.0])  # mm bounding box

    def test_export_stl_through_oracle(self):
        b = _backend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0, y=0, w=10, h=10))
        b.apply(Extrude(sketch="sk1", distance=10))
        self.assertIn("solid", b.export("stl"))

    def test_export_step_is_declared_not_faked(self):
        b = _backend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0, y=0, w=10, h=10))
        b.apply(Extrude(sketch="sk1", distance=10))
        with self.assertRaises(ValueError):
            b.export("step")

    def test_close_deletes_only_the_scratch_document(self):
        client = MockOnshapeClient()
        b = OnshapeBackend(client=client)
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0, y=0, w=10, h=10))
        b.apply(Extrude(sketch="sk1", distance=10))
        did = b.doc.did
        b.close()
        self.assertEqual(client.deleted, [did])

    def test_keep_document_skips_deletion(self):
        client = MockOnshapeClient()
        b = OnshapeBackend(client=client, keep_document=True)
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0, y=0, w=10, h=10))
        b.apply(Extrude(sketch="sk1", distance=10))
        b.close()
        self.assertEqual(client.deleted, [])

    def test_state_digest_stable_across_identical_replay(self):
        def run():
            b = _backend()
            b.apply(NewSketch(plane="XY"))
            b.apply(AddRectangle(sketch="sk1", x=0, y=0, w=10, h=10))
            b.apply(Extrude(sketch="sk1", distance=10))
            return b.state_digest()
        self.assertEqual(run(), run())

    def test_mapping_query_reports_partition(self):
        b = _backend()
        mapping = b.query("mapping")
        self.assertEqual(set(mapping["supported"]), set(SUPPORTED_OPS))
        self.assertIn("extrude", mapping["feature_json"])


class TestPlaneAndBooleanTables(unittest.TestCase):
    def test_plane_table_covers_canonical_planes(self):
        for plane in ("XY", "XZ", "YZ"):
            self.assertIn(plane, PLANE_TO_ONSHAPE)

    def test_boolean_op_table(self):
        self.assertEqual(BOOLEAN_OP["cut"], "SUBTRACT")
        feat = build_boolean_feature("B", "intersect", "A", "C")
        # a bare builder call for INTERSECT keeps both bodies in tools
        params = {p["parameterId"]: p for p in feat["feature"]["parameters"]}
        self.assertEqual(params["operationType"]["value"], "INTERSECT")
        self.assertEqual(len(params["tools"]["queries"]), 2)


if __name__ == "__main__":
    unittest.main()
