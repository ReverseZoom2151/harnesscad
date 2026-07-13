"""Tests for the geometry anomaly detector (anomaly.py).

Covers, deterministically:
  * feature_vector extraction from a raw metrics dict and from a backend,
    including graceful degradation when fields are missing (the stub);
  * AnomalyModel fit/score: a clear outlier (100x aspect ratio) scores
    is_outlier=True and names the offending feature; a normal one is False;
  * IsolationLite multivariate detection of the same outlier;
  * JSON round-trip of the fitted baseline (dict and on-disk);
  * AnomalyCheck INFO-skips on an unfit model and on the stub (unmeasurable),
    WARNs on an outlier, and never emits an ERROR.
"""

import json
import os
import tempfile
import unittest

from harnesscad.core.cisp.ops import NewSketch, AddRectangle, Extrude
from harnesscad.io.backends.stub import StubBackend
from harnesscad.eval.verifiers.verify import Severity
from harnesscad.eval.quality.anomaly import (
    feature_vector, AnomalyModel, AnomalyScore, IsolationLite,
    AnomalyCheck, with_anomaly,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class _MeasuredBackend:
    """A backend that answers 'measure'/'metrics'/'summary' like a real kernel."""

    def __init__(self, bbox, volume=1000.0, surface_area=None,
                 faces=6, edges=12, vertices=8, feature_count=1,
                 solid_present=True):
        self._bbox = list(bbox)
        self._volume = volume
        self._area = surface_area
        self._faces = faces
        self._edges = edges
        self._vertices = vertices
        self._feature_count = feature_count
        self._solid_present = solid_present

    def query(self, q: str) -> dict:
        if q == "summary":
            return {"sketch_count": 1, "entity_count": 1,
                    "feature_count": self._feature_count,
                    "solid_present": self._solid_present}
        if q == "measure":
            return {"volume": self._volume, "bbox": list(self._bbox)}
        if q == "metrics":
            m = {"volume": self._volume, "bbox": list(self._bbox),
                 "faces": self._faces, "edges": self._edges,
                 "vertices": self._vertices}
            if self._area is not None:
                m["surface_area"] = self._area
            return m
        return {}


def _normal_metrics(seed_dim):
    """A tidy box: near-cubic bbox, plausible area, standard box topology."""
    x, y, z = seed_dim
    area = 2 * (x * y + y * z + x * z)
    return {"bbox": [x, y, z], "volume": x * y * z, "surface_area": area,
            "faces": 6, "edges": 12, "vertices": 8, "feature_count": 1}


def _normal_corpus():
    dims = [(10, 10, 10), (12, 10, 8), (9, 11, 10), (10, 9, 11),
            (8, 12, 9), (11, 10, 9), (10, 10, 12), (9, 9, 10)]
    return [feature_vector(_normal_metrics(d)) for d in dims]


def _build_plate(backend):
    for op in (NewSketch(plane="XY"),
               AddRectangle(sketch="sk1", w=20.0, h=10.0),
               Extrude(sketch="sk1", distance=5.0)):
        res = backend.apply(op)
        assert res.ok, res.diagnostics
    return backend


# ---------------------------------------------------------------------------
# feature extraction
# ---------------------------------------------------------------------------
class TestFeatureVector(unittest.TestCase):
    def test_extracts_expected_features_from_dict(self):
        feats = feature_vector(_normal_metrics((20, 10, 5)))
        # bbox ratios: sorted [5,10,20] -> aspect 4, elongation 2, flatness 2
        self.assertAlmostEqual(feats["aspect_ratio"], 4.0)
        self.assertAlmostEqual(feats["elongation"], 2.0)
        self.assertAlmostEqual(feats["flatness"], 2.0)
        self.assertIn("sa_to_vol", feats)
        self.assertIn("log_volume", feats)
        self.assertEqual(feats["faces"], 6.0)
        self.assertEqual(feats["feature_count"], 1.0)

    def test_from_backend_measured(self):
        b = _MeasuredBackend(bbox=(20, 10, 5), volume=1000.0)
        feats = feature_vector(b)
        self.assertAlmostEqual(feats["aspect_ratio"], 4.0)
        self.assertIn("log_volume", feats)

    def test_degrades_gracefully_on_stub(self):
        # Stub answers only 'summary' (no bbox/volume) -> only count features.
        feats = feature_vector(_build_plate(StubBackend()))
        self.assertNotIn("aspect_ratio", feats)
        self.assertNotIn("sa_to_vol", feats)
        # It still surfaces the counts summary can provide.
        self.assertIn("feature_count", feats)
        self.assertIn("entity_count", feats)

    def test_missing_and_degenerate_fields_omitted(self):
        # Zero-volume, zero-min-dim: no sa_to_vol, aspect uses positive dims.
        feats = feature_vector({"bbox": [0.0, 10.0, 20.0], "volume": 0.0})
        self.assertNotIn("log_volume", feats)
        self.assertNotIn("sa_to_vol", feats)
        # aspect uses the smallest *positive* dim (10) -> 20/10 = 2.
        self.assertAlmostEqual(feats["aspect_ratio"], 2.0)

    def test_rejects_bad_input_type(self):
        with self.assertRaises(TypeError):
            feature_vector(42)


# ---------------------------------------------------------------------------
# model fit / score
# ---------------------------------------------------------------------------
class TestAnomalyModel(unittest.TestCase):
    def test_clear_outlier_flagged_and_named(self):
        model = AnomalyModel().fit(_normal_corpus())
        # A 1000 x 10 x 10 slab: aspect ~100 vs baseline ~1.x.
        outlier = feature_vector(_normal_metrics((1000, 10, 10)))
        result = model.score(outlier)
        self.assertIsInstance(result, AnomalyScore)
        self.assertTrue(result.is_outlier)
        self.assertIn("aspect_ratio", result.outlier_features)

    def test_normal_vector_not_outlier(self):
        model = AnomalyModel().fit(_normal_corpus())
        normal = feature_vector(_normal_metrics((10, 10, 10)))
        result = model.score(normal)
        self.assertFalse(result.is_outlier)
        self.assertEqual(result.outlier_features, [])

    def test_iqr_method_also_flags_outlier(self):
        model = AnomalyModel(method="iqr").fit(_normal_corpus())
        outlier = feature_vector(_normal_metrics((1000, 10, 10)))
        self.assertTrue(model.score(outlier).is_outlier)
        normal = feature_vector(_normal_metrics((10, 10, 10)))
        self.assertFalse(model.score(normal).is_outlier)

    def test_deterministic_scoring(self):
        model = AnomalyModel().fit(_normal_corpus())
        vec = feature_vector(_normal_metrics((1000, 10, 10)))
        a = model.score(vec)
        b = model.score(vec)
        self.assertEqual(a.to_dict(), b.to_dict())

    def test_unfit_model_reports_not_fit(self):
        self.assertFalse(AnomalyModel().is_fit)
        self.assertTrue(AnomalyModel().fit(_normal_corpus()).is_fit)

    def test_unknown_method_rejected(self):
        with self.assertRaises(ValueError):
            AnomalyModel(method="bogus")

    def test_feature_absent_from_baseline_is_ignored(self):
        # Baseline built only from count features; a vector adding aspect_ratio
        # must not error and must not be judged on the unknown feature.
        model = AnomalyModel().fit([{"feature_count": 1.0},
                                    {"feature_count": 1.0},
                                    {"feature_count": 2.0}])
        result = model.score({"feature_count": 1.0, "aspect_ratio": 999.0})
        self.assertNotIn("aspect_ratio", result.details)


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------
class TestPersistence(unittest.TestCase):
    def test_dict_round_trip_scores_identically(self):
        model = AnomalyModel().fit(_normal_corpus())
        restored = AnomalyModel.from_dict(json.loads(json.dumps(model.to_dict())))
        self.assertEqual(restored.method, model.method)
        self.assertEqual(restored.baseline.keys(), model.baseline.keys())
        vec = feature_vector(_normal_metrics((1000, 10, 10)))
        self.assertEqual(model.score(vec).to_dict(),
                         restored.score(vec).to_dict())

    def test_save_load_file(self):
        model = AnomalyModel(method="iqr").fit(_normal_corpus())
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            model.save(path)
            loaded = AnomalyModel.load(path)
        finally:
            os.remove(path)
        self.assertEqual(loaded.method, "iqr")
        self.assertTrue(loaded.is_fit)
        vec = feature_vector(_normal_metrics((1000, 10, 10)))
        self.assertEqual(loaded.score(vec).is_outlier, model.score(vec).is_outlier)


# ---------------------------------------------------------------------------
# IsolationLite
# ---------------------------------------------------------------------------
class TestIsolationLite(unittest.TestCase):
    def test_isolates_the_outlier(self):
        iso = IsolationLite(n_trees=64, seed=0).fit(_normal_corpus())
        self.assertTrue(iso.is_fit)
        outlier = feature_vector(_normal_metrics((1000, 10, 10)))
        normal = feature_vector(_normal_metrics((10, 10, 10)))
        self.assertGreater(iso.anomaly_score(outlier), iso.anomaly_score(normal))
        self.assertTrue(iso.is_outlier(outlier))

    def test_deterministic_for_seed(self):
        corpus = _normal_corpus()
        vec = feature_vector(_normal_metrics((1000, 10, 10)))
        s1 = IsolationLite(seed=7).fit(corpus).anomaly_score(vec)
        s2 = IsolationLite(seed=7).fit(corpus).anomaly_score(vec)
        self.assertEqual(s1, s2)

    def test_score_shape_matches(self):
        iso = IsolationLite(seed=0).fit(_normal_corpus())
        result = iso.score(feature_vector(_normal_metrics((1000, 10, 10))))
        self.assertIsInstance(result, AnomalyScore)
        self.assertIn("isolation_score", result.details)


# ---------------------------------------------------------------------------
# AnomalyCheck verifier
# ---------------------------------------------------------------------------
def _by_severity(report, sev):
    return [d for d in report.diagnostics if d.severity is sev]


def _codes(report):
    return {d.code for d in report.diagnostics}


class TestAnomalyCheck(unittest.TestCase):
    def test_info_skips_when_model_none(self):
        report = AnomalyCheck().check(_MeasuredBackend(bbox=(10, 10, 10)), None)
        self.assertIn("anomaly-skipped", _codes(report))
        self.assertEqual(_by_severity(report, Severity.ERROR), [])
        self.assertTrue(report.ok)

    def test_info_skips_when_model_unfit(self):
        report = AnomalyCheck(AnomalyModel()).check(
            _MeasuredBackend(bbox=(10, 10, 10)), None)
        self.assertIn("anomaly-skipped", _codes(report))
        self.assertTrue(report.ok)

    def test_info_skips_on_unmeasurable_stub(self):
        # Baseline built from bbox features; stub yields only counts. Build a
        # baseline that has NO overlap with the stub's features so the vector is
        # non-empty but... actually stub yields feature_count etc. Use a bbox-only
        # baseline and a stub whose feature vector shares nothing -> still scored.
        # To exercise the unmeasurable path we use a backend that answers nothing.
        class _Empty:
            def query(self, q):
                return {}
        model = AnomalyModel().fit(_normal_corpus())
        report = AnomalyCheck(model).check(_Empty(), None)
        self.assertIn("anomaly-unmeasurable", _codes(report))
        self.assertEqual(_by_severity(report, Severity.ERROR), [])
        self.assertTrue(report.ok)

    def test_stub_backend_never_errors(self):
        model = AnomalyModel().fit(_normal_corpus())
        report = AnomalyCheck(model).check(_build_plate(StubBackend()), None)
        self.assertEqual(_by_severity(report, Severity.ERROR), [])
        self.assertTrue(report.ok)

    def test_warns_on_outlier_and_names_feature(self):
        model = AnomalyModel().fit(_normal_corpus())
        backend = _MeasuredBackend(bbox=(1000, 10, 10), volume=100000.0,
                                   surface_area=42000.0)
        report = AnomalyCheck(model).check(backend, None)
        warnings = _by_severity(report, Severity.WARNING)
        self.assertEqual({d.code for d in warnings}, {"geometry-anomaly"})
        self.assertIn("aspect_ratio", warnings[0].message)
        # Advisory only.
        self.assertEqual(_by_severity(report, Severity.ERROR), [])
        self.assertTrue(report.ok)

    def test_clear_on_normal_part(self):
        model = AnomalyModel().fit(_normal_corpus())
        backend = _MeasuredBackend(bbox=(10, 10, 10), volume=1000.0,
                                   surface_area=600.0)
        report = AnomalyCheck(model).check(backend, None)
        self.assertIn("anomaly-clear", _codes(report))
        self.assertEqual(_by_severity(report, Severity.WARNING), [])

    def test_with_anomaly_appends_without_mutating(self):
        base = ["a", "b"]
        result = with_anomaly(base)
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[-1], AnomalyCheck)
        self.assertEqual(result[-1].name, "anomaly")
        self.assertEqual(base, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
