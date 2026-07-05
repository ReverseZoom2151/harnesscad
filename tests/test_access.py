"""Tests for the tool-access / serviceability clearance gate (verifiers.access).

The exact OCCT swept-common path needs cadquery/OCCT, which the dependency-free
test environment may not have; these tests exercise the *pure-python
bbox-corridor fallback* (always available) plus graceful degradation:

  * a hole whose approach corridor is blocked by another instance -> WARNING
    'no-tool-access';
  * a clear hole -> no warning, report ok, INFO 'access-clear';
  * INFO-skip on a StubBackend with no 'assembly' query, and never a crash.
"""

import unittest

from backends.stub import StubBackend
from verifiers.verify import Severity
from verifiers.access import (
    AccessCheck, AccessRules, with_access,
    _collect_features, _corridor_bbox, _overlap_dims, _normalize,
)


def _codes(report):
    return {d.code for d in report.diagnostics}


def _by_severity(report, sev):
    return [d for d in report.diagnostics if d.severity is sev]


class _AccessBackend:
    """Minimal assembly-aware backend: parts (bodies) + serviceable features."""

    def __init__(self, parts, features):
        self._parts = parts
        self._features = features

    def query(self, q: str) -> dict:
        if q == "assembly":
            return {"parts": self._parts, "features": self._features, "mates": []}
        return {}


def _plate(id_, xmin, ymin, zmin, xmax, ymax, zmax):
    return {"id": id_, "bbox": [xmin, ymin, zmin, xmax, ymax, zmax]}


def _hole(id_, x, y, z, owner, axis=None, diameter=6.0):
    f = {"id": id_, "kind": "hole", "position": [x, y, z],
         "diameter": diameter, "part": owner}
    if axis is not None:
        f["axis"] = axis
    return f


class TestRules(unittest.TestCase):
    def test_roundtrip(self):
        r = AccessRules(tool_diameter=10.0, approach_length=40.0, min_clearance=3.0)
        self.assertEqual(AccessRules.from_dict(r.to_dict()), r)

    def test_from_dict_defaults(self):
        r = AccessRules.from_dict(None)
        self.assertEqual(r, AccessRules())
        self.assertEqual(AccessRules.from_dict({"tool_diameter": 5}).tool_diameter, 5.0)


class TestCorridorGeometry(unittest.TestCase):
    def test_corridor_bbox_along_z(self):
        # Feature at origin, +Z, length 30, radius 4.
        bb = _corridor_bbox((0, 0, 0), (0, 0, 1), 30.0, 4.0)
        self.assertEqual(bb, (-4.0, -4.0, -4.0, 4.0, 4.0, 34.0))

    def test_overlap_dims(self):
        a = (0, 0, 0, 10, 10, 10)
        b = (6, 0, 0, 16, 4, 10)
        self.assertEqual(_overlap_dims(a, b), (4.0, 4.0, 10.0))

    def test_normalize(self):
        self.assertEqual(_normalize((0, 0, 5)), (0.0, 0.0, 1.0))
        self.assertEqual(_normalize((0, 0, 0)), (0.0, 0.0, 1.0))  # degenerate -> +Z


class TestAccessFallback(unittest.TestCase):
    """Pure-python bbox-corridor fallback (no cadquery required)."""

    def test_blocked_corridor_flagged(self):
        # Plate at z in [0,5]; hole on its top face at (0,0,5) drilling +Z.
        # An obstruction body sits directly above the hole, in the approach path.
        plate = _plate("plate", -20, -20, 0, 20, 20, 5)
        blocker = _plate("cover", -20, -20, 8, 20, 20, 12)  # spans over the hole
        feats = [_hole("h1", 0, 0, 5, owner="plate")]
        report = AccessCheck().check(
            _AccessBackend([plate, blocker], feats), None)
        codes = _codes(report)
        self.assertIn("no-tool-access", codes)
        warns = _by_severity(report, Severity.WARNING)
        self.assertTrue(any(c.code == "no-tool-access" and "h1" in c.message
                            for c in warns))
        # Advisory only -> never flips the report to not-ok.
        self.assertEqual(_by_severity(report, Severity.ERROR), [])
        self.assertTrue(report.ok)

    def test_clear_corridor_passes(self):
        # Same plate + hole, but the obstruction is far to the side.
        plate = _plate("plate", -20, -20, 0, 20, 20, 5)
        elsewhere = _plate("bracket", 100, 100, 0, 110, 110, 40)
        feats = [_hole("h1", 0, 0, 5, owner="plate")]
        report = AccessCheck().check(
            _AccessBackend([plate, elsewhere], feats), None)
        codes = _codes(report)
        self.assertNotIn("no-tool-access", codes)
        self.assertNotIn("tight-clearance", codes)
        self.assertIn("access-clear", codes)
        self.assertTrue(report.ok)

    def test_tight_clearance_flagged(self):
        # Obstruction clears the tool radius but sits within min_clearance.
        # tool_diameter 8 -> radius 4; min_clearance 2 -> grown radius 6.
        plate = _plate("plate", -20, -20, 0, 20, 20, 5)
        # A wall whose near face is at x=5: 1 mm outside the r=4 core corridor
        # (edge at x=4) but inside the grown r=6 corridor (edge at x=6).
        wall = _plate("wall", 5, -20, 0, 9, 20, 40)
        feats = [_hole("h1", 0, 0, 5, owner="plate")]
        rules = AccessRules(tool_diameter=8.0, approach_length=30.0, min_clearance=2.0)
        report = AccessCheck(rules).check(_AccessBackend([plate, wall], feats), None)
        codes = _codes(report)
        self.assertIn("tight-clearance", codes)
        self.assertNotIn("no-tool-access", codes)
        self.assertTrue(report.ok)

    def test_owner_body_never_blocks_own_feature(self):
        # The plate the hole lives in overlaps the corridor's base but must be
        # excluded as the feature's own host -> not a blockage on its own.
        plate = _plate("plate", -20, -20, 0, 20, 20, 5)
        feats = [_hole("h1", 0, 0, 5, owner="plate")]
        report = AccessCheck().check(_AccessBackend([plate], feats), None)
        self.assertNotIn("no-tool-access", _codes(report))
        self.assertIn("access-clear", _codes(report))
        self.assertTrue(report.ok)

    def test_check_access_direct(self):
        plate = _plate("plate", -20, -20, 0, 20, 20, 5)
        blocker = _plate("cover", -20, -20, 8, 20, 20, 12)
        feats = [_hole("h1", 0, 0, 5, owner="plate")]
        report = AccessCheck().check_access([plate, blocker], feats)
        self.assertIn("no-tool-access", _codes(report))


class TestFeatureCollection(unittest.TestCase):
    def test_top_level_and_per_part(self):
        raw = {
            "parts": [{"id": "p1", "features": [{"id": "fp", "kind": "screw"}]}],
            "features": [{"id": "ft", "kind": "hole"}],
        }
        feats = _collect_features(raw)
        ids = {f["id"] for f in feats}
        self.assertEqual(ids, {"fp", "ft"})
        # Per-part feature inherits its host as owner.
        fp = next(f for f in feats if f["id"] == "fp")
        self.assertEqual(fp["owner"], "p1")

    def test_part_as_serviceable_feature(self):
        raw = {"parts": [{"id": "bolt1", "kind": "bolt", "position": [1, 2, 3]}]}
        feats = _collect_features(raw)
        self.assertEqual(len(feats), 1)
        self.assertEqual(feats[0]["pos"], (1.0, 2.0, 3.0))


class TestDegradation(unittest.TestCase):
    def test_stub_info_skips(self):
        report = AccessCheck().check(StubBackend(), None)
        self.assertIn("access-skipped", _codes(report))
        self.assertTrue(report.ok)

    def test_no_features_info_skips(self):
        report = AccessCheck().check(
            _AccessBackend([_plate("p", 0, 0, 0, 1, 1, 1)], []), None)
        self.assertIn("no-serviceable-features", _codes(report))
        self.assertTrue(report.ok)

    def test_not_measurable_when_no_bbox(self):
        # Obstruction with neither bbox nor shape cannot be tested.
        feats = [_hole("h1", 0, 0, 0, owner="plate")]
        parts = [{"id": "plate"}, {"id": "ghost"}]
        report = AccessCheck().check(_AccessBackend(parts, feats), None)
        self.assertIn("access-not-measurable", _codes(report))
        self.assertTrue(report.ok)

    def test_never_crashes_on_broken_backend(self):
        class _Boom:
            def query(self, q):
                raise RuntimeError("nope")

        report = AccessCheck().check(_Boom(), None)
        self.assertIn("access-skipped", _codes(report))
        self.assertTrue(report.ok)


class TestWithAccess(unittest.TestCase):
    def test_appends_check(self):
        base = ["x"]
        result = with_access(base)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[-1], AccessCheck)
        self.assertEqual(result[-1].name, "access")
        self.assertEqual(base, ["x"])


if __name__ == "__main__":
    unittest.main()
