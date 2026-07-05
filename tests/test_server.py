"""Tests for the CISP server — drives CISPServer.handle programmatically."""

import unittest

from server import CISPServer


PLATE_OPS = [
    {"op": "new_sketch", "plane": "XY"},
    {"op": "add_rectangle", "sketch": "sk1", "x": 0.0, "y": 0.0, "w": 20.0, "h": 10.0},
    {"op": "constrain", "kind": "distance", "a": "e1", "value": 20.0},
    {"op": "constrain", "kind": "distance", "a": "e1", "value": 10.0},
    {"op": "constrain", "kind": "distance", "a": "e1", "value": 20.0},
    {"op": "constrain", "kind": "distance", "a": "e1", "value": 10.0},
    {"op": "extrude", "sketch": "sk1", "distance": 5.0},
]


def _handle(server, method, **params):
    return server.handle({"id": 1, "method": method, "params": params})


class TestInitialize(unittest.TestCase):
    def test_initialize_reports_capabilities_and_ops(self):
        server = CISPServer()
        resp = _handle(server, "initialize")
        self.assertTrue(resp["ok"])
        result = resp["result"]
        self.assertEqual(result["protocol"], "cisp")
        self.assertEqual(result["backend"], "stub")
        self.assertIn("new_sketch", result["ops"])
        self.assertIn("extrude", result["ops"])
        self.assertTrue(result["capabilities"]["applyOps"])


class TestApplyQueryVerifyExport(unittest.TestCase):
    def test_full_flow(self):
        server = CISPServer()

        # initialize
        self.assertTrue(_handle(server, "initialize")["ok"])

        # applyOps with the sample -> ok + a digest
        apply_resp = _handle(server, "applyOps", ops=PLATE_OPS)
        self.assertTrue(apply_resp["ok"])
        result = apply_resp["result"]
        self.assertTrue(result["ok"])
        self.assertEqual(result["applied"], len(PLATE_OPS))
        self.assertTrue(result["digest"])
        self.assertIsNone(result["rejected"])

        # query('summary') -> solid_present
        q = _handle(server, "query", what="summary")
        self.assertTrue(q["ok"])
        self.assertTrue(q["result"]["result"]["solid_present"])
        self.assertEqual(q["result"]["result"]["feature_count"], 1)

        # verify -> ok, no error diagnostics
        v = _handle(server, "verify")
        self.assertTrue(v["ok"])
        self.assertTrue(v["result"]["ok"])

        # export('step') -> content
        e = _handle(server, "export", fmt="step")
        self.assertTrue(e["ok"])
        self.assertEqual(e["result"]["fmt"], "step")
        self.assertIn("step", e["result"]["content"])

    def test_deterministic_digest(self):
        r1 = _handle(CISPServer(), "applyOps", ops=PLATE_OPS)["result"]
        r2 = _handle(CISPServer(), "applyOps", ops=PLATE_OPS)["result"]
        self.assertEqual(r1["digest"], r2["digest"])


class TestErrors(unittest.TestCase):
    def test_unknown_method(self):
        resp = CISPServer().handle({"id": 7, "method": "nope"})
        self.assertFalse(resp["ok"])
        self.assertEqual(resp["error"]["code"], "bad-request")
        self.assertEqual(resp["id"], 7)

    def test_bad_ref_blocks_and_reports(self):
        resp = _handle(CISPServer(), "applyOps",
                       ops=[{"op": "extrude", "sketch": "nope", "distance": 5.0}])
        self.assertTrue(resp["ok"])  # transport ok
        result = resp["result"]
        self.assertFalse(result["ok"])  # model rejected
        self.assertEqual(result["applied"], 0)
        self.assertIsNotNone(result["rejected"])


class TestStdio(unittest.TestCase):
    def test_serve_stdio_roundtrip(self):
        import io
        import json
        server = CISPServer()
        stdin = io.StringIO(json.dumps({"id": 1, "method": "initialize"}) + "\n")
        stdout = io.StringIO()
        server.serve_stdio(stdin=stdin, stdout=stdout)
        line = stdout.getvalue().strip()
        resp = json.loads(line)
        self.assertTrue(resp["ok"])
        self.assertEqual(resp["result"]["protocol"], "cisp")


if __name__ == "__main__":
    unittest.main()
