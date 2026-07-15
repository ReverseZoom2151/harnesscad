"""wire — the JSON envelope + reflective command dispatch. Pure, table of fakes."""

import unittest

from harnesscad.io.cua import wire


def _echo(x, y):
    return {"x": x, "y": y}


def _no_args():
    return {"ok": True}


def _boom():
    raise RuntimeError("kaboom")


def _reports_failure():
    return {"success": False, "error": "handler said no"}


def _takes_kwargs(**kw):
    return {"seen": sorted(kw)}


class RequestTest(unittest.TestCase):
    def test_parse_minimal(self):
        r = wire.Request.from_dict({"command": "screenshot"})
        self.assertEqual(r.command, "screenshot")
        self.assertEqual(r.params, {})

    def test_bad_envelope_raises(self):
        with self.assertRaises(wire.WireError):
            wire.Request.from_dict({"params": {}})
        with self.assertRaises(wire.WireError):
            wire.Request.from_dict(["not", "a", "dict"])
        with self.assertRaises(wire.WireError):
            wire.Request.from_dict({"command": "x", "params": [1, 2]})


class ResponseTest(unittest.TestCase):
    def test_ok_spreads_data(self):
        self.assertEqual(wire.Response.ok(a=1, b=2).to_dict(),
                         {"success": True, "a": 1, "b": 2})

    def test_fail_carries_error(self):
        d = wire.Response.fail("nope", suggestions=("a", "b")).to_dict()
        self.assertEqual(d["success"], False)
        self.assertEqual(d["error"], "nope")
        self.assertEqual(d["suggestions"], ["a", "b"])


class DispatchTest(unittest.TestCase):
    def setUp(self):
        self.d = wire.Dispatcher({
            "move_cursor": _echo, "screenshot": _no_args,
            "explode": _boom, "type_text": lambda text: {"typed": text},
            "guard": _reports_failure, "anything": _takes_kwargs,
        })

    def test_filters_params_to_signature(self):
        # extra 'z' is dropped; handler gets exactly x,y
        resp = self.d.dispatch({"command": "move_cursor",
                                "params": {"x": 3, "y": 4, "z": 99}})
        self.assertEqual(resp.to_dict(), {"success": True, "x": 3, "y": 4})

    def test_alias_resolution(self):
        # 'move' -> 'move_cursor', 'type' -> 'type_text' (default aliases)
        r1 = self.d.dispatch({"command": "move", "params": {"x": 1, "y": 2}})
        self.assertEqual(r1.data, {"x": 1, "y": 2})
        r2 = self.d.dispatch({"command": "type", "params": {"text": "hi"}})
        self.assertEqual(r2.data, {"typed": "hi"})

    def test_unknown_command_suggests(self):
        resp = self.d.dispatch({"command": "scree"})
        self.assertFalse(resp.success)
        self.assertIn("Unknown command", resp.error)
        self.assertIn("screenshot", resp.suggestions)

    def test_handler_exception_is_in_band(self):
        resp = self.d.dispatch({"command": "explode"})
        self.assertFalse(resp.success)
        self.assertIn("kaboom", resp.error)

    def test_missing_required_param_is_in_band(self):
        resp = self.d.dispatch({"command": "move_cursor", "params": {"x": 1}})
        self.assertFalse(resp.success)
        self.assertIn("bad params", resp.error)

    def test_handler_reported_failure_honoured(self):
        resp = self.d.dispatch({"command": "guard"})
        self.assertFalse(resp.success)
        self.assertIn("handler said no", resp.error)

    def test_kwargs_handler_gets_everything(self):
        resp = self.d.dispatch({"command": "anything",
                                "params": {"p": 1, "q": 2}})
        self.assertEqual(resp.data, {"seen": ["p", "q"]})

    def test_none_result_is_bare_success(self):
        d = wire.Dispatcher({"noop": lambda: None}, aliases={})
        self.assertEqual(d.dispatch({"command": "noop"}).to_dict(),
                         {"success": True})


class CatalogTest(unittest.TestCase):
    def test_catalog_reflects_signature_and_aliases(self):
        d = wire.Dispatcher({"move_cursor": _echo},
                            aliases={"move": "move_cursor"})
        cat = d.catalog()
        entry = cat["commands"]["move_cursor"]
        names = [p["name"] for p in entry["params"]]
        self.assertEqual(names, ["x", "y"])
        self.assertTrue(all(p["required"] for p in entry["params"]))
        self.assertEqual(entry["aliases"], ["move"])
        self.assertEqual(cat["aliases"], {"move": "move_cursor"})

    def test_optional_param_default_captured(self):
        d = wire.Dispatcher({"f": lambda a, b=7: {"a": a, "b": b}}, aliases={})
        params = {p["name"]: p for p in d.catalog()["commands"]["f"]["params"]}
        self.assertTrue(params["a"]["required"])
        self.assertFalse(params["b"]["required"])
        self.assertEqual(params["b"]["default"], 7)


if __name__ == "__main__":
    unittest.main()
