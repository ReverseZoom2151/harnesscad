import json
import unittest

from adapters.t2cblender_zoo_api_schema import (
    DEFAULT_BASE_URL,
    SUPPORTED_FORMATS,
    Operation,
    OperationStatus,
    OutputFormat,
    ZooApiError,
    build_poll_request,
    build_submit_request,
    output_key,
    parse_operation,
)


class OutputFormatTest(unittest.TestCase):
    def test_supported_formats(self):
        self.assertEqual(
            set(SUPPORTED_FORMATS), {"fbx", "glb", "gltf", "obj", "ply", "stl"}
        )

    def test_coerce_case_insensitive(self):
        self.assertIs(OutputFormat.coerce("STL"), OutputFormat.stl)
        self.assertIs(OutputFormat.coerce(OutputFormat.glb), OutputFormat.glb)

    def test_coerce_bad(self):
        with self.assertRaises(ZooApiError):
            OutputFormat.coerce("step")

    def test_output_key(self):
        self.assertEqual(output_key("stl"), "source.stl")
        self.assertEqual(output_key(OutputFormat.glb), "source.glb")


class SubmitRequestTest(unittest.TestCase):
    def test_url_and_headers(self):
        req = build_submit_request("a plate", "stl", "TOKEN123")
        self.assertEqual(req.method, "POST")
        self.assertEqual(req.url, DEFAULT_BASE_URL + "/ai/text-to-cad/stl")
        self.assertEqual(req.headers["Authorization"], "Bearer TOKEN123")
        self.assertEqual(req.headers["Content-Type"], "application/json")
        # User-Agent is mandatory to avoid HTTP 403.
        self.assertIn("User-Agent", req.headers)

    def test_body_is_json_prompt(self):
        req = build_submit_request("Create a gear", "glb", "T")
        self.assertEqual(json.loads(req.body.decode("utf-8")), {"prompt": "Create a gear"})

    def test_deterministic_body(self):
        a = build_submit_request("same", "obj", "T")
        b = build_submit_request("same", "obj", "T")
        self.assertEqual(a, b)
        self.assertEqual(a.body, b.body)

    def test_empty_prompt_rejected(self):
        with self.assertRaises(ZooApiError):
            build_submit_request("   ", "stl", "T")

    def test_empty_token_rejected(self):
        with self.assertRaises(ZooApiError):
            build_submit_request("x", "stl", "")

    def test_custom_base_url_trailing_slash(self):
        req = build_submit_request("x", "ply", "T", base_url="http://local/")
        self.assertEqual(req.url, "http://local/ai/text-to-cad/ply")


class PollRequestTest(unittest.TestCase):
    def test_poll(self):
        req = build_poll_request("op-42", "T")
        self.assertEqual(req.method, "GET")
        self.assertEqual(req.url, DEFAULT_BASE_URL + "/async/operations/op-42")
        self.assertEqual(req.headers["Authorization"], "Bearer T")
        self.assertIsNone(req.body)

    def test_empty_id_rejected(self):
        with self.assertRaises(ZooApiError):
            build_poll_request("", "T")


class OperationStatusTest(unittest.TestCase):
    def test_terminal(self):
        self.assertTrue(OperationStatus.completed.is_terminal)
        self.assertTrue(OperationStatus.failed.is_terminal)
        self.assertFalse(OperationStatus.in_progress.is_terminal)
        self.assertFalse(OperationStatus.queued.is_terminal)

    def test_coerce_unknown(self):
        with self.assertRaises(ZooApiError):
            OperationStatus.coerce("exploded")


class ParseOperationTest(unittest.TestCase):
    def test_parse_intermediate(self):
        op = parse_operation({"id": "abc", "status": "in_progress"})
        self.assertEqual(op.id, "abc")
        self.assertFalse(op.is_terminal)
        self.assertEqual(op.outputs, {})

    def test_parse_completed_payload(self):
        op = parse_operation(
            {
                "id": "abc",
                "status": "completed",
                "outputs": {"source.stl": "AAAA"},
            }
        )
        self.assertTrue(op.is_completed)
        self.assertEqual(op.payload_for("stl"), "AAAA")

    def test_payload_missing_key(self):
        op = parse_operation(
            {"id": "abc", "status": "completed", "outputs": {"source.glb": "x"}}
        )
        with self.assertRaises(ZooApiError):
            op.payload_for("stl")

    def test_payload_not_completed(self):
        op = parse_operation({"id": "abc", "status": "queued"})
        with self.assertRaises(ZooApiError):
            op.payload_for("stl")

    def test_parse_failed(self):
        op = parse_operation(
            {"id": "z", "status": "failed", "error": "bad prompt"}
        )
        self.assertTrue(op.is_failed)
        self.assertEqual(op.error, "bad prompt")

    def test_missing_fields(self):
        with self.assertRaises(ZooApiError):
            parse_operation({"status": "queued"})
        with self.assertRaises(ZooApiError):
            parse_operation({"id": "a"})

    def test_bad_outputs_type(self):
        with self.assertRaises(ZooApiError):
            parse_operation({"id": "a", "status": "completed", "outputs": [1, 2]})


if __name__ == "__main__":
    unittest.main()
