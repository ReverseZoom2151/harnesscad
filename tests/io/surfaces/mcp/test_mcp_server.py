"""Unit tests for the real MCP server (surfaces.mcp.server.MCPServer).

Drives ``MCPServer.handle`` in-process (no real stdio) over a StubBackend
session: the initialize handshake + protocol-version negotiation, tools/list
cleaning + ``_meta`` relocation, tools/call success and op-rejection (isError)
paths, unknown-tool errors, resources, prompts rendering, and unknown methods.
Deterministic; no network.
"""

import unittest

from harnesscad.io.backends.stub import StubBackend
from harnesscad.core.loop import HarnessSession
from harnesscad.io.surfaces.mcp.jsonrpc import (
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    RESOURCE_NOT_FOUND,
)
from harnesscad.io.surfaces.mcp.server import (
    PROTOCOL_VERSION,
    SERVER_NAME,
    MCPServer,
)


def _req(id, method, params=None):
    msg = {"jsonrpc": "2.0", "id": id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


class MCPServerTest(unittest.TestCase):
    def setUp(self):
        self.session = HarnessSession(StubBackend())
        self.server = MCPServer(session=self.session)

    # --- initialize -------------------------------------------------------
    def test_initialize_handshake_and_capabilities(self):
        resp = self.server.handle(_req(1, "initialize",
                                       {"protocolVersion": "2025-11-25"}))
        self.assertEqual(resp["id"], 1)
        self.assertEqual(resp["jsonrpc"], "2.0")
        result = resp["result"]
        self.assertEqual(result["protocolVersion"], "2025-11-25")
        self.assertEqual(result["serverInfo"]["name"], SERVER_NAME)
        self.assertIn("version", result["serverInfo"])
        caps = result["capabilities"]
        self.assertEqual(caps["tools"], {"listChanged": False})
        self.assertEqual(caps["resources"], {"listChanged": False})
        self.assertEqual(caps["prompts"], {"listChanged": False})

    def test_initialize_negotiates_supported_older_version(self):
        resp = self.server.handle(_req(1, "initialize",
                                       {"protocolVersion": "2025-06-18"}))
        self.assertEqual(resp["result"]["protocolVersion"], "2025-06-18")

    def test_initialize_falls_back_on_unknown_version(self):
        resp = self.server.handle(_req(1, "initialize",
                                       {"protocolVersion": "1999-01-01"}))
        self.assertEqual(resp["result"]["protocolVersion"], PROTOCOL_VERSION)

    # --- notifications / ping --------------------------------------------
    def test_initialized_notification_returns_none(self):
        self.assertIsNone(
            self.server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))

    def test_bare_notification_without_id_returns_none(self):
        self.assertIsNone(
            self.server.handle({"jsonrpc": "2.0", "method": "notifications/cancelled"}))

    def test_ping_returns_empty(self):
        resp = self.server.handle(_req(7, "ping"))
        self.assertEqual(resp["result"], {})

    # --- tools/list -------------------------------------------------------
    def test_tools_list_cleaned_with_meta_relocation(self):
        resp = self.server.handle(_req(2, "tools/list"))
        tools = resp["result"]["tools"]
        self.assertTrue(tools)
        by_name = {t["name"]: t for t in tools}
        # An op tool keeps only spec keys and relocates non-spec keys under _meta.
        sk = by_name["new_sketch"]
        self.assertIn("inputSchema", sk)
        self.assertIn("outputSchema", sk)
        self.assertIn("annotations", sk)
        self.assertIn("readOnlyHint", sk["annotations"])
        self.assertNotIn("op", sk)
        self.assertNotIn("descriptionComponents", sk)
        self.assertIn("_meta", sk)
        self.assertEqual(sk["_meta"]["com.harnesscad/op"], "new_sketch")
        self.assertIn("com.harnesscad/descriptionComponents", sk["_meta"])

    # --- tools/call -------------------------------------------------------
    def test_tools_call_valid_op_returns_call_result(self):
        resp = self.server.handle(_req(3, "tools/call",
                                       {"name": "new_sketch",
                                        "arguments": {"plane": "XY"}}))
        result = resp["result"]
        self.assertFalse(result["isError"])
        self.assertIn("structuredContent", result)
        self.assertTrue(result["structuredContent"]["ok"])
        self.assertEqual(result["content"][0]["type"], "text")

    def test_tools_call_rejected_op_returns_is_error_with_diagnostics(self):
        # add_circle on a non-existent sketch is rejected by the backend.
        resp = self.server.handle(_req(4, "tools/call",
                                       {"name": "add_circle",
                                        "arguments": {"sketch": "sk_missing", "r": 2.0}}))
        result = resp["result"]
        self.assertTrue(result["isError"])
        self.assertIn("diagnostics", result["structuredContent"])
        self.assertTrue(result["structuredContent"]["diagnostics"])
        self.assertEqual(result["content"][0]["type"], "text")

    def test_tools_call_unknown_tool_is_invalid_params(self):
        resp = self.server.handle(_req(5, "tools/call",
                                       {"name": "does_not_exist", "arguments": {}}))
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], INVALID_PARAMS)

    def test_tools_call_missing_required_param_is_invalid_params(self):
        # extrude requires 'sketch'; omitting it is a validation error.
        resp = self.server.handle(_req(6, "tools/call",
                                       {"name": "extrude", "arguments": {}}))
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], INVALID_PARAMS)

    # --- resources --------------------------------------------------------
    def test_resources_list_and_read(self):
        listed = self.server.handle(_req(8, "resources/list"))["result"]["resources"]
        self.assertTrue(listed)
        uri = listed[0]["uri"]
        read = self.server.handle(_req(9, "resources/read", {"uri": uri}))["result"]
        contents = read["contents"]
        self.assertEqual(contents[0]["uri"], uri)
        self.assertEqual(contents[0]["mimeType"], "application/json")
        self.assertIn("text", contents[0])

    def test_resources_read_unknown_uri_is_resource_not_found(self):
        resp = self.server.handle(_req(10, "resources/read",
                                       {"uri": "cad://model/nope"}))
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], RESOURCE_NOT_FOUND)

    # --- prompts ----------------------------------------------------------
    def test_prompts_list_strips_internal_template(self):
        prompts = self.server.handle(_req(11, "prompts/list"))["result"]["prompts"]
        self.assertTrue(prompts)
        for p in prompts:
            self.assertNotIn("template", p)
            self.assertIn("name", p)

    def test_prompts_get_renders_messages_with_substituted_args(self):
        resp = self.server.handle(_req(12, "prompts/get",
                                       {"name": "cylinder",
                                        "arguments": {"radius": 5, "height": 12}}))
        result = resp["result"]
        messages = result["messages"]
        self.assertEqual(messages[0]["role"], "user")
        text = messages[0]["content"]["text"]
        self.assertEqual(messages[0]["content"]["type"], "text")
        # The resolved op list is embedded as JSON with the arg values substituted.
        self.assertIn("5", text)
        self.assertIn("12", text)
        self.assertIn("new_sketch", text)

    def test_prompts_get_unknown_is_invalid_params(self):
        resp = self.server.handle(_req(13, "prompts/get", {"name": "nope"}))
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], INVALID_PARAMS)

    # --- unknown method ---------------------------------------------------
    def test_unknown_method_is_method_not_found(self):
        resp = self.server.handle(_req(14, "frobnicate"))
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], METHOD_NOT_FOUND)


if __name__ == "__main__":
    unittest.main()
