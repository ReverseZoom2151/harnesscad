"""Tests for the provider-agnostic LLM layer.

No network, no API keys: `MockLLM` implements the `LLM` protocol with canned
responses, and the litellm backend is exercised by monkeypatching
`litellm.completion` (asserting request construction), never by calling out.
"""

import json
import unittest
from typing import List, Optional

from harnesscad.agents.llm.base import (
    LLM,
    Message,
    ToolSpec,
    ToolCall,
    CompletionResult,
    system,
    user,
    assistant,
    messages_to_dicts,
)
from harnesscad.agents.llm.structured import (
    ops_from_json,
    ops_from_completion,
    validate_ops,
    validate_raw,
    OpParseError,
)
from harnesscad.core.cisp.ops import NewSketch, AddRectangle, Constrain, Extrude


# --- a reusable canned-response LLM (shared with test_planner) --------------
def plate_ops_json() -> str:
    ops = (
        [{"op": "new_sketch", "plane": "XY"},
         {"op": "add_rectangle", "sketch": "sk1", "x": 0, "y": 0, "w": 20, "h": 10}]
        + [{"op": "constrain", "kind": "distance", "a": "e1", "value": 10.0} for _ in range(4)]
        + [{"op": "extrude", "sketch": "sk1", "distance": 5.0}]
    )
    return json.dumps(ops)


class MockLLM(LLM):
    """Implements the LLM protocol; returns queued canned CompletionResults.

    Pass either raw JSON strings (wrapped into text responses) or ready
    CompletionResult objects. Records the messages it was called with.
    """

    def __init__(self, responses: List[object]) -> None:
        self._responses = list(responses)
        self.calls: List[List[Message]] = []

    def _next(self) -> CompletionResult:
        item = self._responses.pop(0) if self._responses else CompletionResult(text="[]")
        if isinstance(item, CompletionResult):
            return item
        return CompletionResult(text=str(item))

    def complete(self, messages, tools=None, response_schema=None, **opts) -> CompletionResult:
        self.calls.append(list(messages))
        return self._next()

    def stream(self, messages, tools=None, response_schema=None, **opts):
        yield self._next().text


class TestMessageAndTypes(unittest.TestCase):
    def test_message_helpers_and_to_dict(self):
        self.assertEqual(system("hi").role, "system")
        self.assertEqual(user("hi").role, "user")
        self.assertEqual(assistant("hi").role, "assistant")
        m = Message("tool", "result", name="emit_ops", tool_call_id="abc")
        self.assertEqual(
            m.to_dict(),
            {"role": "tool", "content": "result", "name": "emit_ops", "tool_call_id": "abc"},
        )

    def test_messages_to_dicts_accepts_dicts_and_messages(self):
        out = messages_to_dicts([user("a"), {"role": "system", "content": "b"}])
        self.assertEqual(out[0]["content"], "a")
        self.assertEqual(out[1]["role"], "system")

    def test_toolspec_to_openai_shape(self):
        spec = ToolSpec("emit_ops", "desc", {"type": "object", "properties": {}})
        d = spec.to_openai_tool()
        self.assertEqual(d["type"], "function")
        self.assertEqual(d["function"]["name"], "emit_ops")
        self.assertEqual(d["function"]["description"], "desc")

    def test_completionresult_has_tool_calls(self):
        self.assertFalse(CompletionResult(text="x").has_tool_calls)
        r = CompletionResult(tool_calls=[ToolCall("emit_ops", "{}")])
        self.assertTrue(r.has_tool_calls)

    def test_mockllm_is_llm(self):
        self.assertIsInstance(MockLLM([]), LLM)


class TestStructuredParsing(unittest.TestCase):
    def test_ops_from_json_array(self):
        ops = ops_from_json(plate_ops_json())
        self.assertEqual(len(ops), 7)
        self.assertIsInstance(ops[0], NewSketch)
        self.assertIsInstance(ops[1], AddRectangle)
        self.assertIsInstance(ops[-1], Extrude)
        self.assertEqual(ops[1].w, 20)

    def test_ops_from_json_object_wrapper(self):
        wrapped = json.dumps({"ops": json.loads(plate_ops_json())})
        self.assertEqual(len(ops_from_json(wrapped)), 7)

    def test_single_op_object(self):
        ops = ops_from_json(json.dumps({"op": "new_sketch", "plane": "YZ"}))
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0].plane, "YZ")

    def test_invalid_json_raises(self):
        with self.assertRaises(OpParseError):
            ops_from_json("not json {")

    def test_empty_raises(self):
        with self.assertRaises(OpParseError):
            ops_from_json("   ")

    def test_unknown_op_tag_raises(self):
        with self.assertRaises(OpParseError):
            ops_from_json(json.dumps([{"op": "teleport"}]))

    def test_missing_op_tag_raises(self):
        with self.assertRaises(OpParseError):
            ops_from_json(json.dumps([{"plane": "XY"}]))

    def test_bad_params_raise(self):
        with self.assertRaises(OpParseError):
            ops_from_json(json.dumps([{"op": "new_sketch", "bogus": 1}]))

    def test_ops_from_completion_text(self):
        ops = ops_from_completion(CompletionResult(text=plate_ops_json()))
        self.assertEqual(len(ops), 7)

    def test_ops_from_completion_tool_calls(self):
        tc = ToolCall("emit_ops", plate_ops_json())
        ops = ops_from_completion(CompletionResult(tool_calls=[tc]))
        self.assertEqual(len(ops), 7)
        self.assertIsInstance(ops[2], Constrain)


class TestValidateAndRetry(unittest.TestCase):
    def test_validate_ok(self):
        parsed = validate_ops(CompletionResult(text=plate_ops_json()))
        self.assertTrue(parsed.ok)
        self.assertIsNone(parsed.error)
        self.assertEqual(len(parsed.ops), 7)

    def test_validate_returns_error_not_raises(self):
        parsed = validate_ops(CompletionResult(text="garbage"))
        self.assertFalse(parsed.ok)
        self.assertIsInstance(parsed.error, str)
        self.assertEqual(parsed.ops, [])

    def test_validate_empty_ops_is_error(self):
        parsed = validate_ops(CompletionResult(text="[]"))
        self.assertFalse(parsed.ok)

    def test_validate_raw(self):
        self.assertTrue(validate_raw(plate_ops_json()).ok)
        self.assertFalse(validate_raw("{").ok)


class TestLiteLLMBackendRequestConstruction(unittest.TestCase):
    """Monkeypatch litellm.completion — assert the request, never hit network."""

    def _fake_completion_module(self, sink: dict, response=None):
        import types

        def fake_completion(**kwargs):
            sink.update(kwargs)
            if response is not None:
                return response
            return {
                "choices": [
                    {"message": {"content": "[]", "tool_calls": None},
                     "finish_reason": "stop"}
                ]
            }

        mod = types.ModuleType("litellm")
        mod.completion = fake_completion
        return mod

    def _install(self, module):
        import sys
        sys.modules["litellm"] = module

    def test_complete_builds_model_and_messages(self):
        from harnesscad.agents.llm.litellm_backend import LiteLLMClient

        sink = {}
        self._install(self._fake_completion_module(sink))
        client = LiteLLMClient(model="gpt-4o-mini")
        result = client.complete([system("sys"), user("hi")])
        self.assertEqual(sink["model"], "gpt-4o-mini")
        self.assertEqual(sink["messages"][0], {"role": "system", "content": "sys"})
        self.assertEqual(sink["messages"][1], {"role": "user", "content": "hi"})
        self.assertEqual(sink["temperature"], 0.0)  # temperature=0 default
        self.assertNotIn("stream", sink)
        # And the response is adapted into our CompletionResult.
        self.assertIsInstance(result, CompletionResult)
        self.assertEqual(result.finish_reason, "stop")

    def test_complete_maps_tools(self):
        from harnesscad.agents.llm.litellm_backend import LiteLLMClient

        sink = {}
        self._install(self._fake_completion_module(sink))
        client = LiteLLMClient(model="claude-3-5-sonnet")
        client.complete([user("hi")], tools=[ToolSpec("emit_ops", "d", {"type": "object"})])
        self.assertEqual(sink["tools"][0]["function"]["name"], "emit_ops")
        self.assertEqual(sink["tool_choice"], "auto")

    def test_complete_maps_response_schema(self):
        from harnesscad.agents.llm.litellm_backend import LiteLLMClient

        sink = {}
        self._install(self._fake_completion_module(sink))
        client = LiteLLMClient()
        client.complete([user("hi")], response_schema={"type": "object", "title": "Plan"})
        self.assertEqual(sink["response_format"]["type"], "json_schema")
        self.assertEqual(sink["response_format"]["json_schema"]["name"], "Plan")

    def test_opts_and_overrides_forwarded(self):
        from harnesscad.agents.llm.litellm_backend import LiteLLMClient

        sink = {}
        self._install(self._fake_completion_module(sink))
        client = LiteLLMClient(model="gpt-4o-mini", api_key="sk-test", temperature=0.0)
        client.complete([user("hi")], temperature=0.7)
        self.assertEqual(sink["api_key"], "sk-test")
        self.assertEqual(sink["temperature"], 0.7)  # per-call override wins

    def test_parse_response_extracts_tool_calls(self):
        from harnesscad.agents.llm.litellm_backend import LiteLLMClient

        response = {
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [
                        {"id": "call_1",
                         "function": {"name": "emit_ops", "arguments": plate_ops_json()}}
                    ],
                },
                "finish_reason": "tool_calls",
            }]
        }
        sink = {}
        self._install(self._fake_completion_module(sink, response=response))
        client = LiteLLMClient()
        result = client.complete([user("hi")])
        self.assertTrue(result.has_tool_calls)
        self.assertEqual(result.tool_calls[0].name, "emit_ops")
        ops = ops_from_completion(result)
        self.assertEqual(len(ops), 7)

    def test_stream_yields_text_chunks(self):
        from harnesscad.agents.llm.litellm_backend import LiteLLMClient
        import types

        chunks = [
            {"choices": [{"delta": {"content": "ab"}}]},
            {"choices": [{"delta": {"content": "cd"}}]},
            {"choices": [{"delta": {"content": None}}]},
        ]
        sink = {}

        def fake_completion(**kwargs):
            sink.update(kwargs)
            return iter(chunks)

        mod = types.ModuleType("litellm")
        mod.completion = fake_completion
        self._install(mod)
        client = LiteLLMClient()
        out = "".join(client.stream([user("hi")]))
        self.assertEqual(out, "abcd")
        self.assertTrue(sink["stream"])


if __name__ == "__main__":
    unittest.main()
