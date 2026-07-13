"""Tests for the MCP-style tool-server surface (mcp/ package).

Covers docs/blueprint.md sec.5 (tools=action space, resources=observations,
prompts=op templates, tool-result reward, reset tool, destructive/read-only
annotations) and sec.9 (typed op schema, 5-component descriptions, typed errors)
plus the CADGymEnv reset/step contract.
"""

import json
import unittest

from harnesscad.core.cisp.ops import _REGISTRY
from harnesscad.core.loop import HarnessSession
from harnesscad.io.backends.stub import StubBackend
from harnesscad.io.surfaces.mcp.annotations import (
    TIER_AUTO, TIER_NOTIFY, TIER_REQUIRE, annotate, approval_tier,
)
from harnesscad.io.surfaces.mcp.tools import (
    ToolCatalog, ToolDefinition, ToolResult,
    UnknownToolError, ToolValidationError, ToolExecutionError,
    reward_from_apply,
)
from harnesscad.io.surfaces.mcp.gym import CADGymEnv


PLATE = [
    {"op": "new_sketch", "plane": "XY"},
    {"op": "add_rectangle", "sketch": "sk1", "x": 0.0, "y": 0.0, "w": 20.0, "h": 10.0},
    {"op": "constrain", "kind": "distance", "a": "e1", "value": 20.0},
    {"op": "constrain", "kind": "distance", "a": "e1", "value": 10.0},
    {"op": "constrain", "kind": "distance", "a": "e1", "value": 20.0},
    {"op": "constrain", "kind": "distance", "a": "e1", "value": 10.0},
    {"op": "extrude", "sketch": "sk1", "distance": 5.0},
]


class TestCatalogDerivesOps(unittest.TestCase):
    def setUp(self):
        self.cat = ToolCatalog()

    def test_one_tool_per_op(self):
        op_names = {t.name for t in self.cat.op_tools()}
        self.assertEqual(op_names, set(_REGISTRY.keys()))

    def test_aux_tools_present(self):
        for name in ("measure", "query", "verify", "run_check", "export",
                     "reset", "render"):
            self.assertIn(name, self.cat, name)

    def test_every_op_tool_has_complete_5_part_description(self):
        for t in self.cat.op_tools():
            d = t.description
            self.assertTrue(d.is_complete(), t.name)
            for part in (d.what, d.when, d.when_not, d.side_effects, d.output):
                self.assertTrue(part and part.strip(), t.name)
            # text() renders all five labelled components
            txt = d.text()
            for label in ("When to use", "When NOT", "Side effects", "Output"):
                self.assertIn(label, txt, t.name)

    def test_op_tools_have_typed_params_from_fields(self):
        import dataclasses
        for tag, cls in _REGISTRY.items():
            tool = self.cat[tag]
            pnames = {p.name for p in tool.params}
            fnames = {f.name for f in dataclasses.fields(cls)}
            self.assertEqual(pnames, fnames, tag)
            for p in tool.params:
                self.assertIn(p.type, ("string", "number", "integer",
                                        "boolean", "array"), p.name)

    def test_enums_populated(self):
        constrain = self.cat["constrain"]
        kind = next(p for p in constrain.params if p.name == "kind")
        self.assertIn("distance", kind.enum)
        boolean = self.cat["boolean"]
        bkind = next(p for p in boolean.params if p.name == "kind")
        self.assertEqual(set(bkind.enum), {"union", "cut", "intersect"})

    def test_required_reference_params(self):
        extrude = self.cat["extrude"]
        sketch = next(p for p in extrude.params if p.name == "sketch")
        self.assertTrue(sketch.required)


class TestToMcpJsonSchema(unittest.TestCase):
    def test_to_mcp_is_json_serialisable(self):
        cat = ToolCatalog()
        blob = json.dumps(cat.to_mcp())  # must not raise
        self.assertIn("new_sketch", blob)
        parsed = json.loads(blob)
        self.assertEqual(len(parsed), len(cat))

    def test_mcp_entries_have_schema_shape(self):
        cat = ToolCatalog()
        for entry in cat.to_mcp():
            self.assertIn("name", entry)
            self.assertIn("description", entry)
            self.assertEqual(entry["inputSchema"]["type"], "object")
            self.assertIn("properties", entry["inputSchema"])
            self.assertIn("required", entry["inputSchema"])
            self.assertIn("annotations", entry)

    def test_resources_and_prompts_json_able(self):
        cat = ToolCatalog()
        json.dumps(cat.resources())
        json.dumps(cat.prompts())
        self.assertTrue(any(r["name"] == "feature_tree" for r in cat.resources()))
        self.assertTrue(any(p["name"] == "rectangular_plate" for p in cat.prompts()))


class TestAnnotations(unittest.TestCase):
    def test_export_is_destructive_tier3(self):
        cat = ToolCatalog()
        ann = cat["export"].annotations
        self.assertTrue(ann.destructiveHint)
        self.assertFalse(ann.readOnlyHint)
        self.assertEqual(ann.tier, TIER_REQUIRE)

    def test_measure_is_readonly_tier1(self):
        cat = ToolCatalog()
        ann = cat["measure"].annotations
        self.assertTrue(ann.readOnlyHint)
        self.assertFalse(ann.destructiveHint)
        self.assertEqual(ann.tier, TIER_AUTO)

    def test_modify_op_is_default_tier2(self):
        cat = ToolCatalog()
        ann = cat["extrude"].annotations
        self.assertFalse(ann.readOnlyHint)
        self.assertFalse(ann.destructiveHint)
        self.assertEqual(ann.tier, TIER_NOTIFY)

    def test_reset_destructive_and_render_readonly(self):
        self.assertTrue(annotate("reset").destructiveHint)
        self.assertTrue(annotate("render").readOnlyHint)
        self.assertEqual(approval_tier("render"), TIER_AUTO)
        self.assertEqual(approval_tier("export"), TIER_REQUIRE)


class TestTypedErrors(unittest.TestCase):
    def setUp(self):
        self.cat = ToolCatalog()
        self.session = HarnessSession(StubBackend())

    def test_unknown_tool_raises(self):
        with self.assertRaises(UnknownToolError):
            self.cat.get("does_not_exist")

    def test_missing_required_param_raises(self):
        with self.assertRaises(ToolValidationError):
            self.cat.call("extrude", {"distance": 5.0}, session=self.session)

    def test_bad_enum_raises(self):
        with self.assertRaises(ToolValidationError):
            self.cat.call("boolean", {"kind": "explode", "target": "f1", "tool": "f2"},
                          session=self.session)

    def test_kernel_rejection_raises_execution_error_with_reward(self):
        # extrude a missing sketch -> backend rejects -> typed error, negative reward
        with self.assertRaises(ToolExecutionError) as ctx:
            self.cat.call("extrude", {"sketch": "nope", "distance": 5.0},
                          session=self.session)
        self.assertEqual(ctx.exception.data.get("reward"), -1.0)
        self.assertTrue(ctx.exception.data.get("diagnostics"))


class TestToolResultReward(unittest.TestCase):
    def test_successful_op_returns_toolresult_with_reward(self):
        cat = ToolCatalog()
        session = HarnessSession(StubBackend())
        res = cat.call("new_sketch", {"plane": "XY"}, session=session)
        self.assertIsInstance(res, ToolResult)
        self.assertTrue(res.ok)
        self.assertIn("reward", res.to_dict())
        self.assertGreaterEqual(res.reward, 0.0)
        self.assertIn("reward", res.content)

    def test_verify_tool_reward_reflects_state(self):
        cat = ToolCatalog()
        session = HarnessSession(StubBackend())
        session.apply_ops([__import__("harnesscad.core.cisp.ops", fromlist=["parse_op"]).parse_op(o)
                           for o in PLATE])
        res = cat.call("verify", {}, session=session)
        self.assertTrue(res.ok)
        self.assertGreater(res.reward, 0.0)

    def test_read_resource_reports_state(self):
        cat = ToolCatalog()
        session = HarnessSession(StubBackend())
        tree = cat.read_resource("cad://model/tree", session)
        self.assertIn("summary", tree)
        self.assertEqual(tree["summary"]["feature_count"], 0)


class TestGymEnv(unittest.TestCase):
    def test_reset_returns_compact_obs_without_ground_truth(self):
        env = CADGymEnv()
        obs = env.reset()
        # compact obs keys, and NO leaked target/answer
        self.assertIn("feature_tree", obs)
        self.assertIn("validity", obs)
        self.assertIn("digest", obs)
        for banned in ("target", "answer", "ground_truth", "solution", "expected"):
            self.assertNotIn(banned, obs)
        # empty model at reset
        self.assertEqual(obs["feature_tree"]["summary"]["feature_count"], 0)
        self.assertEqual(obs["feature_tree"]["ops"], [])
        # obs carries no raw image bytes (availability only)
        for v in obs["render"].get("views", {}).values():
            self.assertIsInstance(v, bool)

    def test_step_applies_op_and_returns_quad(self):
        env = CADGymEnv()
        obs, reward, done, info = env.step({"op": "new_sketch", "plane": "XY"})
        self.assertIsInstance(obs, dict)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
        self.assertTrue(info["ok"])
        self.assertEqual(info["applied"], 1)
        self.assertEqual(obs["feature_tree"]["summary"]["sketch_count"], 1)

    def test_step_sane_reward_pass_positive_fail_negative(self):
        env = CADGymEnv()
        # a valid full plate -> positive reward
        _, reward, _, info = env.step(PLATE)
        self.assertTrue(info["ok"])
        self.assertGreater(reward, 0.0)
        # a bad op (extrude missing sketch) -> negative reward, rejected
        env2 = CADGymEnv()
        _, reward2, _, info2 = env2.step({"op": "extrude", "sketch": "nope",
                                          "distance": 5.0})
        self.assertFalse(info2["ok"])
        self.assertLess(reward2, 0.0)
        self.assertIsNotNone(info2["rejected"])

    def test_reset_clears_state(self):
        env = CADGymEnv()
        env.step(PLATE)
        self.assertGreater(env.state()["measurements"]["summary"]["feature_count"], 0)
        obs = env.reset()
        self.assertEqual(obs["feature_tree"]["summary"]["feature_count"], 0)
        self.assertEqual(env.state()["measurements"]["summary"]["feature_count"], 0)
        self.assertEqual(obs["feature_tree"]["ops"], [])

    def test_info_carries_reward_field(self):
        env = CADGymEnv()
        _, reward, _, info = env.step({"op": "new_sketch", "plane": "XY"})
        self.assertEqual(info["reward"], reward)

    def test_action_space_is_op_tools(self):
        env = CADGymEnv()
        self.assertEqual(set(env.action_space()), set(_REGISTRY.keys()))

    def test_max_steps_sets_done(self):
        env = CADGymEnv(max_steps=1)
        _, _, done, _ = env.step({"op": "new_sketch", "plane": "XY"})
        self.assertTrue(done)

    def test_render_never_raises(self):
        env = CADGymEnv()
        env.step(PLATE)
        out = env.render()  # headless stub -> note, no crash
        self.assertIn("note", out)
        self.assertIsInstance(out["any_rendered"], bool)


class TestRewardHelper(unittest.TestCase):
    def test_reward_from_apply_pass_positive(self):
        session = HarnessSession(StubBackend())
        from harnesscad.core.cisp.ops import parse_op
        result = session.apply_ops([parse_op(o) for o in PLATE])
        self.assertTrue(result.ok)
        self.assertGreater(reward_from_apply(result), 0.0)

    def test_reward_from_apply_fail_negative(self):
        session = HarnessSession(StubBackend())
        from harnesscad.core.cisp.ops import parse_op
        result = session.apply_ops([parse_op({"op": "extrude", "sketch": "x",
                                              "distance": 1.0})])
        self.assertFalse(result.ok)
        self.assertEqual(reward_from_apply(result), -1.0)


if __name__ == "__main__":
    unittest.main()
