"""Tests for the trycua-derived model-agnostic capability router."""

import unittest

from harnesscad.agents.cua.capabilities import (
    Action, Capability, CapabilityRouter, Objective, Route, RouteError,
    can_serve, route,
)


class RouteTest(unittest.TestCase):
    def test_step_model_serves_next_action_natively(self):
        r = route(Objective.NEXT_ACTION, [Capability.PREDICT_STEP])
        self.assertEqual(r.kind, "native")
        self.assertEqual(r.steps, (Capability.PREDICT_STEP,))

    def test_click_only_needs_a_planner_for_next_action(self):
        # A grounding-only model cannot plan an action by itself.
        self.assertFalse(can_serve(Objective.NEXT_ACTION, [Capability.PREDICT_CLICK]))
        with self.assertRaises(RouteError):
            route(Objective.NEXT_ACTION, [Capability.PREDICT_CLICK])

    def test_planner_plus_clicker_composes(self):
        r = route(Objective.NEXT_ACTION,
                  [Capability.PLAN, Capability.PREDICT_CLICK])
        self.assertEqual(r.kind, "composed")
        self.assertEqual(r.steps, (Capability.PLAN, Capability.PREDICT_CLICK))

    def test_native_preferred_over_composed(self):
        # With BOTH available, the single-call native loop wins deterministically.
        r = route(Objective.NEXT_ACTION,
                  [Capability.PREDICT_STEP, Capability.PLAN, Capability.PREDICT_CLICK])
        self.assertEqual(r.kind, "native")

    def test_click_target_grounding(self):
        r = route(Objective.CLICK_TARGET, [Capability.PREDICT_CLICK])
        self.assertEqual(r.kind, "grounding")

    def test_click_target_via_step_model(self):
        r = route(Objective.CLICK_TARGET, [Capability.PREDICT_STEP])
        self.assertEqual(r.kind, "native")

    def test_click_target_unroutable_when_blind(self):
        with self.assertRaises(RouteError):
            route(Objective.CLICK_TARGET, [Capability.PLAN])


class RouterNativeTest(unittest.TestCase):
    def test_native_returns_step_action_verbatim(self):
        def step(obs):
            return Action(kind="type", text="Part.makeBox(10,10,10)")

        router = CapabilityRouter({Capability.PREDICT_STEP: step})
        a = router.next_action(observation={"screenshot": "..."})
        self.assertEqual(a.kind, "type")
        self.assertEqual(a.text, "Part.makeBox(10,10,10)")

    def test_native_accepts_dict_from_adapter(self):
        def step(obs):
            return {"kind": "click", "point": [12, 34]}

        router = CapabilityRouter({Capability.PREDICT_STEP: step})
        a = router.next_action(obs := {})
        self.assertEqual(a.kind, "click")
        self.assertEqual(a.point, (12, 34))

    def test_bad_step_return_raises(self):
        router = CapabilityRouter({Capability.PREDICT_STEP: lambda obs: 42})
        with self.assertRaises(RouteError):
            router.next_action({})


class RouterComposedTest(unittest.TestCase):
    def setUp(self):
        self.plan_calls = []
        self.click_calls = []

        def plan(obs):
            self.plan_calls.append(obs)
            return "the Pad button"

        def click(obs, description):
            self.click_calls.append((obs, description))
            return (100, 200)

        self.router = CapabilityRouter({
            Capability.PLAN: plan,
            Capability.PREDICT_CLICK: click,
        })

    def test_composed_chains_plan_into_click(self):
        a = self.router.next_action(observation="frame")
        self.assertEqual(a.kind, "click")
        self.assertEqual(a.point, (100, 200))
        self.assertEqual(a.description, "the Pad button")
        # the planner's description was fed to the grounder
        self.assertEqual(self.click_calls[0][1], "the Pad button")

    def test_empty_plan_is_done(self):
        self.router.handlers[Capability.PLAN] = lambda obs: ""
        a = self.router.next_action("frame")
        self.assertEqual(a.kind, "done")

    def test_click_target_uses_grounder(self):
        a = self.router.click_target("frame", "the OK button")
        self.assertEqual(a.point, (100, 200))
        self.assertEqual(a.kind, "click")


class RouterErrorTest(unittest.TestCase):
    def test_unbound_capability_raises(self):
        # can route (PLAN+PREDICT_CLICK) but bind only PLAN -> handler missing.
        router = CapabilityRouter({Capability.PLAN: lambda obs: "x",
                                   Capability.PREDICT_CLICK: None})
        # PREDICT_CLICK bound to None counts as unbound.
        router.handlers[Capability.PREDICT_CLICK] = None
        with self.assertRaises(RouteError):
            router.next_action("frame")

    def test_no_capabilities_cannot_route(self):
        router = CapabilityRouter({})
        with self.assertRaises(RouteError):
            router.next_action("frame")


if __name__ == "__main__":
    unittest.main()
