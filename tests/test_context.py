"""Tests for the context-management layer (context/manager.py + context/staging.py).

Covers: budget math + pre-flight over/under; assemble pins system + first-user
while the middle is evicted (tool results first); feature_tree_summary is compact
and deterministic; StagingArea round-trips a manifest and renders the selection.
"""

import os
import tempfile
import unittest

from cisp.ops import (
    NewSketch, AddRectangle, AddCircle, Constrain, Extrude, Fillet,
)
from state.opdag import OpDAG
from llm.base import Message, ToolSpec

from context.manager import (
    BudgetReport,
    ContextManager,
    ContextOverflowError,
    HeuristicCounter,
    feature_tree_summary,
)
from context.staging import StagingArea


# --- token counting --------------------------------------------------------
class TestHeuristicCounter(unittest.TestCase):
    def test_empty_is_zero(self):
        self.assertEqual(HeuristicCounter().count(""), 0)

    def test_counts_words_digits_and_punctuation(self):
        c = HeuristicCounter()
        # "extrude" (1 word) + "5" (1 digit-run) + "mm" (1 word) = 3
        self.assertEqual(c.count("extrude 5 mm"), 3)

    def test_beats_4char_rule_on_json(self):
        # The naive len//4 rule badly under-counts punctuation-dense JSON; the
        # heuristic counts each brace/quote/colon, landing much higher.
        c = HeuristicCounter()
        payload = '{"op":"extrude","distance":5}'
        naive = len(payload) // 4
        self.assertGreater(c.count(payload), naive)

    def test_deterministic(self):
        c = HeuristicCounter()
        self.assertEqual(c.count("sketch on XY plane"), c.count("sketch on XY plane"))


# --- budget math -----------------------------------------------------------
class TestBudgetReport(unittest.TestCase):
    def test_total_and_ok_and_overflow(self):
        r = BudgetReport(budget=100, system=10, memory=20, tools=15, history=25, reserved=10)
        self.assertEqual(r.total, 80)
        self.assertTrue(r.ok)
        self.assertEqual(r.overflow, 0)
        self.assertEqual(r.remaining, 20)

    def test_over_budget(self):
        r = BudgetReport(budget=50, system=10, memory=20, tools=15, history=25, reserved=10)
        self.assertEqual(r.total, 80)
        self.assertFalse(r.ok)
        self.assertEqual(r.overflow, 30)


# --- pre-flight ------------------------------------------------------------
class TestPreflight(unittest.TestCase):
    def _msgs(self):
        return [
            Message("system", "you are a CAD agent " * 5),
            Message("user", "make a bracket " * 5),
            Message("assistant", "ok " * 5),
        ]

    def test_under_budget_returns_report_no_raise(self):
        cm = ContextManager(budget=10_000)
        report = cm.preflight(self._msgs())
        self.assertTrue(report.ok)
        self.assertGreater(report.system, 0)
        self.assertGreater(report.history, 0)

    def test_over_budget_raises_with_report(self):
        cm = ContextManager(budget=5)
        with self.assertRaises(ContextOverflowError) as ctx:
            cm.preflight(self._msgs())
        self.assertFalse(ctx.exception.report.ok)
        self.assertGreater(ctx.exception.report.overflow, 0)

    def test_over_budget_non_strict_returns_report(self):
        cm = ContextManager(budget=5)
        report = cm.preflight(self._msgs(), strict=False)
        self.assertFalse(report.ok)

    def test_reserved_and_tools_counted(self):
        cm = ContextManager(budget=10_000)
        tools = [ToolSpec("extrude", "Extrude a sketch profile", {"type": "object"})]
        r0 = cm.preflight(self._msgs(), strict=False)
        r1 = cm.preflight(self._msgs(), tools=tools, reserved=100, strict=False)
        self.assertGreater(r1.tools, 0)
        self.assertEqual(r1.reserved, 100)
        self.assertGreater(r1.total, r0.total)


# --- assemble: pin head, evict middle --------------------------------------
class TestAssemble(unittest.TestCase):
    def test_pins_system_and_first_user(self):
        cm = ContextManager(budget=10_000)
        history = [Message("assistant", "step one"), Message("tool", "result one")]
        out = cm.assemble("SYS", "BRIEF: build a plate", history=history)
        self.assertEqual(out.messages[0].role, "system")
        self.assertEqual(out.messages[0].content, "SYS")
        # first user is pinned right after system (no memory here).
        self.assertEqual(out.messages[1].role, "user")
        self.assertEqual(out.messages[1].content, "BRIEF: build a plate")
        self.assertTrue(out.report.ok)

    def test_memory_block_between_system_and_user(self):
        cm = ContextManager(budget=10_000)
        out = cm.assemble("SYS", "BRIEF", memory="RETRIEVED: prior flange")
        roles = [m.role for m in out.messages]
        self.assertEqual(roles[0], "system")   # pinned system
        self.assertEqual(roles[1], "system")   # memory block
        self.assertEqual(out.messages[1].content, "RETRIEVED: prior flange")
        self.assertEqual(roles[2], "user")     # pinned first user
        self.assertGreater(out.report.memory, 0)

    def test_evicts_middle_keeps_pinned_head_and_tail(self):
        # Tiny budget forces eviction; system + first_user must survive, and the
        # most-recent history message (tail) is preserved.
        cm = ContextManager(budget=40)
        history = [Message("assistant", f"turn {i} " * 3) for i in range(10)]
        history.append(Message("user", "MOST RECENT"))
        out = cm.assemble("SYS", "PINNED BRIEF", history=history)
        contents = [m.content for m in out.messages]
        self.assertIn("SYS", contents)
        self.assertIn("PINNED BRIEF", contents)
        self.assertEqual(out.messages[-1].content, "MOST RECENT")  # tail kept
        self.assertGreater(out.evicted, 0)                          # middle gone
        self.assertLess(len(out.messages), len(history) + 2)

    def test_tool_results_evicted_first(self):
        cm = ContextManager(budget=30)
        # Interleave big tool-result dumps with small assistant turns.
        history = [
            Message("assistant", "a1"),
            Message("tool", "MESHDUMP " * 50),
            Message("assistant", "a2"),
            Message("tool", "LOGDUMP " * 50),
            Message("assistant", "a3 final"),
        ]
        out = cm.assemble("SYS", "BRIEF", history=history)
        kept_roles = [m.role for m in out.messages if m.role == "tool"]
        # the bulky tool dumps are the first to go under pressure.
        self.assertEqual(len(kept_roles), 0)
        # small assistant turns + tail survive.
        self.assertEqual(out.messages[-1].content, "a3 final")

    def test_stable_prefix_reports_system_plus_tools(self):
        cm = ContextManager(budget=10_000)
        tools = [ToolSpec("extrude", "Extrude a profile", {"type": "object"})]
        out = cm.assemble("SYS PROMPT", "BRIEF", tools=tools)
        self.assertEqual(out.stable_prefix_tokens, out.report.system + out.report.tools)
        self.assertGreater(out.report.tools, 0)

    def test_overbudget_pinned_reports_not_ok(self):
        # Even with everything evicted, pinned head + reserved can exceed C.
        cm = ContextManager(budget=3, reserved=100)
        out = cm.assemble("SYS", "BRIEF", history=[Message("assistant", "x")])
        self.assertFalse(out.report.ok)


# --- feature-tree summary --------------------------------------------------
def _sample_ops():
    return [
        NewSketch(plane="XY"),
        AddRectangle(sketch="sk1", x=0.0, y=0.0, w=20.0, h=10.0),
        Constrain(kind="distance", a="e1", value=20.0),
        Extrude(sketch="sk1", distance=5.0),
        AddCircle(sketch="sk1", cx=5.0, cy=5.0, r=2.0),
        Fillet(edges=(1, 2), radius=1.0),
    ]


class TestFeatureTreeSummary(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(feature_tree_summary([]), "Feature tree (empty)")

    def test_accepts_opdag_and_oplist_equivalently(self):
        ops = _sample_ops()
        dag = OpDAG()
        for op in ops:
            dag.append(op)
        self.assertEqual(feature_tree_summary(ops), feature_tree_summary(dag))

    def test_deterministic(self):
        ops = _sample_ops()
        self.assertEqual(feature_tree_summary(ops), feature_tree_summary(ops))

    def test_compact_and_not_brep_dump(self):
        summary = feature_tree_summary(_sample_ops())
        # compact: one line per op + header + totals, no B-rep face/edge listing.
        self.assertLessEqual(len(summary.splitlines()), len(_sample_ops()) + 2)
        self.assertNotIn("face", summary.lower())
        self.assertNotIn("vertex", summary.lower())

    def test_reconstructs_deterministic_ids(self):
        summary = feature_tree_summary(_sample_ops())
        self.assertIn("sk1", summary)     # first sketch
        self.assertIn("e1", summary)      # first entity
        self.assertIn("f1", summary)      # first feature
        self.assertIn("extrude", summary)
        self.assertIn("constrain", summary)

    def test_via_manager_method(self):
        cm = ContextManager(budget=10_000)
        self.assertEqual(
            cm.feature_tree_summary(_sample_ops()),
            feature_tree_summary(_sample_ops()),
        )


# --- staging area ----------------------------------------------------------
class TestStagingArea(unittest.TestCase):
    def test_build_creates_skeleton(self):
        with tempfile.TemporaryDirectory() as d:
            sa = StagingArea(d).build(brief="make a plate", model_tree="sk1: sketch")
            self.assertTrue(sa.exists("01_BRIEF.md"))
            self.assertTrue(sa.exists("02_MODEL/tree.txt"))
            self.assertTrue(os.path.isdir(sa.path("03_DOCS")))
            self.assertTrue(os.path.exists(sa.manifest_path))

    def test_write_read_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            sa = StagingArea(d).build()
            sa.write("03_DOCS/iso.md", "ISO 286 fits")
            self.assertEqual(sa.read("03_DOCS/iso.md"), "ISO 286 fits")

    def test_manifest_round_trips(self):
        with tempfile.TemporaryDirectory() as d:
            sa = StagingArea(d)
            manifest = {
                "manifest": {
                    "brief": "01_BRIEF.md",
                    "model": "02_MODEL/tree.txt",
                    "docs": ["03_DOCS/iso.md", "03_DOCS/dfm.md"],
                }
            }
            sa.build(manifest=manifest)
            loaded = sa.read_manifest()
            self.assertEqual(loaded, manifest)

    def test_render_for_turn_selects_and_orders(self):
        with tempfile.TemporaryDirectory() as d:
            sa = StagingArea(d).build(
                brief="BRIEF BODY", model_tree="MODEL BODY",
            )
            sa.write("03_DOCS/iso.md", "ISO BODY")
            manifest = {
                "manifest": {
                    "brief": "01_BRIEF.md",
                    "model": "02_MODEL/tree.txt",
                    "docs": ["03_DOCS/iso.md"],
                }
            }
            rendered = sa.render_for_turn(manifest)
            # deterministic order: BRIEF, MODEL, DOC.
            self.assertLess(rendered.index("BRIEF BODY"), rendered.index("MODEL BODY"))
            self.assertLess(rendered.index("MODEL BODY"), rendered.index("ISO BODY"))
            self.assertIn("# BRIEF", rendered)
            self.assertIn("# DOC: 03_DOCS/iso.md", rendered)

    def test_render_loads_manifest_from_disk_when_omitted(self):
        with tempfile.TemporaryDirectory() as d:
            sa = StagingArea(d).build(brief="ON DISK BRIEF", model_tree="TREE")
            rendered = sa.render_for_turn()
            self.assertIn("ON DISK BRIEF", rendered)

    def test_render_marks_missing_file(self):
        with tempfile.TemporaryDirectory() as d:
            sa = StagingArea(d).build(brief="B", model_tree="M")
            manifest = {"manifest": {"docs": ["03_DOCS/nope.md"]}}
            rendered = sa.render_for_turn(manifest)
            self.assertIn("(missing)", rendered)

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            sa = StagingArea(d)
            with self.assertRaises(ValueError):
                sa.path("../escape.md")


if __name__ == "__main__":
    unittest.main()
