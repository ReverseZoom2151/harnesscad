"""Tests for the versioned standards knowledge-base (standards/ package).

Covers:
  * RulePack round-trips through JSON (and the tiny YAML-subset reader).
  * Rule.applies_to / StandardsRegistry.active_rules filter by
    material / process / region.
  * StandardsRegistry.rule_versions lists versions and changed_between diffs two
    versions (added / removed / changed).
  * ingest_standard's heuristic pulls >=1 correct typed rule (with clause id)
    from sample clause text, and a deterministic mock-LLM path also works.
  * detect_conflicts flags a contradictory min/max pair on the same parameter and
    passes a consistent set.
All deterministic; no network.
"""

import json
import os
import tempfile
import unittest

from harnesscad.domain.standards.registry import (
    Rule, RulePack, StandardsRegistry, parse_simple_yaml,
)
from harnesscad.domain.standards.ingest import ingest_standard, ingest_heuristic, rule_schema
from harnesscad.domain.standards.conflict import detect_conflicts


def _rule(**kw) -> Rule:
    base = dict(
        id="X:1:1", standard="X", version="1.0",
        parameter="wall thickness", comparator=">=", limit=2.0,
        clause="1", citation="X 1.0, clause 1", scope={},
    )
    base.update(kw)
    return Rule(**base)


# --------------------------------------------------------------------------- #
# Rule / RulePack serialisation
# --------------------------------------------------------------------------- #
class RuleSerialisationTests(unittest.TestCase):
    def test_rule_round_trip(self):
        r = _rule(scope={"material": "aluminum", "process": ["cnc", "mill"]},
                  unit="mm")
        r2 = Rule.from_dict(r.to_dict())
        self.assertEqual(r, r2)

    def test_rule_rejects_bad_comparator(self):
        with self.assertRaises(ValueError):
            _rule(comparator="~=")

    def test_rulepack_json_round_trip(self):
        pack = RulePack(
            name="ISO-2768", version="2023",
            source="unit-test",
            rules=[
                _rule(id="a", standard="ISO-2768", version="2023",
                      parameter="hole diameter", comparator=">=", limit=3.0,
                      clause="3.1", scope={"process": "drill"}, unit="mm"),
                _rule(id="b", standard="ISO-2768", version="2023",
                      parameter="fillet radius", comparator="in",
                      limit=None, values=[0.5, 1.0, 2.0], clause="4.2"),
            ],
        )
        text = pack.to_json()
        back = RulePack.from_json(text)
        self.assertEqual(back.name, "ISO-2768")
        self.assertEqual(back.version, "2023")
        self.assertEqual(len(back.rules), 2)
        self.assertEqual(back.rules[0].parameter, "hole diameter")
        self.assertEqual(back.rules[1].comparator, "in")
        self.assertEqual(back.rules[1].values, [0.5, 1.0, 2.0])
        # Full structural equality of the rule list.
        self.assertEqual([r.to_dict() for r in pack.rules],
                         [r.to_dict() for r in back.rules])

    def test_rulepack_save_load_file(self):
        pack = RulePack(name="X", version="1.0", rules=[_rule()])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pack.json")
            pack.save(path)
            back = RulePack.load(path)
        self.assertEqual(back.rules[0].to_dict(), pack.rules[0].to_dict())

    def test_yaml_subset_reader(self):
        text = (
            "name: ISO-286\n"
            "version: '2010'\n"
            "source: handbook\n"
            "rules:\n"
            "  - id: r1\n"
            "    standard: ISO-286\n"
            "    version: '2010'\n"
            "    parameter: wall thickness\n"
            "    comparator: '>='\n"
            "    limit: 2.0\n"
            "    clause: '5.1'\n"
            "    citation: ISO-286 2010 clause 5.1\n"
            "    scope:\n"
            "      material: aluminum\n"
            "      process: [cnc, mill]\n"
        )
        data = parse_simple_yaml(text)
        pack = RulePack.from_dict(data)
        self.assertEqual(pack.name, "ISO-286")
        self.assertEqual(pack.version, "2010")
        self.assertEqual(len(pack.rules), 1)
        r = pack.rules[0]
        self.assertEqual(r.parameter, "wall thickness")
        self.assertEqual(r.comparator, ">=")
        self.assertEqual(r.limit, 2.0)
        self.assertEqual(r.scope["material"], "aluminum")
        self.assertEqual(r.scope["process"], ["cnc", "mill"])


# --------------------------------------------------------------------------- #
# Registry: active_rules scope filtering
# --------------------------------------------------------------------------- #
class ActiveRulesTests(unittest.TestCase):
    def _registry(self) -> StandardsRegistry:
        pack = RulePack(name="S", version="1.0", rules=[
            _rule(id="alu", standard="S", version="1.0",
                  parameter="p1", scope={"material": "aluminum"}),
            _rule(id="steel", standard="S", version="1.0",
                  parameter="p2", scope={"material": "steel"}),
            _rule(id="cnc-eu", standard="S", version="1.0",
                  parameter="p3", scope={"process": "cnc", "region": "EU"}),
            _rule(id="any", standard="S", version="1.0",
                  parameter="p4", scope={}),
        ])
        return StandardsRegistry().register(pack)

    def test_material_filter(self):
        reg = self._registry()
        ids = {r.id for r in reg.active_rules(material="aluminum")}
        # aluminum rule + the unconstrained rule; NOT the steel-only rule.
        self.assertIn("alu", ids)
        self.assertIn("any", ids)
        self.assertNotIn("steel", ids)

    def test_process_and_region_filter(self):
        reg = self._registry()
        ids = {r.id for r in reg.active_rules(process="cnc", region="EU")}
        self.assertIn("cnc-eu", ids)
        self.assertIn("any", ids)
        # Wrong region excludes the cnc-EU rule.
        ids2 = {r.id for r in reg.active_rules(process="cnc", region="US")}
        self.assertNotIn("cnc-eu", ids2)
        self.assertIn("any", ids2)

    def test_no_filter_returns_all(self):
        reg = self._registry()
        self.assertEqual(len(reg.active_rules()), 4)

    def test_active_rules_deterministic_order(self):
        reg = self._registry()
        a = [r.id for r in reg.active_rules()]
        b = [r.id for r in reg.active_rules()]
        self.assertEqual(a, b)


# --------------------------------------------------------------------------- #
# Registry: versions + changed_between
# --------------------------------------------------------------------------- #
class VersionDiffTests(unittest.TestCase):
    def _registry(self) -> StandardsRegistry:
        v1 = RulePack(name="D", version="1.0", rules=[
            _rule(id="D:1", standard="D", version="1.0",
                  parameter="wall", comparator=">=", limit=2.0, clause="1"),
            _rule(id="D:2", standard="D", version="1.0",
                  parameter="hole", comparator=">=", limit=3.0, clause="2"),
        ])
        v2 = RulePack(name="D", version="2.0", rules=[
            # D:1 changed limit 2.0 -> 2.5
            _rule(id="D:1", standard="D", version="2.0",
                  parameter="wall", comparator=">=", limit=2.5, clause="1"),
            # D:2 removed; D:3 added
            _rule(id="D:3", standard="D", version="2.0",
                  parameter="slot", comparator="<=", limit=10.0, clause="3"),
        ])
        return StandardsRegistry().register(v1).register(v2)

    def test_rule_versions_lists_both(self):
        reg = self._registry()
        self.assertEqual(reg.rule_versions("D"), ["1.0", "2.0"])
        self.assertEqual(reg.latest_version("D"), "2.0")

    def test_changed_between(self):
        reg = self._registry()
        diff = reg.changed_between("D", "1.0", "2.0")
        self.assertEqual([r.id for r in diff.added], ["D:3"])
        self.assertEqual([r.id for r in diff.removed], ["D:2"])
        self.assertEqual(len(diff.changed), 1)
        before, after = diff.changed[0]
        self.assertEqual(before.id, "D:1")
        self.assertEqual(before.limit, 2.0)
        self.assertEqual(after.limit, 2.5)
        self.assertFalse(diff.is_empty)

    def test_active_rules_uses_latest_version_only(self):
        reg = self._registry()
        # Latest is 2.0: 'slot' present, 'hole' (only in 1.0) absent.
        params = {r.parameter for r in reg.active_rules(standard="D")}
        self.assertIn("slot", params)
        self.assertNotIn("hole", params)
        # Pinning the version reaches the old rule.
        params_v1 = {r.parameter for r in reg.active_rules(
            standard="D", version="1.0")}
        self.assertIn("hole", params_v1)

    def test_changed_between_unknown_version_raises(self):
        reg = self._registry()
        with self.assertRaises(KeyError):
            reg.changed_between("D", "1.0", "9.9")


# --------------------------------------------------------------------------- #
# Ingestion (heuristic + mock LLM)
# --------------------------------------------------------------------------- #
_SAMPLE = (
    "3.1 The minimum wall thickness shall be 2 mm.\n"
    "3.2 Hole diameter must be >= 3 mm.\n"
    "3.3 Fillet radius not less than 0.5 mm.\n"
    "3.4 Overall length shall not exceed 250 mm.\n"
    "This introductory sentence has no rule in it.\n"
)


class IngestHeuristicTests(unittest.TestCase):
    def test_pulls_rules_with_clause_ids(self):
        pack = ingest_standard(_SAMPLE, "ACME", "1.0")
        self.assertGreaterEqual(len(pack.rules), 1)
        by_clause = {r.clause: r for r in pack.rules}

        self.assertIn("3.1", by_clause)
        wall = by_clause["3.1"]
        self.assertEqual(wall.parameter, "wall thickness")
        self.assertEqual(wall.comparator, ">=")
        self.assertEqual(wall.limit, 2.0)
        self.assertEqual(wall.unit, "mm")
        self.assertIn("3.1", wall.citation)

    def test_comparators_detected(self):
        pack = ingest_standard(_SAMPLE, "ACME", "1.0")
        by_clause = {r.clause: r for r in pack.rules}
        self.assertEqual(by_clause["3.2"].parameter, "hole diameter")
        self.assertEqual(by_clause["3.2"].comparator, ">=")
        self.assertEqual(by_clause["3.2"].limit, 3.0)
        self.assertEqual(by_clause["3.3"].parameter, "fillet radius")
        self.assertEqual(by_clause["3.3"].comparator, ">=")
        self.assertEqual(by_clause["3.3"].limit, 0.5)
        self.assertEqual(by_clause["3.4"].comparator, "<=")
        self.assertEqual(by_clause["3.4"].limit, 250.0)

    def test_non_rule_sentence_ignored(self):
        rules = ingest_heuristic(
            "This is just prose with no measurable requirement.", "X", "1.0")
        self.assertEqual(rules, [])

    def test_deterministic(self):
        a = ingest_standard(_SAMPLE, "ACME", "1.0").to_json()
        b = ingest_standard(_SAMPLE, "ACME", "1.0").to_json()
        self.assertEqual(a, b)

    def test_rule_schema_shape(self):
        schema = rule_schema()
        self.assertEqual(schema["type"], "object")
        self.assertIn("rules", schema["properties"])


class _MockLLM:
    """Deterministic stand-in for llm.base.LLM returning structured JSON."""

    def __init__(self, payload):
        self._payload = payload

    def complete(self, messages, tools=None, response_schema=None, **opts):
        from harnesscad.agents.llm.base import CompletionResult
        return CompletionResult(text=json.dumps(self._payload))

    def stream(self, messages, tools=None, response_schema=None, **opts):
        yield ""


class IngestLLMTests(unittest.TestCase):
    def test_llm_structured_extraction(self):
        llm = _MockLLM({"rules": [
            {"clause": "7.4", "parameter": "bend radius",
             "comparator": ">=", "limit": 1.5, "unit": "mm"},
        ]})
        pack = ingest_standard("irrelevant text", "LLMSTD", "2.0", llm=llm)
        self.assertEqual(len(pack.rules), 1)
        r = pack.rules[0]
        self.assertEqual(r.parameter, "bend radius")
        self.assertEqual(r.comparator, ">=")
        self.assertEqual(r.limit, 1.5)
        self.assertEqual(r.clause, "7.4")

    def test_llm_failure_falls_back_to_heuristic(self):
        class Boom:
            def complete(self, *a, **k):
                raise RuntimeError("no network")

            def stream(self, *a, **k):
                yield ""

        pack = ingest_standard(_SAMPLE, "ACME", "1.0", llm=Boom())
        # Heuristic still produced rules.
        self.assertGreaterEqual(len(pack.rules), 1)


# --------------------------------------------------------------------------- #
# Conflict detection
# --------------------------------------------------------------------------- #
class ConflictTests(unittest.TestCase):
    def test_flags_min_max_contradiction(self):
        rules = [
            _rule(id="min", parameter="wall", comparator=">=", limit=2.0,
                  clause="1"),
            _rule(id="max", parameter="wall", comparator="<=", limit=1.5,
                  clause="2"),
        ]
        conflicts = detect_conflicts(rules)
        self.assertEqual(len(conflicts), 1)
        c = conflicts[0]
        self.assertEqual(c.parameter, "wall")
        self.assertEqual({c.rule_a.id, c.rule_b.id}, {"min", "max"})
        self.assertIn("unsatisfiable", c.reason)

    def test_consistent_set_has_no_conflict(self):
        rules = [
            _rule(id="min", parameter="wall", comparator=">=", limit=2.0),
            _rule(id="max", parameter="wall", comparator="<=", limit=5.0),
            _rule(id="hole", parameter="hole", comparator=">=", limit=3.0),
        ]
        self.assertEqual(detect_conflicts(rules), [])

    def test_equality_contradiction(self):
        rules = [
            _rule(id="eq2", parameter="pitch", comparator="==", limit=2.0),
            _rule(id="eq3", parameter="pitch", comparator="==", limit=3.0),
        ]
        self.assertEqual(len(detect_conflicts(rules)), 1)

    def test_different_scope_not_compared(self):
        rules = [
            _rule(id="alu", parameter="wall", comparator=">=", limit=2.0,
                  scope={"material": "aluminum"}),
            _rule(id="steel", parameter="wall", comparator="<=", limit=1.5,
                  scope={"material": "steel"}),
        ]
        # Different scopes -> not mutually contradictory.
        self.assertEqual(detect_conflicts(rules), [])

    def test_disjoint_in_sets_conflict(self):
        rules = [
            _rule(id="setA", parameter="thk", comparator="in", limit=None,
                  values=[1.0, 2.0]),
            _rule(id="setB", parameter="thk", comparator="in", limit=None,
                  values=[3.0, 4.0]),
        ]
        self.assertEqual(len(detect_conflicts(rules)), 1)

    def test_deterministic_order(self):
        rules = [
            _rule(id="max", parameter="wall", comparator="<=", limit=1.5),
            _rule(id="min", parameter="wall", comparator=">=", limit=2.0),
        ]
        first = [(c.rule_a.id, c.rule_b.id) for c in detect_conflicts(rules)]
        second = [(c.rule_a.id, c.rule_b.id) for c in detect_conflicts(rules)]
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
