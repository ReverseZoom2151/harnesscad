"""Tests for the spec front-of-pipeline: formalize + interview.

Deterministic and offline: the heuristic parser needs no network, and the one
LLM-path test uses a tiny in-test MockLLM (no provider, no keys).
"""

import json
import unittest

from spec.formalize import (
    Requirement, RequirementSet, formalize, to_contract, requirement_schema,
)
from spec.interview import RequirementsInterview, Question
from contract import Contract


BRIEF = "an aluminium plate 100mm x 50mm x 8mm with 4 holes"


class TestHeuristicFormalize(unittest.TestCase):
    def test_extracts_count_dimensions_material(self):
        rs = formalize(BRIEF)

        counts = rs.by_kind("count")
        self.assertEqual(len(counts), 1)
        self.assertEqual(counts[0].target, 4)
        self.assertEqual(counts[0].label, "hole")

        dims = rs.by_kind("dimension")
        self.assertEqual(len(dims), 3)
        self.assertEqual({d.target for d in dims}, {100.0, 50.0, 8.0})
        self.assertEqual({d.label for d in dims},
                         {"length", "width", "height"})
        self.assertTrue(all(d.unit == "mm" for d in dims))

        materials = rs.by_kind("material")
        self.assertEqual(len(materials), 1)
        self.assertEqual(materials[0].target, "aluminium")

    def test_named_dimension_and_tolerance(self):
        rs = formalize("a bar 120 mm long, +/- 0.05 mm, with 2 slots")
        dims = rs.by_kind("dimension")
        self.assertTrue(any(d.label == "length" and d.target == 120.0
                            for d in dims))
        tols = rs.by_kind("tolerance")
        self.assertEqual(len(tols), 1)
        self.assertAlmostEqual(tols[0].target, 0.05)
        counts = rs.by_kind("count")
        self.assertEqual(counts[0].target, 2)
        self.assertEqual(counts[0].label, "slot")

    def test_empty_brief(self):
        self.assertEqual(len(formalize("")), 0)
        self.assertEqual(len(formalize("   ")), 0)

    def test_requirementset_round_trip(self):
        rs = formalize(BRIEF)
        restored = RequirementSet.from_dict(json.loads(json.dumps(rs.to_dict())))
        self.assertEqual(restored.to_dict(), rs.to_dict())


class TestToContract(unittest.TestCase):
    def test_seeds_contract_from_dict(self):
        rs = formalize(BRIEF)
        d = to_contract(rs)
        # round-trips into a real Contract (and back out again).
        contract = Contract.from_dict(d)
        self.assertEqual(contract.to_dict(),
                         Contract.from_dict(contract.to_dict()).to_dict())

        # dimensions -> bbox axes
        self.assertEqual(contract.bbox["x"].target, 100.0)
        self.assertEqual(contract.bbox["y"].target, 50.0)
        self.assertEqual(contract.bbox["z"].target, 8.0)
        # 4 holes -> hole_count and min_features
        self.assertEqual(contract.hole_count, 4)
        self.assertEqual(contract.min_features, 4)
        # material recorded in the description
        self.assertIn("aluminium", contract.description)

    def test_dimension_tolerance_flows_into_bbox(self):
        rs = formalize("a bar 120 mm long +/- 0.05 mm")
        contract = Contract.from_dict(to_contract(rs))
        self.assertEqual(contract.bbox["x"].target, 120.0)
        self.assertAlmostEqual(contract.bbox["x"].tol, 0.05)


class TestSchema(unittest.TestCase):
    def test_schema_is_json_serialisable(self):
        schema = requirement_schema()
        self.assertEqual(schema["title"], "RequirementSet")
        json.dumps(schema)  # must not raise


class TestInterview(unittest.TestCase):
    def test_vague_brief_asks_for_tolerance_and_quantity(self):
        interview = RequirementsInterview()
        questions = interview.next_questions("a small mounting bracket", k=5)
        fields = {q.field for q in questions}
        self.assertIn("tolerance", fields)
        self.assertIn("quantity", fields)
        # every question is a phrased Question
        self.assertTrue(all(isinstance(q, Question) and q.text for q in questions))

    def test_questions_are_ranked_and_capped(self):
        interview = RequirementsInterview()
        questions = interview.next_questions("a bracket", k=2)
        self.assertEqual(len(questions), 2)
        priorities = [q.priority for q in questions]
        self.assertEqual(priorities, sorted(priorities))

    def test_specified_fields_not_asked(self):
        # a fully-specified brief leaves few gaps; material must not be asked.
        interview = RequirementsInterview()
        brief = ("an aluminium plate 100mm x 50mm x 8mm with 4 holes, "
                 "+/- 0.1 mm, must carry a 200 N load")
        fields = set(interview.missing_fields(brief))
        self.assertNotIn("material", fields)
        self.assertNotIn("tolerance", fields)
        self.assertNotIn("envelope", fields)
        self.assertNotIn("quantity", fields)
        self.assertNotIn("load", fields)

    def test_accepts_requirementset_directly(self):
        rs = formalize("a bracket")  # no dims/material/tol/count
        interview = RequirementsInterview()
        fields = set(interview.missing_fields(rs))
        self.assertIn("material", fields)
        self.assertIn("envelope", fields)


class _MockLLM:
    """Minimal LLM stub: returns a canned structured-output JSON string."""

    def __init__(self, text):
        self._text = text
        self.calls = []

    def complete(self, messages, tools=None, response_schema=None, **opts):
        self.calls.append(messages)
        from llm.base import CompletionResult
        return CompletionResult(text=self._text)

    def stream(self, *a, **k):
        yield self._text


class TestFormalizeLLMPath(unittest.TestCase):
    def test_uses_llm_structured_output(self):
        payload = {
            "requirements": [
                {"kind": "count", "target": 6, "label": "hole"},
                {"kind": "dimension", "target": 200.0, "unit": "mm",
                 "label": "length"},
                {"kind": "material", "target": "titanium", "label": "material"},
            ]
        }
        llm = _MockLLM(json.dumps(payload))
        rs = formalize("some brief the mock ignores", llm=llm)
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(rs.by_kind("count")[0].target, 6)
        self.assertEqual(rs.by_kind("material")[0].target, "titanium")

    def test_falls_back_to_heuristic_when_llm_fails(self):
        class _BadLLM:
            def complete(self, *a, **k):
                raise RuntimeError("provider down")

            def stream(self, *a, **k):
                yield ""

        rs = formalize(BRIEF, llm=_BadLLM())
        # heuristic still runs -> 4 holes recovered
        self.assertEqual(rs.by_kind("count")[0].target, 4)


if __name__ == "__main__":
    unittest.main()
