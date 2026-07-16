"""Built-in Lattice namespace agent cards for HarnessCAD's CAD domain.

Source: Forma-OSS ``blueprint_core/lattice_agents.py``, which pre-builds one
Lattice agent card per Blueprint project namespace (project.meta,
product.electrical, product.bom, ...), each declaring a single
namespace-update capability, a Question/Result schema contract, and a shared
safety posture (namespace-scoped updates only, declared dependencies, human
review for safety-critical changes).

Harness gap filled: ``harnesscad.agents.a2a.lattice`` (the module this one
builds on) provides the generic contract layer -- capabilities, contracts,
cards, registry -- but no concrete roster of agents. This module supplies
that roster, re-mapped from Blueprint's electronics namespaces onto
HarnessCAD's CAD pipeline: geometry owns the CISP op stream, fabrication
owns process/DFM/cost, validation owns the measured gate and differential
oracle, and so on. It complements ``harnesscad.agents.a2a.messages`` and
``harnesscad.agents.a2a.task`` (transport and lifecycle) without duplicating
them: those move messages, this declares who owns which namespace and under
what contract.

Stdlib only, dataclasses + explicit validation (no pydantic), deterministic,
no wall clock, no uuid4.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from harnesscad.agents.a2a.lattice import (
    LatticeAgentCard,
    LatticeCapability,
    LatticeRegistry,
    LatticeSchemaContract,
)


@dataclass
class NamespaceAgentQuestion:
    """The input contract every namespace agent accepts.

    ``namespace`` is the target (e.g. ``product.geometry``), ``prompt`` is the
    user request or upstream-agent instruction, ``current_payload`` is the
    existing namespace payload if any, and ``constraints`` are invariants the
    agent must preserve.
    """

    namespace: str
    prompt: str
    project_context: Dict[str, Any] = field(default_factory=dict)
    current_payload: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.namespace.strip():
            raise ValueError("NamespaceAgentQuestion.namespace must be non-empty")
        if not self.prompt.strip():
            raise ValueError("NamespaceAgentQuestion.prompt must be non-empty")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NamespaceAgentResult:
    """The output contract every namespace agent produces.

    ``payload_patch`` is a structured patch or replacement payload for the
    namespace, ``dependencies`` names other namespaces that should be checked
    rather than silently rewritten, and ``handoff_actions`` are suggested
    downstream actions for the orchestrator.
    """

    namespace: str
    summary: str
    payload_patch: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    validation_notes: List[str] = field(default_factory=list)
    handoff_actions: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.namespace.strip():
            raise ValueError("NamespaceAgentResult.namespace must be non-empty")
        if not self.summary.strip():
            raise ValueError("NamespaceAgentResult.summary must be non-empty")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Plain JSON-Schema-style dicts for the two dataclasses above (the harness has
# no pydantic to derive them from, so they are declared by hand and passed to
# LatticeSchemaContract.from_schemas).
NAMESPACE_QUESTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "title": "NamespaceAgentQuestion",
    "properties": {
        "namespace": {"type": "string"},
        "prompt": {"type": "string"},
        "project_context": {"type": "object"},
        "current_payload": {"type": "object"},
        "constraints": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["namespace", "prompt"],
}

NAMESPACE_RESULT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "title": "NamespaceAgentResult",
    "properties": {
        "namespace": {"type": "string"},
        "summary": {"type": "string"},
        "payload_patch": {"type": "object"},
        "dependencies": {"type": "array", "items": {"type": "string"}},
        "validation_notes": {"type": "array", "items": {"type": "string"}},
        "handoff_actions": {"type": "array", "items": {"type": "object"}},
    },
    "required": ["namespace", "summary"],
}


# The built-in CAD namespace roster. Forma-OSS's electronics namespaces
# (product.electrical, product.bom, product.firmware) are re-mapped onto
# HarnessCAD's CAD pipeline: geometry (CISP op stream), mech (enclosure and
# placements), fabrication (process/DFM/cost), assembly (mates and DOF),
# validation (measured gate, differential oracle, MGC predicates), visuals
# (renders and drawings). Tools and handoffs name real harness capabilities.
NAMESPACE_AGENT_PROFILES: Dict[str, Dict[str, Any]] = {
    "project.meta": {
        "kind": "meta",
        "summary": "Owns project identity, runtime metadata, source usage, and workspace-level state.",
        "inputs": ["project_id", "runtime_metadata", "source_usage", "workflow_state"],
        "outputs": ["project_metadata", "source_usage", "revision_metadata"],
        "tools": ["job metadata", "trace inspection", "runtime debug config"],
        "handoffs": ["harnesscad.a2a.get_task", "harnesscad.a2a.list_tasks"],
    },
    "project.docs": {
        "kind": "docs",
        "summary": "Owns build documentation, human-readable guidance, validation reports, and release notes.",
        "inputs": ["cad_ir", "assembly_steps", "validation_notes", "fabrication_notes"],
        "outputs": ["build_docs", "release_notes", "review_summary"],
        "tools": ["documentation generation", "schema validation"],
        "handoffs": ["harnesscad.lattice.list_agents"],
    },
    "project.history": {
        "kind": "history",
        "summary": "Owns revision lineage, iteration decisions, and namespace change history.",
        "inputs": ["revision", "source_project_id", "iteration_request", "namespace_versions"],
        "outputs": ["version_history", "decision_log", "namespace_revision_map"],
        "tools": ["job metadata", "audit log", "provenance tracking"],
        "handoffs": ["harnesscad.a2a.get_task", "harnesscad.a2a.list_tasks"],
    },
    "product.overview": {
        "kind": "overview",
        "summary": "Owns design intent, requirements, constraints, cost target, and the top-level part description.",
        "inputs": ["prompt", "user_goal", "requirements", "constraints"],
        "outputs": ["overview", "functional_requirements", "missing_info"],
        "tools": ["clarifying questions", "schema validation", "harnesscad.spec.contract"],
        "handoffs": ["harnesscad.spec.contract", "product.geometry.update"],
    },
    "product.geometry": {
        "kind": "geometry",
        "summary": "Owns the CISP op stream, sketches, parametric features, and geometric constraints.",
        "inputs": ["design_intent", "current_ops", "sketch_constraints", "feature_parameters"],
        "outputs": ["op_patch", "sketch_updates", "feature_tree", "constraint_status"],
        "tools": ["CISP op validation", "kernel execution", "constraint solving"],
        "handoffs": ["harnesscad.gate.measure", "harnesscad.oracle.crosscheck"],
    },
    "product.mech": {
        "kind": "mech",
        "summary": "Owns enclosure geometry, component placements, dimensions, clearances, and mechanical constraints.",
        "inputs": ["components", "dimensions", "mounting_constraints", "clearance_rules"],
        "outputs": ["mechanical_notes", "component_placements", "spatial_relationships", "interference_report"],
        "tools": ["placement gates", "interference checks", "mechanical fit checks"],
        "handoffs": ["product.assembly.update", "harnesscad.gate.measure"],
    },
    "product.fabrication": {
        "kind": "fabrication",
        "summary": "Owns fabrication process selection, DFM findings, tolerances, and cost estimation.",
        "inputs": ["geometry_summary", "process_candidates", "material", "quantity"],
        "outputs": ["process_plan", "dfm_findings", "tolerance_notes", "estimated_cost"],
        "tools": ["DFM rule checks", "process selection", "cost estimation"],
        "handoffs": ["product.validation.update", "harnesscad.lattice.get_agent_card"],
    },
    "product.assembly": {
        "kind": "assembly",
        "summary": "Owns mates, degree-of-freedom contracts, assembly sequencing, and builder-facing steps.",
        "inputs": ["components", "mates", "dof_contracts", "mechanical_notes"],
        "outputs": ["assembly_steps", "mate_graph", "dof_report", "danger_flags"],
        "tools": ["mate validation", "DOF analysis", "build planning"],
        "handoffs": ["product.validation.update", "harnesscad.lattice.get_agent_card"],
    },
    "product.validation": {
        "kind": "validation",
        "summary": "Owns the measured gate, differential oracle cross-checks, MGC predicates, and review gates.",
        "inputs": ["op_stream", "spec_contract", "measured_properties", "assembly_steps"],
        "outputs": ["validation_summary", "gate_results", "blocking_issues", "review_gates"],
        "tools": ["harnesscad.gate.measure", "harnesscad.oracle.crosscheck", "harnesscad.spec.contract"],
        "handoffs": ["harnesscad.gate.measure", "harnesscad.oracle.crosscheck"],
    },
    "product.visuals": {
        "kind": "visuals",
        "summary": "Owns rendered imagery, engineering drawings, view metadata, and presentation assets.",
        "inputs": ["cad_ir", "component_placements", "visual_style", "drawing_conventions"],
        "outputs": ["render_metadata", "drawing_views", "image_prompts", "annotation_layout"],
        "tools": ["render generation", "orthographic projection", "drawing annotation"],
        "handoffs": ["project.docs.update", "harnesscad.lattice.list_agents"],
    },
}


def namespace_agent_id(namespace: str) -> str:
    """Stable agent id for a namespace: the namespace itself, lowercased."""
    return namespace.strip().lower()


def namespace_action(namespace: str) -> str:
    """The single update action a namespace agent exposes."""
    return f"{namespace_agent_id(namespace)}.update"


def namespace_contract_id(namespace: str) -> str:
    """The v0 contract id for a namespace agent."""
    return f"{namespace_agent_id(namespace)}.v0"


def namespace_agent_card(
    namespace: str,
    label: str,
    description: str,
    scope: str,
) -> LatticeAgentCard:
    """Build a Lattice agent card for one project or product namespace.

    ``scope`` is the top-level grouping (``project`` or ``product``). All
    cards share the same safety posture: namespace-scoped updates only,
    declared dependencies instead of silent cross-namespace rewrites, and
    mandatory human review for safety-critical or irreversible changes.
    """
    if not namespace.strip():
        raise ValueError("namespace must be non-empty")
    if not label.strip():
        raise ValueError("label must be non-empty")
    if not description.strip():
        raise ValueError("description must be non-empty")
    if not scope.strip():
        raise ValueError("scope must be non-empty")

    profile = NAMESPACE_AGENT_PROFILES.get(namespace, {})
    agent_id = namespace_agent_id(namespace)
    action = namespace_action(namespace)
    agent_kind = profile.get("kind", namespace.rsplit(".", 1)[-1])

    contract = LatticeSchemaContract.from_schemas(
        id=namespace_contract_id(namespace),
        name=f"{label} Contract",
        purpose=f"Update and audit the {namespace} namespace: {description}",
        input_schema=NAMESPACE_QUESTION_SCHEMA,
        output_schema=NAMESPACE_RESULT_SCHEMA,
        extraction_prompt=(
            f"Extract only the information needed to update {namespace}. "
            "Preserve adjacent namespaces unless a dependency note explains "
            "why another agent should be consulted."
        ),
        metadata={
            "namespace": namespace,
            "scope": scope,
            "agent_kind": agent_kind,
        },
    )

    capability = LatticeCapability(
        id=action,
        label=f"{label} Agent",
        description=profile.get("summary", description),
        inputs=profile.get(
            "inputs", ["project_context", "current_payload", "constraints"]
        ),
        outputs=profile.get(
            "outputs", ["payload_patch", "validation_notes", "handoff_actions"]
        ),
        actions=[action, "harnesscad.lattice.get_agent_card"],
    )

    return LatticeAgentCard(
        agent_id=agent_id,
        namespace=namespace,
        name=f"{label} Agent",
        version="0.1.0",
        domain=description,
        summary=profile.get("summary", description),
        capabilities=[capability],
        contracts=[contract],
        runtime_boundary=(
            f"{label} owns the {namespace} namespace contract; HarnessCAD owns "
            "orchestration, kernel execution, provider routing, validation, "
            "persistence, and cross-namespace coordination."
        ),
        tools_needed=profile.get(
            "tools", ["schema validation", "project object inspection"]
        ),
        handoff_actions=profile.get("handoffs", ["harnesscad.lattice.list_agents"]),
        safety_limits=[
            "Return namespace-scoped updates only.",
            "Declare dependencies instead of silently rewriting unrelated namespaces.",
            "Require human review for safety-critical, costly, or irreversible changes.",
        ],
        human_review_triggers=[
            "safety-critical issue",
            "cross-namespace dependency",
            "irreversible fabrication or procurement decision",
            "missing source data",
        ],
        tags=["lattice", "namespace-agent", scope, namespace],
        metadata={
            "namespace": namespace,
            "scope": scope,
            "agent_kind": agent_kind,
        },
    )


# (namespace, label, description, scope) rows for the built-in roster, in a
# stable declared order. Descriptions are the "domain" text on each card.
_DEFAULT_NAMESPACES: List[tuple] = [
    ("project.meta", "Project Meta", "Project identity, runtime metadata, and workspace state.", "project"),
    ("project.docs", "Project Docs", "Build documentation, guidance, and validation reports.", "project"),
    ("project.history", "Project History", "Revision lineage and namespace change history.", "project"),
    ("product.overview", "Product Overview", "Design intent, requirements, and constraints.", "product"),
    ("product.geometry", "Product Geometry", "CISP op stream, sketches, and parametric features.", "product"),
    ("product.mech", "Product Mech", "Enclosure geometry, placements, and mechanical constraints.", "product"),
    ("product.fabrication", "Product Fabrication", "Fabrication process, DFM, tolerances, and cost.", "product"),
    ("product.assembly", "Product Assembly", "Mates, DOF contracts, and assembly sequencing.", "product"),
    ("product.validation", "Product Validation", "Measured gate, differential oracle, and MGC predicates.", "product"),
    ("product.visuals", "Product Visuals", "Renders, engineering drawings, and presentation assets.", "product"),
]


def default_namespace_agent_cards() -> List[LatticeAgentCard]:
    """Build the full built-in roster of CAD namespace agent cards."""
    return [
        namespace_agent_card(namespace, label, description, scope)
        for namespace, label, description, scope in _DEFAULT_NAMESPACES
    ]


__all__ = [
    "NAMESPACE_AGENT_PROFILES",
    "NAMESPACE_QUESTION_SCHEMA",
    "NAMESPACE_RESULT_SCHEMA",
    "NamespaceAgentQuestion",
    "NamespaceAgentResult",
    "default_namespace_agent_cards",
    "namespace_action",
    "namespace_agent_card",
    "namespace_agent_id",
    "namespace_contract_id",
]


# --- selfcheck ---------------------------------------------------------------


def _selfcheck() -> None:
    """Build all default cards and exercise their contracts with asserts."""
    cards = default_namespace_agent_cards()
    assert len(cards) == len(_DEFAULT_NAMESPACES) == 10
    assert {card.namespace for card in cards} == set(NAMESPACE_AGENT_PROFILES)

    registry = LatticeRegistry(cards)
    assert len(registry.list_cards()) == 10

    # Every card has exactly one capability, one contract, and the shared
    # safety posture.
    for card in cards:
        assert len(card.capabilities) == 1
        assert len(card.contracts) == 1
        assert card.capabilities[0].id == namespace_action(card.namespace or "")
        contract = card.contract(namespace_contract_id(card.namespace or ""))
        assert contract.input_schema["title"] == "NamespaceAgentQuestion"
        assert contract.output_schema["title"] == "NamespaceAgentResult"
        assert len(card.safety_limits) == 3
        assert "safety-critical issue" in card.human_review_triggers

    # Discovery: validation agent found by its real harness tool names.
    validation = registry.get("product.validation")
    assert registry.find(tool="gate.measure")
    assert validation in registry.find(tool="oracle.crosscheck")
    geometry = registry.get("product.geometry")
    assert "CISP" in geometry.summary
    assert geometry in registry.find(namespace="product.geometry")
    assert registry.find(domain="mates") == [registry.get("product.assembly")]

    # Helpers.
    assert namespace_agent_id(" Product.Geometry ") == "product.geometry"
    assert namespace_action("product.mech") == "product.mech.update"
    assert namespace_contract_id("product.mech") == "product.mech.v0"

    # Question/Result dataclasses round-trip and validate.
    question = NamespaceAgentQuestion(
        namespace="product.geometry",
        prompt="Add a 3 mm fillet to the top edge loop.",
        constraints=["preserve wall thickness >= 2 mm"],
    )
    result = NamespaceAgentResult(
        namespace="product.geometry",
        summary="Applied fillet op to op stream.",
        payload_patch={"ops": [{"op": "fillet", "radius_mm": 3.0}]},
        dependencies=["product.validation"],
    )
    json.dumps(question.to_dict())
    json.dumps(result.to_dict())
    json.dumps(registry.manifest())

    for bad in (
        lambda: NamespaceAgentQuestion(namespace="", prompt="x"),
        lambda: NamespaceAgentQuestion(namespace="product.mech", prompt="  "),
        lambda: NamespaceAgentResult(namespace="product.mech", summary=""),
        lambda: namespace_agent_card("", "Label", "Desc", "product"),
        lambda: namespace_agent_card("product.x", "Label", "Desc", " "),
    ):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError from invalid construction")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harnesscad.agents.a2a.lattice_agents",
        description="Built-in CAD namespace agent cards for the Lattice contract layer.",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="build all default namespace cards, register them, exercise "
        "discovery and contract lookup with asserts; print PASS on success.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the default roster manifest as JSON (with --selfcheck).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.selfcheck:
        parser.print_help()
        return 0

    try:
        _selfcheck()
    except AssertionError as exc:
        print(f"SELFCHECK FAILED: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(LatticeRegistry(default_namespace_agent_cards()).manifest(), indent=2, sort_keys=True))
    print("PASS: lattice_agents selfcheck (10 CAD namespace cards, registry discovery, validation)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
