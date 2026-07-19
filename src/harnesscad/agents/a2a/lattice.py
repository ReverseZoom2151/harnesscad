"""Lattice -- typed agent-contract manifests for composable domain agents.

Each domain agent publishes a portable *agent card* declaring the namespace
it owns, the capabilities it exposes, its input/output schema contracts, the
tools it may request, its handoff actions, and its safety and human-review
boundaries. A registry lets other agents discover cards by namespace,
domain, capability, or tool.

Harness gap filled: HarnessCAD's A2A layer already has a *transport*
vocabulary -- ``harnesscad.agents.a2a.messages`` (AgentCard/Part/A2AMessage,
the wire format) and ``harnesscad.agents.a2a.task`` (the async task
lifecycle) -- but no typed *contract* layer sitting above it: nothing that
says, in a portable manifest, what an agent's input/output schemas are,
whether those schemas were declared in code or induced from examples, what
safety limits bind the agent, and when a human must review. This module adds
that layer. It complements (and does not duplicate) messages.py/task.py:
the ``AgentCard`` there is a discovery/handshake artefact for the wire;
``LatticeAgentCard`` here is the richer contract manifest an orchestrator
consults before routing work.

The implementation uses dataclasses and explicit validation; callers pass
``now`` strings instead of consulting a wall clock, and run ids are
caller-supplied or derived from a deterministic counter (``f"lat_{n}"``).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

# --- vocabulary -------------------------------------------------------------
# Canonical schema kinds. "declared" contracts are authored in code;
# "induced" and "refined" contracts are produced or updated from documents,
# examples, or run history. Exposed as tuples so routing/validation code never
# hard-codes string literals (cf. messages.PART_KINDS).
SCHEMA_DECLARED = "declared"
SCHEMA_INDUCED = "induced"
SCHEMA_REFINED = "refined"
SCHEMA_KINDS = (SCHEMA_DECLARED, SCHEMA_INDUCED, SCHEMA_REFINED)

RUN_STARTED = "started"
RUN_COMPLETED = "completed"
RUN_FAILED = "failed"
RUN_BLOCKED = "blocked"
RUN_STATUSES = (RUN_STARTED, RUN_COMPLETED, RUN_FAILED, RUN_BLOCKED)

LATTICE_VERSION = "0.1.0"


def _require_text(name: str, value: str) -> str:
    """Validate that ``value`` is a non-empty string; return it stripped."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _require_choice(name: str, value: str, choices: Sequence[str]) -> str:
    if value not in choices:
        raise ValueError(f"{name} must be one of {list(choices)}, got {value!r}")
    return value


def run_id_from_counter(counter: Callable[[], int]) -> str:
    """Build a deterministic run id ``lat_{n}`` from a counter callable.

    The counter is caller-owned (e.g. ``task.monotonic_counter()``) so run
    ids are reproducible -- the Forma-OSS original used ``uuid4``, which the
    harness forbids.
    """
    return f"lat_{counter()}"


@dataclass
class LatticeCapability:
    """A discrete capability another agent can discover and call.

    ``id`` is the stable capability identifier (e.g. ``product.geometry.update``),
    ``inputs``/``outputs`` are natural-language field or concept names, and
    ``actions`` are the callable action names the capability exposes.
    """

    id: str
    label: str
    description: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.id = _require_text("LatticeCapability.id", self.id)
        self.label = _require_text("LatticeCapability.label", self.label)
        self.description = _require_text(
            "LatticeCapability.description", self.description
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LatticeSchemaContract:
    """A schema contract for an agent input/output pair.

    ``schema_kind`` records provenance: ``declared`` contracts are authored in
    code, ``induced`` contracts are produced from documents or examples, and
    ``refined`` contracts have been updated from run history. Schemas are plain
    JSON-Schema-style dicts (no pydantic in the harness), so ``from_schemas``
    replaces the reference's ``from_models`` and takes the dicts directly.
    """

    id: str
    name: str
    purpose: str
    version: str = "0.1.0"
    schema_kind: str = SCHEMA_DECLARED
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    induction_prompt: Optional[str] = None
    extraction_prompt: Optional[str] = None
    examples: List[Dict[str, Any]] = field(default_factory=list)
    review_required: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = _require_text("LatticeSchemaContract.id", self.id)
        self.name = _require_text("LatticeSchemaContract.name", self.name)
        self.purpose = _require_text("LatticeSchemaContract.purpose", self.purpose)
        self.version = _require_text("LatticeSchemaContract.version", self.version)
        _require_choice(
            "LatticeSchemaContract.schema_kind", self.schema_kind, SCHEMA_KINDS
        )

    @classmethod
    def from_schemas(
        cls,
        *,
        id: str,
        name: str,
        purpose: str,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "LatticeSchemaContract":
        """Build a contract from plain dict schemas.

        The Forma-OSS original (``from_models``) derived JSON Schemas from
        pydantic models; the harness is stdlib-only, so callers hand over the
        schema dicts themselves (empty dicts when unconstrained).
        """
        return cls(
            id=id,
            name=name,
            purpose=purpose,
            input_schema=dict(input_schema or {}),
            output_schema=dict(output_schema or {}),
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LatticeAgentCard:
    """A portable contract manifest for a domain agent.

    Richer than the discovery ``AgentCard`` in a2a.messages: this declares the
    namespace the agent owns, its schema contracts, the runtime boundary (what
    the agent owns versus what the host runtime owns), the tools it may
    request, the handoff actions it can suggest, and its safety posture.
    """

    agent_id: str
    name: str
    domain: str
    summary: str
    runtime_boundary: str
    card_type: str = "lattice.agent_card"
    namespace: Optional[str] = None
    version: str = "0.1.0"
    capabilities: List[LatticeCapability] = field(default_factory=list)
    contracts: List[LatticeSchemaContract] = field(default_factory=list)
    tools_needed: List[str] = field(default_factory=list)
    handoff_actions: List[str] = field(default_factory=list)
    safety_limits: List[str] = field(default_factory=list)
    human_review_triggers: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.agent_id = _require_text("LatticeAgentCard.agent_id", self.agent_id)
        self.name = _require_text("LatticeAgentCard.name", self.name)
        self.domain = _require_text("LatticeAgentCard.domain", self.domain)
        self.summary = _require_text("LatticeAgentCard.summary", self.summary)
        self.runtime_boundary = _require_text(
            "LatticeAgentCard.runtime_boundary", self.runtime_boundary
        )
        self.version = _require_text("LatticeAgentCard.version", self.version)

    def contract(self, contract_id: str) -> LatticeSchemaContract:
        """Return the contract with ``contract_id`` or raise ``KeyError``."""
        for item in self.contracts:
            if item.id == contract_id:
                return item
        raise KeyError(f"Unknown Lattice contract: {contract_id}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LatticeRunRecord:
    """Auditable record for one domain-agent invocation.

    ``run_id`` is caller-supplied (or built with ``run_id_from_counter`` for
    the deterministic ``lat_{n}`` form). Timestamps are opaque caller-supplied
    strings -- the harness keeps wall clocks out of logic, so where the
    reference stamped ``utc_now()`` this accepts an optional ``now``.
    """

    run_id: str
    agent_id: str
    action: str
    record_type: str = "lattice.run_record"
    contract_id: Optional[str] = None
    mode: str = "local"
    status: str = RUN_STARTED
    started_at: str = ""
    completed_at: Optional[str] = None
    input_payload: Dict[str, Any] = field(default_factory=dict)
    output_payload: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    handoff_actions: List[Dict[str, Any]] = field(default_factory=list)
    audit_events: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.run_id = _require_text("LatticeRunRecord.run_id", self.run_id)
        self.agent_id = _require_text("LatticeRunRecord.agent_id", self.agent_id)
        self.action = _require_text("LatticeRunRecord.action", self.action)
        _require_choice("LatticeRunRecord.status", self.status, RUN_STATUSES)

    @classmethod
    def completed(
        cls,
        *,
        agent_card: LatticeAgentCard,
        action: str,
        input_payload: Dict[str, Any],
        output_payload: Dict[str, Any],
        run_id: str = "",
        counter: Optional[Callable[[], int]] = None,
        contract_id: Optional[str] = None,
        mode: str = "local",
        warnings: Optional[List[str]] = None,
        handoff_actions: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        now: str = "",
    ) -> "LatticeRunRecord":
        """Build a completed run record.

        Exactly one of ``run_id`` (caller-supplied) or ``counter`` (a
        deterministic counter callable, e.g. ``task.monotonic_counter()``)
        must identify the run. ``now`` is an optional caller-supplied
        timestamp string stamped on both started_at and completed_at.
        """
        if not run_id:
            if counter is None:
                raise ValueError(
                    "LatticeRunRecord.completed requires run_id or counter"
                )
            run_id = run_id_from_counter(counter)
        return cls(
            run_id=run_id,
            agent_id=agent_card.agent_id,
            action=action,
            contract_id=contract_id,
            mode=mode,
            status=RUN_COMPLETED,
            started_at=now,
            completed_at=now or None,
            input_payload=input_payload,
            output_payload=output_payload,
            warnings=list(warnings or []),
            handoff_actions=list(handoff_actions or []),
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LatticeRegistry:
    """In-memory registry of domain-agent cards, keyed by agent_id.

    Discovery surface for the orchestrator: ``find`` matches by namespace,
    domain, capability id/label, or tool-name substring (all case-insensitive
    substring matches, mirroring the Forma-OSS semantics). ``manifest`` emits
    the whole registry as one JSON-serialisable dict.
    """

    def __init__(self, cards: Optional[List[LatticeAgentCard]] = None) -> None:
        self._cards: Dict[str, LatticeAgentCard] = {}
        for card in cards or []:
            self.register(card)

    def register(self, card: LatticeAgentCard) -> LatticeAgentCard:
        self._cards[card.agent_id] = card
        return card

    def get(self, agent_id: str) -> LatticeAgentCard:
        try:
            return self._cards[agent_id]
        except KeyError as exc:
            raise KeyError(f"Unknown Lattice agent: {agent_id}") from exc

    def list_cards(self) -> List[LatticeAgentCard]:
        return [self._cards[key] for key in sorted(self._cards)]

    def find(
        self,
        *,
        namespace: Optional[str] = None,
        domain: Optional[str] = None,
        capability: Optional[str] = None,
        tool: Optional[str] = None,
    ) -> List[LatticeAgentCard]:
        namespace_text = namespace.lower() if namespace else None
        domain_text = domain.lower() if domain else None
        capability_text = capability.lower() if capability else None
        tool_text = tool.lower() if tool else None
        matches: List[LatticeAgentCard] = []

        for card in self.list_cards():
            if namespace_text and namespace_text not in (card.namespace or "").lower():
                continue
            if domain_text and domain_text not in card.domain.lower():
                continue
            if capability_text and not any(
                capability_text in item.id.lower()
                or capability_text in item.label.lower()
                for item in card.capabilities
            ):
                continue
            if tool_text and not any(
                tool_text in item.lower() for item in card.tools_needed
            ):
                continue
            matches.append(card)

        return matches

    def manifest(self) -> Dict[str, Any]:
        return {
            "name": "Lattice",
            "lattice_version": LATTICE_VERSION,
            "agents": [card.to_dict() for card in self.list_cards()],
        }


__all__ = [
    "LATTICE_VERSION",
    "LatticeAgentCard",
    "LatticeCapability",
    "LatticeRegistry",
    "LatticeRunRecord",
    "LatticeSchemaContract",
    "RUN_STATUSES",
    "SCHEMA_KINDS",
    "run_id_from_counter",
]


# --- selfcheck ---------------------------------------------------------------


def _selfcheck() -> None:
    """Exercise the contract layer end-to-end with asserts."""
    capability = LatticeCapability(
        id="product.geometry.update",
        label="Geometry Agent",
        description="Owns the CISP op stream and parametric features.",
        inputs=["prompt", "current_ops"],
        outputs=["op_patch", "validation_notes"],
        actions=["product.geometry.update"],
    )
    contract = LatticeSchemaContract.from_schemas(
        id="product.geometry.v0",
        name="Geometry Contract",
        purpose="Update and audit the product.geometry namespace.",
        input_schema={"type": "object", "properties": {"prompt": {"type": "string"}}},
        output_schema={
            "type": "object",
            "properties": {"op_patch": {"type": "object"}},
        },
        extraction_prompt="Extract only geometry-relevant fields.",
    )
    assert contract.schema_kind == SCHEMA_DECLARED
    assert contract.input_schema["properties"]["prompt"]["type"] == "string"

    card = LatticeAgentCard(
        agent_id="product.geometry",
        namespace="product.geometry",
        name="Geometry Agent",
        domain="Parametric CAD geometry",
        summary="Owns the CISP op stream, sketches, and features.",
        runtime_boundary=(
            "The agent owns the product.geometry contract; HarnessCAD owns "
            "orchestration, kernel execution, validation, and persistence."
        ),
        capabilities=[capability],
        contracts=[contract],
        tools_needed=["harnesscad.gate.measure", "kernel execution"],
        handoff_actions=["harnesscad.oracle.crosscheck"],
        tags=["lattice", "geometry"],
    )
    assert card.contract("product.geometry.v0") is contract
    try:
        card.contract("missing.v0")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for unknown contract id")

    registry = LatticeRegistry([card])
    assert registry.get("product.geometry") is card
    assert registry.find(namespace="geometry") == [card]
    assert registry.find(domain="parametric") == [card]
    assert registry.find(capability="geometry agent") == [card]
    assert registry.find(tool="gate.measure") == [card]
    assert registry.find(namespace="nonexistent") == []
    manifest = registry.manifest()
    assert manifest["lattice_version"] == LATTICE_VERSION
    assert len(manifest["agents"]) == 1
    json.dumps(manifest)  # must be JSON-serialisable

    # Deterministic run ids, no wall clock.
    import itertools

    counter_iter = itertools.count()
    counter = lambda: next(counter_iter)  # noqa: E731
    record = LatticeRunRecord.completed(
        agent_card=card,
        action="product.geometry.update",
        input_payload={"prompt": "add a boss"},
        output_payload={"op_patch": {}},
        counter=counter,
        contract_id="product.geometry.v0",
        now="t0",
    )
    assert record.run_id == "lat_0"
    assert record.status == RUN_COMPLETED
    assert record.started_at == "t0" and record.completed_at == "t0"
    record2 = LatticeRunRecord.completed(
        agent_card=card,
        action="product.geometry.update",
        input_payload={},
        output_payload={},
        run_id="run-explicit",
    )
    assert record2.run_id == "run-explicit"

    # Validation: empty text fields raise ValueError.
    for bad in (
        lambda: LatticeCapability(id="", label="x", description="y"),
        lambda: LatticeSchemaContract(id="c", name="  ", purpose="p"),
        lambda: LatticeSchemaContract(id="c", name="n", purpose="p", schema_kind="bogus"),
        lambda: LatticeAgentCard(
            agent_id="a", name="n", domain="d", summary="s", runtime_boundary=""
        ),
        lambda: LatticeRunRecord(run_id="r", agent_id="a", action="do", status="odd"),
        lambda: LatticeRunRecord.completed(
            agent_card=card, action="a", input_payload={}, output_payload={}
        ),
    ):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError from invalid construction")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harnesscad.agents.a2a.lattice",
        description="Lattice typed agent-contract layer (ported from Forma-OSS).",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="exercise cards, contracts, registry discovery, run records, and "
        "validation errors with asserts; print PASS on success.",
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
    print("PASS: lattice selfcheck (cards, contracts, registry, run records, validation)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
