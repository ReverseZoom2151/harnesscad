"""Three-tier approval model (docs/blueprint.md sec.14, sec.10, sec.5).

The blueprint's human-in-the-loop contract:

  - **Tier-1 AUTO**    read / measure / render  -> auto-proceed, no prompt.
  - **Tier-2 NOTIFY**  modify ops (sketch, extrude, fillet, boolean, ...)
                       -> proceed but surface a notification (alert-fatigue aware).
  - **Tier-3 REQUIRE** export / delete / irreversible -> block until a human
                       approves; emit an ``approval_required`` UIEvent carrying a
                       **risk indicator** and a **dry-run preview** of the
                       predicted geometry change.

Tiers are assigned two ways, both here:
  1. From CISP op identity (``tier_for``) — the mutating ops in cisp/ops.py are
     Tier-2; the query/export surface (backend.query/export in loop.py) maps to
     Tier-1 (measure/render) and Tier-3 (export/delete).
  2. From MCP tool annotations (``tier_from_annotations``) — sec.5 says annotate
     ``export``/``delete`` destructive and ``render``/``measure`` read-only; those
     hints auto-assign REQUIRE / AUTO respectively.

Everything here is pure and stdlib-only: the ``DryRunPreview`` predicts a change
by *describing intent*, never by touching the kernel or mutating state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Union

from harnesscad.core.cisp.ops import Op
from harnesscad.io.surfaces.ui.events import UIEvent


class ApprovalTier(Enum):
    """The three approval tiers, ordered by escalating human involvement."""

    AUTO = 1     # Tier-1: auto-proceed, no human in the loop
    NOTIFY = 2   # Tier-2: proceed, but tell the human
    REQUIRE = 3  # Tier-3: block until a human approves

    def __str__(self) -> str:
        return self.name.lower()


class RiskLevel(Enum):
    """Risk indicator shown alongside an approval request."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3

    def __str__(self) -> str:
        return self.name.lower()


# --- op -> tier classification --------------------------------------------
# Keyword sets over the op/tool *name*. Order of precedence: REQUIRE beats AUTO
# beats the NOTIFY default, so an ambiguous name errs toward more oversight.
_REQUIRE_WORDS = (
    "export", "delete", "irreversible", "purge", "drop", "remove", "overwrite",
)
_AUTO_WORDS = (
    "measure", "render", "query", "read", "view", "inspect", "summary",
    "get", "list",
)

# Explicit MCP-style annotations for the known op/tool surface (sec.5). A real
# MCP server would carry these as tool metadata; we mirror them so ``tier_for``
# can honour annotations first and fall back to keyword classification.
#   read_only  -> AUTO      destructive -> REQUIRE
OP_ANNOTATIONS: Dict[str, Dict[str, bool]] = {
    # queries / observations (loop.py: backend.query / render)
    "measure": {"read_only": True},
    "render": {"read_only": True},
    "query": {"read_only": True},
    "summary": {"read_only": True},
    # mutating CISP ops (cisp/ops.py) — Tier-2 NOTIFY (neither hint set)
    "new_sketch": {},
    "add_point": {},
    "add_line": {},
    "add_circle": {},
    "add_rectangle": {},
    "constrain": {},
    "extrude": {},
    "fillet": {},
    "boolean": {},
    # irreversible surface (loop.py: backend.export) — Tier-3 REQUIRE
    "export": {"destructive": True},
    "export_step": {"destructive": True},
    "export_stl": {"destructive": True},
    "delete": {"destructive": True},
}


def op_name(op: Union[str, Op]) -> str:
    """Normalise an op reference (an ``Op`` instance, class, or name) to a name."""
    if isinstance(op, str):
        return op.strip().lower()
    # Op instance or Op subclass — both expose the ``OP`` tag.
    tag = getattr(op, "OP", None)
    if tag is None:
        raise TypeError(f"cannot derive an op name from {op!r}")
    return str(tag).strip().lower()


def tier_from_annotations(annotations: Dict[str, bool]) -> ApprovalTier:
    """Map MCP-style tool annotations to a tier (sec.5 auto-assignment).

    ``destructive`` -> REQUIRE (Tier-3); ``read_only`` -> AUTO (Tier-1);
    otherwise NOTIFY (Tier-2, the mutating default).
    """
    if annotations.get("destructive"):
        return ApprovalTier.REQUIRE
    if annotations.get("read_only"):
        return ApprovalTier.AUTO
    return ApprovalTier.NOTIFY


def tier_for(op: Union[str, Op]) -> ApprovalTier:
    """Classify a CISP op / tool into an ``ApprovalTier``.

    read/measure/render        -> AUTO
    modify/sketch/extrude/...   -> NOTIFY
    export/delete/irreversible  -> REQUIRE

    Annotations (``OP_ANNOTATIONS``) win when present; otherwise the name is
    matched against the REQUIRE then AUTO keyword sets, defaulting to NOTIFY.
    """
    name = op_name(op)
    if name in OP_ANNOTATIONS:
        return tier_from_annotations(OP_ANNOTATIONS[name])
    if any(w in name for w in _REQUIRE_WORDS):
        return ApprovalTier.REQUIRE
    if any(w in name for w in _AUTO_WORDS):
        return ApprovalTier.AUTO
    return ApprovalTier.NOTIFY


# Risk maps one-to-one onto tier: the more oversight a tier demands, the higher
# the risk indicator the UI shows.
_TIER_RISK = {
    ApprovalTier.AUTO: RiskLevel.LOW,
    ApprovalTier.NOTIFY: RiskLevel.MEDIUM,
    ApprovalTier.REQUIRE: RiskLevel.HIGH,
}


def risk_for(op_or_tier: Union[str, Op, ApprovalTier]) -> RiskLevel:
    """Risk indicator for an op or a tier."""
    tier = op_or_tier if isinstance(op_or_tier, ApprovalTier) else tier_for(op_or_tier)
    return _TIER_RISK[tier]


# --- dry-run preview -------------------------------------------------------
# Per-op intent templates: (summary verb, predicted effect on model state). These
# describe the change without running the kernel — the preview is advisory.
_OP_INTENT: Dict[str, str] = {
    "new_sketch": "add an empty sketch",
    "add_point": "add a point to the sketch",
    "add_line": "add a line to the sketch",
    "add_circle": "add a circle to the sketch",
    "add_rectangle": "add a rectangle to the sketch",
    "constrain": "apply a constraint (reduces sketch DOF)",
    "extrude": "extrude a sketch into a new solid",
    "fillet": "round edges with a fillet",
    "boolean": "combine solids with a boolean",
    "export": "write geometry to a file (irreversible side effect)",
    "delete": "delete geometry (irreversible)",
    "measure": "read a measurement (no change)",
    "render": "render a viewport (no change)",
}


@dataclass
class DryRunPreview:
    """A predicted, non-mutating description of what an op would do.

    ``before``/``after`` are *intent* dicts (a coarse model-state delta), not a
    kernel result — computed by reading the op's fields, never by applying it.
    ``changes`` is a flat list of human-readable predicted effects for a UI card.
    """

    op_name: str
    summary: str
    before: Dict = field(default_factory=dict)
    after: Dict = field(default_factory=dict)
    changes: List[str] = field(default_factory=list)
    mutates: bool = True

    def to_dict(self) -> dict:
        return {
            "op": self.op_name,
            "summary": self.summary,
            "before": self.before,
            "after": self.after,
            "changes": self.changes,
            "mutates": self.mutates,
        }

    @classmethod
    def for_op(cls, op: Union[str, Op],
               before_state: Optional[Dict] = None) -> "DryRunPreview":
        """Build a preview for ``op`` WITHOUT mutating any state.

        ``before_state`` is an optional read-only snapshot (e.g. a backend
        ``summary()``); it is copied, never modified. ``after`` is a shallow
        intent-projected copy so a UI can show a before/after diff.
        """
        name = op_name(op)
        tier = tier_for(op)
        intent = _OP_INTENT.get(name, f"apply {name}")
        mutates = tier is not ApprovalTier.AUTO

        before = dict(before_state) if before_state else {}
        after = dict(before)  # copy — we never touch the original snapshot

        changes: List[str] = []
        # Enrich the summary/changes from op fields when we have an Op instance.
        detail = ""
        if not isinstance(op, str):
            fields = {k: v for k, v in getattr(op, "__dict__", {}).items()}
            if name == "extrude":
                detail = f" (sketch={fields.get('sketch')!r}, distance={fields.get('distance')})"
                after["solids_intent"] = before.get("solids_intent", 0) + 1
            elif name == "fillet":
                detail = f" (radius={fields.get('radius')}, edges={fields.get('edges')})"
            elif name == "boolean":
                detail = f" ({fields.get('kind')}: {fields.get('target')!r} <- {fields.get('tool')!r})"
            elif name == "constrain":
                detail = f" (kind={fields.get('kind')!r}, value={fields.get('value')})"
            elif name in ("add_circle", "add_rectangle", "add_line", "add_point"):
                detail = f" (sketch={fields.get('sketch')!r})"
            elif name == "new_sketch":
                detail = f" (plane={fields.get('plane')!r})"
        summary = f"{intent}{detail}"
        changes.append(summary)
        if mutates:
            after["last_op_intent"] = name
        return cls(op_name=name, summary=summary, before=before, after=after,
                   changes=changes, mutates=mutates)


# --- approval decision + gate ---------------------------------------------
@dataclass
class ApprovalDecision:
    """The gate's verdict on one op."""

    op_name: str
    tier: ApprovalTier
    risk: RiskLevel
    auto_proceed: bool
    requires_approval: bool
    preview: DryRunPreview
    event: Optional[UIEvent] = None  # set for Tier-3 (approval_required)

    def to_dict(self) -> dict:
        return {
            "op": self.op_name,
            "tier": str(self.tier),
            "risk": str(self.risk),
            "auto_proceed": self.auto_proceed,
            "requires_approval": self.requires_approval,
            "preview": self.preview.to_dict(),
            "event": self.event.to_dict() if self.event else None,
        }


class ApprovalGate:
    """Decides whether an op may auto-proceed or needs human approval.

    Tier-1 (AUTO) and Tier-2 (NOTIFY) auto-proceed — NOTIFY additionally surfaces
    a ``status`` notification. Tier-3 (REQUIRE) does NOT auto-proceed: the gate
    builds an ``approval_required`` UIEvent carrying the risk indicator and the
    dry-run preview, and (if an ``emit`` sink is supplied) pushes it there.

    Every emitted event is also retained on ``self.emitted`` for inspection.
    ``batch_evaluate`` groups related Tier-3 approvals into a single
    ``approval_required`` event to fight alert fatigue (sec.14).
    """

    def __init__(self, emit: Optional[Callable[[UIEvent], None]] = None,
                 state_provider: Optional[Callable[[], Dict]] = None) -> None:
        self._emit = emit
        # Optional read-only snapshot source for previews (e.g. session.summary).
        self._state_provider = state_provider
        self.emitted: List[UIEvent] = []

    # --- internals --------------------------------------------------------
    def _push(self, event: UIEvent) -> UIEvent:
        self.emitted.append(event)
        if self._emit is not None:
            self._emit(event)
        return event

    def _snapshot(self) -> Optional[Dict]:
        if self._state_provider is None:
            return None
        try:
            return self._state_provider()
        except Exception:
            return None  # a preview must never break the gate

    # --- single op --------------------------------------------------------
    def evaluate(self, op: Union[str, Op]) -> ApprovalDecision:
        """Classify one op and, for Tier-3, emit an ``approval_required`` event."""
        name = op_name(op)
        tier = tier_for(op)
        risk = _TIER_RISK[tier]
        preview = DryRunPreview.for_op(op, self._snapshot())

        if tier is ApprovalTier.REQUIRE:
            event = UIEvent.approval_required(
                name=name, risk=str(risk), preview=preview.to_dict())
            self._push(event)
            return ApprovalDecision(name, tier, risk, auto_proceed=False,
                                    requires_approval=True, preview=preview,
                                    event=event)

        if tier is ApprovalTier.NOTIFY:
            # Proceeds, but the human is told (alert-fatigue-aware: a status line,
            # not a blocking prompt).
            self._push(UIEvent.status(
                f"proceeding: {name}", tier=str(tier), risk=str(risk)))

        return ApprovalDecision(name, tier, risk, auto_proceed=True,
                                requires_approval=False, preview=preview,
                                event=None)

    def may_proceed(self, op: Union[str, Op]) -> bool:
        """Convenience: True iff the op auto-proceeds (Tier-1/Tier-2)."""
        return self.evaluate(op).auto_proceed

    # --- batching ---------------------------------------------------------
    def batch_evaluate(self, ops: Sequence[Union[str, Op]]) -> List[ApprovalDecision]:
        """Evaluate a related group of ops, batching Tier-3 approvals.

        AUTO/NOTIFY ops are handled as in ``evaluate``. All REQUIRE ops in the
        group are collapsed into ONE ``approval_required`` event whose ``batch``
        field lists each op's preview — so the human approves the set once
        instead of dismissing N prompts.
        """
        decisions: List[ApprovalDecision] = []
        require: List[ApprovalDecision] = []

        for op in ops:
            name = op_name(op)
            tier = tier_for(op)
            risk = _TIER_RISK[tier]
            preview = DryRunPreview.for_op(op, self._snapshot())
            if tier is ApprovalTier.REQUIRE:
                d = ApprovalDecision(name, tier, risk, auto_proceed=False,
                                     requires_approval=True, preview=preview,
                                     event=None)
                require.append(d)
                decisions.append(d)
            elif tier is ApprovalTier.NOTIFY:
                self._push(UIEvent.status(
                    f"proceeding: {name}", tier=str(tier), risk=str(risk)))
                decisions.append(ApprovalDecision(
                    name, tier, risk, True, False, preview, None))
            else:  # AUTO
                decisions.append(ApprovalDecision(
                    name, tier, risk, True, False, preview, None))

        if require:
            # Highest risk in the batch drives the indicator.
            top_risk = max((d.risk for d in require), key=lambda r: r.value)
            batch_payload = [
                {"name": d.op_name, "risk": str(d.risk),
                 "preview": d.preview.to_dict()} for d in require
            ]
            event = UIEvent.approval_required(
                name=f"batch:{len(require)}", risk=str(top_risk),
                preview=require[0].preview.to_dict(), batch=batch_payload)
            self._push(event)
            # Attach the shared batch event to every Tier-3 decision.
            for d in require:
                d.event = event
        return decisions


# ===========================================================================
# The MECHANICAL gate: ApprovalPolicy
# ===========================================================================
# ``ApprovalGate`` above is *advisory*: it classifies and emits, and every caller
# was free to ignore its verdict — and every caller did. ``fleet_blocking=False``
# taught us what an advisory guardrail buys (all of the harm, none of the
# containment). ``ApprovalPolicy`` is the enforcing half:
#
#   * ``decide(...)`` returns an ``ApprovalRecord`` for EVERY action it sees
#     (Tier-1/2 included) and appends it to an audit log. Nothing passes the gate
#     without leaving a record of who let it through and why.
#   * ``require(...)`` RAISES ``ApprovalDenied`` when the record is not approved.
#     A caller that forgets to check a bool cannot silently proceed.
#   * Headless/non-interactive contexts (no human attached: MCP stdio, A2A, CI,
#     an RL rollout) must state a ``HeadlessPolicy`` EXPLICITLY. There is no
#     silent default: the default is REFUSE, and choosing AUTO_APPROVE requires a
#     written ``headless_reason`` that is copied into every record it approves.


class HeadlessPolicy(Enum):
    """What a Tier-3 action does when there is no human approver attached.

    REFUSE        the action is denied and a denial record is written (default).
    AUTO_APPROVE  the action proceeds, and an approval record naming the policy
                  and its ``headless_reason`` is written. Never silent.
    """

    REFUSE = "refuse"
    AUTO_APPROVE = "auto_approve"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ApprovalRecord:
    """One auditable decision: what was asked, who decided, and why."""

    op_name: str
    tier: ApprovalTier
    risk: RiskLevel
    approved: bool
    decided_by: str          # "policy:tier-1" | "human:<principal>" | "policy:headless-*"
    reason: str
    surface: str = ""        # which surface asked (harness / mcp / a2a / acp)
    preview: Optional[DryRunPreview] = None

    def to_dict(self) -> dict:
        return {
            "op": self.op_name,
            "tier": str(self.tier),
            "risk": str(self.risk),
            "approved": self.approved,
            "decided_by": self.decided_by,
            "reason": self.reason,
            "surface": self.surface,
            "preview": self.preview.to_dict() if self.preview else None,
        }


class ApprovalDenied(Exception):
    """Raised by ``ApprovalPolicy.require`` when an action is not approved.

    Carries the ``ApprovalRecord`` so the surface can render an auditable refusal
    instead of inventing its own error text.
    """

    def __init__(self, record: ApprovalRecord) -> None:
        super().__init__(
            f"approval denied for '{record.op_name}' "
            f"(tier={record.tier}, risk={record.risk}): {record.reason}")
        self.record = record


class ApprovalPolicy:
    """The enforcing human-in-the-loop gate. Composed by every write surface.

    ``approver`` is the human channel: a callable taking the op/tool name (or the
    ``Op``) and returning a bool. The ACP surface passes the blocking
    ``session/request_permission`` round-trip here; a CLI passes a prompt.

    With no ``approver``, the process is HEADLESS and ``headless`` decides —
    REFUSE by default. ``HeadlessPolicy.AUTO_APPROVE`` is legal but must be asked
    for by name and must carry a ``headless_reason``; that reason is written into
    every record it approves, so an auto-approved destructive action is always
    attributable to a stated policy rather than to an omission.
    """

    def __init__(
        self,
        approver: Optional[Callable[[Union[str, Op]], bool]] = None,
        *,
        headless: HeadlessPolicy = HeadlessPolicy.REFUSE,
        headless_reason: str = "",
        principal: str = "unknown",
        surface: str = "",
        gate: Optional[ApprovalGate] = None,
    ) -> None:
        if headless is HeadlessPolicy.AUTO_APPROVE and not headless_reason.strip():
            raise ValueError(
                "HeadlessPolicy.AUTO_APPROVE requires a headless_reason: an "
                "unattended destructive action must be attributable to a stated "
                "policy, not to a default")
        self.approver = approver
        self.headless = headless
        self.headless_reason = headless_reason.strip()
        self.principal = principal
        self.surface = surface
        self.gate = gate if gate is not None else ApprovalGate()
        self.audit: List[ApprovalRecord] = []

    # --- construction helpers --------------------------------------------
    @classmethod
    def headless_auto_approve(cls, reason: str, *, principal: str = "policy",
                              surface: str = "") -> "ApprovalPolicy":
        """An explicit, recorded auto-approve policy for an unattended surface."""
        return cls(None, headless=HeadlessPolicy.AUTO_APPROVE,
                   headless_reason=reason, principal=principal, surface=surface)

    @property
    def headless_context(self) -> bool:
        """True when no human approver is attached to this process."""
        return self.approver is None

    def audit_dicts(self) -> List[dict]:
        return [r.to_dict() for r in self.audit]

    # --- the decision -----------------------------------------------------
    def decide(self, op: Union[str, Op], *,
               tier: Optional[ApprovalTier] = None) -> ApprovalRecord:
        """Classify ``op``, decide, RECORD, and return the record.

        ``tier`` overrides classification (the MCP surface passes the tier its
        tool annotations already carry, so the two classifiers cannot drift).
        """
        name = op_name(op)
        t = tier if tier is not None else tier_for(op)
        risk = _TIER_RISK[t]
        preview = DryRunPreview.for_op(op) if not isinstance(op, str) else None

        if t is not ApprovalTier.REQUIRE:
            # Tier-1/Tier-2 auto-proceed — but the decision is still recorded, and
            # Tier-2 still notifies (the gate emits the status event).
            try:
                self.gate.evaluate(op)
            except Exception:  # noqa: BLE001 - a preview must never break the gate
                pass
            return self._record(ApprovalRecord(
                name, t, risk, approved=True,
                decided_by=f"policy:tier-{t.value}",
                reason=("read-only, no human gate required"
                        if t is ApprovalTier.AUTO
                        else "mutating op: proceed with notification (tier-2)"),
                surface=self.surface, preview=preview))

        # Tier-3: surface the risk indicator + dry-run preview, then decide.
        try:
            self.gate.evaluate(op)
        except Exception:  # noqa: BLE001
            pass

        if self.approver is not None:
            approved = bool(self.approver(op))
            return self._record(ApprovalRecord(
                name, t, risk, approved=approved,
                decided_by=f"human:{self.principal}",
                reason=("approved by human" if approved else "denied by human"),
                surface=self.surface, preview=preview))

        if self.headless is HeadlessPolicy.AUTO_APPROVE:
            return self._record(ApprovalRecord(
                name, t, risk, approved=True,
                decided_by="policy:headless-auto-approve",
                reason=f"headless auto-approve (explicit policy): {self.headless_reason}",
                surface=self.surface, preview=preview))

        return self._record(ApprovalRecord(
            name, t, risk, approved=False,
            decided_by="policy:headless-refuse",
            reason=("tier-3 (destructive/irreversible) action attempted in a "
                    "headless context with no human approver; policy=refuse. "
                    "Attach an approver, or construct the surface with "
                    "ApprovalPolicy.headless_auto_approve(reason=...)."),
            surface=self.surface, preview=preview))

    def require(self, op: Union[str, Op], *,
                tier: Optional[ApprovalTier] = None) -> ApprovalRecord:
        """``decide``, but RAISE ``ApprovalDenied`` when the verdict is no.

        This is the mechanical form: a caller cannot forget to check a bool.
        """
        record = self.decide(op, tier=tier)
        if not record.approved:
            raise ApprovalDenied(record)
        return record

    # --- internals --------------------------------------------------------
    def _record(self, record: ApprovalRecord) -> ApprovalRecord:
        self.audit.append(record)
        return record
