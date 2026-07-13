"""Prompt and tool trust-boundary enforcement.

The gate is deliberately deterministic. It does not try to classify arbitrary
language with an LLM; it blocks known instruction-smuggling patterns, enforces
an explicit tool allowlist and trust tier, and emits auditable decisions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Iterable, List, Mapping, Optional


class TrustTier(IntEnum):
    UNTRUSTED = 0
    USER = 1
    PROJECT = 2
    SYSTEM = 3


@dataclass(frozen=True)
class ToolPolicy:
    allowed_tools: frozenset[str]
    minimum_trust: TrustTier = TrustTier.USER
    max_prompt_chars: int = 50_000

    def __post_init__(self) -> None:
        if self.max_prompt_chars <= 0:
            raise ValueError("max_prompt_chars must be positive")


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    code: str
    reason: str
    tool: Optional[str] = None
    sequence: int = 0

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "code": self.code,
            "reason": self.reason,
            "tool": self.tool,
            "sequence": self.sequence,
        }


_INJECTION_PATTERNS = (
    re.compile(r"\bignore\s+(all\s+)?(previous|prior|above)\s+instructions?\b", re.I),
    re.compile(r"\b(system|developer)\s+prompt\b", re.I),
    re.compile(r"\breveal\s+(your\s+)?(instructions?|secrets?|credentials?)\b", re.I),
    re.compile(r"\b(disable|bypass|override)\s+(safety|guardrails?|policy|approval)\b", re.I),
    re.compile(r"<\s*/?\s*(system|developer|tool)\b", re.I),
)


def prompt_risks(text: str) -> List[str]:
    """Return stable risk identifiers for known instruction-smuggling forms."""
    risks = []
    for index, pattern in enumerate(_INJECTION_PATTERNS, 1):
        if pattern.search(text or ""):
            risks.append(f"injection-pattern-{index}")
    return risks


class ToolTrustGate:
    """Authorize prompt/tool actions and retain append-only decisions."""

    def __init__(self, policy: ToolPolicy) -> None:
        self.policy = policy
        self._events: List[GateDecision] = []

    @property
    def events(self) -> List[GateDecision]:
        return list(self._events)

    def _record(self, allowed: bool, code: str, reason: str,
                tool: Optional[str] = None) -> GateDecision:
        event = GateDecision(
            allowed=allowed,
            code=code,
            reason=reason,
            tool=tool,
            sequence=len(self._events) + 1,
        )
        self._events.append(event)
        return event

    def inspect_prompt(self, text: str, trust: TrustTier) -> GateDecision:
        if len(text or "") > self.policy.max_prompt_chars:
            return self._record(False, "prompt-too-large", "prompt exceeds policy limit")
        risks = prompt_risks(text)
        if risks and trust < TrustTier.SYSTEM:
            return self._record(
                False, "prompt-injection",
                "blocked instruction-smuggling indicators: " + ", ".join(risks),
            )
        return self._record(True, "prompt-allowed", "prompt passed deterministic checks")

    def authorize_tool(
        self,
        tool: str,
        trust: TrustTier,
        arguments: Optional[Mapping[str, Any]] = None,
    ) -> GateDecision:
        if tool not in self.policy.allowed_tools:
            return self._record(
                False, "tool-denied", f"tool {tool!r} is not in the allowlist", tool
            )
        if trust < self.policy.minimum_trust:
            return self._record(
                False, "insufficient-trust",
                f"{trust.name.lower()} is below {self.policy.minimum_trust.name.lower()}",
                tool,
            )
        if _contains_control_key(arguments or {}):
            return self._record(
                False, "reserved-argument",
                "arguments contain reserved policy/control keys", tool,
            )
        return self._record(True, "tool-allowed", "tool call satisfies policy", tool)


def _contains_control_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if normalized in {
                "system_prompt", "developer_prompt", "bypass_approval",
                "disable_guardrails", "policy_override",
            }:
                return True
            if _contains_control_key(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_control_key(item) for item in value)
    return False
