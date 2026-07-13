"""The HarnessCAD A2A ``AgentCard`` — the discovery/handshake artefact.

Served at ``GET /.well-known/agent-card.json``. Advertises exactly one skill,
``text-to-cad``: a natural-language brief in, a parametric STEP model out. Built
from the spec-conformant ``a2a.messages`` dataclasses so the wire shape is the
canonical A2A ``AgentCard`` (top-level ``url``/``protocolVersion``/
``preferredTransport``, ``capabilities``, ``defaultInput/OutputModes``, and an
``AgentSkill[]``).
"""

from __future__ import annotations

from typing import Optional

from harnesscad.agents.a2a.messages import AgentCard, AgentSkill

# The A2A protocol binding this server speaks (JSON-RPC 2.0 over HTTP).
PROTOCOL_VERSION = "0.3.0"

# The single skill this agent advertises.
TEXT_TO_CAD_SKILL = AgentSkill(
    id="text-to-cad",
    name="Text to CAD",
    description=(
        "Turn a natural-language design brief into a parametric CAD model and "
        "return it as a STEP file. Backed by the HarnessCAD ReAct harness "
        "(plan -> apply ops -> verify -> repair)."
    ),
    tags=("cad", "step", "parametric"),
    examples=("a 20mm bracket with two M4 holes",),
    inputModes=("text/plain",),
    outputModes=("model/step", "application/json"),
)


def build_agent_card(
    url: str = "http://127.0.0.1:9100/",
    version: Optional[str] = "0.2.1",
) -> AgentCard:
    """Construct the HarnessCAD ``AgentCard`` for the given public ``url``."""
    return AgentCard(
        name="HarnessCAD",
        description=(
            "An agent that generates parametric CAD geometry (STEP) from a "
            "text brief. Exposes the A2A JSON-RPC binding with streaming task "
            "updates."
        ),
        capabilities={"streaming": True, "pushNotifications": False},
        skills=(TEXT_TO_CAD_SKILL.to_dict(),),
        version=version,
        url=url,
        protocolVersion=PROTOCOL_VERSION,
        defaultInputModes=("text/plain",),
        defaultOutputModes=("model/step", "application/json"),
        preferredTransport="JSONRPC",
    )
