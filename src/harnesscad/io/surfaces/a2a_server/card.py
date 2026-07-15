"""The HarnessCAD A2A ``AgentCard`` — the discovery/handshake artefact.

Served at ``GET /.well-known/agent-card.json``. Advertises exactly one skill,
``text-to-cad``: a natural-language brief in, a parametric STEP model out. Built
from the spec-conformant ``a2a.messages`` dataclasses so the wire shape is the
canonical A2A ``AgentCard`` (top-level ``url``/``protocolVersion``/
``preferredTransport``, ``capabilities``, ``defaultInput/OutputModes``, and an
``AgentSkill[]``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from harnesscad.agents.a2a.messages import AgentCard, AgentSkill
from harnesscad.io.surfaces.a2a_server.auth import Authenticator

# The A2A protocol binding this server speaks (JSON-RPC 2.0 over HTTP).
PROTOCOL_VERSION = "0.3.0"

# OpenAPI-aligned security scheme catalogue this server can enforce. The names
# are the keys clients reference from the ``security`` requirement block; the
# ``auth.Authenticator`` returns the same names from ``scheme_names()``.
SECURITY_SCHEMES: Dict[str, Dict[str, Any]] = {
    "bearerAuth": {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": (
            "HMAC-SHA256 (HS256) signed JWT in 'Authorization: Bearer <token>'."
        ),
    },
    "apiKeyAuth": {
        "type": "apiKey",
        "in": "header",
        "name": "API-Key",
        "description": "Shared API key in the 'API-Key' request header.",
    },
}


def build_security(
    authenticator: Optional[Authenticator] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Tuple[Dict[str, Any], ...]]]:
    """Return ``(securitySchemes, security)`` for the Agent Card.

    Advertises the full scheme catalogue so clients can discover what the server
    understands. The ``security`` requirement is populated only when the
    authenticator actually requires auth and has at least one live scheme; each
    live scheme is offered as an independent alternative (Bearer OR API key).
    """
    if authenticator is None or not authenticator.required:
        return dict(SECURITY_SCHEMES), None
    live = authenticator.scheme_names()
    if not live:
        return dict(SECURITY_SCHEMES), None
    requirements = tuple({name: []} for name in live)
    return dict(SECURITY_SCHEMES), requirements

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
    authenticator: Optional[Authenticator] = None,
) -> AgentCard:
    """Construct the HarnessCAD ``AgentCard`` for the given public ``url``.

    When ``authenticator`` is supplied its live schemes drive the card's
    ``securitySchemes``/``security`` block so clients can discover how to
    authenticate; otherwise the schemes are advertised but not required.
    """
    security_schemes, security = build_security(authenticator)
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
        securitySchemes=security_schemes,
        security=security,
    )
