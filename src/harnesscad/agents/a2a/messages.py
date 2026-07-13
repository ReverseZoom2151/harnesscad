"""The A2A (agent-to-agent) message vocabulary — our *internal* wire format.

Per docs/blueprint.md sec.2 and sec.12, agents talk to each other in the A2A
message format *even in-process*, so a remote transport (HTTP + SSE/webhooks) is
a drop-in later without touching agent code. This module is that vocabulary as
frozen, typed, JSON-serialisable dataclasses:

  - ``AgentCard``  — how an agent advertises itself (name, description,
    capabilities, skills, endpoints). The discovery/handshake artefact.
  - ``Part``       — one piece of message content: ``text`` | ``data`` | ``artifact``.
  - ``A2AMessage`` — one turn between agents: a ``role`` ('user' | 'agent'), an
    ordered list of ``Part``s, plus ``contextId``/``taskId``/``metadata`` for
    correlating turns with a task lifecycle (see a2a.task).

Everything is a plain value object with ``to_dict``/``from_dict`` so the same
shapes serialise straight to JSON on a real network hop. stdlib only; no vendor
SDKs — this mirrors the vendor-neutral seam philosophy of llm/base.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# --- part kinds ------------------------------------------------------------
# The canonical content kinds a Part may carry. Exposed so routing/validation
# code never hard-codes string literals (cf. trace.EVENT_KINDS).
PART_TEXT = "text"
PART_DATA = "data"
PART_FILE = "file"
# PART_ARTIFACT is NOT a spec A2A Part kind (the spec Part union is text|file|data).
# We keep it working for back-compat with existing agent code, but new code that
# needs to hand back a produced artefact should use the first-class ``Artifact``
# dataclass below rather than a ``Part`` of kind "artifact".
PART_ARTIFACT = "artifact"
PART_KINDS = (PART_TEXT, PART_DATA, PART_FILE, PART_ARTIFACT)

# The two roles an A2A turn may take. 'user' is the requesting agent (or human
# proxy); 'agent' is the responding agent.
ROLE_USER = "user"
ROLE_AGENT = "agent"


@dataclass(frozen=True)
class Part:
    """One piece of A2A message content.

    A ``Part`` is a tagged union over three variants, discriminated by ``kind``:

      - ``text``     — free text in ``text``.
      - ``data``     — a structured JSON object in ``data`` (e.g. ops, params).
      - ``artifact`` — a produced artefact descriptor in ``artifact``
        (e.g. ``{"name": "part.step", "mimeType": "model/step", "uri": ...}``),
        which is how long solves hand back geometry/mesh/FEA results.

    Use the ``Part.text``/``Part.data``/``Part.artifact`` constructors rather
    than the raw initialiser so the invariant (exactly one payload set for the
    kind) is upheld.
    """

    kind: str
    text: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    file: Optional[Dict[str, Any]] = None
    artifact: Optional[Dict[str, Any]] = None

    # --- variant constructors ------------------------------------------
    @classmethod
    def from_text(cls, text: str) -> "Part":
        return cls(kind=PART_TEXT, text=text)

    @classmethod
    def from_data(cls, data: Dict[str, Any]) -> "Part":
        return cls(kind=PART_DATA, data=dict(data))

    @classmethod
    def from_file(
        cls,
        name: Optional[str] = None,
        mime_type: Optional[str] = None,
        bytes_b64: Optional[str] = None,
        uri: Optional[str] = None,
    ) -> "Part":
        """Build a spec ``file`` Part: ``{kind:"file", file:{name?,mimeType?,bytes?|uri?}}``.

        Per the A2A FilePart, the file payload carries the content either inline
        as base64 (``bytes``) or by reference (``uri``) -- exactly one, mirroring
        FileWithBytes vs FileWithUri. Optional ``name``/``mimeType`` describe it.
        """
        f: Dict[str, Any] = {}
        if name is not None:
            f["name"] = name
        if mime_type is not None:
            f["mimeType"] = mime_type
        if bytes_b64 is not None:
            f["bytes"] = bytes_b64
        if uri is not None:
            f["uri"] = uri
        return cls(kind=PART_FILE, file=f)

    @classmethod
    def from_artifact(cls, artifact: Dict[str, Any]) -> "Part":
        return cls(kind=PART_ARTIFACT, artifact=dict(artifact))

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"kind": self.kind}
        if self.kind == PART_TEXT:
            d["text"] = self.text
        elif self.kind == PART_DATA:
            d["data"] = self.data
        elif self.kind == PART_FILE:
            d["file"] = self.file
        elif self.kind == PART_ARTIFACT:
            d["artifact"] = self.artifact
        else:  # pragma: no cover - defensive; unknown kinds serialise verbatim
            if self.text is not None:
                d["text"] = self.text
            if self.data is not None:
                d["data"] = self.data
            if self.file is not None:
                d["file"] = self.file
            if self.artifact is not None:
                d["artifact"] = self.artifact
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Part":
        return cls(
            kind=d["kind"],
            text=d.get("text"),
            data=d.get("data"),
            file=d.get("file"),
            artifact=d.get("artifact"),
        )


@dataclass(frozen=True)
class Artifact:
    """A first-class A2A ``Artifact`` -- a produced output of a task.

    Per the A2A spec an Artifact is ``{artifactId, name?, description?, parts:
    Part[], metadata?}``. Unlike a ``Part`` of the legacy "artifact" kind (which
    is a single opaque descriptor), an Artifact groups one or more real content
    ``Part``s (text/file/data) under a stable ``artifactId``.
    """

    artifactId: str
    name: Optional[str] = None
    description: Optional[str] = None
    parts: Tuple[Part, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "artifactId": self.artifactId,
            "parts": [p.to_dict() for p in self.parts],
        }
        if self.name is not None:
            d["name"] = self.name
        if self.description is not None:
            d["description"] = self.description
        if self.metadata:
            d["metadata"] = dict(self.metadata)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Artifact":
        return cls(
            artifactId=d["artifactId"],
            name=d.get("name"),
            description=d.get("description"),
            parts=tuple(Part.from_dict(p) for p in (d.get("parts") or ())),
            metadata=dict(d.get("metadata") or {}),
        )


@dataclass(frozen=True)
class AgentSkill:
    """An A2A ``AgentSkill`` descriptor advertised in an AgentCard.

    Shape: ``{id, name, description, tags, examples?, inputModes?,
    outputModes?}``. ``to_dict`` emits the conformant wire shape; use it to
    populate ``AgentCard.skills``.
    """

    id: str
    name: str
    description: str = ""
    tags: Tuple[str, ...] = ()
    examples: Optional[Tuple[str, ...]] = None
    inputModes: Optional[Tuple[str, ...]] = None
    outputModes: Optional[Tuple[str, ...]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
        }
        if self.examples is not None:
            d["examples"] = list(self.examples)
        if self.inputModes is not None:
            d["inputModes"] = list(self.inputModes)
        if self.outputModes is not None:
            d["outputModes"] = list(self.outputModes)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentSkill":
        examples = d.get("examples")
        input_modes = d.get("inputModes")
        output_modes = d.get("outputModes")
        return cls(
            id=d["id"],
            name=d.get("name", ""),
            description=d.get("description", ""),
            tags=tuple(d.get("tags") or ()),
            examples=tuple(examples) if examples is not None else None,
            inputModes=tuple(input_modes) if input_modes is not None else None,
            outputModes=tuple(output_modes) if output_modes is not None else None,
        )


@dataclass(frozen=True)
class AgentCard:
    """An agent's self-description — the A2A discovery/handshake artefact.

    ``capabilities`` advertises transport features (e.g. ``{"streaming": True,
    "pushNotifications": True}``) so a caller knows whether SSE/webhooks are
    available. ``skills`` is an ordered list of skill descriptors (each a plain
    dict, e.g. ``{"id": "extrude", "description": ...}``). ``endpoints`` maps
    logical names to URLs (e.g. ``{"a2a": "https://.../a2a"}``); empty in-process.
    """

    name: str
    description: str = ""
    capabilities: Dict[str, Any] = field(default_factory=dict)
    skills: Tuple[Dict[str, Any], ...] = ()
    version: Optional[str] = None
    # --- spec fields ----------------------------------------------------
    url: Optional[str] = None
    protocolVersion: str = "0.3.0"
    defaultInputModes: Tuple[str, ...] = ()
    defaultOutputModes: Tuple[str, ...] = ()
    preferredTransport: str = "JSONRPC"
    provider: Optional[Dict[str, Any]] = None
    securitySchemes: Optional[Dict[str, Any]] = None
    security: Optional[Tuple[Dict[str, Any], ...]] = None
    additionalInterfaces: Optional[Tuple[Dict[str, Any], ...]] = None
    supportsAuthenticatedExtendedCard: Optional[bool] = None
    # --- deprecated -----------------------------------------------------
    # ``endpoints`` predates the spec's single top-level ``url``. Kept as a
    # deprecated alias: if ``url`` is unset it is populated from ``endpoints``
    # (preferring the "a2a" endpoint, else the first value) in __post_init__.
    endpoints: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.url is None and self.endpoints:
            derived = self.endpoints.get("a2a")
            if derived is None:
                derived = next(iter(self.endpoints.values()))
            object.__setattr__(self, "url", derived)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "protocolVersion": self.protocolVersion,
            "preferredTransport": self.preferredTransport,
            "capabilities": dict(self.capabilities),
            "defaultInputModes": list(self.defaultInputModes),
            "defaultOutputModes": list(self.defaultOutputModes),
            "skills": [dict(s) for s in self.skills],
            "version": self.version,
            "endpoints": dict(self.endpoints),
        }
        if self.provider is not None:
            d["provider"] = dict(self.provider)
        if self.securitySchemes is not None:
            d["securitySchemes"] = dict(self.securitySchemes)
        if self.security is not None:
            d["security"] = [dict(s) for s in self.security]
        if self.additionalInterfaces is not None:
            d["additionalInterfaces"] = [dict(i) for i in self.additionalInterfaces]
        if self.supportsAuthenticatedExtendedCard is not None:
            d["supportsAuthenticatedExtendedCard"] = self.supportsAuthenticatedExtendedCard
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentCard":
        security = d.get("security")
        additional = d.get("additionalInterfaces")
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            capabilities=dict(d.get("capabilities") or {}),
            skills=tuple(dict(s) for s in (d.get("skills") or ())),
            version=d.get("version"),
            url=d.get("url"),
            protocolVersion=d.get("protocolVersion", "0.3.0"),
            defaultInputModes=tuple(d.get("defaultInputModes") or ()),
            defaultOutputModes=tuple(d.get("defaultOutputModes") or ()),
            preferredTransport=d.get("preferredTransport", "JSONRPC"),
            provider=dict(d["provider"]) if d.get("provider") is not None else None,
            securitySchemes=dict(d["securitySchemes"]) if d.get("securitySchemes") is not None else None,
            security=tuple(dict(s) for s in security) if security is not None else None,
            additionalInterfaces=tuple(dict(i) for i in additional) if additional is not None else None,
            supportsAuthenticatedExtendedCard=d.get("supportsAuthenticatedExtendedCard"),
            endpoints=dict(d.get("endpoints") or {}),
        )


@dataclass(frozen=True)
class A2AMessage:
    """One turn between two agents.

    ``role`` is 'user' (requester) or 'agent' (responder). ``parts`` is the
    ordered content of the turn. ``contextId`` groups all turns/tasks of one
    logical conversation; ``taskId`` links the turn to a specific task in that
    context (see a2a.task). ``metadata`` is an opaque dict for routing/telemetry
    (tokens/cost/latency, trust-boundary tags, etc.), mirroring the opaque
    ``data`` dict convention of trace.py.
    """

    role: str
    parts: Tuple[Part, ...] = ()
    contextId: Optional[str] = None
    taskId: Optional[str] = None
    messageId: Optional[str] = None
    referenceTaskIds: Optional[Tuple[str, ...]] = None
    extensions: Optional[Tuple[str, ...]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "role": self.role,
            "parts": [p.to_dict() for p in self.parts],
            "contextId": self.contextId,
            "taskId": self.taskId,
            "messageId": self.messageId,
            "metadata": dict(self.metadata),
            "kind": "message",
        }
        if self.referenceTaskIds is not None:
            d["referenceTaskIds"] = list(self.referenceTaskIds)
        if self.extensions is not None:
            d["extensions"] = list(self.extensions)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "A2AMessage":
        ref = d.get("referenceTaskIds")
        ext = d.get("extensions")
        return cls(
            role=d["role"],
            parts=tuple(Part.from_dict(p) for p in (d.get("parts") or ())),
            contextId=d.get("contextId"),
            taskId=d.get("taskId"),
            messageId=d.get("messageId"),
            referenceTaskIds=tuple(ref) if ref is not None else None,
            extensions=tuple(ext) if ext is not None else None,
            metadata=dict(d.get("metadata") or {}),
        )

    # --- convenience ----------------------------------------------------
    def text(self) -> str:
        """Concatenate all text parts (ignores data/artifact parts)."""
        return "".join(p.text or "" for p in self.parts if p.kind == PART_TEXT)


def user_message(*parts: Part, **kw: Any) -> A2AMessage:
    """Build a 'user' (requesting-agent) turn from parts."""
    return A2AMessage(role=ROLE_USER, parts=tuple(parts), **kw)


def agent_message(*parts: Part, **kw: Any) -> A2AMessage:
    """Build an 'agent' (responding-agent) turn from parts."""
    return A2AMessage(role=ROLE_AGENT, parts=tuple(parts), **kw)
