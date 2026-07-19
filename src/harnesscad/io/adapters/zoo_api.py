"""Deterministic data model for the Zoo / KittyCAD text-to-CAD async API.

The hosted text-to-CAD REST endpoint is driven as follows:

* submit a prompt --  ``POST https://api.zoo.dev/ai/text-to-cad/{format}``
  with body ``{"prompt": ...}``, which returns an operation ``{"id", "status",
  ...}``;
* poll the async operation -- ``GET https://api.zoo.dev/async/operations/{id}``
  until ``status`` becomes ``completed`` or ``failed``;
* on success, read the file bytes from ``outputs["source.{format}"]`` (base64);
* on failure, read ``error``.

This module captures that request/response contract as a pure, offline data
model.  It builds the request descriptors (method, url, headers, json body) and
parses response dictionaries into structured objects.  It performs NO network
I/O, no sleeping, and no Blender interaction -- those (and the actual HTTP
transport / retry loop) are external glue supplied by the caller.  What this
module captures is the deterministic shape of the protocol: endpoint templating,
supported output formats, the ``source.{format}`` output key, and the terminal
status semantics of the async operation.

stdlib-only, deterministic, no wall clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Optional

__all__ = [
    "OutputFormat",
    "OperationStatus",
    "ZooApiError",
    "DEFAULT_BASE_URL",
    "SUPPORTED_FORMATS",
    "output_key",
    "build_submit_request",
    "build_poll_request",
    "parse_operation",
    "Operation",
    "HttpRequest",
]

DEFAULT_BASE_URL = "https://api.zoo.dev"

# The User-Agent header is required by the API; without it Zoo returns HTTP 403.
_DEFAULT_USER_AGENT = "Mozilla/5.0"


class OutputFormat(str, Enum):
    """CAD/mesh formats the text-to-CAD endpoint can emit."""

    fbx = "fbx"
    glb = "glb"
    gltf = "gltf"
    obj = "obj"
    ply = "ply"
    stl = "stl"

    @classmethod
    def coerce(cls, value: "OutputFormat | str") -> "OutputFormat":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError as exc:
            raise ZooApiError("unsupported output format: %r" % (value,)) from exc


SUPPORTED_FORMATS = tuple(f.value for f in OutputFormat)


class OperationStatus(str, Enum):
    """Lifecycle states of a Zoo async operation.

    The addon only distinguishes the two terminal states, but the async API
    surfaces intermediate ones; modelling them lets a poller decide when to
    keep waiting.
    """

    queued = "queued"
    uploaded = "uploaded"
    in_progress = "in_progress"
    completed = "completed"
    failed = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (OperationStatus.completed, OperationStatus.failed)

    @classmethod
    def coerce(cls, value: object) -> "OperationStatus":
        try:
            return cls(str(value))
        except ValueError as exc:
            raise ZooApiError("unknown operation status: %r" % (value,)) from exc


class ZooApiError(ValueError):
    """Raised for malformed requests or unparseable API responses."""


def output_key(output_format: "OutputFormat | str") -> str:
    """Return the key under which the result payload is stored.

    Zoo nests the generated file under ``outputs["source.{format}"]``.
    """
    fmt = OutputFormat.coerce(output_format)
    return "source.%s" % fmt.value


@dataclass(frozen=True)
class HttpRequest:
    """A transport-agnostic description of an HTTP call.

    It is deliberately inert: something else must actually perform it.
    """

    method: str
    url: str
    headers: Mapping[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None


def _auth_header(api_token: str) -> str:
    if not api_token:
        raise ZooApiError("api_token must be a non-empty string")
    return "Bearer %s" % api_token


def build_submit_request(
    prompt: str,
    output_format: "OutputFormat | str",
    api_token: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    user_agent: str = _DEFAULT_USER_AGENT,
) -> HttpRequest:
    """Build the POST request that submits a text-to-CAD prompt.

    The JSON body is serialised deterministically (sorted keys) so the same
    inputs always produce byte-identical output, which keeps the descriptor
    hashable/comparable in tests.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ZooApiError("prompt must be a non-empty string")
    fmt = OutputFormat.coerce(output_format)
    url = "%s/ai/text-to-cad/%s" % (base_url.rstrip("/"), fmt.value)
    # Compact, sorted JSON without importing json at call sites that only need
    # the descriptor shape; json keeps escaping correct for arbitrary prompts.
    import json

    body = json.dumps({"prompt": prompt}, sort_keys=True).encode("utf-8")
    headers = {
        "Authorization": _auth_header(api_token),
        "Content-Type": "application/json",
        "User-Agent": user_agent,
    }
    return HttpRequest("POST", url, headers, body)


def build_poll_request(
    operation_id: str,
    api_token: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    user_agent: str = _DEFAULT_USER_AGENT,
) -> HttpRequest:
    """Build the GET request that polls an async operation by id."""
    if not isinstance(operation_id, str) or not operation_id:
        raise ZooApiError("operation_id must be a non-empty string")
    url = "%s/async/operations/%s" % (base_url.rstrip("/"), operation_id)
    headers = {
        "Authorization": _auth_header(api_token),
        "User-Agent": user_agent,
    }
    return HttpRequest("GET", url, headers, None)


@dataclass(frozen=True)
class Operation:
    """Parsed view of a text-to-CAD operation response."""

    id: str
    status: OperationStatus
    outputs: Mapping[str, str] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    @property
    def is_completed(self) -> bool:
        return self.status is OperationStatus.completed

    @property
    def is_failed(self) -> bool:
        return self.status is OperationStatus.failed

    def payload_for(self, output_format: "OutputFormat | str") -> str:
        """Return the raw (still base64) payload string for *output_format*.

        Raises :class:`ZooApiError` if the operation is not completed or the
        expected output key is missing.
        """
        if not self.is_completed:
            raise ZooApiError(
                "operation %s is not completed (status=%s)"
                % (self.id, self.status.value)
            )
        key = output_key(output_format)
        try:
            return self.outputs[key]
        except KeyError as exc:
            raise ZooApiError("missing output %r in operation %s" % (key, self.id)) from exc


def parse_operation(response: Mapping[str, object]) -> Operation:
    """Parse a submit/poll response dict into an :class:`Operation`.

    Only ``id`` and ``status`` are required; ``outputs``/``error`` default to
    empty/None, matching the intermediate-state responses.
    """
    if "id" not in response:
        raise ZooApiError("response missing 'id'")
    if "status" not in response:
        raise ZooApiError("response missing 'status'")
    outputs = response.get("outputs") or {}
    if not isinstance(outputs, Mapping):
        raise ZooApiError("'outputs' must be an object")
    error = response.get("error")
    return Operation(
        id=str(response["id"]),
        status=OperationStatus.coerce(response["status"]),
        outputs=dict(outputs),
        error=None if error is None else str(error),
    )
