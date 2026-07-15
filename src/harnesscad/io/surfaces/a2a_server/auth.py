"""Request authentication for the A2A server (stdlib-only, opt-in via env).

The A2A enterprise profile authenticates EVERY RPC out-of-band, in standard
HTTP headers, and advertises the accepted schemes on the Agent Card via
``securitySchemes`` + ``security`` (see ``card.build_security``). This module is
the verifier that sits in front of ``handler.dispatch``: it pulls a credential
from the request headers, verifies it, and either returns an ``Principal`` or
raises a typed ``AuthError`` carrying the exact HTTP status (401/403) and the
``WWW-Authenticate`` challenge the transport must echo back.

Two schemes are supported, both credential-only (no passwords ever touched):

  - Bearer / JWT — ``Authorization: Bearer <token>``. The token is a compact
    HS256 JWS (``header.payload.signature``) signed with an HMAC-SHA256 secret
    shared out-of-band. Verification is pure stdlib (``hmac``/``hashlib``/
    ``base64``/``json``): recompute the MAC over ``header.payload``, compare in
    constant time, then enforce ``exp``/``nbf`` if present.
  - API key — ``API-Key: <key>``. Compared in constant time against the set of
    keys configured out-of-band.

Secrets NEVER live in code. They come from the environment:

  - ``HARNESSCAD_A2A_AUTH`` — master switch. Falsy/unset => auth disabled so
    local dev and tests keep working; truthy (``1``/``true``/``yes``/``on``)
    => every request must authenticate. Production sets this.
  - ``HARNESSCAD_A2A_JWT_SECRET`` — the HMAC-SHA256 secret for Bearer/JWT.
  - ``HARNESSCAD_A2A_JWT_ISSUER`` — optional expected ``iss`` claim.
  - ``HARNESSCAD_A2A_JWT_AUDIENCE`` — optional expected ``aud`` claim.
  - ``HARNESSCAD_A2A_API_KEYS`` — comma-separated set of accepted API keys.

Whichever secrets are present decide which schemes are live; if auth is required
but no secret is configured, ``Authenticator`` rejects every request (fail
closed) rather than silently letting traffic through.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

# --- env var names (single source of truth) --------------------------------
ENV_AUTH_REQUIRED = "HARNESSCAD_A2A_AUTH"
ENV_JWT_SECRET = "HARNESSCAD_A2A_JWT_SECRET"
ENV_JWT_ISSUER = "HARNESSCAD_A2A_JWT_ISSUER"
ENV_JWT_AUDIENCE = "HARNESSCAD_A2A_JWT_AUDIENCE"
ENV_API_KEYS = "HARNESSCAD_A2A_API_KEYS"

# Realm advertised in the WWW-Authenticate challenge.
_REALM = "harnesscad-a2a"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


class AuthError(Exception):
    """A typed authentication/authorization failure.

    ``status`` is the HTTP status the transport must return (401 for a missing
    or bad credential, 403 for a well-formed credential that is not permitted).
    ``www_authenticate``, when set, is the value for the ``WWW-Authenticate``
    response header (required by RFC 7235 on a 401).
    """

    def __init__(
        self,
        message: str,
        status: int = 401,
        www_authenticate: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.www_authenticate = www_authenticate


@dataclass(frozen=True)
class Principal:
    """The authenticated caller: which scheme let them in and who they are."""

    scheme: str
    subject: str
    claims: Dict[str, Any] = field(default_factory=dict)


# --- base64url helpers (JWS uses unpadded base64url) ------------------------
def _b64url_decode(segment: str) -> bytes:
    """Decode an unpadded base64url segment, restoring padding first."""
    pad = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + pad)


def _b64url_encode(raw: bytes) -> str:
    """Encode bytes as unpadded base64url text."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def mint_jwt(
    secret: str,
    subject: str,
    *,
    issuer: Optional[str] = None,
    audience: Optional[str] = None,
    ttl_seconds: int = 3600,
    extra_claims: Optional[Mapping[str, Any]] = None,
    now: Optional[int] = None,
) -> str:
    """Mint a compact HS256 JWT signed with ``secret`` (HMAC-SHA256).

    Provided so callers (and ``--selfcheck``) can produce a valid token without
    a third-party library. Not used on the request path.
    """
    issued = int(time.time()) if now is None else int(now)
    header = {"alg": "HS256", "typ": "JWT"}
    payload: Dict[str, Any] = {"sub": subject, "iat": issued}
    if ttl_seconds > 0:
        payload["exp"] = issued + int(ttl_seconds)
    if issuer is not None:
        payload["iss"] = issuer
    if audience is not None:
        payload["aud"] = audience
    if extra_claims:
        payload.update(extra_claims)
    signing_input = (
        _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
        + "."
        + _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    )
    signature = hmac.new(
        secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256
    ).digest()
    return signing_input + "." + _b64url_encode(signature)


def verify_jwt(
    token: str,
    secret: str,
    *,
    issuer: Optional[str] = None,
    audience: Optional[str] = None,
    leeway_seconds: int = 60,
    now: Optional[int] = None,
) -> Dict[str, Any]:
    """Verify a compact HS256 JWT and return its claims, or raise ``AuthError``.

    Checks, in order: three dot-separated segments; ``alg == HS256`` (no
    ``none`` downgrade); constant-time HMAC-SHA256 signature match; ``exp`` not
    past and ``nbf`` not future (with ``leeway_seconds`` slack); and, when
    configured, that ``iss``/``aud`` match.
    """
    parts = token.split(".")
    if len(parts) != 3:
        raise AuthError("malformed JWT", 401, _bearer_challenge("invalid_token"))
    header_seg, payload_seg, signature_seg = parts
    try:
        header = json.loads(_b64url_decode(header_seg))
        payload = json.loads(_b64url_decode(payload_seg))
        signature = _b64url_decode(signature_seg)
    except (ValueError, json.JSONDecodeError):
        raise AuthError("undecodable JWT", 401, _bearer_challenge("invalid_token"))

    if not isinstance(header, dict) or header.get("alg") != "HS256":
        raise AuthError(
            "unsupported JWT alg", 401, _bearer_challenge("invalid_token")
        )
    if not isinstance(payload, dict):
        raise AuthError("invalid JWT payload", 401, _bearer_challenge("invalid_token"))

    signing_input = (header_seg + "." + payload_seg).encode("ascii")
    expected = hmac.new(
        secret.encode("utf-8"), signing_input, hashlib.sha256
    ).digest()
    if not hmac.compare_digest(expected, signature):
        raise AuthError(
            "bad JWT signature", 401, _bearer_challenge("invalid_token")
        )

    current = int(time.time()) if now is None else int(now)
    exp = payload.get("exp")
    if exp is not None and current > int(exp) + leeway_seconds:
        raise AuthError(
            "token expired", 401, _bearer_challenge("invalid_token", "token expired")
        )
    nbf = payload.get("nbf")
    if nbf is not None and current + leeway_seconds < int(nbf):
        raise AuthError(
            "token not yet valid", 401, _bearer_challenge("invalid_token")
        )
    if issuer is not None and payload.get("iss") != issuer:
        raise AuthError("issuer mismatch", 403)
    if audience is not None and not _audience_matches(payload.get("aud"), audience):
        raise AuthError("audience mismatch", 403)
    return payload


def _audience_matches(claim: Any, expected: str) -> bool:
    """A JWT ``aud`` claim may be a string or a list of strings."""
    if isinstance(claim, str):
        return hmac.compare_digest(claim, expected)
    if isinstance(claim, (list, tuple)):
        return any(isinstance(a, str) and hmac.compare_digest(a, expected) for a in claim)
    return False


def _bearer_challenge(error: Optional[str] = None, description: Optional[str] = None) -> str:
    """Build a ``Bearer`` ``WWW-Authenticate`` value (RFC 6750)."""
    parts = [f'realm="{_REALM}"']
    if error is not None:
        parts.append(f'error="{error}"')
    if description is not None:
        parts.append(f'error_description="{description}"')
    return "Bearer " + ", ".join(parts)


def _apikey_challenge() -> str:
    """Build the ``API-Key`` ``WWW-Authenticate`` value."""
    return f'API-Key realm="{_REALM}"'


def _parse_api_keys(raw: Optional[str]) -> Tuple[str, ...]:
    """Split a comma-separated env value into a tuple of non-empty keys."""
    if not raw:
        return ()
    return tuple(k.strip() for k in raw.split(",") if k.strip())


def _match_api_key(candidate: str, valid_keys: Iterable[str]) -> bool:
    """Constant-time membership test against the configured key set.

    Iterates every key (no early return) so timing does not leak which, or how
    many, keys matched.
    """
    matched = False
    for key in valid_keys:
        if hmac.compare_digest(candidate, key):
            matched = True
    return matched


class Authenticator:
    """Verifies A2A requests against the configured schemes.

    Construct with :meth:`from_env` in production; the constructor takes explicit
    config so it stays testable and secret-free. When ``required`` is False the
    authenticator is a no-op (``authenticate`` returns an anonymous principal),
    which keeps local dev and the existing tests running unchanged.
    """

    def __init__(
        self,
        *,
        required: bool = False,
        jwt_secret: Optional[str] = None,
        jwt_issuer: Optional[str] = None,
        jwt_audience: Optional[str] = None,
        api_keys: Iterable[str] = (),
    ) -> None:
        self.required = bool(required)
        self.jwt_secret = jwt_secret or None
        self.jwt_issuer = jwt_issuer or None
        self.jwt_audience = jwt_audience or None
        self.api_keys: Tuple[str, ...] = tuple(api_keys)

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "Authenticator":
        """Build an ``Authenticator`` from environment variables."""
        source = os.environ if env is None else env
        required = (source.get(ENV_AUTH_REQUIRED, "") or "").strip().lower() in _TRUTHY
        return cls(
            required=required,
            jwt_secret=source.get(ENV_JWT_SECRET),
            jwt_issuer=source.get(ENV_JWT_ISSUER),
            jwt_audience=source.get(ENV_JWT_AUDIENCE),
            api_keys=_parse_api_keys(source.get(ENV_API_KEYS)),
        )

    @property
    def bearer_enabled(self) -> bool:
        return self.jwt_secret is not None

    @property
    def api_key_enabled(self) -> bool:
        return len(self.api_keys) > 0

    def scheme_names(self) -> Tuple[str, ...]:
        """The names of the live schemes, for the Agent Card ``security`` block."""
        names = []
        if self.bearer_enabled:
            names.append("bearerAuth")
        if self.api_key_enabled:
            names.append("apiKeyAuth")
        return tuple(names)

    def authenticate(self, headers: Mapping[str, str]) -> Principal:
        """Authenticate one request from its headers.

        ``headers`` is any case-insensitive mapping with ``.get`` (an
        ``http.client.HTTPMessage`` qualifies). Returns a ``Principal`` on
        success; raises ``AuthError`` (401/403) on failure. When auth is not
        required, returns an anonymous principal without inspecting credentials.
        """
        if not self.required:
            return Principal(scheme="none", subject="anonymous")

        if not (self.bearer_enabled or self.api_key_enabled):
            # Required but nothing to verify against: fail closed.
            raise AuthError(
                "server misconfigured: authentication required but no scheme configured",
                status=500,
            )

        authorization = headers.get("Authorization")
        api_key = headers.get("API-Key")

        if authorization:
            return self._authenticate_bearer(authorization)
        if api_key:
            return self._authenticate_api_key(api_key)

        raise AuthError(
            "missing credentials", 401, self._primary_challenge()
        )

    # -- per-scheme ------------------------------------------------------
    def _authenticate_bearer(self, authorization: str) -> Principal:
        if not self.bearer_enabled:
            raise AuthError(
                "bearer auth not supported", 401, self._primary_challenge()
            )
        parts = authorization.split(None, 1)
        if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
            raise AuthError(
                "malformed Authorization header",
                401,
                _bearer_challenge("invalid_request"),
            )
        token = parts[1].strip()
        claims = verify_jwt(
            token,
            self.jwt_secret,  # type: ignore[arg-type]  # guarded by bearer_enabled
            issuer=self.jwt_issuer,
            audience=self.jwt_audience,
        )
        subject = str(claims.get("sub") or "unknown")
        return Principal(scheme="bearerAuth", subject=subject, claims=dict(claims))

    def _authenticate_api_key(self, api_key: str) -> Principal:
        if not self.api_key_enabled:
            raise AuthError(
                "api-key auth not supported", 401, self._primary_challenge()
            )
        if not _match_api_key(api_key.strip(), self.api_keys):
            raise AuthError("invalid API key", 401, _apikey_challenge())
        # Do not echo the key; identify by a short stable fingerprint.
        fingerprint = hashlib.sha256(api_key.strip().encode("utf-8")).hexdigest()[:12]
        return Principal(
            scheme="apiKeyAuth", subject=f"apikey:{fingerprint}"
        )

    def _primary_challenge(self) -> str:
        """The WWW-Authenticate value to offer when no credential was sent."""
        if self.bearer_enabled:
            return _bearer_challenge()
        return _apikey_challenge()
