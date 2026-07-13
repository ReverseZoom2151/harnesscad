"""Host-neutral transactional CAD adapter contract.

The contract describes integration semantics only.  It does not claim support
for any proprietary CAD host.  Concrete connectors can implement this protocol
while tests and local workflows use :mod:`adapters.memory`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable


class Capability(str, Enum):
    READ = "read"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    TRANSACTIONS = "transactions"
    VERIFY = "verify"
    IDEMPOTENCY = "idempotency"


@dataclass(frozen=True)
class AdapterCapabilities:
    host: str
    operations: frozenset[Capability]
    formats: tuple[str, ...] = ()
    transactional: bool = True

    def supports(self, capability: Capability) -> bool:
        return capability in self.operations


@dataclass(frozen=True)
class WriteCommand:
    """A host-neutral mutation.

    ``action`` is ``create``, ``update`` or ``delete``.  ``entity_id`` is the
    stable host-side identity and ``values`` is a JSON-like property mapping.
    """

    action: str
    entity_id: str
    values: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ApplyReceipt:
    idempotency_key: str
    command_digest: str
    staged_revision: str
    replayed: bool = False


@dataclass(frozen=True)
class VerificationIssue:
    code: str
    message: str
    entity_id: Optional[str] = None


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    issues: tuple[VerificationIssue, ...] = ()
    staged_revision: str = ""


@dataclass(frozen=True)
class CommitReceipt:
    transaction_id: str
    revision: str
    applied_keys: tuple[str, ...]


class AdapterError(RuntimeError):
    pass


class TransactionStateError(AdapterError):
    pass


class CapabilityError(AdapterError):
    pass


class IdempotencyConflict(AdapterError):
    pass


class VerificationRequired(AdapterError):
    pass


@runtime_checkable
class CADAdapter(Protocol):
    def capabilities(self) -> AdapterCapabilities: ...

    def read(self, entity_id: Optional[str] = None) -> Any: ...

    def begin(self, transaction_id: Optional[str] = None) -> str: ...

    def apply(self, command: WriteCommand, *, idempotency_key: str) -> ApplyReceipt: ...

    def verify(self) -> VerificationResult: ...

    def commit(self) -> CommitReceipt: ...

    def rollback(self) -> str: ...
