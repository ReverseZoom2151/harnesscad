"""Deterministic in-memory implementation of the CAD adapter contract."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from typing import Any, Callable, Mapping, Optional

from harnesscad.io.adapters.base import (
    AdapterCapabilities,
    ApplyReceipt,
    Capability,
    CapabilityError,
    CommitReceipt,
    IdempotencyConflict,
    TransactionStateError,
    VerificationIssue,
    VerificationRequired,
    VerificationResult,
    WriteCommand,
)


Validator = Callable[[Mapping[str, Mapping[str, Any]]], list[VerificationIssue]]


class MemoryCADAdapter:
    """A real transactional fake for integration tests and offline workflows."""

    def __init__(
        self,
        entities: Optional[Mapping[str, Mapping[str, Any]]] = None,
        *,
        validator: Optional[Validator] = None,
    ) -> None:
        self._entities = copy.deepcopy(dict(entities or {}))
        self._validator = validator or _default_validator
        self._staged: Optional[dict[str, dict[str, Any]]] = None
        self._transaction_id: Optional[str] = None
        self._counter = 0
        self._verified_revision: Optional[str] = None
        self._pending: dict[str, tuple[str, ApplyReceipt]] = {}
        self._committed: dict[str, tuple[str, ApplyReceipt]] = {}

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            host="memory",
            operations=frozenset(Capability),
            formats=("json",),
        )

    def read(self, entity_id: Optional[str] = None) -> Any:
        source = self._staged if self._staged is not None else self._entities
        if entity_id is None:
            return copy.deepcopy(source)
        if entity_id not in source:
            raise KeyError(entity_id)
        return copy.deepcopy(source[entity_id])

    def begin(self, transaction_id: Optional[str] = None) -> str:
        if self._staged is not None:
            raise TransactionStateError("a transaction is already active")
        self._counter += 1
        self._transaction_id = transaction_id or f"tx-{self._counter}"
        if not self._transaction_id.strip():
            raise ValueError("transaction_id must not be empty")
        self._staged = copy.deepcopy(self._entities)
        self._pending = {}
        self._verified_revision = None
        return self._transaction_id

    def apply(self, command: WriteCommand, *, idempotency_key: str) -> ApplyReceipt:
        if self._staged is None:
            raise TransactionStateError("begin a transaction before apply")
        if not idempotency_key or not idempotency_key.strip():
            raise ValueError("idempotency_key must not be empty")
        digest = _digest(asdict(command))

        prior = self._pending.get(idempotency_key) or self._committed.get(idempotency_key)
        if prior is not None:
            prior_digest, receipt = prior
            if prior_digest != digest:
                raise IdempotencyConflict(
                    f"idempotency key {idempotency_key!r} was used for another command"
                )
            return ApplyReceipt(
                receipt.idempotency_key,
                receipt.command_digest,
                receipt.staged_revision,
                replayed=True,
            )

        self._mutate(command)
        self._verified_revision = None
        receipt = ApplyReceipt(idempotency_key, digest, _revision(self._staged))
        self._pending[idempotency_key] = (digest, receipt)
        return receipt

    def verify(self) -> VerificationResult:
        if self._staged is None:
            raise TransactionStateError("no active transaction")
        revision = _revision(self._staged)
        issues = tuple(self._validator(copy.deepcopy(self._staged)))
        if not issues:
            self._verified_revision = revision
        else:
            self._verified_revision = None
        return VerificationResult(not issues, issues, revision)

    def commit(self) -> CommitReceipt:
        if self._staged is None or self._transaction_id is None:
            raise TransactionStateError("no active transaction")
        revision = _revision(self._staged)
        if self._verified_revision != revision:
            raise VerificationRequired("verify the current staged revision before commit")
        txid = self._transaction_id
        keys = tuple(self._pending)
        self._entities = self._staged
        self._committed.update(self._pending)
        self._clear_transaction()
        return CommitReceipt(txid, revision, keys)

    def rollback(self) -> str:
        if self._staged is None or self._transaction_id is None:
            raise TransactionStateError("no active transaction")
        txid = self._transaction_id
        self._clear_transaction()
        return txid

    def revision(self) -> str:
        return _revision(self._entities)

    def _mutate(self, command: WriteCommand) -> None:
        assert self._staged is not None
        action = command.action.lower()
        entity_id = command.entity_id
        if not entity_id:
            raise ValueError("entity_id must not be empty")
        if action == "create":
            if entity_id in self._staged:
                raise ValueError(f"entity {entity_id!r} already exists")
            self._staged[entity_id] = copy.deepcopy(dict(command.values))
        elif action == "update":
            if entity_id not in self._staged:
                raise KeyError(entity_id)
            self._staged[entity_id].update(copy.deepcopy(dict(command.values)))
        elif action == "delete":
            if entity_id not in self._staged:
                raise KeyError(entity_id)
            del self._staged[entity_id]
        else:
            raise CapabilityError(f"unsupported write action {command.action!r}")

    def _clear_transaction(self) -> None:
        self._staged = None
        self._transaction_id = None
        self._pending = {}
        self._verified_revision = None


def _default_validator(
    entities: Mapping[str, Mapping[str, Any]],
) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    for entity_id, values in sorted(entities.items()):
        if not isinstance(values, Mapping):
            issues.append(VerificationIssue(
                "invalid-entity", "entity properties must be a mapping", entity_id
            ))
        elif values.get("valid") is False:
            issues.append(VerificationIssue(
                "host-invalid", "entity is explicitly marked invalid", entity_id
            ))
    return issues


def _digest(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _revision(entities: Mapping[str, Mapping[str, Any]]) -> str:
    return _digest(entities)
