"""Machine-checkable research-corpus audit utilities."""

from audit.closure import (
    ALLOWED_DISPOSITIONS,
    AuditIssue,
    AuditReport,
    validate_register,
)

__all__ = [
    "ALLOWED_DISPOSITIONS",
    "AuditIssue",
    "AuditReport",
    "validate_register",
]
