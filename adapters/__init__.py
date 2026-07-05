"""Host-neutral transactional CAD adapter contracts."""

from adapters.base import (
    AdapterCapabilities,
    ApplyReceipt,
    CADAdapter,
    Capability,
    CommitReceipt,
    VerificationIssue,
    VerificationResult,
    WriteCommand,
)
from adapters.memory import MemoryCADAdapter

__all__ = [
    "AdapterCapabilities",
    "ApplyReceipt",
    "CADAdapter",
    "Capability",
    "CommitReceipt",
    "VerificationIssue",
    "VerificationResult",
    "WriteCommand",
    "MemoryCADAdapter",
]
