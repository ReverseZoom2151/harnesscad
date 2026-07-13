"""Host-neutral transactional CAD adapter contracts."""

from harnesscad.io.adapters.base import (
    AdapterCapabilities,
    ApplyReceipt,
    CADAdapter,
    Capability,
    CommitReceipt,
    VerificationIssue,
    VerificationResult,
    WriteCommand,
)
from harnesscad.io.adapters.memory import MemoryCADAdapter

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
