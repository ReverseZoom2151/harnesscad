"""Explicit supported-feature coverage and OOD approximation policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpressivityReport:
    supported: tuple[str, ...]
    observed: tuple[str, ...]
    unsupported: tuple[str, ...]
    policy: str

    @property
    def reconstructable(self):
        return not self.unsupported


def expressivity_report(observed, supported, *, policy="reject"):
    if policy not in {"reject", "approximate", "external"}:
        raise ValueError("unknown policy")
    observed, supported = set(observed), set(supported)
    return ExpressivityReport(tuple(sorted(supported)), tuple(sorted(observed)),
                              tuple(sorted(observed-supported)), policy)
