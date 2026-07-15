"""Versioned standards knowledge-base for HarnessCAD.

This package turns *written* engineering standards into *machine-readable,
config-driven* rule records, keeps them in a versioned "living codebook", and
reasons over them:

  * :mod:`standards.registry` — the typed :class:`Rule` record, a versioned
    :class:`RulePack` (JSON / tiny-YAML load & save), and the
    :class:`StandardsRegistry` that indexes packs by ``(standard, version)``,
    resolves the *active* rule set for a given material/process/region, lists
    the versions of a standard, and diffs two versions so you can flag when a
    regulation changed.
  * :mod:`standards.ingest` — :func:`ingest_standard` translates clause text
    into typed :class:`Rule` records, either with an injected LLM (structured
    extraction against :func:`rule_schema`) or, with no network, a deterministic
    heuristic parser. Every rule cites the exact clause id.
  * :mod:`standards.conflict` — :func:`detect_conflicts` finds mutually
    contradictory active rules on the same parameter + scope.

Two standards-derived accounting surfaces over a *finished* design --
:mod:`standards.embodied_carbon` (embodied-CO2e tally over a bill of materials)
and :mod:`standards.evidence_bundle` (cited-provenance roll-up over a spec) --
are dispatched by :mod:`standards.accounting`; they are deliberately not
re-exported here because a re-exported ``embodied_carbon`` function would shadow
the submodule of the same name.

Deliberately decoupled from ``verifiers/``: this package *produces* rule records
that a verifier can later *consume*; it never imports the hardcoded rule engines.
Stdlib only, deterministic, no third-party YAML dependency.
"""

from __future__ import annotations

from harnesscad.domain.standards.registry import (
    Rule,
    RulePack,
    StandardsRegistry,
    VersionDiff,
    parse_simple_yaml,
)
from harnesscad.domain.standards.ingest import ingest_standard, rule_schema
from harnesscad.domain.standards.conflict import Conflict, detect_conflicts

__all__ = [
    "Rule",
    "RulePack",
    "StandardsRegistry",
    "VersionDiff",
    "parse_simple_yaml",
    "ingest_standard",
    "rule_schema",
    "Conflict",
    "detect_conflicts",
]
