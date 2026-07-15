"""Release gates — the capabilities the build is not allowed to regress past.

`eval/selftest/` MEASURES the harness (precision per verifier, engine
disagreement, metamorphic laws, field liveness). Nothing enforced any of it: the
oracles were a CLI subcommand, and a rule with precision 0.4 could be merged the
next day. That is exactly the bug that lost `assets/pressure/report.md`.

This package is the enforcement half. A gate:
  * reads a COMMITTED baseline (checked into the repo, reviewed like code),
  * re-measures the live system with the selftest oracles,
  * and exits non-zero when the live system is worse than the baseline.

The rule the whole repository turns on:

    A capability that is not wired to the loop is not a capability, and a
    capability the loop is not gated on is a decoration.
"""

from __future__ import annotations

__all__ = ["precision_floor", "judge_gate", "liveness_floor",
           "heldout_isolation", "warning_channel"]
