"""Self-evaluation — the harness evaluating the HARNESS, not a model.

Everything else under ``harnesscad.eval`` points OUTWARD: ``bench`` scores a
prediction against a gold, ``verifiers`` gate an op stream, ``quality`` analyses
a model. Nothing asked whether the harness itself is correct -- and so nothing
caught that the F-rep ``shell`` op grew the part it was asked to hollow, or that
the verifier fleet rejects a washer.

This package points INWARD. Four oracles, none of which needs a model:

* :mod:`harnesscad.eval.selftest.differential` -- run one op stream on EVERY
  available backend and report where they disagree. Six independent geometry
  engines are the strongest oracle in the repo and it never used them. No ground
  truth is required: a disagreement is, by itself, a bug in at least one engine.
* :mod:`harnesscad.eval.selftest.golden` -- a corpus of parts whose volume, bbox
  and genus are known in CLOSED FORM. This is the oracle that can say which side
  of a disagreement is wrong.
* :mod:`harnesscad.eval.selftest.fleet_audit` -- precision / recall / F1 PER
  VERIFIER over a known-good and a known-bad corpus. An ERROR on a known-good
  part is a false positive; silence on a known-bad part is a false negative.
  The fleet was never held to this metric, and that is why it shipped a rule
  that rejects a washer.
* :mod:`harnesscad.eval.selftest.properties` -- metamorphic / property tests over
  a seeded random corpus: a shell must not grow a part, a cut must not add
  volume, scaling by k must scale volume by k^3, a replay must be identical.

WHAT THESE ORACLES CANNOT DO
----------------------------
The signature they compare -- volume, bbox, genus, watertightness -- is
MANY-TO-ONE. It does not pin down a part. Two parts with the same volume, the
same envelope and the same number of handles can be completely different objects:
move all four holes of a bracket to the wrong corners and every number here is
still perfect. A part that PASSES has not been proved correct; it has failed to be
proved wrong, which is a strictly weaker claim and the only one on offer.

The signature is chosen because it is cheap, engine-independent, and catches the
class of bug that actually shipped (a shell that grew a part, a wall 42% too
thin). It is a sieve, not a proof. Two consequences are load-bearing:

* a bbox check CANNOT prove a shell -- an inward shell can preserve the envelope
  exactly and still leave the wall at ``t/sqrt(3)``, so the shell parts assert the
  exact analytic VOLUME, which pins the wall;
* the tolerances in :mod:`harnesscad.eval.selftest.probe` are only as sound as the
  tessellation underneath them. A mesh-derived volume inherits its exporter's
  tolerance, so a threshold calibrated against our own tessellation error would be
  measuring the harness's meshing, not its geometry. When an exporter tolerance
  changes, the numbers in ``TOLERANCES`` must be re-derived, not re-fitted.

An oracle that overclaims is worse than one that abstains.

Public entry point: :func:`harnesscad.eval.selftest.registry.run` (the
``harnesscad selftest`` subcommand). Every oracle is importable and callable on
its own, is stdlib-only, is deterministic, and SKIPS a backend cleanly when the
tool behind it is not installed.
"""

from __future__ import annotations

__all__ = [
    "probe",
    "differential",
    "golden",
    "fleet_audit",
    "properties",
    "registry",
]
