"""An adversarial auditor of the verifier fleet. It hunts FALSE POSITIVES.

THE JOB
-------
Find op streams that build a CORRECT part -- proven by arithmetic, corroborated by
an engine -- and that some verifier REJECTS anyway. Every hit is a bug in the
fleet, of exactly the kind that cost the pressure experiment eight briefs: a hole
rule that compared a diameter against a plate's THICKNESS fired forty times,
rejected an ordinary washer, caused every regression the harness suffered, and had
a passing unit test the whole time.

The four fleet bugs in ``assets/pressure/report.md`` were found by a human doing
this by hand, once. That is not a process; it is a person, and people get bored.
The book audit endorses exactly one multi-agent pattern for this repository -- a
RED TEAM -- and this is it.

WHERE IT LOOKS
--------------
Not at random. A false positive lives at a RULE'S BOUNDARY, because a boundary is
where an off-by-one lives, and the boundaries are all written down in the rules
themselves (``verifiers/kernel_preflight.py``, ``verifiers/precheck.py``):

  * a fillet at exactly half the smallest extent, and one step below it;
  * a shell at exactly half the smallest extent, and one step below it;
  * a shell at exactly ``min_wall``;
  * a hole exactly tangent to the stock edge, and one step inside it;
  * a hole wider than the plate is thick (the washer -- the bug that shipped);
  * a bore wider than the part is tall (a bearing housing);
  * and the regions no brief covers at all: revolves, patterns, chamfers,
    multi-body booleans, extreme aspect ratios.

``preflight-RADIUS_TOO_LARGE`` fired at r = 3.1 on a 6 mm plate and stayed SILENT
at r = 3.0, which is the true degenerate limit. An off-by-one is invisible to a
corpus of round numbers and obvious to a search that walks up to the boundary and
takes one step either side. That is what ``attacks.py`` does.

THE DISCIPLINE
--------------
**A false positive is REPORTED, NOT FIXED.** ``eval/verifiers/`` belongs to
somebody else, and repairing the thing under test to improve its score is the
definition of a rigged result. The only reason the pressure experiment's negative
result is worth anything is that it was published before anything was fixed.

**A part is only called "actually fine" when it can be PROVEN fine** (see
``oracle.py``): a closed-form volume the part must have, plus an engine that
builds it, watertight, at that volume. When the oracle cannot certify, the attack
is dropped and counted as uncertified -- never silently promoted into a finding.
An adversarial auditor that inflates its own hit count is worth less than none.
"""

from __future__ import annotations

__all__ = ["attacks", "oracle", "run"]
