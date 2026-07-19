---
name: cad-pdd
description: Run the Parts-Driven Development workflow -- certify a part against a Measured Geometric Contract with harnesscad pdd. Use when the user wants a part certified rather than merely built, asks about PDD, MGC, measured contracts or the certification gates, when pdd reports UNCERTIFIED or unbound predicates or [NEEDS CLARIFICATION], or says "prove this part is right", "check it against the spec", "certify this".
---

# Parts-Driven Development

PDD is this project's answer to a limit in spec-driven development: a passing
spec test proves the code matches the spec, never that the spec was right,
because prose specs are checked by more prose. CAD is the exception. A part's
acceptance criteria are measurable quantities of the artifact -- volume,
bounding box, genus, hole count and position, minimum wall, mass, centre of
mass, mobility. So you do not trust the generator; you measure its output and
refuse on mismatch. That is the rung PDD claims: parts as measured source.

## The Measured Geometric Contract

The MGC is the per-part predicate set with tolerances. Its defining rule is
that it states measurable outcomes and never ops -- no HOW in the WHAT. That
restraint is what makes the contract satisfiable by OCCT, Manifold, truck or
frep alike, which is in turn what lets independent kernels cross-check the
same claim. A contract that named ops would only be checkable by the engine
that ran them.

A measurable the brief does not state becomes an unbound predicate and is
reported as `[NEEDS CLARIFICATION]`, not filled with a guessed millimetre.

## Running it

```bash
harnesscad pdd "<brief>" --ops ops.json --backend frep
```

`--ops` is required and takes the op stream -- the plan the model built. The
pipeline never generates it; PDD's separation is that planning and measuring
are different jobs. `brief` is a prose string or a path to a file holding
one. Other flags: `--measurement <json>` supplies contract-keyed measurements
(`volume_mm3`, `bbox_mm`, `genus`, ...) to check against, and when omitted a
best-effort mapping is adapted from the output gate's own measurement;
`--part-id` stamps an id on a contract that has none; `--json` emits the
verdict as JSON. `--backend` defaults to `stub`, so name a real one.

Exit code is 0 only for `PASS`. `FAIL` and `UNCERTIFIED` both exit 1.

## Reading the verdict

```
verdict:  UNCERTIFIED
part:     part
digest:   c26ea1b1d8ca1f1dbfa37fdfb1cb23b219f21e59d6af4ff4f8d964621c877f30
reasons:
  - contract has unbound MEASURED predicate(s): bbox_mm, volume_mm3, genus, ...
  - contract not satisfied (no bound MEASURED predicate passed)
  - differential oracle abstained (fewer than two independent engines)
[NEEDS CLARIFICATION]: bbox_mm, volume_mm3, hole_count, genus, ...
```

Three distinct things are being said and they need distinct responses:

- **unbound predicates** -- the brief did not state these quantities, so the
  contract has nothing to check. Fix by putting the numbers in the brief or
  supplying `--measurement`. Do not fix by removing the predicates.
- **contract not satisfied** -- bound predicates exist and did not pass. This
  is a real geometry failure; go to the `cad-repair` skill.
- **oracle abstained** -- fewer than two independent engines are installed,
  so cross-checking did not happen. An environment gap, not a part defect.

`UNCERTIFIED` is its own verdict. It is not a soft pass and not a fail: the
harness could not certify the part. Reporting it as either is the specific
misreading this workflow is designed to prevent.

## The certification bar

A part is certified only when all four hold: the gate did not refuse it,
independent kernels agree, every measured predicate passes, and every op is
attributed to a real geometry change. Anything less is `UNCERTIFIED`, and the
reasons list says which of the four is missing.

The three gates the project holds itself to are related but broader: an
orphan-provenance gate (every op must move real geometry), a verifier-fleet
mutation score (inject known defects, require the oracle kills them), and an
op x backend x format coverage census with a drift pause-gate. Those are run
through `selftest`, not through `pdd`.

## The honest residual

`pdd` prints this with every verdict, and it should be relayed rather than
trimmed: a PASS means the part passes every measured, bound predicate of its
contract, not that it matches the designer's intent. The oracle is
many-to-one -- volume, bounding box and genus do not pin down a part, and a
feature at the wrong coordinate can change no measured quantity. PDD narrows
the space of acceptable parts; it does not close it. Contract completeness is
the new spec-quality problem: garbage MGC in, faithfully wrong part out.

Which means a PASS is a claim about the contract as much as about the part.
When you report one, report what was in the contract.

## Avoid

- Do not remove, widen or unbind a predicate to turn UNCERTIFIED into PASS.
  That produces a contract that certifies anything, and the certificate is
  the deliverable here.
- Do not supply `--measurement` values taken from the part you just built as
  though they were requirements. Measurements must come from the brief; a
  contract fitted to the output is a tautology, and it will pass for any part.
- Do not fill an unbound predicate with a plausible number to clear
  `[NEEDS CLARIFICATION]`. Ask.
- Do not report PASS without saying how many independent engines agreed. One
  engine passing is not the differential result the verdict format implies.
- Do not describe the three gates or the four certification conditions as met
  when you have only run `pdd`. `pdd` checks the contract; the gates are
  `selftest`'s job.
