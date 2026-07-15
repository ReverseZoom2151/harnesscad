# Parts-Driven Development (PDD): a spec-driven variant for CAD generation

We read the spec-driven development (SDD) paper (Piskala, *From Code to Contract in the
Age of AI Coding Assistants*, arXiv 2602.00180), GitHub's **spec-kit** toolkit
(constitution -> specify -> clarify -> plan -> tasks -> implement -> validate, templates
that constrain the LLM), and the IBM / GitHub / Fowler / Microsoft / Wikipedia writeups.
This is the CAD-generation variant they point to, named **Parts-Driven Development (PDD)**.

> Naming note: the obvious "Component-Driven Development (CDD)" is already taken -- it's
> the established Storybook methodology for building UIs bottom-up from isolated
> components. We use **Parts-Driven Development** instead: a *part* is the atomic unit of
> mechanical CAD, and it maps one-to-one onto our core primitive, the **Measured
> Geometric Contract** (one part = one measured contract; assemblies compose parts through
> mate/DOF contracts).

Short version: **software SDD stops at spec-anchored because its validation is fallible.
CAD is the one domain where the validate phase can be exact, so CAD is the one domain
where spec-as-source is actually reachable for LLM generation. HarnessCAD already built
the exact machinery SDD's validate phase wishes it had. PDD is the name for the pipeline
that already runs, plus the one new rung it unlocks.**

## The thing SDD cannot fix, and CAD can

Every SDD source names the same fatal soft spot:

- The paper, Pitfall "False confidence": *"A passing spec test doesn't guarantee correct
  software -- it only guarantees that the software matches the spec. If the spec is
  wrong, the code will faithfully implement the wrong thing."*
- Fowler's tool review: agents *"frequently ignore detailed specifications or
  hallucinate despite elaborate prompts"*; *"I'd rather review code than all these
  markdown files."*
- This is why the paper's decision tree lands almost everyone at **spec-anchored**, and
  reserves **spec-as-source** for Simulink / certified code-gen -- domains where
  *"trust in generation quality ... has been established."* General software can't get
  there: the validation (tests) is authored by the same fallible process it's meant to
  check, so a green suite is only ever evidence, never proof.

CAD breaks this because the acceptance criteria of a **part** are **measurable quantities
of the produced artifact**, not prose about behavior:

| Software spec (prose, fallibly tested) | Part contract (measured, exactly checked) |
|---|---|
| "user can reset password" | volume = 6000.0 mm3 +/- 1e-3 |
| "handles 1000 concurrent users" | genus = 1 (exactly one through-hole) |
| "photos resized to 1024px" | min wall thickness >= 2.0 mm |
| verified by a test someone wrote | verified by re-reading + re-measuring the file |

The right column needs no human, no fallibly-authored test, and no trust in the
generator -- the check is a measurement of the output. That is the whole game, and it is
why the unit of the methodology is the *part*, not the feature.

## PDD's four phases -- three map cleanly, the fourth is our differentiator

SDD's workflow is Specify -> Plan -> Implement -> Validate (Fig. 2 of the paper). The
harness already has all four:

- **Specify** -> the **Measured Geometric Contract (MGC)** for a part. `domain/spec/`
  already parses part briefs (`part_brief_parser.py`, `design_brief.py`). PDD formalizes
  that parse's output as a machine-checkable predicate set: target volume/bbox/genus,
  hole count + positions, min wall, mass + centre of mass, interference-free, assembly
  mobility/DOF, printability verdict -- each with a tolerance. This is spec-kit's
  Functional Requirements + Success Criteria, except every SC is a *measured quantity with
  an epsilon* rather than a sentence.
- **Plan** -> **CISP**. The typed, content-digested op stream (sketch -> extrude ->
  feature, 22 ops) plus the chosen engine *is* the technical plan -- "how to build it."
  CISP is already the plan-as-contract; spec-kit writes `plan.md`, we emit a CISP program.
- **Implement** -> op execution by any model or engine. Interchangeable by construction
  (the CISP program is the interface); this is where the LLM/policy lives.
- **Validate** -> the **measured output gate (`io/gate.py`) + the differential oracle**.
  A written file is re-read and re-measured or *refused* -- the "third outcome"
  (wrote-it-anyway) never occurs. Multiple independent engines (OCCT via
  cadquery/freecad/build123d, Manifold, truck, frep) cross-check: a disagreement proves a
  bug **with no ground truth**. Where software SDD validates with fallible tests, PDD
  validates by measurement + independent replication.

So the coinage is not aspirational. It names a pipeline that already runs.

## The one new rung: parts-as-measured-source

We add a rung to the paper's rigor spectrum (Fig. 1):

```
Code-First -> Spec-First -> Spec-Anchored -> Spec-as-Source -> PARTS-as-MEASURED-Source
                                              (trust-based)     (measurement-based)
```

- **Spec-as-source** (Tessl / Simulink): humans edit only the spec; code is regenerated;
  drift is eliminated *by construction* -- but adoption *"requires high trust in
  generation quality."* You believe the generator.
- **Parts-as-measured-source** (PDD): humans edit only the MGC; the part is regenerated
  from CISP; and every regeneration is **gated by measurement of the artifact against the
  contract**. You don't trust the generator -- you measure its output and refuse on
  mismatch. This is the Simulink tier *without* needing a certified generator, because the
  certificate is computed from the result, not assumed of the process.

That rung is the contribution. It is reachable in CAD and (per the paper's own argument)
unreachable in general software.

## Mapping spec-kit's machinery onto the harness

spec-kit's power is its **constitution** (immutable principles as phase gates) and its
templates (which constrain the LLM: `[NEEDS CLARIFICATION]` markers, "no HOW in the WHAT",
test-first ordering, simplicity/anti-abstraction gates). Each has a CAD analogue the
harness already enforces:

- **The constitution** = the harness's standing invariants, already live:
  1. *Measured-gate*: no file leaves unmeasured (`io/gate.py`).
  2. *Refuse-with-taint*: an unsupported op taints the session; queries return
     volume=None, never leaked pre-op geometry.
  3. *Soundness tiers*: only PROVEN + MEASURED diagnostics reach the model; HEURISTIC
     never instructs. (severity _|_ soundness -- how-bad vs how-likely are orthogonal.)
  4. *Independent replication*: >= 2 genuinely independent engines or the oracle abstains.
  These are our articles. They are immutable and they are gates, exactly as spec-kit
  intends -- but ours are checked by code, not by an LLM self-reviewing a checklist.

- **`[NEEDS CLARIFICATION]` markers** = the anti-guess rule we *already* deleted a bug to
  enforce. When a brief underspecifies a measurable ("a bracket", no dimensions), the MGC
  must **mark the quantity as unbound**, never let the model infer a magnitude. This is
  precisely the coordinate-space-guessed-from-magnitude anti-pattern we removed
  (`coords.py`/`coordinate.py` now *declare* the space). spec-kit forbids the LLM from
  guessing auth methods; PDD forbids it from guessing millimetres. Same discipline, learned
  the hard way before we had the name.

- **"No HOW in the WHAT"** = the MGC states measurable outcomes (volume, genus, wall),
  never ops. The ops live in the CISP plan. This keeps the contract stable across engines
  -- the same MGC is satisfiable by OCCT, Manifold, truck, or frep, which is exactly why
  the differential oracle can cross-check it.

- **Test-first / contracts-before-code** = the MGC is authored (or extracted from the
  brief) *before* the op stream, and becomes the oracle's target. A part is "done" only
  when every contract predicate measures true -- spec-kit's "a feature is done when its
  scenarios pass," with measurement instead of Gherkin.

## Assemblies: where "parts" earns the name over "spec"

The unit choice pays off at assembly time. An assembly's contract is *compositional*:

- each **part** carries its own MGC (volume, walls, genus...);
- each **mate** carries a **DOF contract** -- and this is now enforceable end-to-end
  because of the joint-taxonomy unification (`domain/geometry/assembly/joint_taxonomy.py`):
  a `Mate(kind=...)` removes a known number of DOF or is *refused*, and the assembly's
  measured mobility must equal the contracted mobility;
- the whole assembly carries an **interference contract** (`domain/assembly/interference.py`):
  measured part-part overlap must be zero (or exactly the contracted press-fit volume).

"Feature-driven" doesn't capture this; "parts-driven" does -- the methodology composes
measured parts into measured assemblies, each level gated by measurement.

## Why this is more than branding (the paper's own challenge)

The paper quotes Finster: *"SDD is not a revolution... it's just BDD with branding."* For
software, fair. For CAD the branding earns its keep because it changes what tier is
reachable:

1. **BDD's Given/When/Then is prose executed by a fallible step-definition.** The MGC's
   "Then volume = 6000.0 +/- 1e-3" is executed by a mesh/BRep measurement. One is a test
   someone wrote; the other is a property of the artifact.
2. **The validate phase closes the loop with no human and no ground truth.** The
   differential oracle turns "does it meet spec?" into "do independent kernels agree the
   measured quantity equals the contracted one?" -- catching even bugs the *contract
   author* didn't foresee (a disagreement is a bug with no oracle). Software SDD's
   false-confidence pitfall is partially defeated, not just renamed.
3. **It yields verified training data for free** (ties to `cua_synthesis.md`): an MGC +
   its gate verdict is a labelled example with p=1.0 by construction. The contract becomes
   not just the source of the part but the source of the reward signal.

## The honest residual (do not oversell -- same discipline as cua_synthesis.md)

- **The MGC is necessary, not sufficient.** volume + bbox + genus + walls do **not** pin a
  unique part (the oracle is many-to-one). A part passing every contracted predicate is a
  part that passes them -- not a proof it matches the designer's intent. The MGC narrows
  the space; the shape metric narrows it further; neither closes it. This is the CAD form
  of the paper's false-confidence pitfall, and it survives. Say so on every report.
- **Contract completeness is the new spec-quality problem.** Garbage MGC in, faithfully-
  wrong part out -- the paper's warning holds. Mitigation is the same: the MGC gets the
  same review as code, and `[NEEDS CLARIFICATION]`-style unbound markers make
  under-specification loud instead of silent.
- **Parts-as-measured-source only applies where we can measure.** Aesthetic/ergonomic
  requirements ("looks premium", "comfortable grip") have no gate; those stay spec-first at
  best. Be explicit about which contract predicates are measured vs advisory -- exactly the
  PROVEN/MEASURED/HEURISTIC split we already ship.

## One-line thesis

Software spec-driven development is stuck at spec-anchored because its validation is
fallible prose-testing; the paper concedes spec-as-source needs trust nobody has for
general code. CAD is the exception: a part's acceptance criteria are measurable quantities
of the artifact, and HarnessCAD already gates every output on measuring them against the
contract and cross-checking independent kernels. **Parts-Driven Development** reaches a
rung software can't -- **parts-as-measured-source** -- where you don't trust the
generator, you measure it and refuse on mismatch. The contract isn't the new source code;
the *measured* contract is.
