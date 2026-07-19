---
name: cad-gate-verdicts
description: Read a harnesscad gate verdict honestly -- apply's ok/rejected/diagnostics block, pdd's PASS/FAIL/UNCERTIFIED, and selftest's findings. Use when the harness has just printed a verdict and you must decide whether the part is good, when a check reports skipped or unmeasurable, when tempted to pass --force or --strict to make output look clean, or when the user asks "did it pass", "is this part valid", "what does UNCERTIFIED mean", "why did it refuse".
---

# Reading a gate verdict

The whole point of this harness is that it measures rather than trusts. That
only pays off if the verdict is read as written. Most of the ways to get a
clean-looking result out of these commands do not involve building a better
part, so the discipline here is about what a verdict does *not* say.

## The three verdict surfaces

**`apply`** builds an op stream and prints a block:

```
ok:       True
applied:  7
digest:   10cdda8514ee871ac39e1f67a237c4694c744e33e6db6148127eb49a39588d7e
diagnostics:
  [info] under-constrained: sketch sk1 is under-constrained (dof=4): ...
```

`ok` is whether every op applied. `applied` is how many did. `digest` is the
content hash of the resulting op program -- two runs printing the same digest
built the same thing, which is how you check a change did anything. When
`ok` is False a `rejected:` line names the exact op that stopped it. Exit
code is 0 on `ok: True`, 1 otherwise.

Add `--verify core` for the three core checks, or `--verify full` to run the
whole discovered verifier fleet over the plan. Note the flag's own wording:
full is *advisory -- reported, not fatal*. A full run that prints problems
still exits 0 if the ops applied. Read the diagnostics.

**`pdd`** prints a certification verdict, one of `PASS`, `FAIL`,
`UNCERTIFIED`, with the reasons that produced it. Exit code is 0 only for
`PASS`; both `FAIL` and `UNCERTIFIED` exit 1. See the `cad-pdd` skill for the
workflow; the reading rule is in this one.

**`selftest`** runs the oracles (`--differential`, `--golden`, `--fleet`,
`--properties`, `--field-liveness`, or `--all`). Its exit code is 0 *by
design even when it finds something* -- the flag help says a finding is this
command working. `--strict` makes findings exit non-zero. So the exit code of
a default `selftest` run tells you nothing about whether the harness is
healthy; only the report does.

## Diagnostic severities

Diagnostics are prefixed `[info]`, `[warn]` or `[error]` and carry a named
code (`bad-value`, `under-constrained`, `dfm-not-yet-measurable`) plus a
`@where` locator. The severity is the harness's own judgement of whether the
built artifact is defective:

- `[error]` -- the op was refused; nothing after it ran.
- `[warn]` -- built, but the harness believes it is wrong.
- `[info]` -- built, and the note is about what was *not* determined.

`[info]` is the one that gets misread. `under-constrained` is an info because
the built geometry is well-defined; it is still a real property of the model.

## Skipped is not passed

This is the failure mode that matters most here. A `--verify full` run over a
backend with no assembly or thickness query prints lines like:

```
[info] access-skipped: tool-access check skipped: backend exposes no 'access' query
[info] assembly-skipped: assembly DOF/mate checks skipped: backend has no 'assembly' query
[info] dfm-not-yet-measurable: thin-wall: true wall thickness not measured
[info] completeness-unmeasurable: metadata coverage not evaluated
[info] simulation-skipped: no load case supplied
```

Those are not five checks passing. They are five checks that did not run, and
the harness is telling you so in its own vocabulary: `-skipped`,
`-unmeasurable`, `-not-yet-measurable`. Treat every one of them as an open
question about the part, not as evidence for it. If wall thickness matters to
the user's part and the backend cannot measure it, the correct report is
"wall thickness was not checked", not "no problems found".

The same applies to the differential oracle: `differential oracle abstained
(fewer than two independent engines)` means cross-checking did not happen.
Installing a second backend is the fix; reporting agreement is not.

## Reporting the verdict

When you relay a result, say which of these you have:

1. Built and every applicable measured predicate passed.
2. Built, and some checks could not run -- name them.
3. Refused -- name the op and the diagnostic code.

State the backend and the digest with the verdict. A pass on `stub` or `frep`
is a much weaker claim than a pass on `cadquery` cross-checked against
`manifold`, and the reader cannot tell which they got unless you say.

## Avoid

- Do not pass `--force` to make a refused artifact appear. It writes the file
  *and* a `<name>.INVALID.json` sidecar naming every failed measurement,
  because the harness expects this to be misused. If you use it for
  debugging, say so and quote the sidecar.
- Do not run `selftest` without `--strict` and then report the exit code as a
  pass. Report the findings.
- Do not relax a tolerance, delete a constraint, drop a required feature, or
  switch to a backend that cannot measure the failing property in order to
  turn a failure into a pass. Each of those changes the question rather than
  the answer, and the harness exists precisely because that substitution is
  invisible in the output.
- Do not paraphrase `UNCERTIFIED` as "passed with warnings". It means the
  harness could not certify the part, which is a different and weaker claim
  than either pass or fail.
- Do not report a verdict you did not run. If a command was not executed
  because a backend or dataset was missing, that is an unknown, and an
  unknown reported as a pass is the one error this harness cannot catch.
