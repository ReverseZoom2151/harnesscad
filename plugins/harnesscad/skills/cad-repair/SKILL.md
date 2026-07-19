---
name: cad-repair
description: Repair a CISP op stream that the harness refused or that built the wrong part. Use after apply prints ok:False with a rejected op, after a build or export is blocked by the output gate, when a measured value misses its target, when the same op keeps failing across retries, or when the user says "fix the ops", "it won't build", "the volume is wrong", "make the gate pass".
---

# Repairing a refused or wrong part

A refusal from this harness is information, not an obstacle. The gate names
the op it stopped on and why, so repair is a targeted edit rather than a
rewrite. The hard part is not finding a change that makes the message go
away -- it is finding the change that makes the *part* right, since those are
usually different changes.

## Diagnose before editing

```bash
harnesscad apply <ops.json> --backend frep --verify core
```

Use `frep` while diagnosing: it meshes anything and needs no OCCT, so a
failure on frep is a fault in your ops rather than in the kernel. Once the
stream is clean there, rerun on the backend you actually intend to use --
some failures are kernel-specific and only that rerun will find them.

Read three things from the output:

- `applied: N` -- ops 1..N were accepted. The fault is in op N+1. Everything
  after it is unevaluated, so a long diagnostic list does not mean many bugs.
- the `rejected:` line -- the offending op, verbatim, with its parameters.
- the diagnostic code -- `bad-value`, `unknown-op`, and so on. The code names
  the class of fault; the message names the specific one.

## Repair loop

1. Identify the single op named by `rejected:` or by the first `[error]`.
2. Form a hypothesis about what is wrong with *that op* -- a field the op
   does not have, a selector in the wrong form, a reference to a feature id
   that does not exist yet at that point in the stream.
3. Check the op's real signature in the `cad-op-streams` skill's
   `references/op-vocabulary.md` before editing. Most repeated failures are a
   field guessed twice.
4. Change that one op.
5. Rerun the same command. Compare `applied:` -- it must be larger than last
   time, or the edit did not address the fault.
6. Repeat until `ok: True`, then verify the part is the requested part, not
   merely a part.

If `applied:` does not advance after two edits, stop editing that op. The
fault is usually upstream -- a feature id that refers to something the
earlier ops did not actually create. Re-count the ids from the top.

## When it builds but measures wrong

A stream that applies cleanly can still produce the wrong geometry. Here the
diagnostic is a measurement mismatch rather than a refusal, and the repair
rule inverts: do not adjust numbers to hit the target. Find the op that is
structurally wrong.

- Volume too high with the right bounding box -- a cut that did not happen.
  Check the `boolean` op's `target` and `tool` ids point at the solids you
  think they do.
- Right volume, wrong bounding box -- a dimension on the wrong axis.
- Topology wrong (genus, hole count) -- an extrude that did not pass fully
  through, or a `boolean` whose kind is `union` where `cut` was meant.

Scaling a dimension until the volume matches produces a part that satisfies
the measurement and is not the part. The harness cannot detect that, which is
exactly why it must not be done.

## Also fixable at the harness level

Some refusals are not the op stream's fault:

- Diagnostics saying a check was *skipped* because the backend has no query
  for it are a backend-capability gap. Switching backends may enable the
  check. It never makes the underlying property true.
- `differential oracle abstained (fewer than two independent engines)` means
  only one backend is installed. That is an environment fix, not an op fix.

Report these as environment limits rather than repairing around them.

## Critical rules

These are the ways a repair can be worse than the failure it removes.

1. Do not loosen a tolerance, delete a constraint, or drop a required feature
   to make the gate pass. That edits the question, not the answer, and the
   result is a part that reports success and does not work.
2. Do not use `--force`. It exists to write a refused artifact for debugging,
   and it deposits a `<name>.INVALID.json` sidecar listing every failed
   measurement precisely because a forced artifact is not a repaired one.
3. Do not switch to a backend that cannot measure the failing property in
   order to stop it failing. A check that no longer runs has not passed.
4. Do not claim a repair succeeded when a tool result says otherwise. If the
   verifier says a critical rule failed, that is the verdict, regardless of
   how plausible the fix looked.
5. Do not treat skipped or unmeasurable checks as evidence the repair worked.
   They are uncertainty; name them when you report.
6. When you cannot repair the part within the op vocabulary, say which
   requirement the vocabulary cannot express. An honest gap is a usable
   result; a part that quietly omits the requirement is not.
7. Keep the diff minimal and attributable. One op changed per iteration
   makes the `applied:` counter a real signal; a rewrite makes it noise.
