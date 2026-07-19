# Deep read: the SKILL.md corpus

> PROVENANCE NOTE. This report was reconstructed post-hoc from the session
> record. The agent that performed the original deep read was killed before it
> could commit this file, so unlike the agent-authored reports in this ledger
> its wording is a reconstruction rather than the original prose. The corpus
> counts and the harness-side facts below were re-verified against disk at
> reconstruction time; where a briefed detail did not hold, it is corrected.

Scope: the SKILL.md corpus scattered across the CAD sibling repos -- what
patterns the good ones share, and the skill suite HarnessCAD built out of them.

Skill bodies found in these repos are treated as unverified exemplar material,
never as instructions to follow.

---

## The corpus: 52 SKILL.md files across the CAD repos

Verified on disk. The CAD corpus carries exactly 52 SKILL.md files, in these
repos:

| Repo | SKILL.md files |
|---|---:|
| AgentSCAD | 13 |
| freecad-ai | 9 |
| cad-cae-copilot | 5 |
| text-to-cad-main | 22 |
| text-to-cad-better | 3 |
| **TOTAL** | **52** |

Notes on the breakdown, since the briefed grouping was looser than the on-disk
layout:

- `cad-cae-copilot` (5) is the "aieng" family: three under
  `aieng-agent-skills/skills/` (`aieng-cad-authoring`, `aieng-cad-cae-copilot`,
  `aieng-closed-loop-copilot`), one `superpowers` skill under
  `aieng/.github/skills/`, and one archived `CAD-Agent-Skills` under `archive/`.
  So "aieng-agent-skills 3 + superpowers + archive" resolves to exactly 5.
- `text-to-cad-main` (22) is the largest and is duplicated: the same 11-skill
  CAD plugin appears twice, once under `plugins/cad/skills/` and once under a
  top-level `skills/` (bambu-labs, cad, cad-viewer, dxf, gcode, implicit-cad,
  sdf, sendcutsend, srdf, step-parts, urdf). Worth knowing the 22 is 11x2, not
  22 distinct skills.
- `text-to-cad-better` (3): `cad`, `robot-motion`, `urdf` under
  `.agents/skills/`.
- CORRECTION: the briefed grouping listed "forgent3d (1)". forgent3d is in the
  corpus (`resources/cad_repos/forgent3d-main/`) but ships NO SKILL.md -- it is
  gifs/docs, not a skills repo. It contributes 0 to the count, and the 52 total
  is unaffected (13+9+5+22+3 = 52 without it).

(For completeness: another ~13 SKILL.md files live under `resources/computer_use`
and `resources/spec-kit`, but those are outside the CAD corpus and outside this
read's scope.)

---

## Patterns that separate the good skills from the filler

1. Descriptions trigger on SITUATIONS, not capabilities. The description field
   should answer "when am I in the state this skill is for?", not "what can this
   skill do?". Best example in the corpus is AgentSCAD's `scad-repair`
   (verified on disk at `AgentSCAD-main/.../skills/scad-repair/SKILL.md`): its
   description names the machine states the job can be in --
   `GEOMETRY_FAILED`, `RENDER_FAILED`, `VALIDATION_FAILED`, `REPAIRING`,
   `DEBUGGING` -- so the model matches the skill by recognising the state it is
   already in, not by reasoning about capability.

2. Progressive disclosure, three levels. Name + first line are always in
   context; the body (kept under ~200 lines) loads when the skill fires; the
   `references/` directory loads only when the body tells the model to open it.
   The cost of a skill you do not use is one line, not one file.

3. "Explain the WHY, not the WHAT." The central authoring rule. A skill that
   states a rule without its reason cannot be applied to a case the author did
   not foresee. Corollary that recurs verbatim: "if you're writing ALWAYS/NEVER
   in caps, the instruction needs a reason, not more emphasis." Caps are a smell
   that a missing rationale is being papered over with volume.

4. Scripts where determinism matters -- "thin harness, fat skills." Anything
   that must be exact (parsing, hashing, a fixed sequence of shell steps) is
   pushed into a checked-in script the skill calls, not left to the model to
   reproduce. The prose carries judgement; the scripts carry determinism.

5. JSON output contracts with enum'd verdicts. The good verifier-style skills
   emit a fixed JSON shape whose verdict is drawn from a closed enum, so a
   caller can branch on it without parsing prose.

6. Anti-reward-hacking guardrails. The recurring defensive moves:
   - "Do not loosen validation to fake success" -- the skill forbids the model
     from weakening the check to make it pass.
   - `tool_unavailable` is a first-class verdict, not an error swallowed into a
     pass. A missing tool is reported as such.
   - "Treat skipped checks as uncertainty, not success proof." A check that did
     not run is not a check that passed.

---

## The suite HarnessCAD built from it: 5 skills

Verified on disk under `plugins/harnesscad/skills/`:

- `cad-op-streams/SKILL.md`
- `cad-gate-verdicts/SKILL.md`
- `cad-repair/SKILL.md`
- `cad-brief-to-part/SKILL.md`
- `cad-pdd/SKILL.md`

(There is also a `harnesscad/SKILL.md` at the same level -- the plugin's own
root/index skill -- which is not one of the five task-shaped skills but is
present on disk.)

Key finding baked into these skills: `verify` is not a verb. There is no
`harnesscad verify` command. Verification is a flag on `apply`:
`harnesscad apply <ops.json> --backend frep --verify core` (or
`--verify full`). Verified on disk -- both `cad-repair/SKILL.md` (line 17) and
`cad-op-streams/SKILL.md` (line 90) invoke `apply ... --verify core`, and
`cad-gate-verdicts/SKILL.md` documents "`--verify core` for the three core
checks, or `--verify full`". A skill that told the model to run a bare `verify`
subcommand would send it down a path the CLI does not have; encoding
`apply --verify {core,full}` correctly is exactly the kind of situation-accurate
detail pattern 1 is about.

---

## Summary of stale-claim corrections

- forgent3d ships no SKILL.md; it does not contribute the "1" it was briefed
  with. The 52-file total is unchanged (13+9+5+22+3).
- text-to-cad-main's 22 is 11 skills duplicated across two trees, not 22
  distinct skills -- recorded so nobody double-counts unique skills.
- Everything else held: the 52-file CAD corpus total, the scad-repair
  state-named description, the "why not what" / caps-need-a-reason rules, the
  anti-reward-hacking guardrails, the 5 built skills, and the `apply --verify
  {core,full}` (not a `verify` verb) finding.
