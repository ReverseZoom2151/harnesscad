"""Soundness tiering — which diagnostics are allowed to instruct the model.

Why this module exists
----------------------
``assets/pressure/report.md`` measured the harness's central claim ("typed
diagnostics beat blind resampling") and the harness LOST by 8.3 points, losing
hardest on the strongest model (-25pp on qwen2.5-coder:14b). Every one of the
harness arm's net losses was a REGRESSION: an attempt the grader had accepted,
which the loop then broke *because the fleet told it to*.

The mechanism is not subtle. **A typed diagnostic is an instruction, and
instructions get obeyed.** A blind loop is un-poisonable because it is deaf. A
typed loop is a lever, and a lever amplifies whichever way it is pushed. The
value of a typed diagnostic is bounded above by its truth, and the tighter a
model's instruction-following, the tighter that bound binds.

The root cause was methodological. Twenty-three verifiers were written, each
with a test asking *"does it FIRE on bad input?"*. Not one asked *"does it stay
SILENT on good input?"*. The fleet optimised recall and never measured
precision. **In a correction loop precision is the only thing that matters: a
missed error leaves you where you were; a false error destroys work.**

The three tiers
---------------
``PROVEN``
    The rule can prove infeasibility from first principles. "Shell thickness t
    with 2t >= the smallest extent leaves literally no cavity" is a theorem
    about offset surfaces, not a guess. A PROVEN rule may never be wrong.

``MEASURED``
    The rule states an observed FACT about built geometry ("the backend reports
    no solid after 4 features"; "the mesh has 12 boundary edges"). Facts cannot
    be false. A MEASURED diagnostic reports; it does not infer.

``HEURISTIC``
    The rule guesses. Most preflight/DFM/standards rules are here. A heuristic
    may be *useful* and still be *wrong*, and "wrong" is the expensive case.

A rule that cannot be *proved* sound is HEURISTIC. Do not flatter it.

The policy
----------
Only PROVEN and MEASURED diagnostics reach the model
(:data:`MODEL_FACING_TIERS`). HEURISTIC diagnostics are still produced, still
logged, still shown to humans and still returned in every ``ApplyOpsResult`` --
they are useful. They are simply not fed back into the retry prompt, because a
wrong instruction is worse than no instruction: it converts a capable model's
instruction-following into a weapon aimed at correct geometry.

This is a *precision* policy, not a *recall* policy. Nothing is silenced. The
model-facing channel is narrowed to the diagnostics that cannot lie.

Phrasing
--------
A diagnostic that does reach the model states the OBSERVATION and its EVIDENCE
first, and carries any imperative as a trailing, clearly-marked SUGGESTION (see
:func:`observe`). "Reduce the radius below 2.5" is an order, and an order is
executed even when it is wrong. "fillet r=8 on a part whose smallest extent is
5" is evidence, and a capable model can reason from evidence. Diagnostic CODES
are never changed here -- tests and the pressure experiment key on them.

Deterministic, stdlib-only, no I/O.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

__all__ = [
    "PROVEN",
    "MEASURED",
    "HEURISTIC",
    "TIERS",
    "MODEL_FACING_TIERS",
    "Soundness",
    "SOUNDNESS",
    "KERNEL_CODES",
    "soundness_of",
    "tier_of",
    "stamp",
    "UNDECLARED",
    "soundness_or_untrusted",
    "model_facing",
    "human_facing",
    "observe",
]

# --- the tiers --------------------------------------------------------------

#: Provable from first principles. May never be wrong.
PROVEN = "proven"
#: An observed fact about built geometry. Cannot be false.
MEASURED = "measured"
#: A guess. May be wrong, and a wrong instruction destroys work.
HEURISTIC = "heuristic"

TIERS: Tuple[str, ...] = (PROVEN, MEASURED, HEURISTIC)

#: The tiers whose diagnostics are allowed into the model's retry prompt.
MODEL_FACING_TIERS: Tuple[str, ...] = (PROVEN, MEASURED)


class Soundness:
    """A verifier's declared soundness: a tier, plus per-code refinements.

    Most verifiers are one tier throughout. Two are not: ``kernel-preflight``
    emits a provable shell theorem *and* an unsound fillet ceiling under the
    same adapter, and they are separable because they carry different CODES.
    Where the reasons are NOT separable by code -- ``precheck`` emits a single
    ``infeasible-plan`` code for a dozen unrelated reasons, some provable and
    some guessed -- the whole verifier takes the weakest tier of any rule it
    contains. That is the honest reading, and it is also forced: the codes may
    not be changed, so the fleet has no way to tell those reasons apart.
    """

    __slots__ = ("default", "by_code", "reason")

    def __init__(self, default: str, by_code: Optional[Mapping[str, str]] = None,
                 reason: str = "") -> None:
        if default not in TIERS:
            raise ValueError(f"unknown soundness tier {default!r}; expected one of {TIERS!r}")
        for code, tier in (by_code or {}).items():
            if tier not in TIERS:
                raise ValueError(
                    f"unknown soundness tier {tier!r} for code {code!r}; expected {TIERS!r}")
        self.default = default
        self.by_code: Dict[str, str] = dict(by_code or {})
        self.reason = reason

    def of(self, code: Optional[str]) -> str:
        if code is None:
            return self.default
        return self.by_code.get(str(code), self.default)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Soundness({self.default!r}, by_code={self.by_code!r})"


# ---------------------------------------------------------------------------
# The table. EVERY verifier the fleet can run appears here, including the two
# core checks in `verify.py` that the session runs directly. A verifier missing
# from this table is a hard error (see registry.discover) -- there is no default
# tier, because "we never thought about it" is exactly the failure mode that
# cost 8 briefs.
# ---------------------------------------------------------------------------

SOUNDNESS: Dict[str, Soundness] = {

    # -- core (verify.py / geometry.py; run by HarnessSession directly) -------

    "sketch-constraint": Soundness(
        PROVEN,
        by_code={"under-constrained": MEASURED},
        reason=(
            "PROVEN: a constraint system whose DOF count is negative has more "
            "independent constraints than degrees of freedom, so no assignment "
            "satisfies it. That is a theorem of linear algebra, not a guess. "
            "The under-constrained WARNING is merely the observed DOF count, so "
            "it is MEASURED."),
    ),
    "solid-presence": Soundness(
        MEASURED,
        reason=(
            "MEASURED: it reports what the backend answered -- features were "
            "applied and the backend says no solid exists. It infers nothing."),
    ),
    "brep-validity": Soundness(
        MEASURED,
        reason=(
            "MEASURED: it forwards the kernel's own validity report "
            "(manifold/watertight/is_valid). The kernel is the ground truth "
            "here; the verifier only relays it."),
    ),

    # -- kernel preflight: the ONE place a real theorem lives ----------------

    "kernel-preflight": Soundness(
        HEURISTIC,
        by_code={
            # 2t >= min_extent means the inward offsets of two opposite faces
            # meet or cross: the cavity is empty by construction. A theorem
            # about offset surfaces. This is the harness's genuine structural
            # win over the blind loop (report.md, "What the harness genuinely
            # wins") and it is fed back.
            "preflight-THICKNESS_TOO_LARGE": PROVEN,
            # A shape of zero volume cannot carry a feature; there is no
            # surface to offset, fillet or shell. Also a theorem.
            "preflight-ZERO_VOLUME": PROVEN,
        },
        reason=(
            "The DEFAULT is HEURISTIC because preflight-RADIUS_TOO_LARGE is "
            "unsound: it compares the fillet radius against half the smallest "
            "extent of the whole-body bounding box, but a fillet is applied to "
            "an EDGE, and the edge need not span that extent. A 50x30x6 plate "
            "filleted at r=3.1 is valid, watertight and correctly bounded, and "
            "the rule rejects it (report.md, fleet hole 4). Two codes are "
            "promoted to PROVEN -- see by_code."),
    ),

    # -- symbolic plan lint --------------------------------------------------

    "precheck": Soundness(
        HEURISTIC,
        reason=(
            "HEURISTIC as a whole, and this is the most consequential entry in "
            "the table. Precheck contains genuinely provable rules (an extrude "
            "of an empty sketch cannot make a solid; a boolean needs two "
            "operands; a reference to a sketch that was never created is "
            "dangling) AND genuine guesses (the shell-vs-inferred-stock rule "
            "infers the stock thickness from the last extrude distance; the "
            "min_wall rule is a manufacturing policy, not a geometric fact; "
            "the hole-vs-material rule reasons about in-plane extents it can "
            "only partially know). Its single unsound rule -- hole diameter "
            "compared against the extrude DEPTH, orthogonal quantities -- fired "
            "40 times in the pressure run and caused EVERY regression. "
            "Precheck emits ONE code, `infeasible-plan`, for all of them, and "
            "the codes may not be changed (tests and the experiment key on "
            "them), so the fleet cannot separate the theorems from the guesses. "
            "A verifier takes the weakest tier of any rule it cannot separate. "
            "Its diagnostics remain ERRORs and remain logged for humans; they "
            "no longer instruct the model."),
    ),

    # -- measured-fact verifiers ---------------------------------------------

    "validity-gate": Soundness(
        MEASURED,
        reason=(
            "MEASURED: it counts boundary edges and non-manifold edges of the "
            "tessellated candidate that was actually built. A mesh with an open "
            "boundary IS open; the count is an observation, not a model."),
    ),
    "shell-envelope": Soundness(
        MEASURED,
        reason=(
            "MEASURED: it compares the BUILT bounding box against the op-stream "
            "envelope and reports the growth it observes -- 'the shelled part's "
            "bbox grew from 60x40x20 to 63x43x23'. The envelope is an exact "
            "upper bound for the op set the check restricts itself to (sketch "
            "primitives, extrude, boolean, hole, fillet, chamfer, shell all "
            "remove material or stay within it), and the verifier ABSTAINS on "
            "any op that could legitimately push geometry outside it (mirror, "
            "pattern, sweep, loft, draft, instances). So a part measured larger "
            "than its envelope is a fact plus a contradiction, not an opinion. "
            "It fires only on a real backend regression (report.md hole 2: the "
            "F-rep two-sided shell dilated every shelled part by t/2 and NO "
            "verifier noticed). "
            "It claims EXACTLY what it can support and no more: it proves the "
            "part did not GROW. It does NOT prove the shell is correct -- an "
            "inward shell can preserve the bounding box exactly and still leave "
            "the wall far too thin, and this verifier would say nothing. "
            "Silence from it is not a certificate."),
    ),
    "dimension-qa": Soundness(
        MEASURED,
        reason=(
            "MEASURED: it subtracts a measured dimension from a nominal one "
            "that the caller supplied. Both numbers are given; the deviation is "
            "arithmetic. It only runs when a target set exists."),
    ),
    "edit-consistency": Soundness(
        MEASURED,
        reason=(
            "MEASURED: it evaluates the declared sketch constraints against the "
            "actual primitive coordinates and reports the residual. A residual "
            "is a measurement of the geometry in hand."),
    ),

    # -- heuristics. Useful, fallible, and not allowed to give orders. --------

    "access": Soundness(
        HEURISTIC,
        reason=("Tool reachability is modelled from bounding boxes and a "
                "nominal tool envelope. Real access depends on the tool, the "
                "fixture and the approach vector, none of which are known."),
    ),
    "assembly": Soundness(
        HEURISTIC,
        reason=("Mate DOF accounting is a model of the joint semantics, not a "
                "solve. An 'unsatisfied' mate may be satisfiable by a solver "
                "the verifier does not run."),
    ),
    "brick-validity": Soundness(
        HEURISTIC,
        reason=("Buildability and stability rest on a support model (what holds "
                "what up). The support model is a physical assumption, so a "
                "counter-example is possible; it is not a theorem."),
    ),
    "clearance-shift": Soundness(
        HEURISTIC,
        reason=("Overlap is tested between AXIS-ALIGNED BOUNDING BOXES. Two "
                "solids whose AABBs overlap need not touch (an L-bracket and a "
                "peg through its notch), so the finding may be a false positive "
                "and the suggested shift may be unnecessary."),
    ),
    "completeness": Soundness(
        HEURISTIC,
        reason=("Release-readiness policy, not geometry: it ERRORs when a part "
                "carries no name, no units, no material, no hole tolerance. "
                "None of those are expressible in the CISP op vocabulary at "
                "all, so it raises hard errors on EVERY correctly-built part in "
                "the known-good corpus. Feeding that back tells the model to fix "
                "something it cannot express, which is the purest form of a "
                "wrong instruction. Kept for humans, who own the PDM record."),
    ),
    "compliance": Soundness(
        HEURISTIC,
        reason=("Process limits (min feature size, overhang angle, regional "
                "rules) are policy parameters of a chosen process, not "
                "properties of the geometry."),
    ),
    "dfm": Soundness(
        HEURISTIC,
        reason=("Aspect-ratio / thin-envelope / oversized are rules of thumb "
                "about an envelope, and the envelope itself is reconstructed "
                "from the op stream rather than measured."),
    ),
    "drag-proxy": Soundness(
        HEURISTIC,
        reason=("A linear surrogate fitted to frontal area. A surrogate is by "
                "definition an approximation of the thing it stands for."),
    ),
    "functional": Soundness(
        HEURISTIC,
        reason=("Function is checked against a declared FunctionalSpec through "
                "a joint model. The model may not capture the mechanism."),
    ),
    "interference": Soundness(
        HEURISTIC,
        reason=("Part-vs-part interference is computed from AABBs when no mesh "
                "is available; AABB overlap does not imply solid overlap. The "
                "verifier itself marks such findings `interference-approx`."),
    ),
    "modal-frequency": Soundness(
        HEURISTIC,
        reason=("A stiffness floor is an engineering target chosen by someone; "
                "falling below it is not an infeasibility."),
    ),
    "plausibility": Soundness(
        HEURISTIC,
        reason=("Fill ratio, aspect ratio and 'absurd size' are calibrated "
                "thresholds. A 2 m x 3 mm strip is implausible for a bracket and "
                "correct for a shim. Note that when the backend has no kernel "
                "the volume is the ENVELOPE's, so the fill-ratio findings are "
                "vacuous by construction -- exactly the kind of number that "
                "sounds measured and is not."),
    ),
    "rim-feasibility": Soundness(
        HEURISTIC,
        reason=("Wheel-rim manufacturability rules (contour continuity, spoke "
                "count, symmetry) encode a manufacturing convention."),
    ),
    "simulation": Soundness(
        HEURISTIC,
        reason=("Analytic beam/buckling formulae applied to a bounding box "
                "under an assumed load path. A closed-form stress estimate on an "
                "assumed section is a guess about the real part, however exact "
                "the arithmetic."),
    ),
    "standability": Soundness(
        HEURISTIC,
        reason=("The centre of mass is taken as the centroid of the bounding "
                "boxes (uniform density, box geometry) and the support polygon "
                "from box corners. Both are approximations of the real solid."),
    ),
    "standards": Soundness(
        HEURISTIC,
        reason=("ISO preferred numbers and standard drill sizes are "
                "CONVENTIONS. A 12 mm plate is not wrong because 12.5 is on the "
                "R10 series. It fired on three of the seven known-good parts."),
    ),
    "tolerance-stack": Soundness(
        HEURISTIC,
        reason=("Worst-case / RSS stack-up over a chain the CALLER declared. "
                "The arithmetic is exact; the chain is a claim about which "
                "dimensions accumulate, and that claim can be wrong."),
    ),
}


# ---------------------------------------------------------------------------
# Codes that do not come from a fleet verifier at all: the kernel's own
# refusals, the backend's rejections, the parser's failures. These are MEASURED
# by construction -- the kernel *actually* refused the op; the parser *actually*
# failed. They must keep reaching the model, or the correction loop goes silent
# and the harness degrades to a blind loop with extra steps.
# ---------------------------------------------------------------------------

KERNEL_CODES: Dict[str, str] = {
    "bad-ref": MEASURED,
    "bad-request": MEASURED,
    "bad-value": MEASURED,
    "degenerate": MEASURED,
    "empty-sketch": MEASURED,
    "empty-solid": MEASURED,
    "internal": MEASURED,
    "invalid-brep": MEASURED,
    "invalid-mesh": MEASURED,
    "kernel-error": MEASURED,
    "kernel-exception": MEASURED,
    "no-solid": MEASURED,
    "not-yet-supported": MEASURED,
    "parse-error": MEASURED,
    "plan-parse-error": MEASURED,
    "unknown-op": MEASURED,
    "unsupported-op": MEASURED,
    # The dispatcher's own "a verifier blew up" note. It is a fact about the
    # fleet, not about the geometry, and the model can do nothing with it.
    "verifier-error": HEURISTIC,
}

#: Every code any tiered verifier is known to emit, mapped to its tier. Built
#: from SOUNDNESS + KERNEL_CODES; used to tier a diagnostic that arrives with no
#: verifier attribution (the core checks, the backend, a replayed trace).
_CODE_INDEX: Dict[str, str] = dict(KERNEL_CODES)
for _name, _s in SOUNDNESS.items():
    for _code, _tier in _s.by_code.items():
        _CODE_INDEX.setdefault(_code, _tier)
# Codes emitted by the MEASURED/PROVEN core checks, which the session runs
# outside the fleet and which therefore never get stamped by the dispatcher.
_CODE_INDEX.setdefault("over-constrained", PROVEN)
_CODE_INDEX.setdefault("under-constrained", MEASURED)


def soundness_of(verifier: Any) -> Soundness:
    """The declared soundness of a verifier (or verifier name).

    Raises KeyError when the verifier is not in the table. That is deliberate:
    a verifier with no declared tier is the bug this module exists to prevent,
    and it must not silently default to "trustworthy".
    """
    name = verifier if isinstance(verifier, str) else str(getattr(verifier, "name", verifier))
    try:
        return SOUNDNESS[name]
    except KeyError:
        raise KeyError(
            f"verifier {name!r} declares no soundness tier. Add it to "
            f"harnesscad.eval.verifiers.soundness.SOUNDNESS. There is no "
            f"default: a rule nobody has audited is not trusted with the "
            f"model's retry prompt.") from None


#: What an undeclared verifier gets at RUNTIME. Not a default tier -- a
#: quarantine. The fleet auto-discovers modules, so a new verifier dropped into
#: the package must not take the loop down; it must also not be trusted. It is
#: therefore treated as HEURISTIC (logged, never fed back) and the test suite
#: (tests/eval/verifiers/test_soundness.py) fails until it is declared here.
UNDECLARED = Soundness(
    HEURISTIC,
    reason="undeclared: this verifier is not in the soundness table, so it is "
           "quarantined as HEURISTIC and never reaches the model.")


def soundness_or_untrusted(verifier: Any) -> Soundness:
    """:func:`soundness_of`, but quarantining an undeclared verifier (see UNDECLARED)."""
    try:
        return soundness_of(verifier)
    except KeyError:
        return UNDECLARED


def tier_of(diag: Any) -> str:
    """The soundness tier of a single diagnostic.

    Resolution order:
      1. the tier the dispatcher stamped on it (it knew the emitting verifier);
      2. the code index (kernel/backend/core codes, and any code a verifier
         promoted by name);
      3. HEURISTIC -- an unrecognised code is untrusted, never trusted. Failing
         closed is the whole point.
    """
    stamped = getattr(diag, "soundness", None)
    if stamped in TIERS:
        return str(stamped)
    if isinstance(diag, dict):
        stamped = diag.get("soundness")
        if stamped in TIERS:
            return str(stamped)
        code = diag.get("code")
    else:
        code = getattr(diag, "code", None)
    if code is None:
        return HEURISTIC
    return _CODE_INDEX.get(str(code), HEURISTIC)


def stamp(diag: Any, verifier: Any) -> Any:
    """Record on *diag* the tier of the verifier that produced it. Returns diag."""
    tier = soundness_or_untrusted(verifier).of(getattr(diag, "code", None))
    try:
        setattr(diag, "soundness", tier)
    except (AttributeError, TypeError):  # pragma: no cover - frozen diagnostic
        pass
    return diag


def model_facing(diags: Iterable[Any],
                 tiers: Iterable[str] = MODEL_FACING_TIERS) -> List[Any]:
    """The diagnostics allowed to instruct the model. THE gate.

    Everything else is dropped from the retry prompt -- not from the log, not
    from the report, not from the human's screen. Only from the one channel
    where being wrong destroys work.
    """
    allowed = set(tiers)
    return [d for d in diags if tier_of(d) in allowed]


def human_facing(diags: Iterable[Any],
                 tiers: Iterable[str] = MODEL_FACING_TIERS) -> List[Any]:
    """The complement of :func:`model_facing`: what humans still get to see."""
    allowed = set(tiers)
    return [d for d in diags if tier_of(d) not in allowed]


# ---------------------------------------------------------------------------
# Phrasing: state facts, not orders.
# ---------------------------------------------------------------------------

#: Marks the imperative half of a message so a reader (human or model) can tell
#: the observation from the advice. Kept short: it is prepended to model input.
SUGGESTION_PREFIX = "SUGGESTION (advisory, not a requirement): "


def observe(observation: str, evidence: str = "", suggestion: str = "") -> str:
    """Compose an evidence-led diagnostic message.

    ``"Reduce the radius below 2.5"`` is an ORDER. A capable model executes an
    order precisely, including when the order is wrong -- that is how the 14b
    read `hole diameter 30 >= wall 8`, changed exactly one field (30 -> 7.5) and
    destroyed a correct washer. ``"fillet r=8 on a part whose smallest extent is
    5; the resulting solid has volume 0"`` is an OBSERVATION with EVIDENCE, and a
    model can reason from evidence rather than merely comply with it.

    So: observation first, evidence attached, imperative last and labelled.
    """
    text = observation.strip()
    ev = evidence.strip()
    if ev:
        text = f"{text}: {ev}" if not text.endswith(":") else f"{text} {ev}"
    if not text.endswith((".", "!", "?")):
        text += "."
    sug = suggestion.strip()
    if sug:
        text = f"{text} {SUGGESTION_PREFIX}{sug}"
    return text
