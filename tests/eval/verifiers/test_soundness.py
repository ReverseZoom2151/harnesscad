"""Soundness tiering: the PRECISION gate the verifier fleet never had.

Twenty-three verifiers were written. Every one of them has a test asking
"does it FIRE on bad input?". Not one asked "does it stay SILENT on good
input?". The fleet optimised recall and never measured precision, and
`assets/pressure/report.md` measured the bill: the typed loop lost to blind
resampling by 8.3 points, losing hardest (-25pp) on the strongest model, because
a false diagnostic is an instruction and a capable model executes it precisely.

In a correction loop, precision is the only thing that matters:
  * a MISSED error leaves you where you were;
  * a FALSE error destroys work.

So the central test here is `TestKnownGoodCorpus`: seven parts any engineer
would sign off, and the assertion that the fleet's trusted channel is SILENT on
all of them. It is the gate the fleet never had.
"""

from __future__ import annotations

import unittest
from typing import Dict, List

from harnesscad.core.cisp.ops import (
    AddCircle, AddRectangle, Extrude, Fillet, Hole, NewSketch, Op, Shell,
)
from harnesscad.core.loop import HarnessSession
from harnesscad.eval.verifiers import soundness
from harnesscad.eval.verifiers.registry import discover, model_state, run_all
from harnesscad.eval.verifiers.verify import Diagnostic, Severity, default_verifiers
from harnesscad.io.backends.frep import FRepBackend


# ---------------------------------------------------------------------------
# The known-good corpus.
#
# Seven parts. Every one of them is a part an engineer would sign off without
# comment, and every one of them is expressible in the CISP op vocabulary. They
# are the ground truth for PRECISION: whatever else the fleet does, it may not
# raise a trusted hard error on any of these.
#
# Two of them are the parts report.md caught the fleet rejecting: the washer
# (80 mm disc, 8 mm thick, 30 mm bore) and the bearing housing (50 mm disc,
# 12 mm thick, 25 mm bore). The old `precheck` hole rule compared the bore
# DIAMETER (in-plane) against the extrude DISTANCE (along Z) -- orthogonal
# quantities -- and called both infeasible. It fired 40 times in the pressure
# run and caused every regression in it.
# ---------------------------------------------------------------------------

def _washer() -> List[Op]:
    """80 mm disc, 8 mm thick, 30 mm bore. A washer. Volume 34,217 mm3."""
    return [
        NewSketch(),
        AddCircle(sketch="sk1", cx=0.0, cy=0.0, r=40.0),
        Extrude(sketch="sk1", distance=8.0),
        Hole(face_or_sketch="sk1", x=0.0, y=0.0, diameter=30.0, through=True),
    ]


def _bearing_housing() -> List[Op]:
    """50 mm disc, 12 mm thick, 25 mm bore. A bearing housing."""
    return [
        NewSketch(),
        AddCircle(sketch="sk1", cx=0.0, cy=0.0, r=25.0),
        Extrude(sketch="sk1", distance=12.0),
        Hole(face_or_sketch="sk1", x=0.0, y=0.0, diameter=25.0, through=True),
    ]


def _flange_bolt_circle() -> List[Op]:
    """100 mm flange, 10 mm thick, 40 mm bore, four M10 bolts on an 80 mm PCD."""
    ops: List[Op] = [
        NewSketch(),
        AddCircle(sketch="sk1", cx=0.0, cy=0.0, r=50.0),
        Extrude(sketch="sk1", distance=10.0),
        Hole(face_or_sketch="sk1", x=0.0, y=0.0, diameter=40.0, through=True),
    ]
    for x, y in ((40.0, 0.0), (-40.0, 0.0), (0.0, 40.0), (0.0, -40.0)):
        ops.append(Hole(face_or_sketch="sk1", x=x, y=y, diameter=10.0, through=True))
    return ops


def _plate_four_holes() -> List[Op]:
    """80 x 60 x 6 plate with four 8 mm holes, 10 mm in from each corner."""
    ops: List[Op] = [
        NewSketch(),
        AddRectangle(sketch="sk1", x=0.0, y=0.0, w=80.0, h=60.0),
        Extrude(sketch="sk1", distance=6.0),
    ]
    for x, y in ((10.0, 10.0), (70.0, 10.0), (10.0, 50.0), (70.0, 50.0)):
        ops.append(Hole(face_or_sketch="sk1", x=x, y=y, diameter=8.0, through=True))
    return ops


def _filleted_thin_plate() -> List[Op]:
    """50 x 30 x 6 plate, corners filleted r=3. Report.md hole 4: this is valid.

    The vertical corner edges are rounded by a CYLINDER of radius 3 whose axis is
    the 6 mm-tall edge, so the radius is bounded by the IN-PLANE extents (50, 30),
    not by the 6 mm thickness. `preflight-RADIUS_TOO_LARGE` compares it against
    half the smallest extent of the whole bbox and rejects it. That rule is
    HEURISTIC, so it may say so -- to a human. It may not say so to the model.
    """
    return [
        NewSketch(),
        AddRectangle(sketch="sk1", x=0.0, y=0.0, w=50.0, h=30.0),
        Extrude(sketch="sk1", distance=6.0),
        Fillet(edges=("e1",), radius=3.0),
    ]


def _shelled_box() -> List[Op]:
    """60 x 40 x 20 box, shelled 3 mm. 2*3 = 6 << 20: a real cavity survives."""
    return [
        NewSketch(),
        AddRectangle(sketch="sk1", x=0.0, y=0.0, w=60.0, h=40.0),
        Extrude(sketch="sk1", distance=20.0),
        Shell(faces=("top",), thickness=3.0),
    ]


def _counterbored_bracket() -> List[Op]:
    """70 x 40 x 10 bracket with two counterbored 8 mm fixing holes."""
    return [
        NewSketch(),
        AddRectangle(sketch="sk1", x=0.0, y=0.0, w=70.0, h=40.0),
        Extrude(sketch="sk1", distance=10.0),
        Hole(face_or_sketch="sk1", x=15.0, y=20.0, diameter=8.0, through=True,
             kind="counterbore"),
        Hole(face_or_sketch="sk1", x=55.0, y=20.0, diameter=8.0, through=True,
             kind="counterbore"),
    ]


KNOWN_GOOD: Dict[str, List[Op]] = {
    "washer": _washer(),
    "bearing_housing": _bearing_housing(),
    "flange_bolt_circle": _flange_bolt_circle(),
    "plate_four_holes": _plate_four_holes(),
    "filleted_thin_plate": _filleted_thin_plate(),
    "shelled_box": _shelled_box(),
    "counterbored_bracket": _counterbored_bracket(),
}


#: ERROR codes a HEURISTIC rule is permitted to raise on a known-good part.
#:
#: There is exactly one, and it is here because it is REVIEWED, not because it
#: is tolerated: `completeness` ERRORs when a part carries no name, no units and
#: no material. None of those three are expressible in the CISP op vocabulary at
#: ALL, so it fires on every part any op stream can possibly build. It is a
#: release-readiness policy aimed at a PDM record, not a statement about
#: geometry -- which is exactly why it is tiered HEURISTIC and never reaches the
#: model. Anything else appearing here is a new false hard error and this test is
#: how you find out.
ALLOWED_HEURISTIC_ERROR_CODES = frozenset({"missing-metadata"})


def _build(ops: List[Op]):
    """Apply an op stream at verify_level='full' and return (result, session)."""
    session = HarnessSession(FRepBackend(), verify_level="full")
    return session.apply_ops(ops), session


def _all_diagnostics(ops: List[Op]) -> List[Diagnostic]:
    """Every diagnostic the product produces for `ops`: core checks + full fleet."""
    result, session = _build(ops)
    diags = list(result.diagnostics)
    # The fleet only auto-runs on an ACCEPTED batch; run it explicitly so a
    # rejected batch is measured too (a rejection on a known-good part is itself
    # the failure this corpus exists to catch).
    state = model_state(session.backend, session.opdag)
    seen = {(d.severity, d.code, d.message) for d in diags}
    for d in run_all(state):
        if (d.severity, d.code, d.message) not in seen:
            diags.append(d)
    return diags


def _errors(diags: List[Diagnostic]) -> List[Diagnostic]:
    return [d for d in diags if d.severity is Severity.ERROR]


# ---------------------------------------------------------------------------
# 1. THE PRECISION GATE
# ---------------------------------------------------------------------------

class TestKnownGoodCorpus(unittest.TestCase):
    """No trusted rule may raise a hard error on a part that is correct."""

    def test_no_proven_or_measured_error_on_any_known_good_part(self):
        """THE test. A PROVEN or MEASURED rule that fires here is not sound.

        These are the only diagnostics that reach the model. A false one among
        them is not a nuisance -- it is an instruction to break a correct part,
        and report.md measured a capable model doing exactly that. A rule that
        trips this assertion must be demoted to HEURISTIC or deleted; it must
        NOT be weakened until it stops firing.
        """
        offenders = []
        for name, ops in sorted(KNOWN_GOOD.items()):
            for d in _errors(_all_diagnostics(ops)):
                tier = soundness.tier_of(d)
                if tier in soundness.MODEL_FACING_TIERS:
                    offenders.append(f"{name}: [{tier}] {d.code}: {d.message}")
        self.assertEqual(
            offenders, [],
            "a PROVEN/MEASURED rule raised a hard ERROR on a KNOWN-GOOD part. "
            "These diagnostics are fed to the model as instructions. Demote the "
            "rule to HEURISTIC or delete it -- do not weaken it:\n  "
            + "\n  ".join(offenders))

    def test_no_unreviewed_hard_error_on_any_known_good_part(self):
        """Even an untrusted rule may not hard-reject a correct part unreviewed.

        An ERROR is a hard reject: at `fleet_blocking=True` it rolls the op back,
        and in every report it reads as "this part is wrong". A HEURISTIC rule
        that ERRORs on a known-good part is a false rejection whether or not the
        model ever hears it, so every such code must be listed and justified in
        ALLOWED_HEURISTIC_ERROR_CODES.

        This is the assertion the broken hole rule failed: it raised
        `infeasible-plan` on the washer and on the bearing housing, and
        `infeasible-plan` is not (and must never be) on the allowlist.
        """
        offenders = []
        for name, ops in sorted(KNOWN_GOOD.items()):
            for d in _errors(_all_diagnostics(ops)):
                if d.code not in ALLOWED_HEURISTIC_ERROR_CODES:
                    offenders.append(
                        f"{name}: [{soundness.tier_of(d)}] {d.code}: {d.message}")
        self.assertEqual(
            offenders, [],
            "a verifier raised a hard ERROR on a KNOWN-GOOD part:\n  "
            + "\n  ".join(offenders))

    def test_every_known_good_part_builds(self):
        """The corpus is only a precision gate if the parts are actually good."""
        for name, ops in sorted(KNOWN_GOOD.items()):
            with self.subTest(part=name):
                result, session = _build(ops)
                self.assertTrue(result.ok, f"{name} did not build: {result.diagnostics}")
                self.assertTrue(session.summary()["solid_present"], name)


# ---------------------------------------------------------------------------
# 2. Every verifier declares a tier
# ---------------------------------------------------------------------------

class TestEveryVerifierDeclaresATier(unittest.TestCase):

    def test_every_discovered_verifier_is_declared(self):
        """No default tier. An unaudited rule is not trusted with the prompt."""
        undeclared = [v.name for v in discover() if v.name not in soundness.SOUNDNESS]
        self.assertEqual(
            undeclared, [],
            "these verifiers declare no soundness tier; add them to "
            f"verifiers.soundness.SOUNDNESS: {undeclared}")

    def test_every_core_verifier_is_declared(self):
        """The core checks run outside the fleet, and are tiered all the same."""
        undeclared = [v.name for v in default_verifiers()
                      if v.name not in soundness.SOUNDNESS]
        self.assertEqual(undeclared, [])

    def test_declared_tiers_are_legal_and_reasoned(self):
        for name, s in sorted(soundness.SOUNDNESS.items()):
            with self.subTest(verifier=name):
                self.assertIn(s.default, soundness.TIERS)
                for code, tier in s.by_code.items():
                    self.assertIn(tier, soundness.TIERS, code)
                # A PROVEN claim without a stated proof is a claim nobody checked.
                self.assertTrue(s.reason.strip(), f"{name} declares a tier with no reason")

    def test_a_verifier_that_declares_nothing_is_quarantined_not_trusted(self):
        """Fail closed: an undeclared verifier is HEURISTIC, never PROVEN."""
        with self.assertRaises(KeyError):
            soundness.soundness_of("a-verifier-nobody-wrote")
        self.assertEqual(
            soundness.soundness_or_untrusted("a-verifier-nobody-wrote").default,
            soundness.HEURISTIC)

    def test_an_unknown_diagnostic_code_is_untrusted(self):
        d = Diagnostic(Severity.ERROR, "something-nobody-tiered", "boom")
        self.assertEqual(soundness.tier_of(d), soundness.HEURISTIC)
        self.assertEqual(soundness.model_facing([d]), [])


# ---------------------------------------------------------------------------
# 3. The feedback channel is gated on tier
# ---------------------------------------------------------------------------

class _RecordingLLM:
    """An LLM that records the prompts it is given and emits nothing useful."""

    def __init__(self) -> None:
        self.prompts: List[str] = []

    def complete(self, messages, tools=None):
        self.prompts.append("\n\n".join(m.content for m in messages))
        return "[]"


class TestFeedbackGate(unittest.TestCase):

    def _prompt_with(self, diags, **kw) -> str:
        from harnesscad.agents.agent.planner import Planner

        llm = _RecordingLLM()
        planner = Planner(llm, **kw)
        msgs = planner.build_messages("a plate", diagnostics=diags)
        return "\n\n".join(m.content for m in msgs)

    def test_heuristic_diagnostics_never_reach_the_model(self):
        heuristics = [
            Diagnostic(Severity.ERROR, "infeasible-plan",
                       "hole diameter 30 mm >= plate/stock wall 8 mm", "op[3]",
                       soundness.HEURISTIC),
            Diagnostic(Severity.ERROR, "missing-metadata",
                       "part carries no name", None, soundness.HEURISTIC),
            Diagnostic(Severity.WARNING, "preflight-RADIUS_TOO_LARGE",
                       "Fillet radius 3 exceeds half the smallest extent (6).",
                       "op[3]:fillet", soundness.HEURISTIC),
            Diagnostic(Severity.WARNING, "non-preferred-dimension",
                       "12 mm is not on the ISO preferred-number series"),
        ]
        self.assertEqual(soundness.model_facing(heuristics), [])
        prompt = self._prompt_with(heuristics)
        for d in heuristics:
            self.assertNotIn(d.code, prompt)
        # ...and with nothing trustworthy to say, the loop says nothing at all
        # rather than inventing an instruction. That is a blind resample, and a
        # blind resample beat the typed loop by 8.3 points.
        self.assertNotIn("PRIOR ATTEMPT FAILED", prompt)

    def test_proven_and_measured_diagnostics_do_reach_the_model(self):
        trusted = [
            Diagnostic(Severity.WARNING, "preflight-THICKNESS_TOO_LARGE",
                       "Shell thickness 9 leaves no cavity (smallest extent 5).",
                       "op[3]:shell", soundness.PROVEN),
            Diagnostic(Severity.ERROR, "empty-solid",
                       "features exist but no solid is present", None,
                       soundness.MEASURED),
            # Unstamped, but the code index knows the kernel's own refusals.
            Diagnostic(Severity.ERROR, "bad-ref", "unknown sketch 'sk9'", "op[1]"),
        ]
        self.assertEqual(len(soundness.model_facing(trusted)), 3)
        prompt = self._prompt_with(trusted)
        self.assertIn("PRIOR ATTEMPT FAILED", prompt)
        for d in trusted:
            self.assertIn(d.code, prompt)

    def test_the_human_channel_keeps_everything(self):
        """Heuristics are dropped from the PROMPT, not from the report."""
        diags = [
            Diagnostic(Severity.ERROR, "missing-metadata", "no name", None,
                       soundness.HEURISTIC),
            Diagnostic(Severity.ERROR, "empty-solid", "no solid", None,
                       soundness.MEASURED),
        ]
        self.assertEqual([d.code for d in soundness.human_facing(diags)],
                         ["missing-metadata"])
        self.assertEqual([d.code for d in soundness.model_facing(diags)],
                         ["empty-solid"])

    def test_the_policy_is_configurable(self):
        """`feedback_tiers=TIERS` restores the (measured, losing) old behaviour."""
        d = [Diagnostic(Severity.ERROR, "infeasible-plan", "hole too big", "op[3]",
                        soundness.HEURISTIC)]
        self.assertNotIn("infeasible-plan", self._prompt_with(d))
        self.assertIn("infeasible-plan",
                      self._prompt_with(d, feedback_tiers=soundness.TIERS))

    def test_the_fleet_stamps_every_diagnostic_it_emits(self):
        _result, session = _build(_filleted_thin_plate())
        diags = run_all(model_state(session.backend, session.opdag))
        self.assertTrue(diags)
        for d in diags:
            self.assertIn(d.soundness, soundness.TIERS,
                          f"{d.code} left the fleet unstamped")


# ---------------------------------------------------------------------------
# 4. The harness's one genuine structural win must survive
# ---------------------------------------------------------------------------

class TestShellTooThickSurvives(unittest.TestCase):
    """`trap_shell_too_thick` is the harness's ONLY structural advantage.

    report.md: on that brief the blind arm halts after ONE attempt on every
    model -- the core verifiers pass, the F-rep backend silently inflates a
    60x40x5 plate into a 44,941 mm3 blob, and the loop never learns anything
    went wrong. The harness arm is told the shell leaves no cavity and the 3b,
    7b and 14b all fix it. That detection is worth 3 briefs and the blind loop
    structurally cannot have it. Breaking it would be a disaster.
    """

    def _trap(self) -> List[Op]:
        # 60 x 40 x 5 plate, shelled 9 mm: 2*9 = 18 >= 5. No cavity can exist.
        return [
            NewSketch(),
            AddRectangle(sketch="sk1", x=0.0, y=0.0, w=60.0, h=40.0),
            Extrude(sketch="sk1", distance=5.0),
            Shell(faces=("top",), thickness=9.0),
        ]

    def test_it_still_fires(self):
        diags = _all_diagnostics(self._trap())
        codes = [d.code for d in diags]
        self.assertIn("preflight-THICKNESS_TOO_LARGE", codes,
                      f"the shell-too-thick detection is GONE. codes={sorted(set(codes))}")

    def test_it_is_proven_and_is_fed_back(self):
        shell = [d for d in _all_diagnostics(self._trap())
                 if d.code == "preflight-THICKNESS_TOO_LARGE"]
        self.assertEqual(len(shell), 1)
        d = shell[0]
        self.assertEqual(soundness.tier_of(d), soundness.PROVEN)
        self.assertIn(d, soundness.model_facing([d]))

        from harnesscad.agents.agent.planner import Planner

        llm = _RecordingLLM()
        msgs = Planner(llm).build_messages("a shelled plate", diagnostics=[d])
        prompt = "\n\n".join(m.content for m in msgs)
        self.assertIn("preflight-THICKNESS_TOO_LARGE", prompt)
        self.assertIn("leaves no cavity", prompt)

    def test_it_states_the_fact_before_the_order(self):
        """The message leads with the observation; the imperative is a suggestion."""
        d = [x for x in _all_diagnostics(self._trap())
             if x.code == "preflight-THICKNESS_TOO_LARGE"][0]
        obs, _, sug = d.message.partition(soundness.SUGGESTION_PREFIX)
        self.assertIn("leaves no cavity", obs)
        self.assertIn("zero volume", obs, "the evidence is missing from the message")
        self.assertIn("2 x 9", obs, "the arithmetic that proves it is missing")
        self.assertTrue(sug, "the imperative must survive, marked as a suggestion")
        self.assertLess(obs.index("leaves no cavity"), len(obs),
                        "the observation must come first")
        # The bare order must NOT be the first thing the model reads.
        self.assertFalse(d.message.startswith("Reduce"))

    def test_the_good_shelled_box_is_silent(self):
        """The same rule, on a box that shells correctly, says nothing at all."""
        codes = [d.code for d in _all_diagnostics(_shelled_box())]
        self.assertNotIn("preflight-THICKNESS_TOO_LARGE", codes)


# ---------------------------------------------------------------------------
# 5. The census, as a test (so the numbers in the report cannot rot)
# ---------------------------------------------------------------------------

class TestCensus(unittest.TestCase):

    def test_the_fleet_is_mostly_heuristic_and_says_so(self):
        tiers = [s.default for s in soundness.SOUNDNESS.values()]
        heuristic = tiers.count(soundness.HEURISTIC)
        self.assertGreater(heuristic, len(tiers) / 2,
                           "most rules guess; a table that says otherwise is flattering itself")

    def test_the_only_proven_claims_are_the_ones_with_a_proof(self):
        """PROVEN is a claim of infeasibility from first principles. Audit it."""
        proven_codes = sorted(
            code
            for s in soundness.SOUNDNESS.values()
            for code, tier in s.by_code.items() if tier == soundness.PROVEN)
        proven_verifiers = sorted(
            name for name, s in soundness.SOUNDNESS.items()
            if s.default == soundness.PROVEN)
        # Shell-leaves-no-cavity and zero-volume are theorems about offsets.
        self.assertEqual(
            proven_codes,
            ["preflight-THICKNESS_TOO_LARGE", "preflight-ZERO_VOLUME"])
        # A negative sketch DOF count is a proof of unsatisfiability.
        self.assertEqual(proven_verifiers, ["sketch-constraint"])


if __name__ == "__main__":
    unittest.main()
