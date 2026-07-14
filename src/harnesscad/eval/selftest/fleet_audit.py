"""Fleet audit — precision, recall and F1 PER VERIFIER. The metric we never had.

Twenty-odd verifiers gate every op stream the harness builds. Not one of them was
ever measured. They were tested -- each has a unit test proving it fires when it
is supposed to -- and a unit test of a verifier is a test of its RECALL on the
cases its author thought of. Nothing measured its PRECISION: how often it fires on
a part that is perfectly fine.

That is not a theoretical gap. It cost the harness its own headline experiment.
A rule comparing a hole's diameter against the plate's THICKNESS (orthogonal
dimensions) rejected an ordinary washer, fired 40 times across the pressure test,
caused every regression, and turned a +3.7 into a -8.3. It had a passing unit
test the whole time.

So: two corpora and a confusion matrix.

* **KNOWN-GOOD** -- parts any engineer signs off on: a washer, a bearing housing,
  a flange on a bolt circle, a shelled box, a filleted plate, a counterbored
  bracket. Every one of them BUILDS, watertight, on the exact B-rep kernels.
  A verifier raising an ERROR here is raising a FALSE POSITIVE. There is no
  appeal: the part is good.
* **KNOWN-BAD** -- deliberate defects with a stated reason: a fillet larger than
  half the thinnest extent, a shell that eats its own stock, a hole wider than
  the material around it, a zero-angle revolve, an extrude of an empty sketch.
  A verifier that stays silent here is a FALSE NEGATIVE.

Per verifier: precision = TP/(TP+FP), recall = TP/(TP+FN), F1 the harmonic mean.
A verifier with recall 1.0 and precision 0.4 is not a good verifier that needs
tuning; it is a rule that costs more than it earns, and the whole point of this
table is that the fleet can no longer hide that behind an aggregate.

The fleet is shown the WHOLE PLAN (see ``probe.plan_opdag``), including the ops a
backend refused to build: the LINT tier reads the op stream, and judging it on
evidence it was never given would manufacture false negatives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from harnesscad.core.cisp.ops import (AddCircle, AddRectangle, Chamfer, Extrude,
                                      Fillet, Hole, NewSketch, Op, Revolve, Shell)
from harnesscad.eval.selftest.probe import (BackendFactory, plan_opdag, resolve)
from harnesscad.eval.verifiers import registry as fleet_registry
from harnesscad.eval.verifiers.verify import Severity

__all__ = ["Case", "KNOWN_GOOD", "KNOWN_BAD", "VerifierScore", "FleetReport",
           "audit", "run", "format_text"]


@dataclass(frozen=True)
class Case:
    """One corpus entry: an op stream, and WHY it is good or bad."""

    name: str
    ops: Tuple[Op, ...]
    good: bool
    why: str


def _sk(plane: str = "XY") -> Op:
    return NewSketch(plane)


# --- the known-good corpus -------------------------------------------------
# Every one of these builds a watertight solid on cadquery AND freecad. Any
# ERROR on any of them is a false positive, full stop.

KNOWN_GOOD: Tuple[Case, ...] = (
    Case("washer_80x8_bore30",
         (_sk(), AddCircle("sk1", 0, 0, 40.0), Extrude("sk1", 8.0),
          Hole("sk1", 0.0, 0.0, 30.0, None, True, "simple")),
         True,
         "an 80 mm disc, 8 mm thick, 30 mm bore. A WASHER. Builds: volume "
         "34557.5, bbox 80x80x8, watertight. THE part the fleet rejected."),
    Case("bearing_housing",
         (_sk(), AddCircle("sk1", 0, 0, 30.0), Extrude("sk1", 25.0),
          Hole("sk1", 0.0, 0.0, 40.0, None, True, "simple")),
         True,
         "a 60 mm boss, 25 mm tall, bored 40 mm for a bearing. The bore is "
         "WIDER than the part is thick, and that is what a bearing housing IS."),
    Case("flange_bolt_circle",
         (_sk(), AddCircle("sk1", 0, 0, 50.0), Extrude("sk1", 10.0),
          Hole("sk1", 0.0, 0.0, 40.0, None, True, "simple"),
          Hole("sk1", 35.0, 0.0, 9.0, None, True, "simple"),
          Hole("sk1", -35.0, 0.0, 9.0, None, True, "simple"),
          Hole("sk1", 0.0, 35.0, 9.0, None, True, "simple"),
          Hole("sk1", 0.0, -35.0, 9.0, None, True, "simple")),
         True,
         "a 100 mm flange, 40 mm bore, four M8 clearance holes on a 70 mm PCD."),
    Case("filleted_thin_plate",
         (_sk(), AddRectangle("sk1", 0, 0, 50, 30), Extrude("sk1", 6.0),
          Fillet((), 2.0)),
         True,
         "a 6 mm plate, corners rounded R2. r < c/2, so the fillet is valid, and "
         "OCCT builds it: 8715.42 mm3."),
    Case("shelled_box_3mm",
         (_sk(), AddRectangle("sk1", 0, 0, 60, 40), Extrude("sk1", 20.0),
          Shell((), 3.0)),
         True,
         "a 60x40x20 enclosure, 3 mm walls. 3 mm of wall in a 20 mm box leaves a "
         "14 mm cavity: entirely feasible."),
    Case("counterbored_bracket",
         (_sk(), AddRectangle("sk1", 0, 0, 80, 40), Extrude("sk1", 12.0),
          Hole("sk1", 20.0, 20.0, 8.0, None, True, "counterbore"),
          Hole("sk1", 60.0, 20.0, 8.0, None, True, "counterbore"),
          Chamfer((), 1.0)),
         True,
         "an 80x40x12 bracket, two counterbored fixing holes, edges broken 1 mm."),
    Case("plate_hole_row",
         (_sk(), AddRectangle("sk1", 0, 0, 120, 30), Extrude("sk1", 6.0),
          Hole("sk1", 20.0, 15.0, 8.0, None, True, "simple"),
          Hole("sk1", 60.0, 15.0, 8.0, None, True, "simple"),
          Hole("sk1", 100.0, 15.0, 8.0, None, True, "simple")),
         True,
         "a 6 mm strip with three 8 mm holes. Each hole is WIDER than the strip is "
         "thick -- normal, and the trap the hole rule fell into."),
    Case("revolved_pulley",
         (_sk(), AddRectangle("sk1", 10, 0, 15, 20),
          Revolve("sk1", (0.0, 0.0, 0.0, 0.0, 1.0, 0.0), 360.0)),
         True,
         "a profile revolved 360 degrees into a ring. Solid, closed, genus 1."),
)


# --- the known-bad corpus --------------------------------------------------
# A defect a competent engineer would refuse to release. The fleet is supposed to
# catch these -- that is the entire reason it exists.

KNOWN_BAD: Tuple[Case, ...] = (
    Case("fillet_larger_than_half_extent",
         (_sk(), AddRectangle("sk1", 0, 0, 50, 30), Extrude("sk1", 6.0),
          Fillet((), 5.0)),
         False,
         "R5 on a 6 mm plate. r > c/2: the fillet cannot close and OCCT throws."),
    Case("shell_consumes_the_stock",
         (_sk(), AddRectangle("sk1", 0, 0, 60, 40), Extrude("sk1", 5.0),
          Shell((), 9.0)),
         False,
         "a 9 mm wall in a 5 mm plate. There is no cavity to leave: the shell "
         "eats the part."),
    Case("shell_thicker_than_half_the_box",
         (_sk(), AddRectangle("sk1", 0, 0, 30, 30), Extrude("sk1", 20.0),
          Shell((), 16.0)),
         False,
         "a 16 mm wall in a 30 mm box. 2t > 30: the walls meet in the middle."),
    Case("hole_wider_than_the_material",
         (_sk(), AddRectangle("sk1", 0, 0, 20, 20), Extrude("sk1", 10.0),
          Hole("sk1", 10.0, 10.0, 30.0, None, True, "simple")),
         False,
         "a 30 mm hole through a 20 mm square. The hole is bigger than the part: "
         "no material survives around it."),
    Case("zero_volume_revolve",
         (_sk(), AddRectangle("sk1", 10, 0, 5, 20),
          Revolve("sk1", (0.0, 0.0, 0.0, 0.0, 1.0, 0.0), 0.0)),
         False,
         "a 0-degree revolve sweeps nothing. Zero volume."),
    Case("extrude_of_an_empty_sketch",
         (_sk(), Extrude("sk1", 10.0)),
         False,
         "an extrude of a sketch with no profile in it. There is nothing to sweep."),
    Case("zero_distance_extrude",
         (_sk(), AddRectangle("sk1", 0, 0, 40, 40), Extrude("sk1", 0.0)),
         False,
         "an extrude of distance 0. A face, not a solid."),
    Case("negative_fillet_radius",
         (_sk(), AddRectangle("sk1", 0, 0, 40, 40), Extrude("sk1", 10.0),
          Fillet((), -3.0)),
         False,
         "a fillet of radius -3. Not a geometry: a typo that must not reach OCCT."),
)


# --- scoring ---------------------------------------------------------------

@dataclass
class VerifierScore:
    """The confusion matrix for ONE verifier over both corpora."""

    name: str
    tier: str
    tp: int = 0     # fired on a known-bad part      (correct)
    fp: int = 0     # fired on a known-good part     (a FALSE ALARM -- the cost)
    fn: int = 0     # silent on a known-bad part     (a MISS)
    tn: int = 0     # silent on a known-good part    (correct)
    oos: int = 0    # out of scope: the verifier's own applies_to() said "not mine"
    false_positives: List[str] = field(default_factory=list)   # part names
    false_negatives: List[str] = field(default_factory=list)
    codes: Dict[str, int] = field(default_factory=dict)         # code -> times fired
    fp_codes: Dict[str, int] = field(default_factory=dict)      # code -> FALSE alarms
    errored: int = 0                                            # verifier crashed

    @property
    def abstained(self) -> bool:
        """It never fired and never had a case in scope: not a bug, just silent."""
        return self.fired == 0 and (self.tp + self.fn) == 0

    @property
    def fires_on_everything(self) -> bool:
        """It rejected EVERY part it saw, good and bad. Then it is information-free.

        Its precision is just the base rate of bad parts in the corpus, its recall
        is a perfect 1.0, and it knows NOTHING: a rule that always says no has the
        same output as a rule with no logic in it. This is the failure mode an
        aggregate "the fleet catches 100% of bad parts" number hides completely,
        and it is worth its own flag because a reader will otherwise credit it.
        """
        return self.fn == 0 and self.tn == 0 and self.fired > 0

    @property
    def fired(self) -> int:
        return self.tp + self.fp

    @property
    def precision(self) -> Optional[float]:
        """Of the parts it rejected, how many deserved it. None = never fired."""
        return None if self.fired == 0 else self.tp / float(self.fired)

    @property
    def recall(self) -> Optional[float]:
        """Of the bad parts, how many it caught. None = no bad part applies."""
        pos = self.tp + self.fn
        return None if pos == 0 else self.tp / float(pos)

    @property
    def f1(self) -> Optional[float]:
        p, r = self.precision, self.recall
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)

    def to_dict(self) -> dict:
        return {"name": self.name, "tier": self.tier,
                "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
                "out_of_scope": self.oos, "abstained": self.abstained,
                "precision": self.precision, "recall": self.recall, "f1": self.f1,
                "false_positives": self.false_positives,
                "false_negatives": self.false_negatives,
                "codes": dict(sorted(self.codes.items())),
                "fp_codes": dict(sorted(self.fp_codes.items())),
                "errored": self.errored}


@dataclass
class FleetReport:
    backend: str = "frep"
    scores: List[VerifierScore] = field(default_factory=list)
    #: part name -> the verifiers that raised an ERROR on it.
    good_rejected_by: Dict[str, List[str]] = field(default_factory=dict)
    bad_caught_by: Dict[str, List[str]] = field(default_factory=dict)
    skipped: str = ""

    # -- fleet-level (the gate the loop actually applies: ANY verifier errors) --
    @property
    def fleet_tp(self) -> int:
        return sum(1 for c in KNOWN_BAD if self.bad_caught_by.get(c.name))

    @property
    def fleet_fn(self) -> int:
        return len(KNOWN_BAD) - self.fleet_tp

    @property
    def fleet_fp(self) -> int:
        return sum(1 for c in KNOWN_GOOD if self.good_rejected_by.get(c.name))

    @property
    def fleet_tn(self) -> int:
        return len(KNOWN_GOOD) - self.fleet_fp

    @property
    def false_positive_rate(self) -> float:
        return self.fleet_fp / float(len(KNOWN_GOOD)) if KNOWN_GOOD else 0.0

    @property
    def informative(self) -> List[VerifierScore]:
        """The rules that actually discriminate: they fired, and not on everything."""
        return [s for s in self.scores if s.fired and not s.fires_on_everything]

    @property
    def informative_tp(self) -> int:
        """Known-bad parts caught by a rule that does NOT reject everything.

        This is the fleet's real recall. Counting a part as "caught" by a rule that
        rejects the washer too is counting it as caught by a rule that would have
        caught anything, including nothing.
        """
        names = {s.name for s in self.informative}
        return sum(1 for c in KNOWN_BAD
                   if names.intersection(self.bad_caught_by.get(c.name) or []))

    @property
    def ok(self) -> bool:
        return self.fleet_fp == 0 and self.fleet_fn == 0

    def to_dict(self) -> dict:
        return {
            "oracle": "fleet",
            "ok": self.ok,
            "backend": self.backend,
            "skipped": self.skipped,
            "known_good": len(KNOWN_GOOD),
            "known_bad": len(KNOWN_BAD),
            "fleet": {"tp": self.fleet_tp, "fp": self.fleet_fp,
                      "fn": self.fleet_fn, "tn": self.fleet_tn,
                      "false_positive_rate": self.false_positive_rate,
                      "informative_tp": self.informative_tp,
                      "fires_on_everything": [s.name for s in self.scores
                                              if s.fires_on_everything]},
            "good_rejected_by": self.good_rejected_by,
            "bad_caught_by": self.bad_caught_by,
            "verifiers": [s.to_dict() for s in self.scores],
        }


def _errors_by_verifier(state: Any, fleet: Sequence[Any]
                        ) -> Dict[str, Tuple[List[str], bool, bool]]:
    """{verifier: (error codes raised, did it crash, is the part in its scope)}.

    ``applies_to`` is the verifier's OWN declaration that a part is its business.
    A brick-stability rule that stays silent about a washer is not missing the
    washer -- it never claimed it. Charging it a false negative would flatter the
    rules that DO claim everything and catch nothing, which is exactly backwards.
    """
    out: Dict[str, Tuple[List[str], bool, bool]] = {}
    for v in fleet:
        name = getattr(v, "name", type(v).__name__)
        try:
            in_scope = bool(v.applies_to(state))
        except Exception:  # noqa: BLE001
            in_scope = True
        # verifiers=[v] is load-bearing: without it run_all re-DISCOVERS the fleet
        # and silently scores nothing for a verifier that was injected rather than
        # discovered -- which would make every synthetic rule look innocent.
        diags = fleet_registry.run_all(state, tiers=fleet_registry.TIERS,
                                       only=[name], verifiers=[v])
        codes = [d.code for d in diags if d.severity is Severity.ERROR]
        # run_all turns a crashed verifier into a WARNING 'verifier-error'; that is
        # not an ERROR, so it would silently score as a clean pass. Count it.
        crashed = any(d.code == "verifier-error" for d in diags)
        out[name] = (codes, crashed, in_scope)
    return out


def audit(backend: str = "frep",
          good: Sequence[Case] = KNOWN_GOOD,
          bad: Sequence[Case] = KNOWN_BAD,
          fleet: Optional[Sequence[Any]] = None,
          factory: Optional[BackendFactory] = None) -> FleetReport:
    """Run the whole verifier fleet over both corpora and score every rule."""
    report = FleetReport(backend=backend)
    be, skip = resolve(backend, factory)
    if be is None:
        report.skipped = skip
        return report
    the_fleet = list(fleet) if fleet is not None else fleet_registry.discover()
    scores: Dict[str, VerifierScore] = {
        getattr(v, "name", type(v).__name__): VerifierScore(
            getattr(v, "name", type(v).__name__), getattr(v, "tier", "lint"))
        for v in the_fleet
    }

    for case in list(good) + list(bad):
        engine, _ = resolve(backend, factory)
        from harnesscad.core.loop import HarnessSession
        session = HarnessSession(engine, verify_level="core")
        try:
            session.apply_ops(list(case.ops))
        except Exception:  # noqa: BLE001 - a backend crash is not the fleet's fault
            pass
        # The fleet judges the PLAN, so it sees every op -- including the ones the
        # backend refused to build.
        state = fleet_registry.model_state(engine, plan_opdag(case.ops))
        fired = _errors_by_verifier(state, the_fleet)
        raisers: List[str] = []
        for name, (codes, crashed, in_scope) in fired.items():
            score = scores[name]
            if crashed:
                score.errored += 1
            if codes:
                raisers.append(name)
                for c in codes:
                    score.codes[c] = score.codes.get(c, 0) + 1
            if case.good:
                # A false alarm is a false alarm whether or not the rule claims the
                # part: it FIRED, and the loop will act on it.
                if codes:
                    score.fp += 1
                    score.false_positives.append(case.name)
                    for c in codes:
                        score.fp_codes[c] = score.fp_codes.get(c, 0) + 1
                elif in_scope:
                    score.tn += 1
                else:
                    score.oos += 1
            else:
                if codes:
                    score.tp += 1
                elif in_scope:
                    score.fn += 1
                    score.false_negatives.append(case.name)
                else:
                    score.oos += 1
        if case.good:
            report.good_rejected_by[case.name] = sorted(raisers)
        else:
            report.bad_caught_by[case.name] = sorted(raisers)

    report.scores = sorted(scores.values(), key=lambda s: (s.tier, s.name))
    return report


def run(backend: str = "frep",
        factory: Optional[BackendFactory] = None) -> FleetReport:
    return audit(backend=backend, factory=factory)


def _pct(v: Optional[float]) -> str:
    return "  n/a" if v is None else "%5.2f" % v


def format_text(report: FleetReport) -> str:
    lines: List[str] = []
    lines.append("FLEET AUDIT -- precision / recall PER VERIFIER")
    lines.append("=" * 78)
    if report.skipped:
        lines.append("skipped: " + report.skipped)
        return "\n".join(lines)
    lines.append("%d known-good parts (every ERROR here is a FALSE POSITIVE), "
                 "%d known-bad (every silence is a MISS). backend=%s"
                 % (len(KNOWN_GOOD), len(KNOWN_BAD), report.backend))
    lines.append("")
    lines.append("FLEET AS A GATE (the loop rejects when ANY verifier errors)")
    lines.append("  known-good rejected : %d / %d   (false-positive rate %.0f%%)"
                 % (report.fleet_fp, len(KNOWN_GOOD),
                    100.0 * report.false_positive_rate))
    lines.append("  known-bad  caught   : %d / %d   (recall %.0f%%)"
                 % (report.fleet_tp, len(KNOWN_BAD),
                    100.0 * report.fleet_tp / max(len(KNOWN_BAD), 1)))
    always = [s for s in report.scores if s.fires_on_everything]
    if always:
        lines.append("")
        lines.append("  WARNING: %s reject(s) EVERY part in both corpora."
                     % ", ".join(s.name for s in always))
        lines.append("  A rule that always says no has the precision of the base "
                     "rate and the information content of a coin that only lands")
        lines.append("  on tails. Discount it, and the fleet's real recall is "
                     "%d / %d (%.0f%%) -- everything above that line is the "
                     "aggregate flattering itself."
                     % (report.informative_tp, len(KNOWN_BAD),
                        100.0 * report.informative_tp / max(len(KNOWN_BAD), 1)))
    lines.append("")
    lines.append("%-22s %-8s %3s %3s %3s %3s  %5s %5s %5s  %s"
                 % ("verifier", "tier", "TP", "FP", "FN", "TN",
                    "prec", "rec", "F1", "false positives"))
    lines.append("-" * 78)
    for s in report.scores:
        if s.abstained:
            continue
        lines.append("%-22s %-8s %3d %3d %3d %3d  %s %s %s  %s"
                     % (s.name, s.tier, s.tp, s.fp, s.fn, s.tn,
                        _pct(s.precision), _pct(s.recall), _pct(s.f1),
                        ", ".join(s.false_positives[:3])))
    silent = [s.name for s in report.scores if s.abstained]
    if silent:
        lines.append("")
        lines.append("abstained on all %d parts (fired on nothing, and claimed "
                     "nothing): %s" % (len(KNOWN_GOOD) + len(KNOWN_BAD),
                                       ", ".join(silent)))
    lines.append("")
    lines.append("KNOWN-BAD -- who caught what")
    lines.append("-" * 78)
    for case in KNOWN_BAD:
        raisers = report.bad_caught_by.get(case.name) or []
        lines.append("  %-32s %s" % (case.name,
                                     ", ".join(raisers) if raisers
                                     else "*** CAUGHT BY NOTHING ***"))
    lines.append("")
    lines.append("FALSE POSITIVES -- good parts the fleet rejected")
    lines.append("-" * 78)
    any_fp = False
    for case in KNOWN_GOOD:
        raisers = report.good_rejected_by.get(case.name) or []
        if not raisers:
            continue
        any_fp = True
        lines.append("  %-24s rejected by: %s" % (case.name, ", ".join(raisers)))
        lines.append("      %s" % case.why)
        for s in report.scores:
            if case.name in s.false_positives:
                for code in sorted(s.fp_codes):
                    lines.append("      %s -> %s" % (s.name, code))
    if not any_fp:
        lines.append("  none. Every known-good part passed the fleet.")
    lines.append("")
    lines.append("FALSE NEGATIVES -- bad parts nothing caught")
    lines.append("-" * 78)
    any_fn = False
    for case in KNOWN_BAD:
        if report.bad_caught_by.get(case.name):
            continue
        any_fn = True
        lines.append("  %-24s caught by NOTHING" % case.name)
        lines.append("      %s" % case.why)
    if not any_fn:
        lines.append("  none. Every known-bad part was caught by something.")
    return "\n".join(lines)
