"""Preference pairs with a GROUND-TRUTH label -- DPO and KTO, oracle-adjudicated.

A correct part and an incorrect part on the SAME brief is a preference pair, and
the preference is not an opinion. No LLM judge, no human annotator, no learned
reward model, no position bias, no verbosity bias, no self-enhancement bias. The
label is a measurement.

``data/dataengine/preference/dpo_pairs.py`` already builds pairs from
``{'code', 'reward'}`` samples and is imported by nothing but its test. This module
does not reimplement it -- it supplies the reward the oracle computes and calls it.

THE LABELLER IS THE ORACLE, NEVER THE FLEET
===========================================
This is the rule and it is not negotiable. The book, DPO failure mode #4 (H2 line
1129):

    "Data quality: Noisy labels poison training. Unlike PPO which averages over
    many samples, DPO memorizes individual pairs."

The verifier fleet has a MEASURED false-positive problem: ``precheck`` compared a
hole's diameter against the plate's THICKNESS -- orthogonal dimensions -- fired 40
times in the pressure run and caused every one of the eight regressions. A
fleet-labelled pair set would contain, as a *chosen* example, the 14b's 8 mm hole
in a brief that demanded 12 mm, and as *rejected*, the correct part. DPO memorises
pairs. We would be teaching the model to reject washers, one gradient step at a
time, and it would learn it perfectly.

So: ``chosen`` and ``rejected`` are decided by :func:`selftrain.ledger.certify` and
by nothing else. The fleet's diagnostics ride along in the record as *data*.

THE REWARD IS ORDINAL, AND SAYS SO
==================================
Pairs need only an ordering. :func:`pair_reward` gives one, lexicographically:
built < applies < gate-passes < envelope-correct < shape-correct. Ties carry no
preference signal and are dropped (``dpo_pairs.preference_pair`` returns ``None``).
The magnitudes are NOT calibrated and must not be read as a value function.

KTO IS THE BETTER FIRST BET
===========================
:func:`kto_records` needs only *unpaired binary* labels -- "this stream is good /
bad" -- which the oracle emits for free on every stream ever generated, including
the ones with no sibling on the same brief. The book (H2 line 2952): KTO is "more
robust than DPO to noise", and we have measured noise. The counts differ sharply:
DPO needs two candidates on one brief; KTO needs one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from harnesscad.agents.selftrain.trajectory import Trajectory
from harnesscad.data.dataengine.preference import dpo_pairs

__all__ = [
    "pair_reward",
    "group_by_brief",
    "build_dpo",
    "build_kto",
    "preference_stats",
    "ROBUST_DPO_NOTE",
]


ROBUST_DPO_NOTE = (
    "If DPO is run, use loss_type='robust' (H2 sec. 6.9.2) with label_smoothing "
    "set to the fleet's MEASURED false-positive rate from "
    "eval/selftest/fleet_audit.py -- HarnessCAD is one of very few systems that "
    "can measure its own epsilon instead of guessing it. But note that this "
    "corpus's labels come from the ORACLE, not the fleet, so the correct epsilon "
    "for THESE pairs is the oracle's error rate, which is bounded below by the "
    "shape metric's size-blindness and is not zero. See selftrain.ledger."
)


def pair_reward(verdict: Dict[str, Any]) -> float:
    """An ORDINAL score over certificates. Used only to order a pair.

    0  did not build / the kernel refused the plan
    1  built and applied, but the gate refused it (malformed, or it betrayed its
       own declared intent)
    2  gate-clean, but the brief's envelope says the part is wrong
    3  envelope-correct, but the shape metric says the geometry is not the
       reference's -- the many-to-one cell
    4  correct on every instrument that can speak
    """
    if not verdict.get("apply_ok"):
        return 0.0
    if not verdict.get("gate_ok"):
        return 1.0
    if not verdict.get("envelope_ok"):
        return 2.0
    if not verdict.get("shape_ok"):
        return 3.0
    return 4.0


def group_by_brief(trajectories: Iterable[Trajectory]
                   ) -> Dict[str, List[Trajectory]]:
    """Candidates sharing a brief. A pair may only ever be formed WITHIN a group."""
    groups: Dict[str, List[Trajectory]] = {}
    for t in trajectories:
        if not t.parse_ok or not t.ops:
            continue
        groups.setdefault(t.brief_id, []).append(t)
    return groups


def _sample(t: Trajectory) -> dict:
    import json
    return {
        "code": json.dumps([dict(o) for o in t.ops], sort_keys=True, indent=2),
        "reward": pair_reward(t.verdict or {}),
        "trajectory_id": t.trajectory_id,
        "model": t.model,
        "loop": t.loop,
        "shape_iou": (t.verdict or {}).get("shape_iou"),
    }


@dataclass
class DPORecord:
    schema: str = "selftrain/dpo/1"
    brief_id: str = ""
    prompt: str = ""
    chosen: str = ""
    rejected: str = ""
    chosen_reward: float = 0.0
    rejected_reward: float = 0.0
    chosen_id: str = ""
    rejected_id: str = ""
    label_source: str = "oracle (gate+envelope+shape); NEVER the verifier fleet"

    def to_dict(self) -> dict:
        return {
            "schema": self.schema, "brief_id": self.brief_id,
            "prompt": self.prompt, "chosen": self.chosen,
            "rejected": self.rejected, "chosen_reward": self.chosen_reward,
            "rejected_reward": self.rejected_reward,
            "chosen_id": self.chosen_id, "rejected_id": self.rejected_id,
            "label_source": self.label_source,
        }


def build_dpo(trajectories: Iterable[Trajectory], *,
              strict: bool = True,
              dedup: bool = True) -> List[DPORecord]:
    """Every ordered pair, within a brief, whose two members the oracle separates.

    ``strict`` (the default) keeps ONLY pairs whose chosen member is fully
    certified (reward 4). A pair of two wrong answers ordered by "less wrong" is a
    preference over failure modes, and training on it teaches the model to prefer
    a well-formed wrong part -- which is precisely the reward-hacking direction the
    ledger warns about. Set ``strict=False`` to keep them and see how many there
    are; the manifest reports both counts.

    Pair construction is delegated to ``dataengine.preference.dpo_pairs``
    (``all_preference_pairs``: every i<j, ordered by reward, ties dropped) so there
    is exactly one implementation of it in the repository.
    """
    out: List[DPORecord] = []
    seen: set = set()
    for brief_id, group in sorted(group_by_brief(trajectories).items()):
        if len(group) < 2:
            continue
        prompt = group[0].prompt or group[0].brief_text
        samples = [_sample(t) for t in group]
        for chosen, rejected in dpo_pairs.all_preference_pairs(samples):
            if strict and chosen["reward"] < 4.0:
                continue
            if dedup:
                key = (brief_id, chosen["code"], rejected["code"])
                if key in seen:
                    continue
                seen.add(key)
            out.append(DPORecord(
                brief_id=brief_id, prompt=prompt,
                chosen=chosen["code"], rejected=rejected["code"],
                chosen_reward=float(chosen["reward"]),
                rejected_reward=float(rejected["reward"]),
                chosen_id=str(chosen["trajectory_id"]),
                rejected_id=str(rejected["trajectory_id"]),
            ))
    return out


@dataclass
class KTORecord:
    schema: str = "selftrain/kto/1"
    trajectory_id: str = ""
    brief_id: str = ""
    prompt: str = ""
    completion: str = ""
    desirable: bool = False
    reward: float = 0.0

    def to_dict(self) -> dict:
        return {
            "schema": self.schema, "trajectory_id": self.trajectory_id,
            "brief_id": self.brief_id, "prompt": self.prompt,
            "completion": self.completion, "desirable": self.desirable,
            "reward": self.reward,
        }


def build_kto(trajectories: Iterable[Trajectory]) -> List[KTORecord]:
    """Unpaired binary labels. Every parsed stream is one record. No pairing."""
    out: List[KTORecord] = []
    for t in trajectories:
        if not t.parse_ok or not t.ops:
            continue
        r = pair_reward(t.verdict or {})
        s = _sample(t)
        out.append(KTORecord(
            trajectory_id=t.trajectory_id, brief_id=t.brief_id,
            prompt=t.prompt or t.brief_text, completion=s["code"],
            desirable=(r >= 4.0), reward=r,
        ))
    return out


def preference_stats(dpo: Sequence[DPORecord],
                     kto: Sequence[KTORecord]) -> Dict[str, Any]:
    desirable = sum(1 for k in kto if k.desirable)
    briefs = {d.brief_id for d in dpo}
    return {
        "dpo_pairs": len(dpo),
        "dpo_briefs_covered": len(briefs),
        "kto_records": len(kto),
        "kto_desirable": desirable,
        "kto_undesirable": len(kto) - desirable,
        "kto_imbalance": (desirable / len(kto)) if kto else 0.0,
        "robust_dpo": ROBUST_DPO_NOTE,
    }
