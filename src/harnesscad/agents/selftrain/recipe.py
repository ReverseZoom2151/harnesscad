"""The training recipe and its cost. Stated, not run.

The instruction was explicit: produce the corpus and the numbers, state the
recipe, and STOP. Building the dataset is the load-bearing part and it is useful
even if we never train -- a graded trajectory corpus is an eval set, a regression
suite and a debugging record before it is ever a gradient.

So this module is a *declaration*. It imports no torch, adds no dependency, and
runs nothing. ``pyproject.toml`` still says ``dependencies = []`` and that is a
feature until the day it is not.

WHY THESE NUMBERS
=================
The book's go/no-go gate before spending a dollar on RL (H2 sec. 10.6): "If
pass@1 < 5%, RL will likely fail. If pass@k < 20%, RL will struggle." The pressure
run's own data answers it: qwen2.5-coder:14b blind pass@1 = **66.7%**; pooled
across the ladder, **33.3%**. That is inside the Goldilocks band GRPO wants
(20-80%, H2 sec. 7.4) and an order of magnitude above the floor. **The pressure
report is, unintentionally, an RL green light.**

The ORDER is the book's, and it is not negotiable either (H2 sec. 3.3 of this
package's audit): RFT first, then KTO/DPO, and GRPO only if the first two move
pass@1 by 10+ points. GRPO costs 100x more and, per H2 sec. 8.5.4, has to beat
Best-of-N at equal compute before it is worth anything -- and Best-of-N with the
differential oracle as the selector is an arm the harness can run today for the
price of inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

__all__ = ["Stage", "RECIPE", "total_cost", "PREREQUISITES", "format_recipe"]


@dataclass(frozen=True)
class Stage:
    """One training stage, its config, and what it costs to find out."""

    order: int
    name: str
    method: str
    base_model: str
    config: Dict[str, object]
    dataset: str
    gpu_hours: float
    usd: float
    rationale: str
    stop_condition: str

    def to_dict(self) -> dict:
        d = {
            "order": self.order, "name": self.name, "method": self.method,
            "base_model": self.base_model, "config": dict(self.config),
            "dataset": self.dataset, "gpu_hours": self.gpu_hours,
            "usd": self.usd, "rationale": self.rationale,
            "stop_condition": self.stop_condition,
        }
        return d


#: Things that must be TRUE before stage 1 runs. Every one of them is a lesson
#: from the pressure run, and every one of them is cheaper than a GPU-hour.
PREREQUISITES: Tuple[str, ...] = (
    "eval/selftest/fleet_audit.py false_positive_rate == 0, enforced in CI. A "
    "model trained against a blind oracle learns to exploit the blindness; a "
    "model trained on FLEET labels learns to reject washers. The fleet is not "
    "the labeller here, but it is still the loop's feedback channel.",
    "A HELD-OUT brief corpus written by a different hand (eval/corpus/). Training "
    "and evaluating on the same 12 briefs measures memorisation. There is "
    "currently no held-out set and any pass@1 gain on these briefs means nothing "
    "without one.",
    "The Best-of-N / oracle-selection arm, run at matched compute (H2 sec. 8.5.4: "
    "'Always compare your RL method against Best-of-N with the same compute "
    "budget'). If Oracle-BoN at N=3 already lands near 70%, a fine-tune that "
    "reaches 50% is a regression dressed as a result.",
    "A shape metric that can resolve a defect smaller than 2% of part volume. The "
    "current IoU scores an 8mm-for-12mm hole at 0.963 and calls it a match; an "
    "RFT set filtered by it will contain wrong parts. Chamfer or a feature-level "
    "diff would close this and both modules exist unused "
    "(eval/bench/geometry/chamfer.py).",
)


RECIPE: Tuple[Stage, ...] = (
    Stage(
        order=1,
        name="RFT / STaR",
        method="Supervised fine-tuning on oracle-certified op streams (QLoRA)",
        base_model="Qwen2.5-Coder-7B-Instruct",
        config={
            "peft": "QLoRA, 4-bit NF4, r=16, alpha=32, dropout=0.05",
            "target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
            "epochs": 3,
            "lr": 1e-4,
            "scheduler": "cosine, 3% warmup",
            "batch": "1 x grad_accum 16 (effective 16)",
            "max_seq_len": 2048,
            "completion_only_masking": True,
            "library": "trl.SFTTrainer + peft",
            "precision": "bf16",
        },
        dataset="assets/selftrain/rft.jsonl (policy=full)",
        gpu_hours=1.0,
        usd=2.0,
        rationale=(
            "The book ranks it first: cheapest, most robust use of a verifiable "
            "reward. The corpus is FREE -- it is the exhaust of an experiment "
            "already run. One 7B QLoRA epoch over a few hundred short op streams "
            "is under an hour on a 24 GB card."),
        stop_condition=(
            "pass@1 on the HELD-OUT corpus does not beat the base model by >=5 "
            "points -> stop. The dataset is too small or too narrow, and no "
            "amount of RL will fix a distribution problem."),
    ),
    Stage(
        order=2,
        name="KTO",
        method="Unpaired binary preference optimisation on oracle labels",
        base_model="the stage-1 RFT adapter",
        config={
            "peft": "same QLoRA adapter, continued",
            "beta": 0.1,
            "desirable_weight": "set to balance the measured good/bad imbalance",
            "epochs": 1,
            "lr": 5e-6,
            "library": "trl.KTOTrainer",
        },
        dataset="assets/selftrain/kto.jsonl",
        gpu_hours=1.0,
        usd=2.0,
        rationale=(
            "KTO before DPO. It needs only unpaired binary labels, which the "
            "oracle emits for EVERY stream (DPO needs two candidates on one "
            "brief and most briefs do not have a separating pair). The book: "
            "KTO is 'more robust than DPO to noise', and our labels have "
            "measured noise -- the shape metric is size-blind."),
        stop_condition=(
            "Any drop in gate-pass rate -> stop. A preference method that makes "
            "the model produce MORE malformed geometry is optimising the wrong "
            "thing."),
    ),
    Stage(
        order=3,
        name="DPO (Robust)",
        method="Pairwise preference optimisation, label-smoothed",
        base_model="the stage-1 RFT adapter",
        config={
            "beta": 0.1,
            "loss_type": "robust",
            "label_smoothing": "epsilon = the MEASURED oracle error rate, not a guess",
            "epochs": 1,
            "lr": 5e-6,
            "library": "trl.DPOTrainer",
        },
        dataset="assets/selftrain/dpo.jsonl (strict: chosen is fully certified)",
        gpu_hours=1.0,
        usd=2.0,
        rationale=(
            "Only if KTO underperforms. DPO memorises individual pairs, which "
            "makes it the LEAST forgiving method for a system with a measured "
            "false-positive problem -- so it runs third, on strict pairs, with "
            "robust loss."),
        stop_condition=(
            "The pair count is the binding constraint, not the compute. Fewer "
            "than ~200 separating pairs is not a DPO dataset."),
    ),
    Stage(
        order=4,
        name="GRPO",
        method="Group-relative policy optimisation with the oracle as R",
        base_model="the best adapter from stages 1-3",
        config={
            "group_size": 8,
            "reward": "R = 1.0 if ledger.certify(brief, ops).accepted else 0.0, "
                      "+ 0.1 * tool_reward.format_reward, "
                      "+ beta * mean(divergence.step_rewards)",
            "clip": "DAPO Clip-Higher (data/dataengine/reward/asymmetric_clip.py, "
                    "already implemented, zero callers)",
            "advantage": "group-relative (data/dataengine/reward/expert_advantage.py, "
                         "already implemented, zero callers)",
            "generation": "vLLM (60-70% of RLHF wall-clock is generation)",
            "steps": 3000,
            "library": "trl.GRPOTrainer",
        },
        dataset="the brief corpus, filtered to a 20-80% pass rate (Goldilocks)",
        gpu_hours=336.0,           # ~2 A100-weeks
        usd=4000.0,
        rationale=(
            "The book's decision tree resolves in one step: 'Do you have "
            "verifiable rewards? -> GRPO'. Yes: six engines. Three of DAPO's five "
            "components are already written in this repo as orphaned pure-Python "
            "formulas. What is missing is a policy, a gradient and vLLM."),
        stop_condition=(
            "DO NOT START HERE. If RFT + KTO on the oracle do not move pass@1 by "
            "10+ points, GRPO will not save it -- and it costs 300x more. Also, "
            "per H2 sec. 8.5.4, it must beat Oracle-Best-of-N at equal compute, "
            "and that arm has not been run."),
    ),
)


def total_cost(stages: Tuple[Stage, ...] = RECIPE,
               through: int = 3) -> Dict[str, float]:
    """Cost of running the recipe through stage ``through`` (default: not GRPO)."""
    kept = [s for s in stages if s.order <= through]
    return {
        "stages": float(len(kept)),
        "gpu_hours": sum(s.gpu_hours for s in kept),
        "usd": sum(s.usd for s in kept),
    }


def format_recipe(stages: Tuple[Stage, ...] = RECIPE) -> str:
    """A plain-text rendering. No emoji, no colour, no clock."""
    lines: List[str] = ["PREREQUISITES (all cheaper than a GPU-hour):"]
    for i, p in enumerate(PREREQUISITES, 1):
        lines.append("  %d. %s" % (i, p))
    lines.append("")
    for s in stages:
        lines.append("STAGE %d: %s -- %s" % (s.order, s.name, s.method))
        lines.append("  base:    %s" % s.base_model)
        lines.append("  data:    %s" % s.dataset)
        lines.append("  cost:    %.0f GPU-hour(s), ~$%.0f" % (s.gpu_hours, s.usd))
        for k, v in s.config.items():
            lines.append("  %-12s %s" % (k + ":", v))
        lines.append("  why:     %s" % s.rationale)
        lines.append("  STOP IF: %s" % s.stop_condition)
        lines.append("")
    c = total_cost()
    lines.append("Stages 1-3 total: %.0f GPU-hours, ~$%.0f."
                 % (c["gpu_hours"], c["usd"]))
    return "\n".join(lines)
