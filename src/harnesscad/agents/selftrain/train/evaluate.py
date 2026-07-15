"""THE POINT: fine-tune vs base vs Oracle-Best-of-N, on briefs we did not train on.

The book's instruction is in bold and it is not optional (H2 sec. 8.5.4): *"Always
compare your RL method against Best-of-N with the same compute budget."* A
fine-tune that reaches 50% is a regression if un-tuned Best-of-N at 3 samples
already reaches 70%. So this module runs three arms through ONE generation code
path on ONE base model, so the only thing that differs is the thing under test:

  base@1        the un-tuned base, greedy, one draw.
  finetune@1    base + the LoRA adapter, greedy, one draw. Same decode, same prompt.
  oracle-bon@N  the un-tuned base, N draws at T>0, a REFERENCE-FREE selector
                (the output gate -- the only oracle available on a user's brief)
                picks one. Costs Nx the inference of finetune@1; that is the
                "matched compute" the comparison is about.

Every arm is graded by :func:`ledger.certify` -- the conjunction (gate AND envelope
AND shape), the same labeller the training data was built with. The eval briefs are
the 16 pressure briefs the corpus NEVER trained on (``HELDOUT_BRIEFS``); a gain on
the 12 training briefs would be memorisation and is worth nothing. Where an op
translator exists, the fully-different-hand ``eval/corpus`` split can also be run.

Statistics, never bare percentages: Wilson score intervals on each rate, and
McNemar's exact test on the paired per-brief outcomes between arms.

The base model is loaded through ``sft.load_base`` (HF, 4-bit) rather than Ollama's
GGUF, because the fine-tune's adapter sits on THAT model and a fair base-vs-tuned
comparison must hold the base weights fixed. Ollama is a different quantisation and
would confound the two variables.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: The 16 pressure briefs held out of training. Same grader, zero op-vocabulary
#: mismatch, never sampled into the RFT/KTO/DPO sets.
HELDOUT_BRIEFS: Tuple[str, ...] = (
    "bar_100x10x10", "chamfer_plate_2mm", "disc_bore", "disc_d30_h8",
    "fillet_block_5mm", "fillet_plate_3mm", "flange_thick", "plate_60x40x5",
    "plate_hole_centre", "plate_hole_offcentre", "plate_square_25",
    "plate_thin_80x50x2", "shell_deep_4mm", "shell_tray_2mm", "slotted_block",
    "spacer_bore",
)

Solver = Callable[[str], List[dict]]


# --------------------------------------------------------------------------- #
# statistics -- Wilson interval and McNemar's exact test, pure python
# --------------------------------------------------------------------------- #
def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Wilson score interval for a binomial proportion. Returns (p, lo, hi)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact (binomial) McNemar p-value on discordant pairs (b, c).

    ``b`` = arm-A-right/arm-B-wrong, ``c`` = arm-A-wrong/arm-B-right. Exact test
    is used because discordant counts here are tiny and the chi-square
    approximation is invalid at this scale.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)

    def binom(n_, k_):
        return math.comb(n_, k_)

    tail = sum(binom(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


# --------------------------------------------------------------------------- #
# grading one arm over the held-out briefs
# --------------------------------------------------------------------------- #
@dataclass
class ArmResult:
    name: str = ""
    n: int = 0
    accepted: int = 0
    gate_pass: int = 0
    envelope_pass: int = 0
    shape_pass: int = 0
    parse_fail: int = 0
    per_brief: Dict[str, bool] = field(default_factory=dict)
    detail: Dict[str, dict] = field(default_factory=dict)
    generations: int = 0            # total model calls -- the compute meter

    @property
    def rate(self) -> float:
        return self.accepted / self.n if self.n else 0.0

    def to_dict(self) -> dict:
        p, lo, hi = wilson(self.accepted, self.n)
        return {"arm": self.name, "n": self.n, "accepted": self.accepted,
                "pass@1_or_selected": round(self.rate, 4),
                "wilson95": [round(lo, 4), round(hi, 4)],
                "gate_pass": self.gate_pass, "envelope_pass": self.envelope_pass,
                "shape_pass": self.shape_pass, "parse_fail": self.parse_fail,
                "generations": self.generations,
                "per_brief": dict(sorted(self.per_brief.items()))}


def grade_solver(name: str, solver: Solver,
                 brief_ids: Sequence[str] = HELDOUT_BRIEFS,
                 gens_per_brief: int = 1) -> ArmResult:
    """Run ``solver`` on each brief, grade with ``ledger.certify`` (full conjunction)."""
    from harnesscad.eval.pressure import briefs as B
    from harnesscad.agents.selftrain import ledger

    res = ArmResult(name=name, n=len(brief_ids))
    for bid in brief_ids:
        br = B.brief_by_id(bid)
        try:
            ops = solver(br.text)
        except Exception as exc:  # noqa: BLE001
            res.per_brief[bid] = False
            res.detail[bid] = {"error": str(exc)}
            res.parse_fail += 1
            res.generations += gens_per_brief
            continue
        res.generations += gens_per_brief
        if not ops:
            res.per_brief[bid] = False
            res.parse_fail += 1
            res.detail[bid] = {"parse": "empty"}
            continue
        cert = ledger.certify(br, ops)
        res.per_brief[bid] = bool(cert.accepted)
        res.gate_pass += int(cert.gate_ok)
        res.envelope_pass += int(cert.envelope_ok)
        res.shape_pass += int(cert.shape_ok)
        res.accepted += int(cert.accepted)
        res.detail[bid] = {"accepted": cert.accepted, "gate": cert.gate_ok,
                           "envelope": cert.envelope_ok, "shape": cert.shape_ok,
                           "iou": cert.shape_iou}
    return res


def compare(a: ArmResult, b: ArmResult) -> dict:
    """McNemar on the paired per-brief outcomes of two arms."""
    both = sorted(set(a.per_brief) & set(b.per_brief))
    a_only = sum(1 for k in both if a.per_brief[k] and not b.per_brief[k])
    b_only = sum(1 for k in both if b.per_brief[k] and not a.per_brief[k])
    agree_pass = sum(1 for k in both if a.per_brief[k] and b.per_brief[k])
    agree_fail = sum(1 for k in both if not a.per_brief[k] and not b.per_brief[k])
    return {"arm_a": a.name, "arm_b": b.name,
            "a_wins": a_only, "b_wins": b_only,
            "both_pass": agree_pass, "both_fail": agree_fail,
            "mcnemar_p_exact": round(mcnemar_exact(a_only, b_only), 4),
            "delta_rate": round(a.rate - b.rate, 4)}


# --------------------------------------------------------------------------- #
# solvers
# --------------------------------------------------------------------------- #
class HFGenerator:
    """One base model, optionally with a LoRA adapter, behind a brief->ops call.

    Loading the base once and swapping the adapter would be ideal; simpler and
    safe here is one instance per arm. Greedy decoding for the @1 arms (deterministic
    given the weights); sampled for Best-of-N.
    """

    def __init__(self, base_model: str, adapter: Optional[str] = None,
                 max_new_tokens: int = 768) -> None:
        from harnesscad.agents.selftrain.train import require, sft
        require()
        import torch  # noqa: F401
        self.model, self.tok = sft.load_base(base_model, quantized=True)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def _generate(self, brief_text: str, *, do_sample: bool, temperature: float,
                  seed: Optional[int]) -> str:
        import torch
        from harnesscad.agents.selftrain.train import data as data_mod

        if seed is not None:
            torch.manual_seed(seed)
        msgs = data_mod.messages_for(brief_text)
        prompt = self.tok.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=(temperature if do_sample else None),
                top_p=(0.95 if do_sample else None),
                pad_token_id=self.tok.pad_token_id)
        text = self.tok.decode(out[0][enc["input_ids"].shape[1]:],
                               skip_special_tokens=True)
        return text

    def solve_greedy(self, brief_text: str) -> List[dict]:
        from harnesscad.eval.pressure import model as pmodel
        from harnesscad.agents.selftrain.train.generate import strip_think
        raw = self._generate(brief_text, do_sample=False, temperature=1.0, seed=None)
        parsed = pmodel.extract_ops(strip_think(raw))
        return pmodel.ops_to_dicts(parsed) if parsed.ok else []

    def draws(self, brief_text: str, n: int, temperature: float = 0.8,
              base_seed: int = 20260714) -> List[List[dict]]:
        from harnesscad.eval.pressure import model as pmodel
        from harnesscad.agents.selftrain.train.generate import strip_think
        out: List[List[dict]] = []
        for i in range(n):
            raw = self._generate(brief_text, do_sample=True,
                                 temperature=temperature, seed=base_seed + i)
            parsed = pmodel.extract_ops(strip_think(raw))
            out.append(pmodel.ops_to_dicts(parsed) if parsed.ok else [])
        return out


def best_of_n_solver(gen: "HFGenerator", n: int, temperature: float = 0.8):
    """A brief->ops solver that draws N and returns the gate-selected candidate.

    The selector is the output gate -- REFERENCE-FREE, the only oracle available on
    a brief a user typed (envelope and shape need a hand-written answer key). The
    gate accepts ~76% of streams (it is many-to-one; see ledger.py), so a
    gate-selected Best-of-N will sometimes pick a well-formed WRONG part. That is the
    honest production selector, and reporting it is the point.
    """
    from harnesscad.eval.pressure import metrics as metrics_mod
    from harnesscad.eval.pressure import briefs as B

    # Map brief text back to a Brief so the gate can replay the ops.
    text_to_brief = {B.brief_by_id(b).text: B.brief_by_id(b) for b in
                     [x.id for x in B.BRIEFS]}

    def solve(brief_text: str) -> List[dict]:
        candidates = gen.draws(brief_text, n, temperature=temperature)
        brief = text_to_brief.get(brief_text)
        best = []
        for ops in candidates:
            if not ops:
                continue
            if brief is None:
                return ops
            g = metrics_mod.grade(brief, list(ops), shape=False)
            if g.gate_ok and g.apply_ok:
                return ops           # first gate-passing candidate wins
            if not best:
                best = ops
        return best

    return solve


def ollama_solver(model: str, seed: int = 20260714, temperature: float = 0.0):
    """A brief->ops solver backed by an Ollama model (used for reference points)."""
    from harnesscad.eval.pressure import prompts, model as pmodel
    from harnesscad.agents.selftrain.train.generate import strip_think

    def solve(brief_text: str) -> List[dict]:
        cli = pmodel.OllamaClient(model, seed=seed, temperature=temperature)
        msgs = [{"role": "system", "content": prompts.SYSTEM_PROMPT},
                {"role": "user", "content": prompts.user_prompt(brief_text)}]
        raw = cli.complete(msgs, attempt=1, seed=seed, temperature=temperature)
        parsed = pmodel.extract_ops(strip_think(raw))
        return pmodel.ops_to_dicts(parsed) if parsed.ok else []

    return solve
