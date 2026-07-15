"""STAGE 3 -- DPO (Robust). Pairwise preference optimisation, run only if KTO stalls.

The recipe puts DPO third and hedges it (``recipe.RECIPE[2]``): DPO memorises
individual pairs, which makes it the LEAST forgiving method for a corpus with a
MEASURED false-positive problem. So it runs last, on strict pairs (``chosen`` is
fully certified by the conjunction, ``rejected`` is separated by the ordinal
``preference.pair_reward``), with the ROBUST loss.

Two numbers here are NOT free parameters:

* ``loss_type="robust"`` (H2 sec. 6.9.2). Standard DPO assumes noiseless
  preferences; ours are not noiseless, because the shape metric is size-blind
  (the 8mm-for-12mm hole). Robust DPO is the label-noise-aware variant.
* ``label_smoothing`` = the MEASURED oracle error rate, never a guessed constant.
  ``preference.ROBUST_DPO_NOTE`` spells out where it comes from and why zero is
  wrong. This trainer REFUSES to invent one: if the caller does not pass a
  measured epsilon it raises, rather than silently smoothing by a made-up number.

BINDING CONSTRAINT (recipe stop condition): the pair count, not the compute. There
are 66 DPO pairs over 6 briefs in the current corpus; fewer than ~200 separating
pairs is not a DPO dataset, and this module says so in the report rather than
producing a confident adapter off 66 pairs. It does not refuse to run -- the caller
decides -- but the warning rides in the result so no downstream reader mistakes a
66-pair run for a finished stage.

All heavy imports are deferred; the module imports on a core-only machine and its
pure helpers are unit-tested there.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from harnesscad.agents.selftrain.train import require
from harnesscad.agents.selftrain.train.sft import BASE_MODEL, SEED, seed_everything

#: Below this many separating pairs, DPO is memorising, not generalising. Stated by
#: the recipe; enforced as a WARNING in the result, not a hard refusal.
MIN_PAIRS_FOR_DPO = 200


@dataclass
class DPOResult:
    output_dir: str = ""
    base_model: str = ""
    init_adapter: Optional[str] = None
    records: int = 0
    briefs: int = 0
    beta: float = 0.1
    loss_type: str = "robust"
    label_smoothing: float = 0.0
    steps: int = 0
    train_runtime_s: float = 0.0
    loss_curve: List[Dict[str, float]] = field(default_factory=list)
    final_loss: Optional[float] = None
    peak_vram_gb: Optional[float] = None
    pair_count_ok: bool = True
    warnings: List[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return {"stage": "dpo-robust", "output_dir": self.output_dir,
                "base_model": self.base_model, "init_adapter": self.init_adapter,
                "records": self.records, "briefs": self.briefs, "beta": self.beta,
                "loss_type": self.loss_type, "label_smoothing": self.label_smoothing,
                "steps": self.steps,
                "train_runtime_s": round(self.train_runtime_s, 1),
                "loss_curve": self.loss_curve, "final_loss": self.final_loss,
                "peak_vram_gb": self.peak_vram_gb,
                "pair_count_ok": self.pair_count_ok, "warnings": self.warnings,
                "note": self.note}


def train_dpo(data_path: str, output_dir: str, *,
              label_smoothing: float,
              model_name: str = BASE_MODEL,
              init_adapter: Optional[str] = None,
              epochs: int = 1, lr: float = 5e-6, beta: float = 0.1,
              loss_type: str = "robust",
              max_length: int = 1024, grad_accum: int = 16) -> DPOResult:
    """Run stage-3 robust DPO, continuing from the stage-1 adapter if given.

    ``label_smoothing`` is REQUIRED and must be the measured oracle error rate
    (``preference.ROBUST_DPO_NOTE``); passing a guess defeats the point of the
    robust loss. It must lie in ``[0, 0.5)`` -- DPO's label smoothing is a
    flip-probability, and >= 0.5 inverts the preference.
    """
    require()
    if not (0.0 <= label_smoothing < 0.5):
        raise ValueError(
            "label_smoothing must be the MEASURED oracle error rate in [0, 0.5); "
            "got %r. See selftrain.preference.ROBUST_DPO_NOTE -- do not guess it."
            % label_smoothing)

    import torch
    from peft import get_peft_model, prepare_model_for_kbit_training, PeftModel
    from transformers import TrainerCallback
    from trl import DPOConfig, DPOTrainer

    from harnesscad.agents.selftrain.train import data as data_mod
    from harnesscad.agents.selftrain.train.sft import load_base, _lora_config

    seed_everything()
    os.makedirs(output_dir, exist_ok=True)

    model, tok = load_base(model_name, quantized=True)
    model = prepare_model_for_kbit_training(model)
    if init_adapter:
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, _lora_config())
    model.config.use_cache = False

    ds, stats = data_mod.dpo_dataset(data_path, tok)

    warnings: List[str] = []
    pair_ok = stats.records >= MIN_PAIRS_FOR_DPO
    if not pair_ok:
        warnings.append(
            "only %d separating pairs (< %d): the recipe's binding constraint. "
            "This adapter is a memorisation risk, not a finished DPO stage."
            % (stats.records, MIN_PAIRS_FOR_DPO))

    cfg = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        beta=beta,
        loss_type=loss_type,
        label_smoothing=label_smoothing,
        max_length=max_length,
        max_prompt_length=max_length // 2,
        report_to=[],
        seed=SEED,
    )

    loss_curve: List[Dict[str, float]] = []

    class _LossCB(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                loss_curve.append({"step": int(state.global_step),
                                   "loss": float(logs["loss"]),
                                   "epoch": float(logs.get("epoch", 0.0))})

    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, callbacks=[_LossCB()])

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    out = trainer.train()
    peak = (torch.cuda.max_memory_allocated() / 1e9
            if torch.cuda.is_available() else None)

    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)

    res = DPOResult(
        output_dir=output_dir, base_model=model_name, init_adapter=init_adapter,
        records=stats.records, briefs=stats.briefs, beta=beta,
        loss_type=loss_type, label_smoothing=label_smoothing,
        steps=int(trainer.state.global_step),
        train_runtime_s=float(out.metrics.get("train_runtime", 0.0)),
        loss_curve=loss_curve,
        final_loss=(loss_curve[-1]["loss"] if loss_curve else None),
        peak_vram_gb=(round(peak, 2) if peak else None),
        pair_count_ok=pair_ok, warnings=warnings,
        note="robust DPO; eps=%.4f (measured oracle error); continued from %s; "
             "seed=%d" % (label_smoothing, init_adapter or "(fresh LoRA)", SEED),
    )
    with open(os.path.join(output_dir, "dpo_result.json"), "w", encoding="utf-8") as fh:
        json.dump(res.to_dict(), fh, indent=2)
    return res


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Stage-3 robust DPO trainer")
    ap.add_argument("--data", default="assets/selftrain/dpo.jsonl")
    ap.add_argument("--out", default="assets/selftrain/adapters/dpo")
    ap.add_argument("--init-adapter", default=None)
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--label-smoothing", type=float, required=True,
                    help="the MEASURED oracle error rate; see preference.ROBUST_DPO_NOTE")
    args = ap.parse_args(argv)
    res = train_dpo(args.data, args.out, model_name=args.model,
                    init_adapter=args.init_adapter, epochs=args.epochs,
                    label_smoothing=args.label_smoothing)
    print(json.dumps(res.to_dict(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
