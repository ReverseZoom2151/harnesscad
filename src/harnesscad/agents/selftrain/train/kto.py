"""STAGE 2 -- KTO. Unpaired binary preference optimisation on oracle labels.

The recipe orders KTO before DPO for a stated reason (H2 sec. 6.9): KTO needs only
an unpaired good/bad bit, which the oracle emits for EVERY stream, where DPO needs
two candidates on one brief that the oracle separates -- and most briefs have no
such pair. KTO is also, per the book, "more robust than DPO to noise", and these
labels have measured noise: the shape metric is size-blind (the 8mm-for-12mm hole).

Config from ``recipe.RECIPE[1]``: continue the stage-1 QLoRA adapter, beta 0.1,
1 epoch, lr 5e-6, ``desirable_weight`` set to balance the measured good/bad
imbalance (208 records, 42 desirable / 166 undesirable -> up-weight the rare
desirable class so the loss is not dominated by "reject").

STOP CONDITION (recipe, enforced by the caller/report, not silently here): any drop
in gate-pass rate versus the stage-1 model means KTO is producing MORE malformed
geometry and must be discarded.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from harnesscad.agents.selftrain.train import require
from harnesscad.agents.selftrain.train.sft import BASE_MODEL, SEED, seed_everything


@dataclass
class KTOResult:
    output_dir: str = ""
    base_model: str = ""
    init_adapter: Optional[str] = None
    records: int = 0
    desirable: int = 0
    undesirable: int = 0
    desirable_weight: float = 1.0
    undesirable_weight: float = 1.0
    steps: int = 0
    train_runtime_s: float = 0.0
    loss_curve: List[Dict[str, float]] = field(default_factory=list)
    final_loss: Optional[float] = None
    peak_vram_gb: Optional[float] = None
    note: str = ""

    def to_dict(self) -> dict:
        return {"stage": "kto", "output_dir": self.output_dir,
                "base_model": self.base_model, "init_adapter": self.init_adapter,
                "records": self.records, "desirable": self.desirable,
                "undesirable": self.undesirable,
                "desirable_weight": self.desirable_weight,
                "undesirable_weight": self.undesirable_weight,
                "steps": self.steps,
                "train_runtime_s": round(self.train_runtime_s, 1),
                "loss_curve": self.loss_curve, "final_loss": self.final_loss,
                "peak_vram_gb": self.peak_vram_gb, "note": self.note}


def _balanced_weights(des: int, undes: int) -> tuple:
    """desirable_weight / undesirable_weight so the two classes contribute equally.

    TRL recommends desirable_weight*n_des ~ undesirable_weight*n_undes, within [1,1.5]
    of each other. We fix undesirable_weight=1.0 and lift desirable_weight to the
    ratio, capped so it stays in TRL's sane band.
    """
    if des == 0:
        return 1.0, 1.0
    ratio = undes / des
    dw = max(1.0, min(ratio, 1.5 * ratio))     # keep it the true ratio, >=1
    return round(dw, 3), 1.0


def train_kto(data_path: str, output_dir: str, *,
              model_name: str = BASE_MODEL,
              init_adapter: Optional[str] = None,
              epochs: int = 1, lr: float = 5e-6, beta: float = 0.1,
              max_length: int = 1024, grad_accum: int = 16) -> KTOResult:
    """Run stage-2 KTO, continuing from the stage-1 adapter if given."""
    require()
    import torch
    from peft import get_peft_model, prepare_model_for_kbit_training, PeftModel
    from transformers import TrainerCallback
    from trl import KTOConfig, KTOTrainer

    from harnesscad.agents.selftrain.train import data as data_mod
    from harnesscad.agents.selftrain.train.sft import load_base, _lora_config

    seed_everything()
    os.makedirs(output_dir, exist_ok=True)

    model, tok = load_base(model_name, quantized=True)
    model = prepare_model_for_kbit_training(model)
    if init_adapter:
        # Continue the stage-1 adapter, trainable.
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, _lora_config())
    model.config.use_cache = False

    ds, stats = data_mod.kto_dataset(data_path, tok)
    dw, uw = _balanced_weights(stats.desirable or 0, stats.undesirable or 0)

    cfg = KTOConfig(
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
        desirable_weight=dw,
        undesirable_weight=uw,
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

    trainer = KTOTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, callbacks=[_LossCB()])

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    out = trainer.train()
    peak = (torch.cuda.max_memory_allocated() / 1e9
            if torch.cuda.is_available() else None)

    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)

    res = KTOResult(
        output_dir=output_dir, base_model=model_name, init_adapter=init_adapter,
        records=stats.records, desirable=stats.desirable or 0,
        undesirable=stats.undesirable or 0, desirable_weight=dw, undesirable_weight=uw,
        steps=int(trainer.state.global_step),
        train_runtime_s=float(out.metrics.get("train_runtime", 0.0)),
        loss_curve=loss_curve,
        final_loss=(loss_curve[-1]["loss"] if loss_curve else None),
        peak_vram_gb=(round(peak, 2) if peak else None),
        note="beta=%.2f; continued from %s; seed=%d" % (
            beta, init_adapter or "(fresh LoRA)", SEED),
    )
    with open(os.path.join(output_dir, "kto_result.json"), "w", encoding="utf-8") as fh:
        json.dump(res.to_dict(), fh, indent=2)
    return res


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Stage-2 KTO trainer")
    ap.add_argument("--data", default="assets/selftrain/kto.jsonl")
    ap.add_argument("--out", default="assets/selftrain/adapters/kto")
    ap.add_argument("--init-adapter", default=None)
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args(argv)
    res = train_kto(args.data, args.out, model_name=args.model,
                    init_adapter=args.init_adapter, epochs=args.epochs)
    print(json.dumps(res.to_dict(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
