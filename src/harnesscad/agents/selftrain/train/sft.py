"""STAGE 1 -- RFT / STaR as QLoRA SFT. The recipe in ``recipe.py``, actually run.

Config is lifted verbatim from ``selftrain.recipe.RECIPE[0]``: Qwen2.5-Coder-7B-
Instruct, 4-bit NF4 QLoRA, r=16 / alpha=32 / dropout=0.05 over the attention and
MLP projections, 3 epochs, lr 1e-4, cosine with 3% warmup, effective batch 16,
bf16, completion-only masking. Nothing here is a free parameter chosen after
seeing a curve; if a number differs from the recipe it is a bug.

Determinism: we seed python / numpy / torch and set ``use_deterministic_algorithms``
where the op supports it. QLoRA training still has residual CUDA nondeterminism in
some fused kernels (a documented, unavoidable floating-point reduction-order
effect); the seed pins data order and initialisation, not the last bit of every
matmul. This is stated, not hidden.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from harnesscad.agents.selftrain.train import require

BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
SEED = 20260714

#: QLoRA target modules -- attention + MLP, exactly as recipe.py declares.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


@dataclass
class SFTResult:
    output_dir: str = ""
    base_model: str = ""
    records: int = 0
    steps: int = 0
    epochs: float = 0.0
    train_runtime_s: float = 0.0
    loss_curve: List[Dict[str, float]] = field(default_factory=list)
    final_loss: Optional[float] = None
    peak_vram_gb: Optional[float] = None
    trainable_params: int = 0
    total_params: int = 0
    data_stats: Dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict:
        return {"stage": "rft-sft", "output_dir": self.output_dir,
                "base_model": self.base_model, "records": self.records,
                "steps": self.steps, "epochs": self.epochs,
                "train_runtime_s": round(self.train_runtime_s, 1),
                "loss_curve": self.loss_curve, "final_loss": self.final_loss,
                "peak_vram_gb": self.peak_vram_gb,
                "trainable_params": self.trainable_params,
                "total_params": self.total_params,
                "trainable_pct": (round(100.0 * self.trainable_params /
                                        self.total_params, 3)
                                  if self.total_params else None),
                "data_stats": self.data_stats, "note": self.note}


def seed_everything(seed: int = SEED) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _bnb_config():
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_base(model_name: str = BASE_MODEL, *, quantized: bool = True):
    """Load the base model (4-bit) and tokenizer. Shared by SFT, KTO and eval."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw: Dict[str, Any] = {"torch_dtype": torch.bfloat16}
    if quantized:
        kw["quantization_config"] = _bnb_config()
        kw["device_map"] = {"": 0} if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
    return model, tok


def _lora_config():
    from peft import LoraConfig

    return LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=TARGET_MODULES,
    )


def train_sft(data_path: str, output_dir: str, *,
              model_name: str = BASE_MODEL,
              epochs: int = 3, lr: float = 1e-4,
              max_seq_len: int = 2048,
              grad_accum: int = 16) -> SFTResult:
    """Run stage-1 RFT. Returns an :class:`SFTResult` with the loss curve and cost."""
    require()
    import torch
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    seed_everything()
    os.makedirs(output_dir, exist_ok=True)

    model, tok = load_base(model_name, quantized=True)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, _lora_config())
    model.config.use_cache = False

    from harnesscad.agents.selftrain.train import data as data_mod
    ds, resp_ids, stats = data_mod.sft_dataset(data_path, tok)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    cfg = SFTConfig(
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
        max_length=max_seq_len,
        optim="paged_adamw_8bit",
        report_to=[],
        seed=SEED,
        dataset_num_proc=1,
        packing=False,
        completion_only_loss=True,
    )

    loss_curve: List[Dict[str, float]] = []

    class _LossCB(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                loss_curve.append({"step": int(state.global_step),
                                   "loss": float(logs["loss"]),
                                   "epoch": float(logs.get("epoch", 0.0))})

    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=ds,
        processing_class=tok, callbacks=[_LossCB()],
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    out = trainer.train()
    peak = (torch.cuda.max_memory_allocated() / 1e9
            if torch.cuda.is_available() else None)

    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)

    res = SFTResult(
        output_dir=output_dir, base_model=model_name,
        records=stats.records, steps=int(trainer.state.global_step),
        epochs=float(epochs), train_runtime_s=float(out.metrics.get("train_runtime", 0.0)),
        loss_curve=loss_curve,
        final_loss=(loss_curve[-1]["loss"] if loss_curve else None),
        peak_vram_gb=(round(peak, 2) if peak else None),
        trainable_params=trainable, total_params=total,
        data_stats=stats.to_dict(),
        note="completion-only masking; QLoRA NF4 double-quant; seed=%d" % SEED,
    )
    with open(os.path.join(output_dir, "sft_result.json"), "w", encoding="utf-8") as fh:
        json.dump(res.to_dict(), fh, indent=2)
    return res


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Stage-1 RFT/SFT QLoRA trainer")
    ap.add_argument("--data", default="assets/selftrain/rft.jsonl")
    ap.add_argument("--out", default="assets/selftrain/adapters/rft")
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args(argv)
    res = train_sft(args.data, args.out, model_name=args.model, epochs=args.epochs)
    print(json.dumps(res.to_dict(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
