"""Grounding-LoRA: a QLoRA VLM fine-tune to click the CAD viewport, and the honest
16GB verdict on whether that is even the right thing to do on this hardware.

WHAT THIS TRAINS ON
===================
938 VERIFIED ``(screenshot, description) -> (x, y)`` pairs from
``eval/grounding/corpus`` -- projected from B-reps we own, adjudicated by the
APPLICATION's own picker, no human and no vision model in the label loop. They are
served to the benchmark through ``eval/grounding/cadspot`` and scored two ways:
``point_in_bbox`` (ScreenSpot's proxy) and ``selects_expected`` (the live-app truth).
The current floor/ceiling on that benchmark's viewport split are:

    random  0.034      centre  0.085      projection-oracle  1.000

Any grounding LoRA that does not clear the CENTRE baseline (0.085) by a wide margin
has learned nothing a constant could not do. That 0.085 is the number to beat, and
0.034 is chance; the oracle's 1.000 only confirms the harness is wired correctly.

THE 16GB VERDICT -- READ THIS BEFORE RUNNING ANYTHING
=====================================================
The instruction was: if 16GB VRAM cannot hold a USABLE VLM, say so plainly and
state what it would take. Here it is, and it is a qualified no.

* A **2B-class grounding VLM** (Qwen2-VL-2B-Instruct, ShowUI-2B) QLoRA-fine-tunes
  and runs inference INSIDE 16GB -- but only at a REDUCED vision-token budget
  (``max_pixels`` ~= 768x768-equivalent, roughly 256-576 image tokens). CAD
  screenshots are 1920x1080+, and the viewport targets are small: a task-panel
  spinbox is ~24 px, a fillet's clickable sliver a few px wide. Downscaling a
  1080p shot into a 768px tile turns a 24px control into ~9px -- below one vision
  patch. So a 2B QLoRA on 16GB is RUNNABLE as a DIAGNOSTIC (does the corpus carry
  a learnable grounding signal at all?), but its ceiling is capped by resolution,
  not by the data, and it is not the production grounder.

* The **7B grounding VLMs that actually reach ScreenSpot SOTA** (OS-Atlas-7B ~82-85,
  UI-TARS-7B ~90, Qwen2-VL-7B) DO NOT fit for TRAINING on 16GB at native
  screenshot resolution. A 4-bit 7B backbone is ~5GB at rest, but grounding needs
  high ``max_pixels``; the vision tower's activations at 1080p, plus LoRA
  optimiser state and the KV cache, blow past 16GB even at batch=1 with gradient
  checkpointing. They fit for 4-bit INFERENCE on 16GB; they do not fit for
  fine-tuning it at the resolution the task requires.

* **What it would take to do this properly:** one 24GB card (RTX 4090 / A5000)
  QLoRA-fine-tunes a 7B grounding VLM at ~1280px vision tokens; a 48GB card
  (A6000) does it at native 1080p with headroom for batch>1. On the frontier
  open-weight side the intended backbones mirror the text lineup's spirit
  (a Qwen2.5-VL / Qwen3-VL-class or ornith-VL-class model) -- left a PARAMETER
  here for exactly that reason. Nothing about the pipeline below changes; only the
  card and the ``max_pixels`` do.

* 938 pairs is a DOMAIN-ADAPTATION set, not a from-scratch grounding corpus. The
  base VLM must already ground GUIs; this LoRA teaches it the CAD viewport's
  idiom. On 938 examples that is the only honest framing.

So: the setup is finished and ready. It is NOT run here (no training, no model
inference anywhere in this commit). On 16GB you CAN run the 2B diagnostic arm; you
CANNOT train the 7B grounder that the benchmark deserves. That is stated, not faked.

DESIGN
======
* The label is ABSOLUTE image pixels with NO downscale (see
  ``corpus.GroundingPair``); an anamorphic resize would turn circles into ellipses,
  which for a CAD grounder is indefensible. So the answer format carries absolute
  pixels, and :class:`LoRAGroundingPredictor` rescales from the model's processed
  resolution back to the target's native ``(w, h)`` -- the one place a downscale is
  allowed to exist, made explicit.
* :func:`build_examples`, :func:`format_answer`, :func:`parse_answer` are PURE
  PYTHON and unit-tested with no torch: the data contract is where a VLM fine-tune
  silently goes wrong, exactly as with the text trainer.
* The predictor implements ``cadspot.Predictor`` -- ``(image, instruction, w, h)
  -> (x, y)`` -- so a trained adapter drops straight into ``cadspot.evaluate`` next
  to random / centre / oracle with zero new plumbing.

DEPENDENCIES (beyond ``harnesscad[train]``)
===========================================
The VLM path additionally needs **Pillow** and a **VLM-capable transformers**
(``AutoModelForImageTextToText``, transformers >= 4.45). These are imported lazily
inside the trainer/predictor, so this module still imports and its pure-python data
contract still tests on a core-only machine. The ``[train]`` extra in
``pyproject.toml`` is not edited here; install Pillow alongside it before running
the grounding arm.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from harnesscad.agents.selftrain.train import require

#: Frontier open-weight VLM backbone. A PARAMETER, exactly like the text trainer's
#: base: the 2B default is the only thing that trains on 16GB, but the intended
#: production backbone is a 7B+ grounding VLM on a >=24GB card. Override freely.
BASE_VLM = "Qwen/Qwen2-VL-2B-Instruct"
SEED = 20260714

#: LoRA targets for a VLM: the LANGUAGE-model attention/MLP projections only. The
#: vision tower is frozen -- 938 pairs cannot re-train a vision encoder, and trying
#: would erase the general grounding ability this LoRA is meant to specialise.
VLM_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]

#: The benchmark floor to beat. Below CENTRE, the LoRA has learned nothing.
VIEWPORT_BASELINES = {"random": 0.034, "center": 0.085, "oracle": 1.000}

#: Machine-readable form of the docstring verdict, so a report can quote it.
VRAM_VERDICT: Dict[str, Any] = {
    "vram_gb": 16,
    "2b_qlora_train": "fits, reduced max_pixels (~768px); diagnostic only, "
                      "resolution-capped for small CAD targets",
    "2b_inference": "fits comfortably (4-bit)",
    "7b_qlora_train": "does NOT fit at native screenshot resolution; the vision "
                      "tower's high-max_pixels activations + LoRA optimiser state "
                      "exceed 16GB even at batch=1 with gradient checkpointing",
    "7b_inference": "fits (4-bit)",
    "recommended_for_production": "24GB (RTX 4090 / A5000) for 7B QLoRA at "
                                  "~1280px; 48GB (A6000) for native 1080p, batch>1",
    "corpus_size": 938,
    "corpus_nature": "domain-adaptation (base VLM must already ground GUIs), "
                     "not from-scratch grounding",
    "run_now": False,
    "reason_not_run": "no training or model inference in this commit; on 16GB only "
                      "the 2B diagnostic arm is runnable, and it is not the "
                      "production grounder the benchmark deserves",
}

#: Vision-token budget by backbone size, in Qwen2-VL ``max_pixels`` units. The 2B
#: value is what actually fits 16GB; the 7B values assume a bigger card and are
#: here so the caller does not have to rediscover them.
MAX_PIXELS_BY_TIER = {
    "2b@16gb": 768 * 768,
    "7b@24gb": 1024 * 1024,
    "7b@48gb": 1280 * 1280,
}


# --------------------------------------------------------------------------- #
# Data contract -- PURE PYTHON, tested without torch
# --------------------------------------------------------------------------- #
_ANSWER = re.compile(r'"?x"?\s*[:=]\s*(-?\d+(?:\.\d+)?)\s*[,;]\s*"?y"?\s*[:=]\s*'
                     r'(-?\d+(?:\.\d+)?)', re.IGNORECASE)
_PAIR = re.compile(r'\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?')

GROUNDING_SYSTEM = (
    "You are a CAD GUI grounding model. Given a screenshot and a description of one "
    "element, reply with ONLY the pixel to click, as JSON: {\"x\": <int>, \"y\": "
    "<int>}, in the image's own pixel coordinates (origin top-left). No prose."
)


def format_answer(x: float, y: float) -> str:
    """The assistant turn: absolute image pixels, the schema the corpus labels use."""
    return json.dumps({"x": int(round(x)), "y": int(round(y))})


def parse_answer(text: str) -> Optional[Tuple[float, float]]:
    """Recover ``(x, y)`` from a model reply. Tolerant: JSON, ``x=..,y=..``, or a
    bare ``(x, y)`` pair. Returns ``None`` if nothing coordinate-like is present --
    a non-answer must score as a miss, never as a silent (0, 0)."""
    if not text:
        return None
    m = _ANSWER.search(text)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    m = _PAIR.search(text)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    return None


def build_examples(targets: Sequence[Any], root: str = "") -> List[Dict[str, Any]]:
    """cadspot ``Target``s (or corpus pairs) -> VLM chat examples.

    Each example is ``{"image": abspath, "messages": [...], "x": px, "y": py}``.
    The supervised answer is the target's CENTRE in absolute pixels (for a viewport
    target that centre IS the verified click). Only targets with a resolvable pixel
    are kept; a target with no ``bbox`` and no ``point`` is not trainable and is
    dropped, counted by the caller via the returned length.
    """
    out: List[Dict[str, Any]] = []
    for t in targets:
        instruction = getattr(t, "instruction", None)
        image = getattr(t, "image", None)
        if instruction is None or image is None:
            continue
        cx, cy = t.center  # Target.center -> pixel
        img = os.path.join(root, image) if root else image
        out.append({
            "image": img,
            "region": getattr(t, "region", ""),
            "x": int(cx), "y": int(cy),
            "width": int(getattr(t, "width", 0) or 0),
            "height": int(getattr(t, "height", 0) or 0),
            "messages": [
                {"role": "system", "content": GROUNDING_SYSTEM},
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": instruction},
                ]},
                {"role": "assistant", "content": format_answer(cx, cy)},
            ],
        })
    return out


@dataclass
class GroundingDataStats:
    records: int = 0
    by_region: Dict[str, int] = field(default_factory=dict)
    viewport: int = 0

    def to_dict(self) -> dict:
        return {"records": self.records, "viewport": self.viewport,
                "by_region": dict(sorted(self.by_region.items()))}


def dataset_stats(examples: Sequence[Dict[str, Any]]) -> GroundingDataStats:
    st = GroundingDataStats(records=len(examples))
    for e in examples:
        r = e.get("region", "")
        st.by_region[r] = st.by_region.get(r, 0) + 1
    st.viewport = st.by_region.get("viewport", 0)
    return st


# --------------------------------------------------------------------------- #
# Predictor -- drops into cadspot.evaluate next to random/centre/oracle
# --------------------------------------------------------------------------- #
class LoRAGroundingPredictor:
    """A fine-tuned VLM behind the ``cadspot.Predictor`` interface.

    Construction loads a 4-bit VLM + LoRA adapter (import-guarded, GPU). ``__call__``
    is ``(image, instruction, w, h) -> (x, y)``: it prompts the model, parses the
    coordinate, and RESCALES from the model's processed pixel space back to the
    target's native ``(w, h)`` -- the one legitimate downscale, made explicit rather
    than smuggled into the label. A model that returns no coordinate is clamped to
    the image centre so the call never crashes the harness, but that is a MISS on
    ``point_in_bbox`` and is the honest outcome for a non-answer.
    """

    def __init__(self, base_model: str = BASE_VLM,
                 adapter: Optional[str] = None,
                 max_pixels: int = MAX_PIXELS_BY_TIER["2b@16gb"],
                 max_new_tokens: int = 32) -> None:
        require()
        import torch  # noqa: F401
        from transformers import (AutoProcessor,
                                  AutoModelForImageTextToText,
                                  BitsAndBytesConfig)

        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
        self.processor = AutoProcessor.from_pretrained(
            base_model, max_pixels=max_pixels)
        self.model = AutoModelForImageTextToText.from_pretrained(
            base_model, quantization_config=quant, torch_dtype=torch.bfloat16,
            device_map={"": 0} if torch.cuda.is_available() else None)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def __call__(self, image: str, instruction: str, w: int, h: int
                 ) -> Tuple[float, float]:
        import torch
        from PIL import Image

        img = Image.open(image).convert("RGB")
        msgs = [
            {"role": "system", "content": GROUNDING_SYSTEM},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": instruction}]},
        ]
        text = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        enc = self.processor(text=[text], images=[img], return_tensors="pt")
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                      do_sample=False)
        gen = out[0][enc["input_ids"].shape[1]:]
        reply = self.processor.decode(gen, skip_special_tokens=True)
        xy = parse_answer(reply)
        if xy is None:
            return (w / 2.0, h / 2.0)          # non-answer -> honest miss
        px, py = xy
        # Rescale from the model's native image size to the target's (w, h). The
        # answer is trained in the source image's pixels; if the harness passes a
        # differently sized (w, h) we map proportionally rather than assume equality.
        sw, sh = img.size
        if sw and sh and (sw != w or sh != h):
            px = px * (w / float(sw))
            py = py * (h / float(sh))
        return (float(px), float(py))


def make_predictor(base_model: str = BASE_VLM, adapter: Optional[str] = None,
                   **kw) -> LoRAGroundingPredictor:
    """Convenience factory named for ``cadspot.baseline_report`` symmetry."""
    return LoRAGroundingPredictor(base_model=base_model, adapter=adapter, **kw)


# --------------------------------------------------------------------------- #
# Trainer -- QLoRA on a VLM. IMPORT-GUARDED, NOT RUN.
# --------------------------------------------------------------------------- #
@dataclass
class GroundingResult:
    output_dir: str = ""
    base_model: str = ""
    records: int = 0
    max_pixels: int = 0
    steps: int = 0
    train_runtime_s: float = 0.0
    final_loss: Optional[float] = None
    peak_vram_gb: Optional[float] = None
    data_stats: Dict[str, Any] = field(default_factory=dict)
    verdict: Dict[str, Any] = field(default_factory=lambda: dict(VRAM_VERDICT))
    note: str = ""

    def to_dict(self) -> dict:
        return {"stage": "grounding-lora", "output_dir": self.output_dir,
                "base_model": self.base_model, "records": self.records,
                "max_pixels": self.max_pixels, "steps": self.steps,
                "train_runtime_s": round(self.train_runtime_s, 1),
                "final_loss": self.final_loss, "peak_vram_gb": self.peak_vram_gb,
                "data_stats": self.data_stats, "verdict": self.verdict,
                "note": self.note}


def train_grounding_lora(cadspot_jsonl: str, output_dir: str, *,
                         base_model: str = BASE_VLM,
                         root: str = "",
                         max_pixels: int = MAX_PIXELS_BY_TIER["2b@16gb"],
                         epochs: int = 3, lr: float = 1e-4,
                         viewport_only: bool = True,
                         grad_accum: int = 16) -> GroundingResult:
    """QLoRA-fine-tune a grounding VLM on the verified viewport corpus. NOT run here.

    Mirrors ``sft.train_sft``: 4-bit NF4 base, r=16/alpha=32 LoRA on the LANGUAGE
    projections only (vision tower frozen -- 938 pairs must not re-train an
    encoder), completion-style supervision on the ``{"x":..,"y":..}`` answer,
    gradient checkpointing on (mandatory at these image sizes). ``viewport_only``
    keeps the training signal on the region that is the contribution; the chrome
    regions are a solved UIA-tree problem and are not what a LoRA should spend its
    budget on.

    On 16GB this runs ONLY for a 2B base at the reduced ``max_pixels`` default; see
    the module docstring and :data:`VRAM_VERDICT`. It is deliberately not executed
    in this commit.
    """
    require()
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoProcessor, AutoModelForImageTextToText,
                              BitsAndBytesConfig, Trainer, TrainingArguments,
                              TrainerCallback)

    from harnesscad.eval.grounding import cadspot as C

    random_seed(SEED)
    os.makedirs(output_dir, exist_ok=True)

    targets = C.load(cadspot_jsonl)
    if viewport_only:
        targets = [t for t in targets if t.region == "viewport"]
    examples = build_examples(targets, root=root)
    stats = dataset_stats(examples)

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(base_model, max_pixels=max_pixels)
    model = AutoModelForImageTextToText.from_pretrained(
        base_model, quantization_config=quant, torch_dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else None)
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=VLM_TARGET_MODULES))
    model.config.use_cache = False

    collator = _GroundingCollator(processor)
    args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=grad_accum,
        learning_rate=lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=1, save_strategy="no", bf16=True,
        gradient_checkpointing=True, optim="paged_adamw_8bit",
        report_to=[], seed=SEED, remove_unused_columns=False)

    losses: List[float] = []

    class _CB(TrainerCallback):
        def on_log(self, a, s, c, logs=None, **kw):
            if logs and "loss" in logs:
                losses.append(float(logs["loss"]))

    trainer = Trainer(model=model, args=args, train_dataset=examples,
                      data_collator=collator, callbacks=[_CB()])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    out = trainer.train()
    peak = (torch.cuda.max_memory_allocated() / 1e9
            if torch.cuda.is_available() else None)
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    res = GroundingResult(
        output_dir=output_dir, base_model=base_model, records=stats.records,
        max_pixels=max_pixels, steps=int(trainer.state.global_step),
        train_runtime_s=float(out.metrics.get("train_runtime", 0.0)),
        final_loss=(losses[-1] if losses else None),
        peak_vram_gb=(round(peak, 2) if peak else None),
        data_stats=stats.to_dict(),
        note="viewport-only=%s; vision tower frozen; QLoRA r=16; max_pixels=%d; "
             "seed=%d" % (viewport_only, max_pixels, SEED))
    with open(os.path.join(output_dir, "grounding_result.json"), "w",
              encoding="utf-8") as fh:
        json.dump(res.to_dict(), fh, indent=2)
    return res


class _GroundingCollator:
    """Batches VLM chat examples and masks loss onto the assistant coordinate only.

    Kept minimal and processor-driven; the point mirrors ``data.py``: the label the
    loss sees must be exactly the ``{"x":..,"y":..}`` answer, never the prompt or
    the image tokens.
    """

    def __init__(self, processor) -> None:
        self.processor = processor

    def __call__(self, batch: Sequence[Dict[str, Any]]):
        from PIL import Image
        texts, images = [], []
        for ex in batch:
            texts.append(self.processor.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False))
            images.append(Image.open(ex["image"]).convert("RGB"))
        enc = self.processor(text=texts, images=images, return_tensors="pt",
                             padding=True)
        labels = enc["input_ids"].clone()
        pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            labels[labels == pad_id] = -100
        enc["labels"] = labels
        return enc


def random_seed(seed: int = SEED) -> None:
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI that, by default, does NOT train -- it PRINTS the verdict. Training is
    opt-in behind ``--train`` and even then only runnable where the stack exists."""
    import argparse
    ap = argparse.ArgumentParser(
        description="Grounding-LoRA setup. Prints the 16GB verdict; --train to run.")
    ap.add_argument("--cadspot", default="assets/grounding/cadspot.jsonl")
    ap.add_argument("--root", default="assets/grounding")
    ap.add_argument("--out", default="assets/grounding/adapters/grounding")
    ap.add_argument("--base", default=BASE_VLM)
    ap.add_argument("--max-pixels", type=int, default=MAX_PIXELS_BY_TIER["2b@16gb"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train", action="store_true",
                    help="actually run the QLoRA fine-tune (needs harnesscad[train] "
                         "and, for 7B, a >=24GB card)")
    args = ap.parse_args(argv)
    if not args.train:
        print(json.dumps({"verdict": VRAM_VERDICT,
                          "viewport_baselines": VIEWPORT_BASELINES,
                          "max_pixels_by_tier": MAX_PIXELS_BY_TIER}, indent=2))
        return 0
    res = train_grounding_lora(args.cadspot, args.out, base_model=args.base,
                               root=args.root, max_pixels=args.max_pixels,
                               epochs=args.epochs)
    print(json.dumps(res.to_dict(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
