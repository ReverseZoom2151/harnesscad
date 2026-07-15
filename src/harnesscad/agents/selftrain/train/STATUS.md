# selftrain/train -- STATUS

**Training apparatus is READY. It has NOT been run. It runs the moment the
frontier-expanded dataset exists, and not before.**

Nothing in this subpackage has been trained and no model has been run to produce
it. Every module import-guards its heavy dependencies (`harnesscad[train]`), and the
whole core suite imports and tests with no torch, no peft, no trl, no bitsandbytes.

## Why it is not run yet

The certified corpora exist (commit `eef643b`): **13 RFT / 66 DPO / 208 KTO** from
223 real trajectories. Thirteen RFT records is not a fine-tuning set -- it is a
smoke test. Training a 7B QLoRA on 13 op streams would produce a number, and the
number would be meaningless. So the apparatus is finished and parked.

The corpus is grown by **re-sampling the FRONTIER lineup** on the training briefs
(never the 16 held-out eval briefs) and re-certifying every draw with
`ledger.certify` -- the gate AND envelope AND shape conjunction, **never the
verifier fleet**. The old model lineup is DELETED; it is not run to grow anything.

## Lineup (base + candidates)

* **Text planner base / candidate lineup:** the frontier open-weight coders
  **qwen3.6 (27b / 35b)** and **ornith (9b / 35b)**. The base is a PARAMETER on
  every entry point (`sft.BASE_MODEL` default is the prior `Qwen2.5-Coder-7B-Instruct`,
  overridable via `--model` / `model_name=`); update the default to the frontier
  base once chosen.
* **Sampling lineup for growing the corpus** (`generate.py`): the same non-reasoning
  frontier coders. Reasoning models are refused by name -- their think-budget
  behaviour is miscounted by a bare-JSON harness.

## What is ready

| piece | module | state |
|-------|--------|-------|
| Stage 1 RFT / STaR (QLoRA SFT, r=16, 3 epochs, completion-only masking) | `sft.py` | ready, not run |
| Stage 2 KTO (unpaired oracle labels, balanced weights) | `kto.py` | ready, not run |
| Stage 3 DPO (robust loss, label_smoothing = MEASURED oracle error, REQUIRED) | `dpo.py` | ready, not run |
| Data pipeline (chat-templated, prompt-masked; RFT/KTO/DPO loaders) | `data.py` | ready |
| Corpus growth (sample frontier lineup, certify, emit RFT+KTO) | `generate.py` | ready, not run |
| Eval harness: base@1 vs finetune@1 vs Oracle-BoN@N at matched compute | `evaluate.py`, `experiment.py` | ready, not run |
| Grounding-LoRA setup + 16GB verdict | `grounding.py` | ready; see verdict below |

Every stage's `base_model` / `model_name` is a parameter; the eval harness is
parameterised by base + adapter + BoN N-list, so any lineup member can be scored
without editing code.

## The two honest rules, encoded

1. **Matched-compute comparison is the point, not a formality.** The eval harness
   scores `finetune@1` (1 generation/brief) against `oracle-bon@N` (N generations)
   off the SAME base, with Wilson CIs and McNemar's exact test on paired outcomes.
   **If Oracle-BoN at N=3 already lands near 70%, a fine-tune reaching 50% is a
   regression dressed as a result** -- and the harness reports it as such rather
   than hiding it. This is in `evaluate.py`'s and `experiment.py`'s docstrings and
   in the emitted report's `note` field.
2. **The labeller is `ledger.certify`, never the verifier fleet.** Training data
   and every eval arm are graded by the gate+envelope+shape conjunction. The fleet
   is the loop's feedback channel, not its ground truth; a model trained on fleet
   labels learns to reject washers.

## Grounding-LoRA: the 16GB verdict (do NOT fake, NOT trained)

938 verified `(screenshot, description) -> (x, y)` pairs (`eval/grounding/corpus`),
scored on `eval/grounding/cadspot.py`. Viewport baselines: **random 0.034, centre
0.085, oracle 1.000** -- 0.085 is the floor to beat.

**16GB cannot train the grounder the benchmark deserves.** In full in
`grounding.py`'s docstring and `grounding.VRAM_VERDICT`:

* A **2B-class VLM** (Qwen2-VL-2B / ShowUI-2B) QLoRA-trains and infers inside 16GB,
  but only at a reduced vision-token budget (~768px). CAD screenshots are 1080p+
  and viewport targets are tiny (a ~24px spinbox, a few-px fillet sliver);
  downscaling to fit destroys exactly the small-target precision the viewport split
  needs. Runnable as a **diagnostic**, resolution-capped, not the production grounder.
* The **7B grounding VLMs that reach ScreenSpot SOTA** (OS-Atlas-7B, UI-TARS-7B,
  Qwen2-VL-7B) **do NOT fit for training on 16GB** at native screenshot resolution:
  the vision tower's high-`max_pixels` activations plus LoRA optimiser state exceed
  16GB even at batch=1 with gradient checkpointing. They fit for 4-bit *inference*.
* **What it takes:** one **24GB** card (RTX 4090 / A5000) for 7B QLoRA at ~1280px;
  **48GB** (A6000) for native 1080p, batch>1. Backbone is a PARAMETER (`BASE_VLM`),
  intended to track a frontier open-weight VLM (Qwen2.5-VL / ornith-VL class).

The setup (`build_examples`, `LoRAGroundingPredictor` -> plugs straight into
`cadspot.evaluate`, `train_grounding_lora`) is finished and import-guarded. It is
**not run**; `grounding.main` PRINTS the verdict by default and only trains behind
`--train`.

## To run it (later, once the corpus is frontier-grown)

```
python -m harnesscad.agents.selftrain.train.generate --models qwen3.6:35b,ornith:35b   # grow corpus
python -m harnesscad.agents.selftrain.train.sft   --data assets/selftrain/rft.jsonl --model <frontier-base>
python -m harnesscad.agents.selftrain.train.kto   --data assets/selftrain/kto.jsonl --init-adapter <rft-adapter>
python -m harnesscad.agents.selftrain.train.dpo   --data assets/selftrain/dpo.jsonl --label-smoothing <measured-eps>
python -m harnesscad.agents.selftrain.train.experiment --base <frontier-base> --adapter <adapter> --bon 3,5
```
