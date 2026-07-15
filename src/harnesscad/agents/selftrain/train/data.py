"""Turn the certified JSONL corpora into tokenised, prompt-masked training tensors.

ONE formatting decision governs everything: a training example must look, byte for
byte, like an inference example, or the fine-tune optimises a distribution the
model is never asked to produce. So every record here is rendered through the SAME
two-message prompt the pressure experiment used at inference --
``eval.pressure.prompts.SYSTEM_PROMPT`` + ``user_prompt(brief_text)`` -- and the
assistant turn is the certified op stream. The stored ``prompt`` field in the JSONL
is only the bare brief text; the system prompt (the op schema, the feasibility
rule) is re-attached here so train and test share it.

COMPLETION-ONLY MASKING. The recipe says it and it matters: loss is computed on
the assistant tokens only. Training the model to reproduce the (fixed, identical)
system prompt would spend the tiny gradient budget on text the model never has to
generate. TRL's ``DataCollatorForCompletionOnlyLM`` masks the prompt for us,
keyed on the assistant header the chat template emits.

All heavy imports are deferred so the module imports on a core-only machine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _messages(brief_text: str, completion: Optional[str] = None) -> List[Dict[str, str]]:
    """The exact two-turn (optionally three-turn) chat the model saw at inference."""
    from harnesscad.eval.pressure import prompts

    msgs = [{"role": "system", "content": prompts.SYSTEM_PROMPT},
            {"role": "user", "content": prompts.user_prompt(brief_text)}]
    if completion is not None:
        msgs.append({"role": "assistant", "content": completion})
    return msgs


# --------------------------------------------------------------------------- #
# RFT / SFT
# --------------------------------------------------------------------------- #
@dataclass
class DataStats:
    records: int = 0
    desirable: Optional[int] = None
    undesirable: Optional[int] = None
    briefs: int = 0
    mean_completion_chars: float = 0.0

    def to_dict(self) -> dict:
        d = {"records": self.records, "briefs": self.briefs,
             "mean_completion_chars": round(self.mean_completion_chars, 1)}
        if self.desirable is not None:
            d["desirable"] = self.desirable
            d["undesirable"] = self.undesirable
        return d


def sft_dataset(path: str, tokenizer):
    """Load an RFT/SFT JSONL into a prompt/completion HF ``Dataset``.

    Returns ``(dataset, response_template_ids, stats)``. The dataset has two
    columns -- ``prompt`` (the chat-templated system+user turns, with a generation
    prompt) and ``completion`` (the assistant op stream). TRL's ``SFTTrainer`` with
    ``completion_only_loss=True`` masks the prompt exactly on this schema, so loss
    falls only on the op stream the model must actually produce. That is more robust
    than masking a single flattened ``text`` field, whose boundary TRL would have to
    re-discover.
    """
    from datasets import Dataset

    rows = _read_jsonl(path)
    prompts_c: List[str] = []
    completions: List[str] = []
    briefs = set()
    total_chars = 0
    for r in rows:
        brief_text = r.get("prompt") or r.get("brief_text") or ""
        completion = r["completion"]
        prompt_text = tokenizer.apply_chat_template(
            _messages(brief_text), tokenize=False, add_generation_prompt=True)
        prompts_c.append(prompt_text)
        completions.append(completion)
        briefs.add(r.get("brief_id", brief_text))
        total_chars += len(completion)
    stats = DataStats(records=len(rows), briefs=len(briefs),
                      mean_completion_chars=(total_chars / len(rows)) if rows else 0.0)
    ds = Dataset.from_dict({"prompt": prompts_c, "completion": completions})
    return ds, _response_template_ids(tokenizer), stats


def _response_template_ids(tokenizer) -> List[int]:
    """Token ids of the assistant-turn header the chat template emits.

    Qwen2.5 uses ``<|im_start|>assistant\\n``. We derive it from the template
    rather than hardcoding, so a tokenizer change cannot silently break masking.
    """
    with_assistant = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True)
    without = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=False)
    header = with_assistant[len(without):] if with_assistant.startswith(without) else \
        "<|im_start|>assistant\n"
    ids = tokenizer.encode(header, add_special_tokens=False)
    return ids


# --------------------------------------------------------------------------- #
# KTO
# --------------------------------------------------------------------------- #
def kto_dataset(path: str, tokenizer):
    """Load a KTO JSONL into the ``{prompt, completion, label}`` schema TRL wants.

    ``prompt`` is the chat-templated system+user turns with a generation prompt;
    ``completion`` is the assistant op stream; ``label`` is the oracle's desirable
    bit. TRL's ``KTOTrainer`` consumes exactly these three columns.
    """
    from datasets import Dataset

    rows = _read_jsonl(path)
    prompts_c: List[str] = []
    completions: List[str] = []
    labels: List[bool] = []
    briefs = set()
    des = 0
    for r in rows:
        brief_text = r.get("prompt") or r.get("brief_text") or ""
        prompt_text = tokenizer.apply_chat_template(
            _messages(brief_text), tokenize=False, add_generation_prompt=True)
        prompts_c.append(prompt_text)
        completions.append(r["completion"])
        lab = bool(r["desirable"])
        labels.append(lab)
        des += int(lab)
        briefs.add(r.get("brief_id", brief_text))
    stats = DataStats(records=len(rows), briefs=len(briefs), desirable=des,
                      undesirable=len(rows) - des)
    ds = Dataset.from_dict({"prompt": prompts_c, "completion": completions,
                            "label": labels})
    return ds, stats


def dpo_dataset(path: str, tokenizer):
    """Load a DPO JSONL into ``{prompt, chosen, rejected}`` (chat-templated prompt)."""
    from datasets import Dataset

    rows = _read_jsonl(path)
    prompts_c, chosen, rejected = [], [], []
    briefs = set()
    for r in rows:
        brief_text = r.get("prompt") or r.get("brief_text") or ""
        prompt_text = tokenizer.apply_chat_template(
            _messages(brief_text), tokenize=False, add_generation_prompt=True)
        prompts_c.append(prompt_text)
        chosen.append(r["chosen"])
        rejected.append(r["rejected"])
        briefs.add(r.get("brief_id", brief_text))
    stats = DataStats(records=len(rows), briefs=len(briefs))
    ds = Dataset.from_dict({"prompt": prompts_c, "chosen": chosen,
                            "rejected": rejected})
    return ds, stats


def messages_for(brief_text: str) -> List[Dict[str, str]]:
    """Public helper: the inference message list for a brief (system+user)."""
    return _messages(brief_text)
