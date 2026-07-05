"""Reproducible multimodal generation provenance."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json


@dataclass(frozen=True)
class GenerationManifest:
    model: str
    checkpoint_digest: str
    prompt_digest: str
    image_digest: str | None
    temperature: float
    top_p: float
    seed: int | None
    maximum_tokens: int
    provider_version: str
    finish_reason: str

    @classmethod
    def create(cls, *, model, checkpoint: bytes, prompt: str, image: bytes | None,
               temperature, top_p, seed, maximum_tokens, provider_version,
               finish_reason):
        digest = lambda value: hashlib.sha256(value).hexdigest()
        return cls(model, digest(checkpoint), digest(prompt.encode()),
                   digest(image) if image is not None else None,
                   float(temperature), float(top_p), seed, int(maximum_tokens),
                   provider_version, finish_reason)

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()
