"""Content-addressed cache for frozen-backbone embeddings."""

from __future__ import annotations

import hashlib
import json


def embedding_key(data: bytes, *, checkpoint: str, preprocessing: dict,
                  precision: str = "float32") -> str:
    metadata = json.dumps({
        "checkpoint": checkpoint, "preprocessing": preprocessing,
        "precision": precision,
    }, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(metadata + b"\0" + data).hexdigest()


class EmbeddingCache:
    def __init__(self, storage=None):
        self.storage = storage if storage is not None else {}

    def get_or_compute(self, data: bytes, encoder, *, checkpoint: str,
                       preprocessing: dict, precision="float32"):
        key = embedding_key(data, checkpoint=checkpoint,
                            preprocessing=preprocessing, precision=precision)
        if key in self.storage:
            return self.storage[key], True, key
        value = encoder(data)
        self.storage[key] = value
        return value, False, key
